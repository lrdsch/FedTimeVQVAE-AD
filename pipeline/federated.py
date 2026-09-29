"""STEP 2 — in-process federated orchestrator for Stage 1 (codebook federation).

Implements the true-FL Stage-1 of Method A:

  * client = entity. One Stage1VQVAE per client, materialized once, optimizer
    built once and KEPT ALIVE across rounds (encoder/decoder train locally).
  * the VQ codebook is GLOBAL and frozen during local training
    (`collect_stats_only=True`): each client accumulates raw per-code sufficient
    statistics (n_j, m_j) against the broadcast codebook WITHOUT mutating it.
  * each round: clients upload (n_j, m_j); the server aggregates them
    (Σ n_j, Σ m_j → e_j = M/smoothed(N) = exact pooled k-means M-step,
    Proposition 1), optionally revives globally-dead codes, and broadcasts the
    new codebook back to every client.

No Flower, no subprocesses, no message bus — a single Python process holding a
list of client models. This matches the scale (client=entity, a handful of
silos) and keeps the codebook aggregation (custom, not weight-FedAvg) trivial.

Because the codebook is frozen within a round, the existing commitment loss
(MSE(z, sg(broadcast_code))) already anchors each local encoder to the global
vocabulary — i.e. the FedProto anchor is implicit here; the explicit anchor loss
(STEP 4) only matters once the codebook is allowed to move locally.

CAVEAT — that paragraph describes merge='suffstat' (and 'fedavg'). Under merge='local'
the commitment target is each client's OWN EMA codebook, which chases that client's own
encoder, so the implicit global anchor is GONE: the only cross-client signal left is
whatever `fed_encoder` shares. That is the point of the mode, and also its main risk (the
encoder is free to drift between aggregations).

The ENCODER can be federated on top of that, by one of three algorithms selected with
`enc_fed_algo` — FedAvg (weight-space), FedProx (weight-space + a local proximal anchor)
or FedProto (function-space: no weight averaging, per-code prototypes only). See the
"encoder federation" section below and documentation/research/FED_ENCODER_ALGOS.md; the arms are
`federated_enc_{fedavg,fedprox,fedproto,commoninit}` in pipeline/federated_eval.py.

Smoke run:
    python pipeline/federated.py --rounds 3 --local-epochs 1
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))    # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))           # pipeline/ (stage1/stage2 use bare imports)

from config import Config, apply_dataset_overrides, apply_env_overrides
from data import make_dataloaders, _build_loader
from model.vector_quantizer import SharedVectorQuantizer
from stage1 import Stage1VQVAE
from stage2 import Stage2System
from utils import resolve_path, force_utf8_stdout

# ─── client selection (entity ids are dataset-specific: fed_0.. vs uni_00..) ───

def _dataset_root(cfg: Config) -> Path:
    return resolve_path(cfg.paths.raw_data) / cfg.dataset.name


def all_clients(cfg: Config) -> list[str]:
    """Every entity on disk, sorted. Beats hard-coding `fed_0..fed_5`: the
    univariate benchmark names its clients `uni_00..` and the count is a build
    parameter (47 as of the 2026-07-16 rebuild), so it is read from disk."""
    return sorted(p.stem for p in (_dataset_root(cfg) / "train").glob("*.npy"))


def load_clusters(cfg: Config) -> dict[str, list[str]]:
    """`{cluster: [entity_id, ...]}` for a clustered build (toy_fed_uni), else `{}`."""
    path = _dataset_root(cfg) / "clusters.json"
    if not path.exists():
        return {}
    import json as _json
    return _json.loads(path.read_text(encoding="utf-8"))


def resolve_clients(cfg: Config, clients: str | None, cluster: str | None) -> list[str]:
    """`--cluster M1_rotary` (one machine type) selects the federation; `--clients a,b`
    NARROWS it; neither given ⇒ every entity.

    Federating a cluster is the intended use of the univariate benchmark: within a
    cluster the clients share a morphology and a base period, so a shared codebook
    and prior body are meaningful. Across clusters they are not.

    `--clients` used to be IGNORED whenever `--cluster` was given. Intersecting instead
    lets one `local` model be trained per process while keeping the cluster in the
    artefact path — which is what makes fine-grained scheduling possible: the `local`
    arm is 78 independent models, and packing them one-per-job leaves no idle slots at
    the tail of a sweep. `centralized` still needs the whole cluster in one process,
    so it simply omits `--clients`.
    """
    if cluster:
        clusters = load_clusters(cfg)
        if not clusters:
            raise SystemExit(f"{cfg.dataset.name} has no clusters.json — --cluster is unavailable.")
        if cluster not in clusters:
            raise SystemExit(f"unknown cluster {cluster!r}; known: {sorted(clusters)}")
        members = list(clusters[cluster])
        if clients:
            want = [c.strip() for c in clients.split(",") if c.strip()]
            missing = [c for c in want if c not in members]
            if missing:
                raise SystemExit(f"--clients {missing} are not in cluster {cluster!r}: {members}")
            return want
        return members
    if clients:
        return [c.strip() for c in clients.split(",") if c.strip()]
    return all_clients(cfg)


@dataclass
class ClientState:
    entity_id: str
    cfg: Config
    model: Stage1VQVAE
    opt: torch.optim.Optimizer
    data: object                       # Dataloaders
    # Per-client adaptive loss scale, kept alive across rounds like `opt`. Never aggregated.
    scaler: torch.amp.GradScaler = field(default_factory=lambda: _make_scaler())
    # Optional warmup+cosine over the WHOLE run (see `enc_sched`). None = constant LR,
    # which is what every federated arm on disk was trained with.
    sched: "torch.optim.lr_scheduler.LambdaLR | None" = None
    # B1 alignment (probe-based, permutation-free): SHARED public probe + broadcast
    # consensus encoding `z_ref`; local training pulls this client's probe-encoding to it.
    probe: "torch.Tensor | None" = None
    z_ref: "torch.Tensor | None" = None
    align_weight: float = 0.0
    # ── ENCODER-federation extras (arms federated_enc_fedprox / federated_enc_fedproto).
    # Zero/None for every other arm ⇒ the local-training loop takes the exact old path.
    prox_mu: float = 0.0                          # FedProx μ
    prox_form: str = "loss"                       # "loss" (in-objective) | "decoupled" (post-step)
    prox_ref: "dict[str, torch.Tensor] | None" = None   # w^t (start-of-round global encoder), fp32
    proto_weight: float = 0.0                     # FedProto λ
    proto_code_weight: str = "uniform"            # per-code loss weighting: uniform | count
    protos: "list[torch.Tensor] | None" = None    # per RVQ stage: (K, D) global prototypes
    proto_mask: "list[torch.Tensor] | None" = None      # per stage: (K,) bool — code has support
    # Per-round telemetry, written by _local_train_stage1.
    last_prox: float = float("nan")               # mean ‖w − w^t‖²
    last_proto: float = float("nan")              # mean prototype term
    last_obj: float = float("nan")                # mean PENALISED objective (vs the task-only mean)
    last_prox_ratio: float = float("nan")         # ‖μ(w−w^t)‖ / ‖∇L_task‖ — 'loss' form only
    last_prox_pull: float = float("nan")          # 1−(1−lr·μ)^steps — 'decoupled' form only
    last_steps: int = 0
    last_skipped: int = 0                         # fp16 steps the GradScaler threw away

    @property
    def vqs(self) -> list[SharedVectorQuantizer]:
        # One inner VQ for a single codebook; one PER STAGE for a Residual-VQ. The
        # federation loops over these — each stage merges by the same k-FED M-step.
        return self.model.quantizer.inner_vqs()

    @property
    def vq(self) -> SharedVectorQuantizer:
        return self.model.quantizer.inner_vqs()[0]    # stage-0 / the single shared VQ


# ─── helpers ─────────────────────────────────────────────────────────────────

def _amp_dtype() -> torch.dtype | None:
    """Autocast dtype for the (compute-heavy) training forwards, or None for fp32.

    `FEDVQ_AMP = auto | fp16 | bf16 | off`  (default: auto)

    `auto` picks **fp16 below Ampere and bf16 from Ampere on**. This matters: Turing
    (sm_75, e.g. the Quadro RTX 8000 on this node) has fp16 tensor cores but *no*
    bf16 ones, so bf16 there is emulated and measurably SLOWER than plain fp32 —
    measured on this node: matmul 4096^3 fp32 22.9ms / fp16 3.4ms / bf16 40.2ms.

    Do NOT gate this on `torch.cuda.is_bf16_supported()`: it returns True on Turing
    (emulation counts as "supported"). Compute capability is the only honest guard.

    Applies ONLY to the neural forwards — the VQ sufficient-statistic accumulation
    stays fp32 (autocast disabled inside the quantizer) so Prop. 1 remains exact,
    and eval / metric code is left in fp32 for precision.

    `FEDVQ_BF16=0` is still honoured as a legacy alias for `FEDVQ_AMP=off`.
    """
    if not torch.cuda.is_available():
        return None
    if os.environ.get("FEDVQ_BF16", "1") == "0":          # legacy kill-switch
        return None
    mode = os.environ.get("FEDVQ_AMP", "auto").lower()
    if mode in {"off", "0", "false", "no", "fp32"}:
        return None
    if mode == "fp16":
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    if mode != "auto":
        raise ValueError(f"FEDVQ_AMP must be auto|fp16|bf16|off, got {mode!r}")
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def _amp():
    dtype = _amp_dtype()
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_scaler() -> torch.amp.GradScaler:
    """Loss scaler for the local client loops. fp16 needs it (gradients underflow
    without it); bf16 and fp32 do not, and `enabled=False` is a pure pass-through.

    The scale is per-client adaptive state. It is deliberately NOT part of anything
    the server aggregates — FedAvg / the suff-stat merge touch model params only.
    """
    return torch.amp.GradScaler("cuda", enabled=_amp_dtype() is torch.float16)


def _client_devices(n: int) -> list[torch.device]:
    """Where each client's model lives. OPT-IN multi-GPU, DEFAULT OFF.

    With `FEDVQ_DEVICES` unset this returns `[cuda] * n` — i.e. exactly the single
    device the code used before, so every code path is unchanged. Set
    `FEDVQ_DEVICES=0,1` (or `all`) to round-robin the clients over those GPUs.

    This is a MEMORY-CAPACITY win, not a wall-clock one: clients are still trained
    strictly one at a time (`for ci, c in enumerate(clients)`), each preceded by its
    own `torch.manual_seed`, and the server merge is an exact sum on CPU. Placement
    therefore cannot change the result. Do NOT parallelize the client loop with
    threads or CUDA streams to chase wall-clock — that WOULD reorder the RNG stream
    and break the determinism the ablations rely on.

    SCOPE: honoured by `federated_stage1` / `federated_stage2` (i.e. `pipeline/federated.py`).
    `pipeline/federated_eval.py` builds its own single-device baselines and has NOT been
    adapted; do not set FEDVQ_DEVICES for the RQ1 eval harness.

    NOT VALIDATED (2026-07-09): shipped default-off. See documentation/operations/SPEEDUP_PLAN.md.
    """
    if not torch.cuda.is_available():
        return [torch.device("cpu")] * n
    spec = os.environ.get("FEDVQ_DEVICES", "").strip()
    if not spec:
        return [torch.device("cuda")] * n            # unchanged path
    ids = (list(range(torch.cuda.device_count())) if spec.lower() == "all"
           else [int(x) for x in spec.split(",") if x.strip() != ""])
    if not ids:
        raise ValueError(f"FEDVQ_DEVICES={spec!r} resolved to no device ids")
    for i in ids:
        if i >= torch.cuda.device_count():
            raise ValueError(f"FEDVQ_DEVICES={spec!r}: cuda:{i} not visible "
                             f"({torch.cuda.device_count()} device(s))")
    devs = [torch.device(f"cuda:{ids[i % len(ids)]}") for i in range(n)]
    print(f"[fed] client placement: " + ", ".join(f"{i}->{d}" for i, d in enumerate(devs)))
    return devs


def _perplexity_from_counts(N: torch.Tensor) -> float:
    p = N / N.sum().clamp_min(1.0)
    nz = p > 0
    return float(torch.exp(-(p[nz] * torch.log(p[nz])).sum()))


def _build_client(base_cfg: Config, entity_id: str, device: torch.device,
                  anchor_weight: float = 0.0, collect_stats: bool = True,
                  local_codebook: bool = False) -> ClientState:
    cfg = copy.deepcopy(base_cfg)
    cfg.dataset.entity_id = entity_id
    data = make_dataloaders(cfg, stage="stage1")

    example = next(iter(data.train_loader))["inputs"][:1].cpu()
    model = Stage1VQVAE(cfg)
    model.eval()                                   # materialize without touching the VQ
    with torch.no_grad():
        model(example)
    model.to(device)

    # collect_stats=True  → suff-stat merge: codebook frozen locally, accumulate (n_j, m_j).
    # collect_stats=False → codebook updated locally via EMA (fedavg-of-the-codebook, or
    #                       merge='local' where it is never aggregated at all).
    # Loop over inner VQs so a Residual-VQ configures every stage identically.
    for vq in model.quantizer.inner_vqs():
        vq.collect_stats_only = collect_stats
        # A FEDERATED codebook starts from one common broadcast, so local k-means must never
        # re-seed it — `initialized=True` disables the seeding. A LOCAL codebook
        # (merge='local') must instead be the `local` baseline's quantizer EXACTLY: seeded by
        # k-means on this client's own first training batch, then EMA-updated with dead-code
        # expiry. `train_local` gets that by leaving `initialized` False
        # (federated_eval._materialized_stage1), and so do we.
        vq.initialized.fill_(not local_codebook)
        vq.anchor_weight = float(anchor_weight)     # STEP 4 ablation knob (0 = off)

    # `weight_decay` was left at torch's AdamW default, which happens to equal
    # cfg.training.weight_decay (0.01) — so the config knob was inert but not visibly wrong.
    # Passed explicitly now: a future config change must not silently fail to apply here.
    # Note for FedProx: AdamW's decoupled decay is itself a proximal pull toward 0, applied
    # WITHOUT the preconditioner that the in-loss prox goes through (see `_prox_term`).
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,   # built ONCE, kept alive
                            weight_decay=float(getattr(cfg.training, "weight_decay", 0.01)),
                            fused=(torch.device(device).type == "cuda"))
    return ClientState(entity_id=entity_id, cfg=cfg, model=model, opt=opt, data=data)


def _local_train_stage1(client: ClientState, n_epochs: int, device: torch.device) -> float:
    """Train encoder/decoder locally for n_epochs; accumulate round stats. Returns mean loss.

    RETURN CONTRACT — the returned mean is the loss AS EVERY PRE-EXISTING ARM REPORTED IT
    (stage-1 objective, including the B1 probe-alignment term when that arm is on), with the
    NEW FedProx/FedProto penalties excluded. They are added to the backward objective but
    never to this number: `mean_loss` is the one training curve compared across arms and
    against every result already on disk, so a penalised value would silently make the
    trio's convergence incomparable with `local`, `centralized` and `federated_cb_only`.
    The fully penalised objective is kept separately in `client.last_obj`.
    """
    model, opt = client.model, client.opt
    model.train()
    for vq in client.vqs:
        vq.reset_round_stats()
    total, count = 0.0, 0
    obj_sum = 0.0
    prox_sum, proto_sum, steps = 0.0, 0.0, 0    # `steps` counts EVERY step (telemetry denominator)
    proto_n = 0        # steps on which the prototype term EXISTED (see the NaN rule below)
    ratio_sum, ratio_n, skipped = 0.0, 0, 0
    # Resolve the proximal (param, ref) pairs ONCE: a per-step `dict(named_parameters())`
    # rebuild would dominate the term's own cost on a 52-tensor encoder.
    prox_pairs = ([(dict(model.named_parameters())[k], r) for k, r in client.prox_ref.items()]
                  if (client.prox_mu > 0 and client.prox_ref) else [])
    for _ in range(n_epochs):
        for batch in client.data.train_loader:
            x = batch["inputs"].to(device, non_blocking=True)
            with _amp():                       # fp16/bf16 forward (VQ stats stay fp32)
                out = model(x)
                loss = out["losses"]["loss"]
                if client.align_weight > 0 and client.z_ref is not None:
                    # Encode the SHARED public probe (encoder-only forward, eval-mode BN so
                    # probe data never pollutes running stats; grad still flows to conv weights)
                    # and pull it toward the broadcast consensus. Aligns the tokenizer FUNCTION
                    # on a common input — permutation-free, no weight averaging.
                    enc = model.encoder
                    was = enc.training; enc.eval()
                    z_probe = enc(model.transform(client.probe.to(device)))
                    if was:
                        enc.train()
                    loss = loss + client.align_weight * \
                        ((z_probe.float() - client.z_ref.to(z_probe.device).float()) ** 2).mean()
            # The TASK loss, captured before any federation penalty is bolted on.
            task = loss
            # encoder-federation regularisers
            # Deliberately OUTSIDE `_amp()`: both are fp32. The distances are small and
            # accumulate over tens of thousands of coordinates, where an fp16 reduction
            # loses the low-order bits the penalty is made of.
            steps += 1
            try:
                if prox_pairs and client.prox_form == "loss":
                    prox = _prox_term(prox_pairs)                       # ‖w − w^t‖²
                    loss = loss + 0.5 * client.prox_mu * prox
                    prox_sum += float(prox.detach())
                elif prox_pairs:                                       # "decoupled": post-step
                    with torch.no_grad():                              # telemetry only — no graph
                        prox_sum += float(_prox_term(prox_pairs))
                if client.proto_weight > 0:
                    # ALWAYS called when the arm is on: it also releases the tokens the VQ
                    # stashed this step (None on round 0 if the seeding is disabled).
                    proto = _proto_term(client)
                    if proto is not None:
                        loss = loss + client.proto_weight * proto
                        proto_sum += float(proto.detach())
                        proto_n += 1
            finally:
                # Never leave a graph pinned on the VQ if the block above raised.
                for vq in client.vqs:
                    vq._last_tokens = vq._last_indices = None
            opt.zero_grad(set_to_none=True)
            client.scaler.scale(loss).backward()
            if prox_pairs and client.prox_form == "loss":
                # 'loss' form ONLY: in the decoupled form the prox never enters p.grad, so
                # this ratio would describe a quantity that plays no part in the update
                # (and is anti-monotone in the anchor's real strength) — `prox_pull_frac`
                # is that form's diagnostic instead.
                # Unscale here (rather than inside `step`) so the gradient norms below are
                # the true ones. `step` detects the already-unscaled state and does not
                # repeat the work, so the update itself is unchanged.
                client.scaler.unscale_(opt)
                r = _prox_grad_ratio(prox_pairs, client.prox_mu, client.prox_form)
                if math.isfinite(r):
                    ratio_sum += r; ratio_n += 1
            prev_scale = client.scaler.get_scale()
            client.scaler.step(opt)            # skipped iff grads overflowed under fp16
            client.scaler.update()
            applied = client.scaler.get_scale() >= prev_scale     # stage1.py's idiom
            if not applied:
                skipped += 1
            elif client.sched is not None:
                # A skipped step moved no weights, so the schedule must not advance either
                # — the same rule `_converged_loop` follows in pipeline/federated_eval.py.
                client.sched.step()
            if prox_pairs and client.prox_form == "decoupled" and applied:
                # AdamW-style DECOUPLED proximal step: w ← w + lr·μ·(w^t − w). Bypasses
                # Adam's preconditioner, which otherwise divides the in-loss prox gradient
                # by sqrt(v̂) and reduces μ to a rotation of a step whose magnitude is ≈ lr.
                with torch.no_grad():
                    lr_t = opt.param_groups[0]["lr"]
                    for p, ref in prox_pairs:
                        p.add_(ref.to(p.dtype) - p, alpha=lr_t * client.prox_mu)
            # `task is loss` for every arm without a federation penalty (i.e. all the
            # pre-existing ones): reuse the value instead of paying a SECOND device→host
            # sync per step just to read the same number.
            lv = float(task.detach())
            obj = lv if loss is task else float(loss.detach())
            if math.isfinite(lv):              # an overflowed step must not poison the mean
                total += lv; obj_sum += obj; count += 1
    client.last_prox = prox_sum / steps if (steps and client.prox_mu > 0) else float("nan")
    # NaN when the term did not EXIST this round — never 0.0. Under `--fedproto-no-seed`
    # there is no prototype at round 0 (`_proto_term` returns None on every step), and a
    # reported `proto_term = 0.0` would read as "the anchor was applied and the encoder
    # already sat exactly on the global prototypes" — the opposite of "the arm was
    # unfederated for this round". Every other non-applicable telemetry field in this file
    # is NaN; this one now is too. The DENOMINATOR is still `steps` (not `proto_n`) so any
    # round that did apply the term reports the value it always reported.
    client.last_proto = (proto_sum / steps
                         if (steps and proto_n and client.proto_weight > 0) else float("nan"))
    client.last_obj = obj_sum / count if count else float("nan")
    client.last_steps, client.last_skipped = steps, skipped
    # ── "is the proximal term doing anything?" — ONE diagnostic PER FORM ──────────
    # These two are NOT comparable to each other and must never be gated by one threshold.
    #
    # 'loss'      ‖μ(w−w^t)‖ / ‖∇_w L_task‖ — the prox gradient against the data gradient.
    #             Below ~1e-2 the term is provably negligible, and a "FedProx ≡ FedAvg"
    #             reading would be a statement about the solver, not about federation.
    #             Note it is roughly LINEAR in μ, so it says nothing about whether the μ
    #             that was actually reported bites — read it at the reported μ, not at the
    #             largest μ in the sweep.
    # 'decoupled' the fraction of the round's drift the contraction removes. Each applied
    #             step multiplies (w − w^t) by (1 − lr·μ) — note the lr: at lr=1e-3 even
    #             μ=10 removes only 1% of the drift per step, so this is the number that
    #             says whether the anchor has any authority over a round.
    client.last_prox_ratio = (ratio_sum / ratio_n
                              if (ratio_n and client.prox_form == "loss") else float("nan"))
    if client.prox_mu > 0 and client.prox_form == "decoupled":
        a = min(1.0, float(opt.param_groups[0]["lr"]) * client.prox_mu)
        client.last_prox_pull = 1.0 - (1.0 - a) ** max(steps - skipped, 0)
    else:
        client.last_prox_pull = float("nan")
    # count == 0 ⇒ every step overflowed. `0.0` would read as a perfect loss.
    return total / count if count else float("nan")


@torch.no_grad()
def _val_loss_stage1(client: ClientState, device: torch.device) -> float:
    """Mean stage-1 loss on this client's held-out `val`. Side-effect free.

    `SharedVectorQuantizer.forward` gates BOTH the round-stat accumulation and the
    local EMA / dead-code expiry on `self.training` (vector_quantizer.py:287), so an
    `eval()` forward neither pollutes the sufficient statistics the server is about to
    merge nor moves the codebook. Called after the broadcast, i.e. it scores the model
    the round actually produced.
    """
    loader = getattr(client.data, "val_loader", None)
    if loader is None:
        return float("nan")
    client.model.eval()
    total, count = 0.0, 0
    for batch in loader:
        x = batch["inputs"].to(device, non_blocking=True)
        # fp32, NOT `_amp()`: this number drives `select_on_val`, i.e. WHICH round's weights
        # the run ships, so it must not depend on the training precision. That is the rule
        # the baselines' own validation follows by construction (federated_eval._converged_loop:
        # "Validation stays fp32 on purpose: it drives early stopping and weight selection").
        # The federated path used to validate under autocast, making the two criteria
        # incomparable. In practice the decision rarely changes (the last round is almost
        # always the best — see the truncation note in documentation/research/FED_ENCODER_ALGOS.md),
        # but a selection criterion that silently differs between arms is not defensible.
        lv = float(client.model(x)["losses"]["loss"].detach())
        if math.isfinite(lv):
            total += lv; count += 1
    # NaN, never 0.0: a diverged client must not win the `val_loss < best_val` test at :497.
    return total / count if count else float("nan")


def _val_seed(entity_id: str) -> int:
    """Deterministic per-client seed for the FIXED val oracle — stable across rounds,
    runs and processes (blake2b, not hash())."""
    import hashlib
    return int.from_bytes(hashlib.blake2b(f"s2val:{entity_id}".encode(),
                                          digest_size=6).digest(), "big")


@torch.no_grad()
def _val_loss_prior(c: "Stage2ClientState", device: torch.device,
                    val_mode: str = "legacy") -> float:
    """Mean prior loss on this client's held-out `val`. Stage-1 is frozen here.

    `val_mode`:
      "legacy"  (default, every run on disk) — masks re-drawn from the GLOBAL RNG at every
                eval, under `_amp()` autocast. Measured per-eval noise σ ≈ 0.006-0.018 nats
                (random MaskGIT masks, model/prior.py `_mask_tokens`) against
                min_delta=1e-4: selection among plateau rounds is jitter, and at small-τ
                regimes the patience stop becomes a noise-record process.
      "fixed"   — the SAME masks at every eval (fork_rng + per-client blake2b seed, so the
                TRAINING RNG stream is untouched and trajectories stay byte-identical) and
                fp32 forward (mirror of the stage-1 rule at `_val_loss`: the number drives
                select_on_val, it must not depend on training precision).
                Prerequisite for every small-τ run; opt-in via --fed-s2-val fixed."""
    loader = getattr(c.data, "val_loader", None)
    if loader is None:
        return float("nan")
    c.s2.stage1.eval(); c.s2.prior.eval()
    fixed = (val_mode == "fixed")
    devs = [device] if device.type == "cuda" else []
    rng_ctx = torch.random.fork_rng(devices=devs) if fixed else nullcontext()
    with rng_ctx:
        if fixed:
            torch.manual_seed(_val_seed(c.entity_id))
        total, count = 0.0, 0
        for batch in loader:
            batch["inputs"] = batch["inputs"].to(device, non_blocking=True)
            with (nullcontext() if fixed else _amp()):
                tokens = c.s2.tokens_from_batch(batch)
                c.s2._inform_latent_shape()
                lv = float(c.s2.prior(tokens).loss.detach())
            if math.isfinite(lv):
                total += lv; count += 1
    # NaN, never 0.0: a diverged client must not win the `val_loss < best_val` test at :658.
    return total / count if count else float("nan")


def _mean_finite(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return sum(vals) / len(vals) if vals else float("nan")


@torch.no_grad()
def _server_merge(counts, sums, eps, threshold, revive: bool, *,
                  ema_state: dict | None = None, ema_decay: float | None = None):
    """Aggregate client stats → new codebook (Prop. 1) + optional server-side
    dead-code revival from live centroids (data-free, privacy-clean).

    `ema_decay is None` (the default) ⇒ the plain per-round merge: the codebook is
    M/smoothed(N) with N,M the aggregated round stats, and the broadcast (N,M) are
    those same round stats. Every existing arm takes this path → byte-identical.

    `ema_decay is not None` ⇒ the `federated_cb_only_ema` regime: the server keeps a
    running EMA of the AGGREGATE sufficient statistics in the mutable `ema_state`
    dict {"C","S","w"} (zero-initialized by the caller, one per RVQ stage):

        C ← γ·C + (1−γ)·N ,   S ← γ·S + (1−γ)·M ,   e_j = S_j / smoothed(C_j)

    and BROADCASTS (C, S) in place of (N, M). γ=0 reproduces the plain merge exactly
    (C=N, S=M). Dead-code detection uses the BIAS-CORRECTED Ĉ = C/(1−γ^(t+1)): a code
    with EMA history is not killed for one idle round (the intended memory), while the
    fixed integer `threshold` is not tripped by the cold-start damping of the raw C.
    The codebook and the broadcast still use the raw (C, S) — the correction cancels in
    S/smoothed(C) and only matters for the magnitude-sensitive threshold.

    EXACTNESS, DELIMITED. "The merge is an exact function of the aggregated sufficient
    statistics" (Prop. 1) is a claim about `weight = M/smoothed(N)` — and it holds ONLY for
    `revive=False`, or for `revive=True` with no dead code. The revival branch below
    overwrites `weight[dead]` with `weight[pick] + 0.01·randn`, so with even one dead code
    the returned codebook depends additionally on (i) the global RNG stream at call time and
    (ii) the `dead` mask, i.e. on the fixed integer `threshold`, not on (N, M) alone. Two
    servers given identical (N, M) then disagree. This is a deliberate, data-free repair of
    an unused dictionary entry — nothing client-side leaks into it — but it must not be
    described as part of the additive/exact aggregation when the property is being claimed
    (paper text, `scripts/fed_regression_unittest.py`): test exactness with revive off."""
    weight, N, M = SharedVectorQuantizer.merge_round_stats(counts, sums, eps=eps)
    dead_count = N                          # what the fixed dead-code threshold is compared to
    if ema_decay is not None:
        C = ema_decay * ema_state["C"] + (1.0 - ema_decay) * N
        S = ema_decay * ema_state["S"] + (1.0 - ema_decay) * M
        ema_state["C"], ema_state["S"] = C, S
        weight = SharedVectorQuantizer.codebook_from_stats(C, S, eps=eps)
        N, M = C, S                                    # broadcast the raw accumulators (spec)
        # Adam-style bias correction, dead-code test ONLY. C is damped by the cold-start
        # weight w_t = 1−γ^(t+1) (C started at 0), so a healthy code has C ≈ w_t·N ≪ N in
        # the first rounds and `C < threshold` would spuriously flag it. Ĉ = C/w_t restores
        # the effective per-round count. w_t is itself the EMA of the constant 1.
        w = ema_decay * float(ema_state.get("w", 0.0)) + (1.0 - ema_decay)
        ema_state["w"] = w
        dead_count = C / max(w, eps)
    dead = dead_count < threshold
    n_dead = int(dead.sum())
    if revive and n_dead > 0:
        live = (~dead).nonzero(as_tuple=True)[0]
        if len(live) > 0:
            pick = live[torch.randint(len(live), (n_dead,))]
            weight[dead] = weight[pick] + 0.01 * torch.randn_like(weight[dead])
    return weight, N, M, n_dead


@torch.no_grad()
def _server_union_recluster(counts, sums, prev_cb: torch.Tensor, eps: float,
                            threshold: float, lloyd_iters: int = 5):
    """k-FED-style alternative to the pooled M-step (`_server_merge`), built for the
    MINORITY-MOTIF failure measured on ucr_170 (2026-08-08): the pooled merge computes
    e_j = Σ_i m_ij / Σ_i n_ij, so a rare motif seen well by ONE small client (p1, 10%)
    is count-diluted into the majority's centroid and the merged codebook mis-encodes it.

    Here instead: per-client M-step centroids (m_ij/n_ij, count-filtered) → UNION →
    deterministic farthest-point seeding → `lloyd_iters` weighted Lloyd steps → the final
    centers are SLOT-MATCHED to the previous codebook (greedy nearest, deterministic).
    Farthest-point is the minority-preserving step: a rare-but-distinct client centroid is
    far from everything and gets picked as a seed, where count-weighted anything ignores it.

    Deterministic BY CONSTRUCTION: no RNG anywhere (seeding starts from the highest-count
    centroid; ties in argmax/argmin resolve by index). No revival branch: unclaimed
    capacity is reassigned by the reclustering itself; slots left unmatched when the union
    is smaller than K keep the previous round's vector and broadcast N=0 (counted dead).

    Returns (weight, N, M, n_dead) with the same contract as `_server_merge`:
    N_k = summed member counts, M_k = weight_k · N_k (fixed point of M/smoothed(N))."""
    K, D = prev_cb.shape
    pts, wts = [], []
    for n, m in zip(counts, sums):                       # one (K,), (K,D) pair per client
        n = n.float().cpu(); m = m.float().cpu()
        live = n >= float(threshold)                     # same liveness bar as the dead test
        if live.any():
            pts.append(m[live] / n[live].unsqueeze(1).clamp_min(eps))
            wts.append(n[live])
    if not pts:                                          # degenerate: nobody used anything
        N = torch.zeros(K)
        return prev_cb.clone(), N, prev_cb * N.unsqueeze(1), K
    P = torch.cat(pts); W = torch.cat(wts)               # (U, D), (U,)
    U = P.shape[0]
    n_seeds = min(K, U)
    # deterministic farthest-point seeding
    seed_idx = [int(W.argmax())]
    d2 = ((P - P[seed_idx[0]]) ** 2).sum(1)
    for _ in range(n_seeds - 1):
        i = int(d2.argmax())
        seed_idx.append(i)
        d2 = torch.minimum(d2, ((P - P[i]) ** 2).sum(1))
    C = P[seed_idx].clone()                              # (n_seeds, D)
    # ── weighted Lloyd ──────────────────────────────────────────────────────────────
    for _ in range(lloyd_iters):
        assign = torch.cdist(P, C).argmin(1)
        for k in range(n_seeds):
            mask = assign == k
            if mask.any():                               # empty cluster keeps its seed
                C[k] = (P[mask] * W[mask].unsqueeze(1)).sum(0) / W[mask].sum().clamp_min(eps)
    assign = torch.cdist(P, C).argmin(1)
    Nc = torch.zeros(n_seeds)
    for k in range(n_seeds):
        Nc[k] = W[assign == k].sum()
    # ── slot-match to the previous codebook (greedy nearest, deterministic) ─────────
    # Functionally optional (the encoder sees vectors, not ids, and the prior trains only
    # after the FINAL round) — but without it `cb_drift` telemetry reads as permutation
    # noise and the resume re-seat asserts would compare unrelated slots.
    cost = torch.cdist(C, prev_cb.float().cpu())         # (n_seeds, K)
    weight = prev_cb.clone().float()
    N = torch.zeros(K)
    free_c = list(range(n_seeds)); free_s = list(range(K))
    while free_c:
        sub = cost[free_c][:, free_s]
        flat = int(sub.argmin())
        ci, si = free_c[flat // len(free_s)], free_s[flat % len(free_s)]
        weight[si] = C[ci]; N[si] = Nc[ci]
        free_c.remove(ci); free_s.remove(si)
    M = weight * N.unsqueeze(1)
    n_dead = int((N < threshold).sum())
    return weight.to(prev_cb.dtype), N, M, n_dead


@torch.no_grad()
def _token_usage_hist(model: Stage1VQVAE, probe: torch.Tensor, K: int, device: torch.device) -> torch.Tensor:
    model.eval()
    _, indices, _ = model.encode_tokens(probe.to(device))     # (B, C, F*W)
    counts = torch.bincount(indices.reshape(-1), minlength=K).float()
    return counts / counts.sum().clamp_min(1.0)


def _load_public_probe(cfg: Config, n_windows: int = 64,
                       cluster: str | None = None) -> torch.Tensor | None:
    """Window the PUBLIC probe series (data-independent of every client) into a
    (B, C, W) batch, per-channel standardised to match the training scale.

    `cluster` selects `probe/<cluster>.npy` (toy_fed_uni writes one per machine
    type). Use it whenever the federation is scoped to a cluster: the global
    `probe.npy` concatenates all six morphologies, so tokenizing it with a
    cluster-trained encoder measures agreement on data no client has ever seen.
    """
    probe_dir = resolve_path(cfg.paths.raw_data) / cfg.dataset.name / "probe"
    path = probe_dir / f"{cluster}.npy" if cluster else probe_dir / "probe.npy"
    if not path.exists():
        if cluster:
            print(f"[fed] no per-cluster probe at {path}; falling back to probe.npy")
            path = probe_dir / "probe.npy"
        if not path.exists():
            return None
    arr = np.load(path).astype("float32")                            # (T, C)
    arr = (arr - arr.mean(0, keepdims=True)) / (arr.std(0, keepdims=True) + 1e-8)
    W = cfg.dataset.window_length
    starts = np.linspace(0, len(arr) - W, n_windows).astype(int)
    wins = np.stack([arr[s:s + W].T for s in starts])                # (B, C, W)
    return torch.from_numpy(wins)


def _mean_pairwise_js(hists: list[torch.Tensor]) -> float:
    def js(p, q):
        m = 0.5 * (p + q)
        def kl(a, b):
            nz = a > 0
            return float((a[nz] * (torch.log(a[nz]) - torch.log(b[nz].clamp_min(1e-12)))).sum())
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
    vals = [js(hists[i], hists[j]) for i in range(len(hists)) for j in range(i + 1, len(hists))]
    return sum(vals) / max(len(vals), 1)


# ─── encoder federation (probe: align the TOKENIZER, not just the dictionary) ──
# A shared codebook over divergent local encoders is incoherent — codeword k means
# a different thing per client. Federating the encoder makes the codebook-facing map
# agree, which is also the precondition the suff-stat (Prop.1) merge assumes.
#
# WHICH TENSORS are shared is `fed_encoder` (off | full | partial | neck); HOW they are
# federated is `enc_fed_algo`:
#
#   fedavg   — server averages the shared encoder tensors (data-size weighted) and
#              broadcasts. The textbook baseline (McMahan et al. 2017).
#   fedprox  — identical aggregation, PLUS a local proximal penalty (μ/2)‖w − w^t‖²
#              anchoring each client to the round's broadcast w^t (Li et al. 2020).
#              μ=0 reduces EXACTLY to fedavg.
#   fedproto — NO weight averaging at all (Tan et al. 2022): clients exchange only
#              per-class PROTOTYPES and each client pulls its own representation toward
#              the global prototype of the class it assigned. Here the "classes" are the
#              codebook indices of the frozen broadcast codebook, so the prototype message
#              is exactly the (n_j, m_j) the suff-stat merge already uploads → zero extra
#              communication. The WHOLE encoder is still federated (every encoder parameter
#              receives gradient from the prototype term), but in FUNCTION space, not weight
#              space — which is the axis the matched-fusion evidence says matters.
#
# ── FedProto non-degeneracy (read this before touching the loss) ──────────────
# `SharedVectorQuantizer.anchor_weight` (arm `federated_anchor`) is documented in
# model/vector_quantizer.py as degenerate: its target IS the commitment target, so it
# only rescales `commitment_weight`. The prototype term here avoids that trap twice:
#
#  (a) FORM. Commitment is a PER-SAMPLE pull, mean_i‖z_i − e_{k(i)}‖²; expanding around
#      the per-code batch mean μ_k gives  mean_i‖z_i − μ_{k(i)}‖² + Σ_k (n_k/n)‖μ_k − e_k‖²
#      = WITHIN + BETWEEN. The prototype term is the BETWEEN part alone, so it moves the
#      class centroid without shrinking within-class spread — not a multiple of commitment
#      (its gradient w.r.t. z_i is proportional to (μ_k − p̄_k): the SAME vector for every
#      sample of class k, where commitment gives each sample its own residual).
#  (b) TARGET. With `agg="count"` the aggregate p̄_k = Σ_j m_j^k / Σ_j n_j^k is EXACTLY the
#      Prop.1 merged codebook e_k (up to Laplace smoothing) — same target, different form.
#      With the default `agg="uniform"` (mean over the clients that used code k) the target
#      differs from e_k whenever clients contribute unequal counts, i.e. under exactly the
#      heterogeneity this study is about, and it equalises client influence — the reason
#      FedProto's own aggregation is reported per-class rather than per-sample. Keep
#      `count` available as the paper-faithful ablation and report both.
# Prototypes are necessarily STALE by one round (they summarise the round that just ended),
# which is the same staleness FedProto itself has; round 0 therefore has no term at all.

def _bn_module_prefixes(model) -> set:
    """Names of every BatchNorm module in the model — used to tell a BN layer's learnable
    affine (gamma/beta) apart from an ordinary conv weight/bias, which the key name alone
    cannot do."""
    import torch.nn as _nn
    bn_types = (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d, _nn.SyncBatchNorm)
    return {name for name, mod in model.named_modules() if isinstance(mod, bn_types)}


def _encoder_shared_keys(model, mode: str, split_at: int = 2, bn: str = "buffers_local") -> list[str]:
    """Encoder params to FedAvg. BN running stats stay LOCAL (FedBN: they are each
    client's data statistics, not learnable weights). In 'partial' mode the first
    `split_at` blocks (the raw-signal front-end) also stay local; the late blocks +
    the ProjectBlock into codebook-space are shared, so the codebook-facing map agrees.
    In 'neck' mode ONLY the codebook-facing projection (the final block's `.proj.`) is
    shared — the purest alignment neck A_G, every encoder block H_k stays local.

    `bn` selects the BatchNorm regime — three genuinely different things, and the middle one
    is the one the literature calls FedBN:

      "buffers_local" (default)  average the BN AFFINE parameters (gamma, beta) with the rest
                                 of the encoder, keep only the running statistics local. This
                                 is NOT FedBN: it shares 14 of this encoder's tensors that
                                 FedBN would keep local. Call it "BN buffers local".
      "fedbn"                    keep the WHOLE BN layer client-local — gamma, beta AND the
                                 running statistics. This is FedBN (Li et al., ICLR 2021,
                                 arXiv:2102.07623), whose claim is precisely that the entire
                                 BN layer must stay local under feature shift.
      "shared" federate the running statistics too ("everything the encoder"
                                 has"). They are POOLED, not averaged — see `_pool_encoder_bn`.

    `num_batches_tracked` is excluded in every mode: it is an int64 counter, and a float
    average copied back into an int64 buffer silently truncates. It only scales BN's own
    momentum-free warmup and is never a learned quantity, so leaving it local is the honest
    choice, not an omission."""
    if bn not in {"buffers_local", "fedbn", "shared"}:
        raise ValueError(f"unknown bn mode {bn!r} (use 'buffers_local', 'fedbn' or 'shared')")
    bn_mods = _bn_module_prefixes(model) if bn == "fedbn" else set()
    def is_bn_stat(k):
        return k.endswith(("running_mean", "running_var", "num_batches_tracked"))
    def blk(k):
        p = k.split(".")                         # encoder.encoder.<idx>.…
        return int(p[2]) if len(p) > 2 and p[2].isdigit() else -1
    out = []
    for k in model.state_dict():
        if not k.startswith("encoder."):
            continue
        if k.endswith("num_batches_tracked"):
            continue
        if is_bn_stat(k) and bn != "shared":
            continue
        if bn == "fedbn" and k.rsplit(".", 1)[0] in bn_mods:
            continue                             # gamma/beta stay local too — that IS FedBN
        if mode == "neck":
            # share only the final codebook-facing linear map (…proj.weight/bias)
            if ".proj." not in k:
                continue
        elif mode == "partial" and 0 <= blk(k) < split_at:
            continue
        out.append(k)
    return out


@torch.no_grad()
def _broadcast_encoder(clients, devices, shared_keys, ref_idx: int = 0) -> None:
    """Common init before round 0 — FedAvg of conv nets needs a shared start."""
    ref = clients[ref_idx].model.state_dict()
    for c, d in zip(clients, devices):
        sd = c.model.state_dict()
        for k in shared_keys:
            sd[k].copy_(ref[k].detach().to(d, dtype=sd[k].dtype))


def _attach_cosine_schedules(clients, rounds: int, local_epochs: int, warmup_rate: float) -> None:
    """Give every client the warmup+cosine schedule the `local`/`centralized` baselines get.

    WHY THIS IS A KNOB AND NOT THE DEFAULT. The federated round loop has always trained at a
    CONSTANT learning rate, while `local` trains under warmup+cosine with val-driven early
    stopping (federated_eval._converged_loop). So a federated arm compared against `local` is
    handicapped by its RECIPE, not only by federation — but turning the schedule on by default
    would make the arm incomparable with every federated result already on disk, which was
    produced at constant LR. Both comparisons are legitimate and they need different settings:

        vs `local`               -> enc_sched='cosine'  (recipe matched, federation isolated)
        vs `federated_cb_only`   -> enc_sched='none'    (matches the existing federated arms)

    Report which one a table used; do not mix them in one table.

    The horizon is the FULL planned budget (rounds x local_epochs x batches), not one round:
    a per-round cosine would decay to zero and restart every round, which is a different
    (cyclic) schedule, not the baseline's.
    """
    for c in clients:
        total = max(1, rounds * local_epochs * len(c.data.train_loader))
        warm = max(1, int(total * warmup_rate))

        def lr_lambda(step: int, _t=total, _w=warm) -> float:
            if step < _w:
                return step / _w
            progress = (step - _w) / max(_t - _w, 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        c.sched = torch.optim.lr_scheduler.LambdaLR(c.opt, lr_lambda)


# ─── resume: continue a finished run instead of restarting it ─────────────────

def _resume_bundle_path(d) -> "Path":
    return Path(d) / "_fed_resume.pt"


@torch.no_grad()
def _save_resume_bundle(clients, out_dir, rounds_done: int, server_ema, merge: str) -> None:
    """Everything a LATER run needs to continue this one exactly.

    The per-client `stage1.ckpt` written by the eval harness already carries the model AND
    the whole VQ state (codebook, ema_cluster_size, ema_embed_sum, initialized) because they
    are registered buffers. What it does NOT carry — and what makes the difference between
    *continuing* and *restarting with warm weights* — is here:

      * the AdamW moment estimates. This repo deliberately keeps one optimizer alive across
        rounds; dropping its state resets β₁/β₂ and gives every client a transient of ~1/(1−β₁)
        ≈ 10 steps of unpreconditioned updates per resume.
      * the GradScaler scale, so fp16 does not re-search it from 65536.
      * `rounds_done`, without which a cross-round schedule (and the server EMA's cold-start
        bias correction) would restart from t=0.
      * the server-side codebook EMA accumulators, for the `federated_cb_only_ema` regime.
        (They are also recoverable from any client's broadcast EMA buffers, but only up to
        the bias-correction weight `w`, which lives on the server alone.)
      * the LR scheduler's step counter (`enc_sched='cosine'`). `rounds_done` alone does NOT
        restore it: `_attach_cosine_schedules` builds a fresh LambdaLR whose `last_epoch` is
        -1→0, so a continuation would replay the warmup from t=0 and jump the LR back to its
        peak — while this docstring promised the opposite. The lambda closure is not saved
        (LambdaLR.state_dict drops it by design) and does not need to be: the schedule is
        rebuilt from `rounds`/`local_epochs` before the load, and only `last_epoch` /
        `_step_count` come from here. `None` per client when no schedule is attached, which
        is every arm on disk — so this key changes nothing for them.

    format 2 adds `sched`. Both directions stay compatible because the reader takes every
    key through `.get()`: a format-1 bundle loads (and warns if a schedule is active), and
    an older reader ignores the new key.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": 2,
        "rounds_done": int(rounds_done),
        "merge": merge,
        "entities": [c.entity_id for c in clients],
        "opt": [{k: v for k, v in c.opt.state_dict().items()} for c in clients],
        "scaler": [c.scaler.state_dict() for c in clients],
        "sched": [(c.sched.state_dict() if c.sched is not None else None) for c in clients],
        "server_ema": ([{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in st.items()}
                        for st in server_ema] if server_ema is not None else None),
    }, _resume_bundle_path(out_dir))
    n_sched = sum(c.sched is not None for c in clients)
    print(f"[fed] resume bundle written -> {_resume_bundle_path(out_dir)} "
          f"(rounds_done={rounds_done}; per-client optimizer + scaler"
          f"{f' + {n_sched} LR schedule(s)' if n_sched else ''}"
          f"{' + server codebook EMA' if server_ema is not None else ''})")


def _load_resume(clients, devices, resume_dir, cfg, merge: str, server_ema) -> int:
    """Restore a previous run into these freshly built clients. Returns `rounds_done`.

    Two tiers, because runs that predate the resume bundle still exist on disk and are worth
    continuing:

      FULL  `_fed_resume.pt` present -> model + optimizer + scaler + round counter (+ server
            EMA, + LR schedule from format 2 on) are all restored; continuing is equivalent
            to having run longer.
      WEIGHTS-ONLY  only the per-entity `stage1.ckpt`s -> the model and the entire VQ state
            are restored, but the optimizer restarts. That is a genuine perturbation, not a
            formality, so it is reported loudly rather than mentioned in a docstring.

    A format-1 bundle carries no schedule state. That is silent only for `enc_sched='none'`
    (every arm on disk); with a schedule attached the continuation WOULD replay the warmup,
    so it is reported as loudly as the weights-only tier instead.
    """
    resume_dir = Path(resume_dir)
    missing = [c.entity_id for c in clients
               if not (resume_dir / c.entity_id / "stage1.ckpt").exists()]
    if missing:
        raise SystemExit(f"--resume-from {resume_dir}: no stage1.ckpt for {missing}. "
                         f"Point it at the arm directory of the run to continue "
                         f"(…/seed<N>/<arm-tag>/), which holds one sub-directory per client.")
    for c, d in zip(clients, devices):
        state = torch.load(resume_dir / c.entity_id / "stage1.ckpt", map_location="cpu",
                           weights_only=False)["state_dict"]
        c.model.load_state_dict(state, strict=True)
        c.model.to(d)
        for v in c.vqs:
            v.reset_round_stats()          # a fresh round starts with empty accumulators

    bundle_path = _resume_bundle_path(resume_dir)
    n_sched = sum(c.sched is not None for c in clients)
    if not bundle_path.exists():
        print(f"[fed] RESUME (weights only) <- {resume_dir}\n"
              f"      !! No _fed_resume.pt: the AdamW moment estimates and the GradScaler "
              f"state of the original run are NOT on disk, so each client restarts its "
              f"optimizer. Expect a transient of ~10 steps/client, and do NOT present the "
              f"result as 'the same run trained longer' — it is a warm-started continuation. "
              f"Runs from this version onward write the bundle, so their resumes are exact."
              + (f"\n      !! The LR schedule also restarts at t=0 (warmup replayed, LR back "
                 f"to its peak) for all {n_sched} client(s)." if n_sched else ""))
        return 0

    b = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if b.get("entities") != [c.entity_id for c in clients]:
        raise SystemExit(f"--resume-from {resume_dir}: the bundle was written for clients "
                         f"{b.get('entities')}, but this run has "
                         f"{[c.entity_id for c in clients]}. Optimizer state is per-client and "
                         f"positional; refusing to mis-assign it.")
    if b.get("merge") != merge:
        raise SystemExit(f"--resume-from {resume_dir}: the bundle was written under "
                         f"merge={b.get('merge')!r} but this run uses merge={merge!r}. "
                         f"The codebook regimes are not interchangeable mid-run.")
    for c, ost, sst in zip(clients, b["opt"], b["scaler"]):
        c.opt.load_state_dict(ost)
        c.scaler.load_state_dict(sst)
    # LR schedules (format 2+). The lambda is rebuilt by `_attach_cosine_schedules` before
    # this call — only the counter comes from the bundle. `.get()` because format-1 bundles
    # predate the key; a run with no schedule attached restores nothing either way, which is
    # why this is a no-op for every arm on disk.
    sched_states = b.get("sched")
    n_sched_restored = 0
    if n_sched and sched_states is not None:
        for c, sd in zip(clients, sched_states):
            if c.sched is None or sd is None:
                continue
            c.sched.load_state_dict(sd)
            # `load_state_dict` restores `last_epoch`/`_last_lr` but does NOT push the LR
            # back into the optimizer, and the first local step reads `param_groups[0]['lr']`
            # before any `sched.step()`. (The optimizer state_dict above already carries the
            # same value; writing it from the schedule keeps the two in agreement even for a
            # bundle whose optimizer groups were edited.)
            for g, lr in zip(c.opt.param_groups, c.sched.get_last_lr()):
                g["lr"] = lr
            n_sched_restored += 1
    elif n_sched:
        print(f"[fed] !! RESUME: {n_sched} client(s) have an LR schedule but the bundle is "
              f"format {b.get('format')} (no 'sched' key): the schedule RESTARTS at t=0, so "
              f"the warmup is replayed and the LR jumps back to its peak. The continuation "
              f"is not 'the same run trained longer' — re-run from scratch, or report the "
              f"restart. Bundles written from format 2 on carry the schedule.")
    done = int(b.get("rounds_done", 0))
    print(f"[fed] RESUME (full) <- {resume_dir}: {done} rounds already done; optimizer + "
          f"scaler{f' + {n_sched_restored} LR schedule(s)' if n_sched_restored else ''}"
          f"{' + server codebook EMA' if b.get('server_ema') else ''} restored.")
    return done


def _round_seed(seed: int, rnd: int, ci: int, stage: int = 1) -> int:
    """Per-(seed, round, client) RNG seed, with NO cross-seed aliasing.

    The old schedule was `seed*1000 + rnd*100 + ci` (stage 2: `seed*7000 + …`). Under it
    (seed 7, round 10, client 0) and (seed 8, round 0, client 0) resolve to the SAME value,
    so `--seeds 0,1,2` over a 90-round budget gave three replicates that shared their entire
    batch-ordering stream, merely shifted by 10 rounds. Each run stayed valid; what was wrong
    is that the seed-to-seed variance — the thing multi-seed exists to estimate — was
    understated, and a paired test across seeds was not testing what it claimed.

    A LINEAR schedule cannot fix this by picking better strides: `a*seed + b*rnd + ci`
    aliases whenever `a` is reachable as `b*Δrnd + Δci`, and making `a` large enough to
    outrun the round range overflows the seed type. (Measured while writing this: primes
    1_000_003/100_003 still collide at Δseed=1, Δrnd=10, Δci=-27.) So the mixing is a hash,
    which has no structured aliasing at all: blake2b over the tuple, taken to 63 bits.
    Deterministic and portable — same (stage, seed, round, client) gives the same value on
    any machine and any run, which is what reproducibility needs.
    """
    key = f"{stage}:{seed}:{rnd}:{ci}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") >> 1


def _client_weights(clients) -> list[float]:
    """FedAvg's n_k — the number of training windows each client holds."""
    return [float(len(c.data.train_dataset)) for c in clients]


@torch.no_grad()
def _fedavg_encoder(clients, devices, shared_keys) -> None:
    """Data-size-weighted average of the shared encoder tensors, broadcast back.

    BN running statistics are NOT handled here even when they are in `shared_keys`:
    averaging `running_var` linearly is wrong (see `_pool_encoder_bn`)."""
    shared_keys = [k for k in shared_keys if not k.endswith(("running_mean", "running_var"))]
    ws = _client_weights(clients)
    wsum = sum(ws) or 1.0
    avg = {}
    for k in shared_keys:
        acc = None
        for w, c in zip(ws, clients):
            t = (w / wsum) * c.model.state_dict()[k].detach().cpu().float()
            acc = t if acc is None else acc + t
        avg[k] = acc
    for c, d in zip(clients, devices):
        sd = c.model.state_dict()
        for k in shared_keys:
            sd[k].copy_(avg[k].to(d, dtype=sd[k].dtype))


@torch.no_grad()
def _flat_encoder(client, keys) -> torch.Tensor:
    """The shared encoder tensors of one client as a single flat fp32 CPU vector.
    CPU because clients may sit on different GPUs and stacking across devices raises."""
    sd = client.model.state_dict()
    return torch.cat([sd[k].detach().float().reshape(-1).cpu() for k in keys])


@torch.no_grad()
def _encoder_drift(clients, keys, weights=None, prev: "list | None" = None) -> dict:
    """Client-drift diagnostics on the shared encoder, measured BEFORE aggregation.

    Returns {'drift', 'drift_rel', 'step', 'flat'}:

      drift      mean_j ‖w_j − w̄‖ with w̄ the SAME weighted centroid `_fedavg_encoder`
                 uses. An unweighted centroid would measure the distance to a point the
                 server never forms — and wsd's clients span 5.8× in size, so the two
                 differ materially.
      drift_rel  drift / ‖w̄‖ — scale-free, hence comparable across arms and datasets.
      step       mean_j ‖w_j^r,pre − w_j^{r−1},POST‖ — how far LOCAL TRAINING moved each
                 client this round, measured from the state it actually started the round
                 in. That is the cross-arm-comparable channel; `drift` is not, being
                 per-round for fedavg/fedprox (the clients are re-synced every round) but
                 CUMULATIVE since init for fedproto (they never are).
                 The caller must therefore pass `prev` = the POST-aggregation flats of the
                 previous round (see `_flat_all`). Passing the pre-aggregation ones would
                 fold the server's own correction into the "local movement" for exactly the
                 arms that aggregate, and compare two different quantities again.

    `keys` should be the shared PARAMETER names: with `--fed-enc-bn shared` the O(1)
    `running_var` entries would otherwise swamp O(1e-2) conv weights in the norm."""
    if not keys:
        return {"drift": float("nan"), "drift_rel": float("nan"),
                "step": float("nan"), "flat": None}
    flat = torch.stack([_flat_encoder(c, keys) for c in clients])
    w = torch.ones(len(clients)) if weights is None else torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum().clamp_min(1e-12)
    centroid = (w.unsqueeze(1) * flat).sum(0)
    d = float((flat - centroid).norm(dim=1).mean())
    step = float((flat - torch.stack(prev)).norm(dim=1).mean()) if prev is not None else float("nan")
    return {"drift": d, "drift_rel": d / max(float(centroid.norm()), 1e-12), "step": step}


@torch.no_grad()
def _flat_all(clients, keys) -> "list | None":
    """Per-client flat encoders — the `prev` argument of `_encoder_drift`, captured AFTER
    aggregation so `enc_step` measures local movement rather than local movement plus the
    server's correction."""
    return [_flat_encoder(c, keys) for c in clients] if keys else None


@torch.no_grad()
def _pool_encoder_bn(clients, devices, weights, bn_keys: list[str]) -> None:
    """Federate BN running statistics by POOLING, not by havingaging (`--fed-enc-bn shared`).

    A linear mean of `running_var` is wrong and this repo already measured it wrong
    (scripts/fa_bnstats.py): variance is not linear in the mixture. The law of total
    variance gives the statistic the pooled data would have produced,

        mean̄  = Σ_j w_j · mean_j
        var̄   = Σ_j w_j · (var_j + mean_j²) − mean̄²      (within + between)

    which is the same decomposition the suff-stat codebook merge uses, applied to BN.
    Keys are paired by prefix, so a module whose `running_mean` is shared but whose
    `running_var` is not (impossible today) would simply be skipped rather than mixed."""
    if not bn_keys:
        return
    ws = torch.tensor([float(w) for w in weights]); ws = ws / ws.sum().clamp_min(1e-12)
    prefixes = sorted({k.rsplit(".", 1)[0] for k in bn_keys
                       if k.endswith(("running_mean", "running_var"))})
    for pre in prefixes:
        km, kv = f"{pre}.running_mean", f"{pre}.running_var"
        if km not in bn_keys or kv not in bn_keys:
            continue
        means = torch.stack([c.model.state_dict()[km].detach().float().cpu() for c in clients])
        varis = torch.stack([c.model.state_dict()[kv].detach().float().cpu() for c in clients])
        m_bar = (ws.unsqueeze(1) * means).sum(0)
        v_bar = (ws.unsqueeze(1) * (varis + means ** 2)).sum(0) - m_bar ** 2
        v_bar = v_bar.clamp_min(0.0)                  # fp error can push a tiny variance below 0
        for c, d in zip(clients, devices):
            sd = c.model.state_dict()
            sd[km].copy_(m_bar.to(d, dtype=sd[km].dtype))
            sd[kv].copy_(v_bar.to(d, dtype=sd[kv].dtype))


# ── FedProx ──────────────────────────────────────────────────────────────────

def _shared_param_names(model, shared_keys: list[str]) -> list[str]:
    """The subset of `shared_keys` that are PARAMETERS (buffers carry no gradient, so a
    proximal penalty on them is meaningless — and BN buffers are FedBN-local anyway)."""
    params = dict(model.named_parameters())
    return [k for k in shared_keys if k in params]


@torch.no_grad()
def _snapshot_prox_ref(model, names: list[str]) -> dict:
    """w^t — the start-of-round global encoder, kept fp32 on the client's own device."""
    p = dict(model.named_parameters())
    return {k: p[k].detach().float().clone() for k in names}


def _prox_term(pairs: list) -> torch.Tensor:
    """‖w − w^t‖² over the pre-resolved (param, w^t) pairs (the caller applies μ/2).

    TWO FORMS, because the local optimizer here is not the FedProx paper's SGD:

      "loss"       the paper's objective, F_k(w) + (μ/2)‖w − w^t‖². This repo keeps ONE
                   AdamW alive per client across rounds, so the proximal gradient
                   μ(w − w^t) is divided by Adam's per-coordinate √v̂ — μ then only
                   ROTATES an update whose magnitude is ≈ lr, and at the start of a round
                   (w = w^t) the pull is exactly zero while the surviving β₁ momentum
                   pushes the client straight back off the aggregate. Faithful, but it can
                   be a no-op at small μ; `last_prox_ratio` is what proves which.
      "decoupled"  the AdamW-consistent analogue: skip the objective and contract after
                   the step, w ← w + lr·μ·(w^t − w). Preconditioner-free, so the drift is
                   multiplied by (1 − lr·μ) per applied step — note the lr: at lr=1e-3
                   even μ=10 removes only 1% of the drift per step. The knob that means
                   "fraction removed per step" is lr·μ, NOT μ (see `last_prox_pull`).

    Either way μ here is NOT numerically comparable to a μ from an SGD FedProx table.

    THE EMPTY-PAIRS GUARD IS DEFENCE IN DEPTH, NOT THE CONTRACT. It cannot fire from this
    repo: `_local_train_stage1` builds `prox_pairs` once and calls this only under
    `if prox_pairs`, so an empty set silently means "no proximal term" there, not an error.
    The condition worth failing on — μ>0 while the shared set contains no PARAMETER, i.e. a
    FedProx arm that is really FedAvg — is therefore checked where it is decidable and where
    a run can still be stopped: `federated_stage1`'s setup, on `enc_param_names`. Keep this
    raise as the last line of defence for a future caller that does not pre-filter."""
    if not pairs:
        raise ValueError("_prox_term called with no (param, ref) pairs — the caller must "
                         "not enable FedProx when the shared parameter set is empty. "
                         "(Unreachable from _local_train_stage1, which calls under "
                         "`if prox_pairs`; the real check is the setup guard on "
                         "enc_param_names in federated_stage1.)")
    acc = None
    for p, r in pairs:
        d = p.float() - r
        t = (d * d).sum()
        acc = t if acc is None else acc + t
    return acc


@torch.no_grad()
def _prox_grad_ratio(pairs: list, mu: float, form: str) -> float:
    """‖μ(w − w^t)‖ / ‖∇_w L_task‖ over the shared params — "is μ actually doing anything?"

    Must be called AFTER `scaler.unscale_()`, or the denominator carries the loss scale.
    In the "loss" form the stored gradient is ∇L_task + μ(w − w^t), so the data gradient
    is recovered by subtracting the (analytically known) proximal part."""
    pull_sq = None
    data_sq = None
    for p, r in pairs:
        if p.grad is None:
            continue
        pull = mu * (p.detach().float() - r)
        g = p.grad.detach().float()
        data = g - pull if form == "loss" else g
        a, b = (pull * pull).sum(), (data * data).sum()
        pull_sq = a if pull_sq is None else pull_sq + a
        data_sq = b if data_sq is None else data_sq + b
    if pull_sq is None:
        return float("nan")
    # ONE device→host sync per step, not one per tensor.
    num, den = float(pull_sq.sqrt()), float(data_sq.sqrt())
    if not math.isfinite(den) or not math.isfinite(num):
        return float("nan")        # fp16 overflow: the step is about to be skipped anyway
    return num / den if den > 0 else float("inf")


# ── FedProto ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _aggregate_prototypes(counts_s: list, sums_s: list, agg: str = "uniform",
                          min_count: float = 0.0):
    """Per-code global prototypes from the clients' round stats → ((K, D), mask (K,)).

    `counts_s[j]` = n_j (K,), `sums_s[j]` = m_j (K, D) — the SAME message the Prop.1 merge
    already receives, so FedProto adds no communication here.

      uniform : p̄_k = mean over the clients that actually used code k of m_j^k / n_j^k
                (one vote per client — the aggregation that is NOT the codebook)
      count   : p̄_k = Σ_j m_j^k / Σ_j n_j^k = the merged codebook itself (paper-faithful
                weighting; kept as the degeneracy ablation, see the section header)

    `mask` is False for codes no client used this round: they have no prototype and are
    excluded from the loss rather than pulled toward a meaningless zero.

    `min_count` additionally drops codes with less than that much TOTAL support. Set it to
    the dead-code threshold and it also protects against the one stale-target case: a code
    below the threshold is exactly the one `_server_merge` may have REVIVED by overwriting
    its codeword with a live centroid, after which its prototype still describes where the
    codeword used to be, and the clients would be pulled toward a region the codebook has
    left. A prototype estimated from a couple of tokens is noise regardless."""
    if agg not in {"uniform", "count"}:
        raise ValueError(f"unknown prototype aggregation {agg!r} (use 'uniform' or 'count')")
    N = torch.stack(counts_s).float()                      # (J, K)
    M = torch.stack(sums_s).float()                        # (J, K, D)
    used = N > 0                                           # (J, K)
    live = used.any(0) & (N.sum(0) >= float(min_count))
    p_count = M.sum(0) / N.sum(0).clamp_min(1.0).unsqueeze(1)
    per = M / N.clamp_min(1.0).unsqueeze(-1)               # (J, K, D) per-client prototypes
    per = per * used.unsqueeze(-1)                         # zero out the unused ones
    p_uniform = per.sum(0) / used.sum(0).clamp_min(1).unsqueeze(1).float()
    # `gap` is the run-time evidence for the degeneracy analysis: it is ‖p̄_uniform − p̄_count‖
    # relative to ‖p̄_count‖ = the merged codebook. Near 0 ⇒ the two aggregations coincide on
    # THIS cohort and the arm is, empirically, a reweighted commitment term — a fact that has
    # to be reported, not discovered later. Free: both vectors come from the same stats.
    gap = float((p_uniform[live] - p_count[live]).norm()
                / p_count[live].norm().clamp_min(1e-12)) if bool(live.any()) else float("nan")
    return (p_count if agg == "count" else p_uniform), live, gap


def _proto_term(client: ClientState) -> "torch.Tensor | None":
    """λ-free FedProto loss for the CURRENT step: Σ_k w_k ‖μ_k(batch) − p̄_k‖².

    Consumes the tokens the VQ stashed during this step's forward (grad-carrying) and
    ALWAYS clears them, so the autograd graph is never held past `backward()` — including
    on round 0 without seeding, where there is no prototype yet and this returns None.

    STAGE 0 ONLY. A Residual-VQ's later stages quantize `residual = residual − q` with a
    straight-through `q`, so `d(residual)/d(latent) == 0` (model/vector_quantizer.py, the
    residual loop): tokens from stage ≥1 are gradient-DEAD and a prototype term built on
    them would be a silent no-op that still costs the memory. The caller therefore only
    sets `export_proto_tokens` on `vqs[0]`; this loop is written to hold even if it did not.

    `code_weight` decides who the term is FOR:
      uniform : w_k = 1/|codes in batch|. Every code counts the same — the prototype
                reading, and the one that makes this the pure BETWEEN-class term.
                Beware: a code seen once in a batch then gets ~N/(n_sel) times the pull
                the commitment loss would give it, and rare codes are the anomaly signal
                (the detector scores prior-NLL over token streams), so λ must be swept.
      count   : w_k ∝ n_k. Restores commitment's token-frequency weighting WITHOUT
                restoring its per-sample form — decouples the loss weighting from the
                target aggregation (`enc_proto_agg`), which are independent choices."""
    protos, mask = client.protos, client.proto_mask
    if protos is None:
        for vq in client.vqs:
            vq._last_tokens = vq._last_indices = None
        return None
    terms = []
    for s, vq in enumerate(client.vqs):
        z, idx = vq._last_tokens, vq._last_indices
        vq._last_tokens = vq._last_indices = None          # release the graph, always
        if z is None or idx is None or s >= len(protos):
            continue
        p, m = protos[s], mask[s]
        zf = z.reshape(-1, z.shape[-1]).float()            # (N, D) — grad-carrying
        i = idx.reshape(-1)
        K, D = p.shape
        ones = torch.ones(i.shape[0], device=zf.device, dtype=zf.dtype)
        cnt = torch.zeros(K, device=zf.device, dtype=zf.dtype).index_add(0, i, ones)
        # out-of-place index_add: in-place on a zeros tensor works too but silently
        # depends on the leaf not requiring grad — this form is safe either way.
        tot = torch.zeros(K, D, device=zf.device, dtype=zf.dtype).index_add(0, i, zf)
        sel = (cnt > 0) & m.to(zf.device)
        if not bool(sel.any()):
            continue
        mu = tot[sel] / cnt[sel].unsqueeze(1)              # per-code batch mean
        sq = ((mu - p.to(zf.device)[sel]) ** 2).mean(1)    # per-code, averaged over D
        if client.proto_code_weight == "count":
            w = cnt[sel]
            terms.append((sq * w).sum() / w.sum())
        else:
            terms.append(sq.mean())
    if not terms:
        return None
    assert len(terms) == 1, (f"prototype term built from {len(terms)} RVQ stages; only stage 0 "
                             f"carries gradient (see the docstring) — check export_proto_tokens.")
    return terms[0]


# ─── orchestrator ────────────────────────────────────────────────────────────

def _make_stage1_viz(client_entities, base_cfg):
    """Opt-in training-viz recorder, or None. Gated ENTIRELY on TVQ_VIZ: when the
    env var is unset (every existing run), `lib.train_viz` is never imported and
    this returns None, so the training path is byte-identical and untouched. See
    lib/train_viz.py for the env knobs and the capture-in-loop/render-offline split.
    """
    if os.environ.get("TVQ_VIZ", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        from lib.train_viz import FedStage1Viz
        return FedStage1Viz.from_env(list(client_entities), base_cfg)
    except Exception as e:                       # a viz init failure must never block training
        print(f"[fed:viz] disabled (init failed: {type(e).__name__}: {e})")
        return None


def federated_stage1(
    base_cfg: Config,
    client_entities: Sequence[str],
    rounds: int,
    local_epochs: int,
    *,
    revive_dead: bool = True,
    anchor_weight: float = 0.0,
    dead_thr: int | None = None,
    seed: int = 7,
    merge: str = "suffstat",
    cb_server_ema_decay: float | None = None,   # None = plain per-round merge (every existing arm).
                                                # float γ = server keeps a cross-round EMA of the
                                                # aggregate codebook stats (federated_cb_only_ema).
    fed_encoder: str = "off",          # "off" | "full" | "partial" | "neck" — WHICH encoder tensors are shared
    enc_split_at: int = 2,             # 'partial': keep the first N encoder blocks (front-end) local
    # HOW the shared encoder is federated (only read when fed_encoder != "off").
    # "fedavg" reproduces every pre-existing encoder-federation arm byte-for-byte.
    enc_fed_algo: str = "fedavg",      # "fedavg" | "fedprox" | "fedproto"
    enc_bn: str = "buffers_local",     # "buffers_local" | "fedbn" | "shared" (see _encoder_shared_keys)
    enc_prox_mu: float = 0.0,          # FedProx μ (0 ⇒ identical to fedavg)
    enc_prox_form: str = "loss",       # "loss" (paper objective) | "decoupled" (AdamW-consistent)
    enc_proto_weight: float = 0.0,     # FedProto λ (0 ⇒ no prototype pull)
    enc_proto_agg: str = "uniform",    # prototype TARGET: "uniform" | "count" (= the codebook)
    enc_proto_code_weight: str = "uniform",   # prototype LOSS weighting per code: uniform | count
    enc_proto_seed_round0: bool = True,       # seed round-0 prototypes from the broadcast codebook
    enc_proto_fedavg: bool = False,    # hybrid ablation: prototypes AND weight averaging
    enc_sched: str = "none",           # "none" (constant LR, every arm on disk) | "cosine"
    resume_from: "str | Path | None" = None,   # continue a finished run instead of restarting
    resume_out: "str | Path | None" = None,    # where to write this run's resume bundle
    resume_rounds_done: int = 0,       # rounds the SOURCE run did, for a weights-only resume
                                       # (a full bundle carries the true count and wins)
    patience_rounds: int = 0,          # 0 = off (every existing arm). >0 = stop once the
                                       # cohort val loss has not improved for N rounds, so a
                                       # generous --s1-rounds budget can be spent safely.
                                       # (warmup+cosine over the full budget = the recipe the
                                       # `local`/`centralized` baselines get). See
                                       # `_attach_cosine_schedules` for which to use when.
    fed_align: str = "off",            # "off" | "probe_mse" — B1: align encoders on a shared probe (no weight avg)
    align_weight: float = 0.0,         # B1 alignment loss weight (lambda)
    align_cluster: str | None = None,  # which probe/<cluster>.npy to use as the shared yardstick
    # OFF by default: the `local`/`centralized` baselines train for a fixed number of
    # epochs with no validation-based restore, so a federated arm that restores its
    # best round would be reported at its best while the baselines are reported at
    # their last. Turn on only when the baselines get the same treatment.
    select_on_val: bool = False,
) -> tuple[list[ClientState], torch.Tensor, list[dict]]:
    # Per-client window/batch shapes are fixed across a round, so the cuDNN autotuner
    # pays for itself. (This used to be `= False`, which silently disabled it.)
    torch.backends.cudnn.benchmark = True
    devices = _client_devices(len(list(client_entities)))
    device = devices[0]              # server-side / print device (== cuda when unset)
    K = base_cfg.quantizer.codebook_size
    eps = base_cfg.quantizer.eps
    if dead_thr is None:
        dead_thr = max(1, base_cfg.quantizer.threshold_ema_dead_code)

    print(f"[fed] device={device} clients={list(client_entities)} "
          f"rounds={rounds} local_epochs={local_epochs} K={K} "
          f"anchor_weight={anchor_weight} revive={revive_dead} dead_thr={dead_thr}")
    if patience_rounds > 0:
        # Printed because a silently-unplumbed patience would burn the whole round budget
        # before anyone noticed — the failure is invisible in the per-round log otherwise.
        print(f"[fed] convergence stop ARMED: patience={patience_rounds} rounds, "
              f"min_delta={float(getattr(base_cfg.training, 'early_stopping_min_delta', 0.0)):g} "
              f"(needs select_on_val, which --protocol converged turns on: "
              f"{'ON' if select_on_val else 'OFF -> the stop will be DISABLED'})")

    cb_local = (merge == "local")
    # `union_recluster` shares the ENTIRE client-side contract with `suffstat` (frozen
    # broadcast codebook during local training, per-round (n, m) stats): the two differ
    # only in the server step — pooled M-step vs k-FED union-recluster.
    stats_merge = merge in ("suffstat", "union_recluster")
    clients = [_build_client(base_cfg, e, devices[i], anchor_weight=anchor_weight,
                             collect_stats=stats_merge, local_codebook=cb_local)
               for i, e in enumerate(client_entities)]
    if merge == "union_recluster":
        print("[fed] codebook merge: UNION-RECLUSTER (k-FED style) — per-client M-step "
              "centroids → count-filtered union → deterministic farthest-point seeding + "
              "weighted Lloyd → slot-matched broadcast. Preserves MINORITY-client motifs "
              "that the count-pooled M-step dilutes (the ucr_170 FP mechanism, 2026-08-08).")
        if revive_dead:
            print("[fed] NOTE union_recluster: `revive_dead` has NO revival branch here — "
                  "unclaimed capacity is reassigned by the reclustering itself, and unmatched "
                  "slots keep the previous vector with N=0 (counted dead, never resampled). "
                  "The flag is inert by design, not silently dropped.")

    if merge == "fedavg" and len(clients[0].vqs) > 1:
        # REFUSE rather than half-federate. The suff-stat path loops `c.vqs` and merges every
        # Residual-VQ stage; the fedavg path below uses `c.vq` (== vqs[0]) throughout, so with
        # more than one stage the later stages' codebooks would be EMA-updated locally by each
        # client and never aggregated or broadcast — the clients would silently diverge on
        # them. The post-merge broadcast assert also only inspects stage 0, so nothing would
        # catch it, and `merge='fedavg'` is the primitive of two REPORTING rows
        # (federated_fedavg_cb_only, federated_fedavg_cb_sharedprior).
        #
        # Not fixed by federating the later stages instead: this arm is a deliberate strawman
        # whose whole point is that the codebook moves locally, so "the right way to weight-
        # average a residual dictionary" is not a question it should be answering. Refusing
        # keeps a silently wrong number from ever being produced; implement the loop only if a
        # multi-stage weight-averaged arm is genuinely wanted.
        raise ValueError(
            f"merge='fedavg' federates STAGE 0 ONLY, but this quantizer has "
            f"{len(clients[0].vqs)} Residual-VQ stages ({base_cfg.quantizer.name!r}). The "
            f"later stages would train locally and never be aggregated, and the broadcast "
            f"assert would not notice. Use merge='suffstat' (which merges every stage) or "
            f"run this arm with a single-stage quantizer.")

    if cb_local:
        # merge='local': the DICTIONARY is never federated. Each client k-means-seeds its own
        # codebook on its own first batch and EMA-updates it, exactly as the `local` baseline
        # does — so the only thing crossing the network is whatever `fed_encoder` shares.
        # No common broadcast: seeding every client from one random codebook would already be
        # a (weak) form of dictionary federation and would blur what this variant isolates.
        global_cbs = [v.codebook.weight.detach().clone() for v in clients[0].vqs]
        print("[fed] codebook: LOCAL — never aggregated, never broadcast (k-means init + EMA "
              "per client, identical to the `local` baseline's quantizer). Reported "
              "codebook/perplexity/dead diagnostics are CLIENT-0's, not a global object.")
    else:
        # Server init: broadcast one common (random) codebook — data-independent.
        # Per stage for a Residual-VQ (one common codebook per stage); a single-codebook VQ
        # has exactly one stage, so this is byte-identical to the original single broadcast.
        global_cbs = [v.codebook.weight.detach().clone() for v in clients[0].vqs]
        for c in clients:
            for v, g in zip(c.vqs, global_cbs):
                v.set_codebook(g.to(device))
    global_cb = global_cbs[0]

    if enc_sched == "cosine":
        _attach_cosine_schedules(clients, rounds, local_epochs,
                                 float(getattr(base_cfg.training, "warmup_rate", 0.1)))
        print(f"[fed] LR schedule: warmup+cosine over the full budget "
              f"({rounds} rounds x {local_epochs} local epochs) — RECIPE-MATCHED to the "
              f"`local` baseline, and therefore NOT comparable with federated arms trained "
              f"at constant LR.")
    elif enc_sched != "none":
        raise ValueError(f"unknown enc_sched {enc_sched!r} (use 'none' or 'cosine')")

    # Opt-in training-visualization capture (default OFF; see _make_stage1_viz).
    viz = _make_stage1_viz(client_entities, base_cfg)

    # Optional: federate the ENCODER too (align the tokenizer so the shared codebook
    # is coherent). Broadcast a common init before round 0 — FedAvg needs a shared start.
    if fed_encoder != "off" and enc_fed_algo not in {"fedavg", "fedprox", "fedproto"}:
        # Validated BEFORE the `if enc_keys` block: a typo must never fall through to a
        # silently unfederated encoder.
        raise ValueError(f"unknown enc_fed_algo {enc_fed_algo!r} (use 'fedavg', 'fedprox' or 'fedproto')")
    enc_keys = (_encoder_shared_keys(clients[0].model, fed_encoder, enc_split_at, enc_bn)
                if fed_encoder != "off" else [])
    if fed_encoder != "off" and not enc_keys:
        # Reachable: 'neck' selects on the substring '.proj.', and an encoder whose project
        # block is an nn.Identity has none. Falling through would train an arm advertised as
        # 'federated encoder' that is in fact `federated_cb_only` — an invisible false null.
        raise SystemExit(f"fed_encoder={fed_encoder!r} selected 0 encoder tensors "
                         f"(split_at={enc_split_at}) — the encoder would NOT be federated. "
                         f"Check the mode against the model's parameter names.")
    enc_param_names: list[str] = []
    # `fedproto` federates in function space: the shared tensors are still identified
    # (they define the common init and the drift telemetry) but they are NOT averaged,
    # unless the hybrid ablation asks for it.
    enc_avg = bool(enc_keys) and (enc_fed_algo != "fedproto" or enc_proto_fedavg)
    if enc_keys:
        # Common start for EVERY algorithm — including fedproto: without it the arms would
        # differ by their random inits as well as by their federation, and no comparison
        # between them would be attributable. (FedProto tolerates heterogeneous models by
        # design; here the models are identical by construction, so a common init is free.)
        _broadcast_encoder(clients, devices, enc_keys)
        enc_param_names = _shared_param_names(clients[0].model, enc_keys)
        detail = {"fedavg": "server weight-average",
                  "fedprox": f"server weight-average + local prox mu={enc_prox_mu}",
                  "fedproto": (f"NO weight averaging; prototype pull lambda={enc_proto_weight} "
                               f"agg={enc_proto_agg}"
                               + (" + weight-average (hybrid)" if enc_proto_fedavg else ""))}[enc_fed_algo]
        bn_label = {"buffers_local": "BN affine SHARED, running stats local (NOT FedBN)",
                    "fedbn": "whole BN layer local (FedBN, Li et al. ICLR'21)",
                    "shared": "BN affine shared + running stats POOLED"}[enc_bn]
        if not enc_avg:
            # THE BANNER MUST NOT OVERSTATE WHAT RUNS. `_fedavg_encoder` and, inside the same
            # branch, `_pool_encoder_bn` are both gated on `enc_avg`, which pure fedproto
            # (no --fedproto-fedavg) sets False: the round loop never runs a server weight
            # step, so nothing about BN is federated after the common init — neither the
            # affine parameters nor the running statistics. Printing "affine shared /
            # stats POOLED" there would describe code that is skipped.
            #
            # Rewritten rather than REFUSED, on purpose: `--fed-enc-bn` also selects WHICH
            # tensors enter `enc_keys`, and under fedproto that set still does real work —
            # it defines the common init and the denominator of the drift telemetry
            # (`enc_drift`, `enc_step`). So `fedproto --fed-enc-bn shared` is a meaningful,
            # if easily misread, configuration; only the banner was wrong.
            bn_label = {
                "buffers_local": "BN affine shared AT INIT ONLY, running stats local",
                "fedbn": "whole BN layer local (FedBN) — and nothing else is averaged either",
                "shared": ("BN affine + running stats shared AT INIT ONLY — NOT pooled per "
                           "round (_pool_encoder_bn runs only in the aggregation step this "
                           "algo skips; use --fedproto-fedavg for the hybrid that pools)"),
            }[enc_bn]
        print(f"[fed] encoder federation={fed_encoder} algo={enc_fed_algo}: {len(enc_keys)} shared "
              f"tensors ({len(enc_param_names)} params), {detail}; {bn_label}"
              f"{' + first %d blocks local' % enc_split_at if fed_encoder == 'partial' else ''}")
        if enc_fed_algo == "fedproto" and enc_proto_weight > 0:
            if merge != "suffstat":
                raise ValueError(
                    f"enc_fed_algo='fedproto' is incompatible with merge={merge!r}.\n"
                    "This is a SEMANTIC obstacle, not a missing feature. FedProto aggregates one\n"
                    "prototype PER CLASS, and here the class of a latent vector is the codebook\n"
                    "index it was assigned. That aggregation is only meaningful while index k\n"
                    "denotes the same thing on every client, which is exactly what the shared\n"
                    "broadcast codebook guarantees and what a per-client codebook destroys:\n"
                    "with local dictionaries, averaging client A's code-7 centroid with client\n"
                    "B's code-7 centroid averages two unrelated regions of latent space (the\n"
                    "usual FL permutation/alignment problem, here at the dictionary level).\n"
                    "Use merge='suffstat' for fedproto, or federate the encoder with\n"
                    "fedavg/fedprox if you want a local codebook.")
            if enc_proto_code_weight not in {"uniform", "count"}:
                # The last unvalidated knob of the trio. `_proto_term` branches
                # `if code_weight == "count": … else: …`, so a typo does not fail — it runs
                # the UNIFORM weighting under a label nobody would read again, exactly the
                # silent-fallback failure the enc_fed_algo / enc_prox_form / enc_proto_agg
                # checks exist to prevent.
                raise ValueError(f"unknown enc_proto_code_weight {enc_proto_code_weight!r} "
                                 f"(use 'uniform' or 'count')")
            if len(clients[0].vqs) > 1:
                print(f"[fed] WARNING: Residual-VQ with {len(clients[0].vqs)} stages — the prototype "
                      f"term uses STAGE 0 ONLY. Later stages quantize a straight-through residual, "
                      f"so their tokens carry no gradient to the encoder.")
            if enc_proto_agg == "count":
                # Twin of the fedprox mu-below-the-floor warning below, and for the same
                # reason: an arm that cannot carry information must not be reported as a null.
                # Under agg='count' the prototype target is p̄_k = Σ_j m_j^k / Σ_j n_j^k, which
                # is the Prop.1 merge `e = M/smoothed(N)` on the SAME round's (N, M) — the
                # codebook this loop has already broadcast to every client (measured with
                # this repo's own functions: ‖p_count − e‖/‖e‖ = 3.3e-08, and the paper's
                # 2e-8 on the toy). The pull then targets a vector the commitment loss is
                # already pulling toward, so nothing crosses the network that
                # `federated_cb_only` + common-init does not already send.
                print(f"[fed] WARNING: fedproto agg='count' — the prototype target IS the merged "
                      f"codebook (rel err ~1e-8), so this arm carries NO cross-client information "
                      f"beyond federated_cb_only + common init: it is a reweighted commitment term "
                      f"under a FedProto label. Check proto_agg_gap in the round log — the "
                      f"pre-registered kill rule (documentation/research/FED_ENCODER_ALGOS.md) is 5%, and "
                      f"'count' pins the gap at ~0 BY CONSTRUCTION. The non-degenerate reading is "
                      f"agg='uniform' — one vote per client, which is what the authors' released "
                      f"`proto_aggregation` computes even though Eq. 6 of the paper is written "
                      f"count-weighted; here that textual reading is the degenerate one.")
            for c in clients:
                c.proto_weight = float(enc_proto_weight)
                c.proto_code_weight = enc_proto_code_weight
                c.vqs[0].export_proto_tokens = True   # stage 0 ONLY (see _proto_term)
        elif enc_fed_algo == "fedproto":
            print("[fed] NOTE: fedproto with lambda=0 and no hybrid averaging — the encoder gets a "
                  "COMMON INIT and is then never synchronised. This is the `commoninit` null arm, "
                  "not a federated encoder.")
        if enc_fed_algo == "fedprox":
            if enc_prox_form not in {"loss", "decoupled"}:
                raise ValueError(f"unknown enc_prox_form {enc_prox_form!r} (use 'loss' or 'decoupled')")
            if enc_prox_mu > 0 and not enc_param_names:
                # `_prox_term`'s stated contract, checked where it is decidable. The shared
                # SET is non-empty (guarded above) but may hold only buffers — no gradient,
                # no anchor — and then `_local_train_stage1` builds an empty `prox_pairs`,
                # skips the penalty and reports prox_term=NaN, i.e. an arm labelled FedProx
                # that IS FedAvg. Refuse instead of producing that row.
                raise SystemExit(
                    f"fedprox mu={enc_prox_mu} but the shared encoder set contains 0 "
                    f"PARAMETERS ({len(enc_keys)} tensors, all buffers) — the proximal "
                    f"anchor has nothing to pull and this arm would be plain FedAvg. "
                    f"Check --fed-enc-scope/--fed-enc-bn against the model's parameter names.")
            if enc_prox_mu <= 0:
                print("[fed] NOTE: fedprox with mu=0 — identical to fedavg by construction.")
            elif enc_prox_form == "decoupled":
                # TWIN OF THE 'loss' WARNING BELOW, for the form whose real knob is lr*mu.
                # Each APPLIED step contracts the drift by (1 - lr*mu), so a round removes
                # 1-(1-lr*mu)^steps of it — the pre-registered `prox_pull_frac >= 5%` gate
                # (documentation/research/FED_ENCODER_ALGOS.md). At the default lr=1e-3 and mu=0.01
                # that is 0.015%/round: ~330x under the gate, and until now silent.
                #
                # `steps` is the OPTIMISTIC estimate local_epochs x len(train_loader) of
                # client 0 (the real count drops with GradScaler skips and varies per client,
                # and under enc_sched='cosine' the average lr is BELOW the peak) — so the
                # warning fires only when even the friendliest arithmetic is under the gate.
                # `initial_lr` and not the live lr: a freshly attached warmup schedule reads
                # lr=0 at t=0, which would make every run look dead.
                g0 = clients[0].opt.param_groups[0]
                lr0 = float(g0.get("initial_lr", g0["lr"]))
                est_steps = max(1, local_epochs * len(clients[0].data.train_loader))
                a = min(1.0, lr0 * enc_prox_mu)
                pull = 1.0 - (1.0 - a) ** est_steps
                if pull < 0.05:
                    print(f"[fed] WARNING: fedprox form='decoupled' with mu={enc_prox_mu} at "
                          f"lr={lr0:g} removes an estimated {pull:.3%} of the drift per round "
                          f"(<= {est_steps} steps) — under the pre-registered prox_pull_frac >= 5% "
                          f"gate, so the anchor has no authority over a round and 'FedProx ties "
                          f"FedAvg' would be a statement about the solver. The knob is lr*mu "
                          f"(={a:.2g}), NOT mu: reach 5% by raising mu by decades. Read "
                          f"prox_pull_frac in the round log for the measured value.")
            elif enc_prox_form == "loss" and enc_prox_mu < 0.1:
                # Measured on toy: mu=0.1 gives prox_grad_ratio 1.7e-4..5.5e-3, i.e. the
                # penalty is 2-3 orders of magnitude under the data gradient. Anything
                # smaller is a no-op, and a no-op reported as "FedProx ties FedAvg" is a
                # false null. Warn at setup rather than only after the fact in the logs.
                print(f"[fed] WARNING: fedprox form='loss' with mu={enc_prox_mu} is likely BELOW the "
                      f"detection floor (AdamW divides the prox gradient by sqrt(v_hat)). Check "
                      f"prox_grad_ratio in the round log; if it stays under 1e-2 this arm is a "
                      f"solver no-op, not a null result. Raise mu or use form='decoupled'.")
            for c in clients:
                c.prox_mu = float(enc_prox_mu)
                c.prox_form = enc_prox_form
    enc_prox_ref_needed = bool(enc_keys) and enc_fed_algo == "fedprox" and enc_prox_mu > 0
    enc_proto_on = bool(enc_keys) and enc_fed_algo == "fedproto" and enc_proto_weight > 0
    # NOTE: the round-0 prototype seeding used to sit HERE, and that was a bug on the resume
    # path — `_load_resume` runs further down and never touches `client.protos`, so a resumed
    # fedproto run entered round 0 pulling toward the codebook of the freshly BUILT client
    # (random init), not the restored one. It is now done in `_seed_round0_protos()` below,
    # after the resume. Non-resumed runs are byte-identical: the codebook they seed from is
    # the same broadcast one either way.

    # B1: probe-based alignment. Encoders stay LOCAL; a per-round consensus encoding of a
    # SHARED public probe is broadcast and each client is pulled toward it (permutation-free).
    align_on = fed_align != "off" and align_weight > 0
    if align_on:
        probe = _load_public_probe(base_cfg, n_windows=64, cluster=align_cluster)
        if probe is None:
            print("[fed] align: no public probe found -> alignment DISABLED")
            align_on = False
        else:
            for c in clients:
                c.probe = probe; c.align_weight = float(align_weight)
            print(f"[fed] alignment ON (probe_mse): weight={align_weight}, probe={tuple(probe.shape)}, "
                  f"cluster={align_cluster}")

    # Server-side codebook EMA (federated_cb_only_ema): one {"C","S"} accumulator per
    # RVQ stage, zero-initialized and kept alive ACROSS rounds. CPU (the merge runs on
    # CPU). None ⇒ every other arm keeps the plain per-round merge, untouched.
    server_ema = None
    if cb_server_ema_decay is not None and merge != "suffstat":
        raise ValueError(f"cb_server_ema_decay={cb_server_ema_decay} is only defined for "
                         f"merge='suffstat' (it smooths the AGGREGATE sufficient statistics); "
                         f"got merge={merge!r}. Silently ignoring it would mislabel the run.")
    if cb_server_ema_decay is not None and merge == "suffstat":
        if not 0.0 <= cb_server_ema_decay < 1.0:
            # γ=1 leaves C,S pinned at 0 (codebook → all-zeros); γ<0 / γ>1 are meaningless.
            raise ValueError(f"cb_server_ema_decay must be in [0, 1), got {cb_server_ema_decay}")
        server_ema = [{"C": torch.zeros_like(v.ema_cluster_size.detach().cpu()),
                       "S": torch.zeros_like(v.ema_embed_sum.detach().cpu()),
                       "w": 0.0}                    # cold-start weight 1−γ^(t+1) for bias correction
                      for v in clients[0].vqs]
        print(f"[fed] server codebook EMA ON: gamma={cb_server_ema_decay} "
              f"({len(server_ema)} stage(s)); e_j = S_j / smoothed(C_j)")

    # RESUME: overwrite the freshly built clients with a previous run's state. Done AFTER
    # `server_ema` is allocated (the bundle refills it) and AFTER the encoder-federation setup
    # (which only sets flags), but BEFORE the round loop and before any schedule is stepped.
    rounds_done = 0
    if resume_from is not None:
        rounds_done = _load_resume(clients, devices, resume_from, base_cfg, merge, server_ema)
        if rounds_done == 0 and resume_rounds_done > 0:
            # Weights-only resume: the source predates the resume bundle, so it cannot say how
            # far it got. Without this the continuation would label its rounds 0..N-1 — a
            # history that reads as a fresh run when it is in fact rounds 30..30+N-1, which is
            # exactly the misreading this experiment must not invite. Declared by the caller.
            rounds_done = int(resume_rounds_done)
            print(f"[fed] source round count DECLARED as {rounds_done} (--resume-rounds-done); "
                  f"the continuation is labelled from there.")
        if stats_merge:
            # Re-seat the server's view of the codebook from the restored clients, and check
            # they really do all hold the same one (they should: it was broadcast).
            global_cbs = [v.codebook.weight.detach().cpu().clone() for v in clients[0].vqs]
            for c in clients:
                for v, g in zip(c.vqs, global_cbs):
                    assert torch.allclose(v.codebook.weight.detach().cpu(), g), \
                        (f"resumed clients disagree on the codebook — {c.entity_id} differs "
                         f"from {clients[0].entity_id}. The checkpoints are not from one run.")
            global_cb = global_cbs[0]
        print(f"[fed] continuing: rounds {rounds_done} -> {rounds_done + rounds} "
              f"({rounds} more)")

    # Round-0 prototypes, seeded from the codebook the clients ACTUALLY hold — i.e. after the
    # resume, not before it. Without any seeding the encoder is unfederated for the whole of
    # round 0 (prototypes only exist once a round of stats has been collected), so a 3-round
    # smoke would federate in 2 of the 3 rounds and the arm's budget would not match its siblings.
    # The broadcast codebook IS the count-aggregated prototype by definition, so seeding with
    # it is the arm's own definition under agg='count' and an explicit, logged warm start
    # under 'uniform'.
    if enc_proto_on and enc_proto_seed_round0:
        for c, d in zip(clients, devices):
            c.protos = [c.vqs[0].codebook.weight.detach().clone().to(d)]
            c.proto_mask = [torch.ones(K, dtype=torch.bool, device=d)]
        print(f"[fed] fedproto: round-0 prototypes seeded from the broadcast codebook "
              f"(agg={enc_proto_agg}, code_weight={enc_proto_code_weight}"
              f"{', post-resume' if resume_from is not None else ''})")

    history: list[dict] = []
    best_val, best_round, best_cb, best_states = float("inf"), None, None, None
    # Codebook-drift telemetry: ‖e^t − e^(t-1)‖ per round. Round 0 measures movement from
    # the random broadcast init; later rounds the round-to-round step. The server EMA should
    # visibly DAMP this vs γ=0 — the direct evidence that the EMA smooths the k-FED M-step.
    prev_cb = global_cb.detach().cpu().clone()
    prev_enc = None
    stale = 0                       # rounds since the last val improvement (patience counter)
    min_delta = float(getattr(base_cfg.training, "early_stopping_min_delta", 0.0))                 # last round's per-client flat encoders (for `enc_step`)
    for r in range(rounds):
        losses: list[float] = []
        proto_cov = float("nan")        # fraction of codes carrying a prototype (fedproto only)
        proto_gap = float("nan")        # ‖p̄_uniform − p̄_count‖/‖p̄_count‖ — the degeneracy witness
        # FedProx: w^t is the encoder the clients START this round from — i.e. the state
        # right after round r-1's aggregation (or after the common init at r=0). Snapshot
        # HERE, before any local step, or the anchor would be the client's own drifted
        # weights and the proximal term would be identically zero at the start of a round.
        if enc_prox_ref_needed:
            for c in clients:
                c.prox_ref = _snapshot_prox_ref(c.model, enc_param_names)
        if align_on:
            # Consensus encoding of the shared probe from the CURRENT (pre-training) encoders,
            # broadcast as this round's alignment target. Encoder-only, eval-mode BN, no grad.
            with torch.no_grad():
                zs = []
                for c, d in zip(clients, devices):
                    enc = c.model.encoder; enc.eval()
                    zs.append(enc(c.model.transform(c.probe.to(d))).detach().float().cpu())
                    enc.train()
                z_ref = torch.stack(zs).mean(0)
            for c, d in zip(clients, devices):
                c.z_ref = z_ref.to(d)
        if stats_merge:
            # Per STAGE: each Residual-VQ stage merges by its own independent k-FED M-step
            # (each has its own frozen broadcast codebook + round stats). A single-codebook
            # VQ has one stage → identical to the original single-codebook merge.
            n_stages = len(clients[0].vqs)
            # CPU master copies (per stage): clients may live on different GPUs, and
            # `torch.allclose` RAISES (not returns False) across devices.
            cb_broadcasts = [v.codebook.weight.detach().cpu().clone() for v in clients[0].vqs]
            counts = [[] for _ in range(n_stages)]
            sums = [[] for _ in range(n_stages)]
            for ci, (c, d) in enumerate(zip(clients, devices)):
                # ABSOLUTE round index: a resumed run must continue the RNG stream, not replay
                # round 0's batch order. `rounds_done` is 0 for every non-resumed run, so this
                # is byte-identical to the previous behaviour there.
                torch.manual_seed(_round_seed(seed, rounds_done + r, ci, stage=1))
                losses.append(_local_train_stage1(c, local_epochs, d))   # trains ALL stages at once
                if viz is not None:                    # opt-in: recon + latent dump (eval/no_grad, RNG-safe)
                    viz.capture_client(c, r, d)
                for s, v in enumerate(c.vqs):
                    # CORRECTNESS: each stage's codebook must be UNCHANGED by local training.
                    assert torch.allclose(v.codebook.weight.cpu(), cb_broadcasts[s]), \
                        f"client {c.entity_id} mutated stage-{s} codebook during local training!"
                    n, m = v.pull_round_stats()
                    counts[s].append(n.cpu()); sums[s].append(m.cpu())
            if merge == "suffstat":
                merged = [_server_merge(counts[s], sums[s], eps, dead_thr, revive_dead,
                                        ema_state=(server_ema[s] if server_ema is not None else None),
                                        ema_decay=(cb_server_ema_decay if server_ema is not None else None))
                          for s in range(n_stages)]
            else:                                # union_recluster (no EMA, no revive: guarded above)
                merged = [_server_union_recluster(counts[s], sums[s],
                                                  prev_cb=cb_broadcasts[s],
                                                  eps=eps, threshold=dead_thr)
                          for s in range(n_stages)]
            global_cbs = [mg[0] for mg in merged]    # per-stage CPU master
            n_dead = sum(int(mg[3]) for mg in merged)
            for c, d in zip(clients, devices):        # carry the aggregated EMA state too
                for s, v in enumerate(c.vqs):
                    weight_s, N_s, M_s, _ = merged[s]
                    v.set_codebook(weight_s.to(d), ema_cluster_size=N_s.to(d), ema_embed_sum=M_s.to(d))
            global_cb = global_cbs[0]                 # stage-0 representative for history/return
            N, M = merged[0][1], merged[0][2]         # stage-0 usage diag (history/select_on_val)
            if enc_proto_on:
                # FedProto server step: aggregate the SAME (n_j, m_j) into per-code global
                # prototypes and broadcast them for the NEXT round (they summarise the round
                # that just ended — the one-round staleness FedProto has by construction).
                # Stage 0 only: later RVQ stages are gradient-dead (see `_proto_term`).
                p0, live0, proto_gap = _aggregate_prototypes(counts[0], sums[0], enc_proto_agg,
                                                             min_count=dead_thr)
                for c, d in zip(clients, devices):
                    c.protos = [p0.to(d)]
                    c.proto_mask = [live0.to(d)]
                proto_cov = float(live0.float().mean())   # fraction of codes with a prototype
        elif merge == "fedavg":
            # (A)-ABLATION baseline: clients update the codebook LOCALLY via EMA
            # (collect_stats_only=False), then the server naively FedAvg's the VQ
            # state (data-size weighted) — the obvious-but-naive FL of the codebook,
            # contrasted against the Prop.1 sufficient-statistic merge.
            #
            # The broadcast EMA state must satisfy TWO constraints:
            #
            #  (i) counts ≫ threshold_ema_dead_code. The previous version reseeded
            #      ema_cluster_size=1, which is BELOW the threshold (2), so on the
            #      FIRST local step after every broadcast `_expire_dead_codes`
            #      declared ~46% of the just-averaged codebook dead and overwrote it
            #      with random samples from the current client's batch (measured:
            #      267 of 347 total replacements happened on those first steps). The
            #      baseline was handicapped by self-inflicted codebook destruction,
            #      not by weight-averaging — which is what the ablation must isolate.
            #
            # (ii) the broadcast must be a FIXED POINT of the first local EMA step,
            #      i.e. ema_embed_sum / smoothed(ema_cluster_size) == global_cb. So we
            #      reinstall sums as `global_cb * counts`, NOT the data-size average of
            #      the clients' raw ema_embed_sum. Averaging the raw sums would make the
            #      implied centroid Σm/Σn — a count-weighted pooled centroid, i.e. an
            #      approximation of the Prop.1 merge itself — which would blur the very
            #      contrast this arm exists to measure. Weight-averaging stays pure.
            #
            # What still separates this from `suffstat`: here the codebook MOVES during
            # local training (local EMA + dead-code expiry), so each client's token
            # assignments are made against a divergent, client-specific codebook. That
            # divergence is contribution (A)'s target, and it is preserved.
            ws = []
            for ci, (c, d) in enumerate(zip(clients, devices)):
                torch.manual_seed(_round_seed(seed, rounds_done + r, ci, stage=1))
                losses.append(_local_train_stage1(c, local_epochs, d))
                ws.append(float(len(c.data.train_dataset)))
            N = torch.stack([c.vq.ema_cluster_size.detach().cpu() for c in clients]).sum(0)  # usage diag
            wsum = sum(ws)
            def _fedavg(get) -> torch.Tensor:
                # Accumulate on CPU: summing tensors that live on different GPUs raises.
                return sum((w / wsum) * get(c).detach().cpu() for w, c in zip(ws, clients))
            global_cb = _fedavg(lambda c: c.vq.codebook.weight)          # the naive FedAvg (CPU)
            avg_size = _fedavg(lambda c: c.vq.ema_cluster_size)          # realistic counts (i)
            avg_sum = global_cb * avg_size.unsqueeze(1)                  # fixed point (ii)
            for c, d in zip(clients, devices):
                c.vq.set_codebook(global_cb.to(d), ema_cluster_size=avg_size.to(d).clone(),
                                  ema_embed_sum=avg_sum.to(d).clone())
            n_dead = int((N < dead_thr).sum())
        elif merge == "local":
            # NO dictionary federation at all: train, and that is the whole round. Each
            # client's VQ moved under its own EMA + dead-code expiry during
            # `_local_train_stage1`, and nothing is merged or broadcast.
            for ci, (c, d) in enumerate(zip(clients, devices)):
                torch.manual_seed(_round_seed(seed, rounds_done + r, ci, stage=1))
                losses.append(_local_train_stage1(c, local_epochs, d))
                if viz is not None:
                    viz.capture_client(c, r, d)
            # Diagnostics only. There is no global codebook: N is the SUM of the per-client
            # EMA usage counts (so `perplexity` reads the pooled usage of K DIFFERENT
            # dictionaries — a cohort-level spread statistic, NOT one dictionary's), and
            # `global_cb`/`cb_drift` track CLIENT 0 as a representative.
            N = torch.stack([c.vq.ema_cluster_size.detach().cpu() for c in clients]).sum(0)
            global_cb = clients[0].vq.codebook.weight.detach().cpu().clone()   # clone: .cpu() is a
            # no-op alias when already on CPU, and the caller must not hold client 0's live weight
            n_dead = int((N < dead_thr).sum())
        else:
            raise ValueError(f"unknown merge {merge!r} (use 'suffstat', 'union_recluster', "
                             f"'fedavg' or 'local')")

        # CORRECTNESS: every client now holds the identical global codebook. Skipped for
        # merge='local', where divergent per-client dictionaries are the POINT.
        # Compared on CPU — `allclose` across devices raises.
        for c in (clients if not cb_local else []):
            assert torch.allclose(c.vq.codebook.weight.cpu(), global_cb.cpu()), "broadcast mismatch!"

        # Align the encoders → the shared codebook becomes coherent (and Prop.1's
        # comparable-assignment precondition holds). AFTER the codebook merge, so the
        # frozen-codebook assert above is untouched.
        # Drift is read BEFORE aggregation: post-average it is 0 by construction for
        # fedavg/fedprox, so only the pre-average value distinguishes the algorithms.
        dstat = _encoder_drift(clients, enc_param_names, _client_weights(clients), prev_enc) \
            if enc_keys else {"drift": float("nan"), "drift_rel": float("nan"),
                              "step": float("nan")}
        if enc_avg:
            _fedavg_encoder(clients, devices, enc_keys)
            if enc_bn == "shared":
                _pool_encoder_bn(clients, devices, _client_weights(clients), enc_keys)
        # AFTER aggregation: this is the state the clients will start the next round from,
        # so next round's `enc_step` is pure local movement for every arm alike.
        prev_enc = _flat_all(clients, enc_param_names)

        # Model selection on the aggregated model, scored on each client's held-out
        # `val` and averaged UNIFORMLY over clients. Uniform, not n-weighted: the
        # cluster ships ONE codebook, and a size-weighted criterion would let the
        # largest silo decide when the cohort has converged.
        val_loss = _mean_finite([_val_loss_stage1(c, d) for c, d in zip(clients, devices)]) \
            if select_on_val else float("nan")

        # Bias-correct the dead diagnostic when the server EMA is on (raw C is cold-start
        # damped); perplexity is scale-invariant so it reads the same on C or Ĉ.
        dead_ref = N if server_ema is None else N / max(server_ema[0]["w"], eps)
        cur_cb = global_cb.detach().cpu()
        cb_drift = float((cur_cb - prev_cb).norm())
        prev_cb = cur_cb.clone()
        if viz is not None and merge == "suffstat":    # opt-in: per-round global codebook + EMA state
            viz.capture_codebook(r, global_cbs, merged, server_ema, cb_drift, N)
        rec = {
            "round": rounds_done + r,        # absolute across resumes, so histories concatenate
            "mean_loss": sum(losses) / len(losses),
            "val_loss": val_loss,
            "perplexity": _perplexity_from_counts(N),
            "dead_frac": float((dead_ref < dead_thr).float().mean()),
            "n_dead_revived": n_dead,
            "cb_drift": cb_drift,       # ‖e^t − e^(t-1)‖ — server EMA should damp this vs γ=0
            # Encoder-federation telemetry (NaN unless the corresponding arm is on).
            "enc_drift": dstat["drift"],        # mean ‖w_j − w̄‖ BEFORE aggregation
            "enc_drift_rel": dstat["drift_rel"],   # …relative to ‖w̄‖ (scale-free)
            "enc_step": dstat["step"],         # mean ‖w_j^r − w_j^{r-1}‖ — cross-arm comparable
            "mean_obj": _mean_finite([c.last_obj for c in clients]),   # task loss + penalties
            "prox_term": _mean_finite([c.last_prox for c in clients]),
            "prox_grad_ratio": _mean_finite([c.last_prox_ratio for c in clients]),   # 'loss' form
            "prox_pull_frac": _mean_finite([c.last_prox_pull for c in clients]),     # 'decoupled'
            "proto_term": _mean_finite([c.last_proto for c in clients]),
            "proto_coverage": proto_cov,
            "proto_agg_gap": proto_gap,
            "steps_per_client": [c.last_steps for c in clients],
            "skipped_steps": sum(c.last_skipped for c in clients),
        }
        history.append(rec)
        enc_msg = ""
        if enc_keys:
            enc_msg = f" enc_drift={dstat['drift']:.3f}"
            if math.isfinite(rec["prox_term"]):
                enc_msg += f" prox={rec['prox_term']:.4g}"
                if math.isfinite(rec["prox_grad_ratio"]):
                    enc_msg += f" g_ratio={rec['prox_grad_ratio']:.2e}"
                if math.isfinite(rec["prox_pull_frac"]):
                    enc_msg += f" pull={rec['prox_pull_frac']:.1%}/round"
            if math.isfinite(rec["proto_term"]):
                enc_msg += (f" proto={rec['proto_term']:.4g} cov={proto_cov:.1%} "
                            f"agg_gap={proto_gap:.1%}")
        # Absolute round index, matching rec['round'], so a resumed log reads as a
        # continuation ("round 30, 31, …") rather than restarting the count at 0.
        print(f"[fed:{merge}] round {rounds_done + r}: loss={rec['mean_loss']:.4f} "
              + (f"val={val_loss:.4f} " if math.isfinite(val_loss) else "")
              + f"perplexity={rec['perplexity']:.1f}/{K} "
              f"dead={rec['dead_frac']:.1%} revived={n_dead} drift={cb_drift:.3f}{enc_msg}")

        # Snapshot rather than early-stop: every round in the budget is still run, so
        # the gradient-step count stays identical across conditions (the ablation needs
        # that), and `val` only decides WHICH round's weights we keep.
        improved = math.isfinite(val_loss) and val_loss < best_val - min_delta
        if math.isfinite(val_loss) and val_loss < best_val:
            best_val, best_round = val_loss, r
            best_cb = global_cb.detach().clone()
            best_states = [copy.deepcopy(c.model.state_dict()) for c in clients]
        # ── convergence stop ─────────────────────────────────────────────────
        # Rounds are expensive (8-18 h for 30 on wsd), and the arms have historically been
        # TRUNCATED rather than converged — val was still falling 1-5%/round at the last
        # round of every converged cb_only run on disk. Patience lets a generous round budget
        # be requested safely: the run stops when it actually flattens and says so, instead
        # of either stopping early by construction or burning the whole budget past the knee.
        # Mirrors the baselines' own criterion (federated_eval._converged_loop) at round
        # granularity. OFF by default, so no existing arm changes.
        if patience_rounds > 0 and math.isfinite(val_loss):
            stale = 0 if improved else stale + 1
            if stale >= patience_rounds:
                print(f"[fed:{merge}] CONVERGED: no val improvement > {min_delta:g} for "
                      f"{stale} rounds (best={best_val:.5f} at round {rounds_done + best_round}); "
                      f"stopping at round {rounds_done + r} of {rounds_done + rounds - 1}.")
                rec["converged_stop"] = True
                break
        elif patience_rounds > 0:
            print("[fed] patience_rounds set but val is not finite — is select_on_val on? "
                  "Convergence stop DISABLED for this run.")

    # ── TRUNCATION ALARM ────────────────────────────────────────────────────────
    # A run whose BEST round is its LAST round was still improving when the budget ran out:
    # it is truncated, not converged. That is not a nuance — measured 2026-07-24 on wsd c3,
    # a 30-round cb_only run left 4 of 6 entities with a near-random detector (VUS-PR
    # 0.03-0.10), and continuing to convergence took the cluster from 0.16 to 0.69. Two
    # independent 30-round runs both showed the broken value, so it reproduces. Every
    # federated result on disk from before that date carries this risk, and nothing in the
    # output said so. Now it does, loudly, in every run that ends this way.
    if math.isfinite(best_val) and best_round == rounds - 1 and rounds > 1:
        print("\n" + "!" * 78)
        print("!! TRUNCATED, NOT CONVERGED: the best validation round IS the last round "
              f"({rounds_done + best_round}).")
        print("!! The model was still improving when the round budget ran out, so this run "
              "reports an")
        print("!! UNDERTRAINED model. On wsd c3 that difference was VUS-PR 0.16 vs 0.69.")
        print("!! Fix: raise --s1-rounds and add --fed-patience-rounds N so the run stops "
              "where it")
        print("!! actually flattens, or continue this one with --resume-from.")
        print("!" * 78 + "\n", flush=True)
        history[-1]["truncated"] = True

    if best_states is not None and best_round != rounds - 1:
        for c, sd in zip(clients, best_states):
            c.model.load_state_dict(sd)
        global_cb = best_cb
        print(f"[fed:{merge}] restored round {best_round} (val={best_val:.4f}); "
              f"last round was {history[-1]['val_loss']:.4f}")
    if best_round is not None:
        history[best_round]["selected"] = True

    if viz is not None:                                # opt-in: flush codebook history to disk
        viz.finalize()

    if resume_out is not None:
        # Written for EVERY federated run, not only resumed ones: the whole point is that a
        # run one later decides to extend can be continued exactly, and that decision is
        # normally made after seeing the curve.
        _save_resume_bundle(clients, resume_out, rounds_done + rounds, server_ema, merge)

    # Leave every model in the state the rest of the pipeline expects: stage 2, the
    # checkpointer and `detect.py` all run this same VQ, and a live export flag would keep
    # stashing tensors nobody consumes (and a stale `_last_tokens` would pin a dead graph).
    for c in clients:
        for v in c.vqs:
            v.export_proto_tokens = False
            v._last_tokens = v._last_indices = None

    # Return on the primary device, as before (callers assume a CUDA tensor when CUDA is up).
    return clients, global_cb.to(device), history


# ─── STEP 3: federated Stage 2 (partially-personalized MaskGIT prior) ─────────

# Prior parameters kept LOCAL (personalized) — they encode silo-specific
# normality. Everything else (token/freq/time embeddings, transformer, pred head)
# is the C-independent shared body, FedAvg'd across clients.
#
# At C=1 `channel_embedding` is an nn.Embedding(1, d): one vector added to every
# token, i.e. a constant the LayerNorm/bias can absorb. It carries no
# personalization there — `output_bias` (one row per (c, f, w) position) is the
# whole (B) arm. `federated_stage2` warns rather than silently reporting a
# head-divergence number driven by a dead parameter.
LOCAL_PRIOR_PREFIXES = ("channel_embedding", "output_bias")


@dataclass
class Stage2ClientState:
    entity_id: str
    cfg: Config
    s2: Stage2System
    opt: torch.optim.Optimizer
    data: object
    n_windows: int
    # Prior-training loader at `batch_size_stage2`. `data.train_loader` is built by
    # `_build_client` with stage="stage1", so every prior loop that iterated it was
    # silently training at the STAGE-1 batch size and `batch_size_stage2` was dead
    # config. Same windows, same order source — only the batching differs.
    s2_loader: object = None
    # Per-client adaptive loss scale, kept alive across rounds like `opt`. Never aggregated.
    scaler: torch.amp.GradScaler = field(default_factory=lambda: _make_scaler())
    # Optional warmup+cosine over the WHOLE run (see `enc_sched`). None = constant LR,
    # which is what every federated arm on disk was trained with.
    sched: "torch.optim.lr_scheduler.LambdaLR | None" = None
    # ── FedProx on the SHARED prior body (port of the stage-1 encoder mechanics) ──
    # μ=0 keeps every field inert: `prox_ref` is never set, the train loops build an
    # empty pair list and the update path is bit-identical to plain FedAvg.
    prox_mu: float = 0.0
    prox_form: str = "decoupled"                  # "decoupled" | "loss" — see _prox_term
    prox_ref: "dict[str, torch.Tensor] | None" = None   # w^t (start-of-round broadcast body), fp32
    last_prox: float = float("nan")               # mean נw − w^t南2 per step
    last_prox_ratio: float = float("nan")         # 'loss' form diagnostic (see _prox_grad_ratio)
    last_prox_pull: float = float("nan")          # 'decoupled' form diagnostic (drift removed/round)


def _prior_shared_keys(prior, local_prefixes: tuple[str, ...] = LOCAL_PRIOR_PREFIXES) -> list[str]:
    # `str.startswith(())` is False for every key, so local_prefixes=() ⇒ ALL keys
    # shared ⇒ the fully-shared "naive federation" baseline arm.
    return [k for k in prior.state_dict().keys() if not k.startswith(local_prefixes)]


def _build_stage2_client(s1c: ClientState, device: torch.device) -> Stage2ClientState:
    cfg = s1c.cfg
    example = next(iter(s1c.data.train_loader))["inputs"][:1].to(device)
    s2 = Stage2System(cfg, s1c.model)              # freezes stage1 (global codebook, already on device)
    s2.prior.to(device)                            # eager prior modules → device BEFORE materialize
    s2.materialize(example)                        # discovers C,F,W; builds prior weights (on device)
    s2.to(device)                                  # belt-and-braces: lazy 3D-pos embeddings too
    opt = torch.optim.AdamW(s2.prior.parameters(), lr=cfg.training.lr,  # built ONCE, kept alive
                            fused=(torch.device(device).type == "cuda"))
    n_windows = len(s1c.data.train_dataset)
    s2_loader = _build_loader(s1c.data.train_dataset, cfg.dataset.batch_size_stage2,
                              cfg.dataset.num_workers, shuffle=True)
    return Stage2ClientState(s1c.entity_id, cfg, s2, opt, s1c.data, n_windows,
                             s2_loader=s2_loader)


def _prox_pairs_prior(c: Stage2ClientState) -> list:
    """(param, w^t) pairs for the prior, resolved ONCE per round (stage-1 twin at
    _local_train_stage1: a per-step dict(named_parameters()) rebuild would dominate)."""
    if not (c.prox_mu > 0 and c.prox_ref):
        return []
    params = dict(c.s2.prior.named_parameters())
    return [(params[k], r) for k, r in c.prox_ref.items()]


def _prox_step_prior(c: Stage2ClientState, prox_pairs: list, loss, prox_sum: float):
    """The pre-backward half of the prox mechanics, shared by both prior train loops.
    Returns (loss, prox_sum). Deliberately OUTSIDE `_amp()`: the distance is made of
    low-order bits an fp16 reduction would lose (same rule as the stage-1 twin)."""
    if prox_pairs and c.prox_form == "loss":
        prox = _prox_term(prox_pairs)                       # ‖w − w^t‖²
        loss = loss + 0.5 * c.prox_mu * prox
        prox_sum += float(prox.detach())
    elif prox_pairs:                                        # "decoupled": telemetry only here
        with torch.no_grad():
            prox_sum += float(_prox_term(prox_pairs))
    return loss, prox_sum


def _prox_finish_prior(c: Stage2ClientState, prox_sum: float, ratio_sum: float,
                       ratio_n: int, steps: int, skipped: int) -> None:
    """Per-round telemetry, one diagnostic PER FORM (they are NOT comparable — see the
    stage-1 twin's comment block): 'loss' → prox_grad_ratio, 'decoupled' → prox_pull_frac."""
    c.last_prox = prox_sum / steps if (steps and c.prox_mu > 0) else float("nan")
    c.last_prox_ratio = (ratio_sum / ratio_n
                         if (ratio_n and c.prox_form == "loss") else float("nan"))
    if c.prox_mu > 0 and c.prox_form == "decoupled":
        a = min(1.0, float(c.opt.param_groups[0]["lr"]) * c.prox_mu)
        c.last_prox_pull = 1.0 - (1.0 - a) ** max(steps - skipped, 0)
    else:
        c.last_prox_pull = float("nan")


def _local_train_prior(c: Stage2ClientState, n_epochs: int, device: torch.device) -> float:
    c.s2.stage1.eval()
    c.s2.prior.train()
    total, count = 0.0, 0
    prox_pairs = _prox_pairs_prior(c)
    prox_sum, ratio_sum, ratio_n, steps, skipped = 0.0, 0.0, 0, 0, 0
    for _ in range(n_epochs):
        for batch in c.s2_loader:              # batch_size_stage2, not the stage-1 loader
            batch["inputs"] = batch["inputs"].to(device, non_blocking=True)
            with _amp():                       # fp16/bf16 forward on the transformer prior
                tokens = c.s2.tokens_from_batch(batch)    # frozen stage1 → (B, L)
                c.s2._inform_latent_shape()
                out = c.s2.prior(tokens)
                loss = out.loss
            task = loss                        # reported loss stays the TASK loss (arm-comparable)
            steps += 1
            loss, prox_sum = _prox_step_prior(c, prox_pairs, loss, prox_sum)
            c.opt.zero_grad(set_to_none=True)
            c.scaler.scale(loss).backward()
            if prox_pairs and c.prox_form == "loss":
                # true-gradient diagnostic; `step` detects the unscaled state (stage-1 twin)
                c.scaler.unscale_(c.opt)
                rr = _prox_grad_ratio(prox_pairs, c.prox_mu, c.prox_form)
                if math.isfinite(rr):
                    ratio_sum += rr; ratio_n += 1
            prev_scale = c.scaler.get_scale()
            c.scaler.step(c.opt)               # skipped iff grads overflowed under fp16
            c.scaler.update()
            applied = c.scaler.get_scale() >= prev_scale
            if not applied:
                skipped += 1
            if prox_pairs and c.prox_form == "decoupled" and applied:
                # AdamW-style DECOUPLED proximal step: w ← w + lr·μ·(w^t − w), bypassing
                # Adam's preconditioner (rationale in _prox_term's docstring).
                with torch.no_grad():
                    lr_t = c.opt.param_groups[0]["lr"]
                    for p, ref in prox_pairs:
                        p.add_(ref.to(p.dtype) - p, alpha=lr_t * c.prox_mu)
            lv = float(task.detach())
            if math.isfinite(lv):              # an overflowed step must not poison the mean
                total += lv; count += 1
    _prox_finish_prior(c, prox_sum, ratio_sum, ratio_n, steps, skipped)
    # count == 0 ⇒ every step overflowed. `0.0` would read as a perfect loss.
    return total / count if count else float("nan")


def _cycle(loader):
    """Endless minibatch stream over for finite loader (for step-granular training)."""
    while True:
        for b in loader:
            yield b


def _local_train_prior_steps(c: Stage2ClientState, n_steps: int, device: torch.device) -> float:
    """Run exactly `n_steps` optimizer steps (the τ-step FedSGD regime), cycling
    the client loader. A persistent generator on the client keeps the minibatch
    stream continuous across rounds so τ=1 approximates one large centralized
    minibatch per aggregation."""
    c.s2.stage1.eval()
    c.s2.prior.train()
    gen = getattr(c, "_step_gen", None)
    if gen is None:
        gen = _cycle(c.s2_loader)              # batch_size_stage2: a "step" must mean the
        c._step_gen = gen                      # same amount of data as in every other arm
    total, count = 0.0, 0
    prox_pairs = _prox_pairs_prior(c)
    prox_sum, ratio_sum, ratio_n, steps, skipped = 0.0, 0.0, 0, 0, 0
    for _ in range(n_steps):
        batch = next(gen)
        batch["inputs"] = batch["inputs"].to(device, non_blocking=True)
        with _amp():
            tokens = c.s2.tokens_from_batch(batch)
            c.s2._inform_latent_shape()
            out = c.s2.prior(tokens)
            loss = out.loss
        task = loss
        steps += 1
        loss, prox_sum = _prox_step_prior(c, prox_pairs, loss, prox_sum)
        c.opt.zero_grad(set_to_none=True)
        c.scaler.scale(loss).backward()
        if prox_pairs and c.prox_form == "loss":
            c.scaler.unscale_(c.opt)
            rr = _prox_grad_ratio(prox_pairs, c.prox_mu, c.prox_form)
            if math.isfinite(rr):
                ratio_sum += rr; ratio_n += 1
        prev_scale = c.scaler.get_scale()
        c.scaler.step(c.opt)
        c.scaler.update()
        applied = c.scaler.get_scale() >= prev_scale
        if not applied:
            skipped += 1
        if prox_pairs and c.prox_form == "decoupled" and applied:
            with torch.no_grad():
                lr_t = c.opt.param_groups[0]["lr"]
                for p, ref in prox_pairs:
                    p.add_(ref.to(p.dtype) - p, alpha=lr_t * c.prox_mu)
        lv = float(task.detach())
        if math.isfinite(lv):
            total += lv; count += 1
    _prox_finish_prior(c, prox_sum, ratio_sum, ratio_n, steps, skipped)
    return total / count if count else float("nan")


@torch.no_grad()
def _fedavg_shared(state_dicts: list[dict], weights: list[int], shared_keys: list[str]) -> dict:
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    agg = {}
    for k in shared_keys:
        acc = None
        for wi, sd in zip(w.tolist(), state_dicts):
            # `.cpu()` — clients may live on different GPUs; adding across devices raises.
            # `load_state_dict` copies back to each client's device, so a CPU agg is fine.
            term = sd[k].float().cpu() * wi
            acc = term if acc is None else acc + term
        agg[k] = acc.to(state_dicts[0][k].dtype)
    return agg


def federated_stage2(
    stage1_clients: list[ClientState],
    base_cfg: Config,
    rounds: int,
    local_epochs: int,
    *,
    seed: int = 7,
    local_prefixes: tuple[str, ...] = LOCAL_PRIOR_PREFIXES,
    select_on_val: bool = False,   # see federated_stage1: keeps arms symmetric
    server_momentum: float = 0.0,  # FedAvgM: server-side momentum on the prior pseudo-gradient (0 = plain FedAvg)
    tau_steps: int = 0,            # >0 ⇒ FedSGD: τ optimizer STEPS per round (not epochs). The R1 regime.
    server_opt: str = "sgd",       # "sgd" (plain/FedAvgM) | "fedadam" (adaptive server optimizer, FedOpt)
    server_lr: float = 0.01,       # FedAdam server step size η
    server_b1: float = 0.9,        # FedAdam betas/eps — the old hardcoded values, now knobs.
    server_b2: float = 0.99,       # NB no bias correction (Reddi et al. FedOpt Alg. 2): the
    server_eps: float = 1e-3,      # transient |m|/√v peaks ≈2.13 at ~round 13, so the real
                                   # per-coordinate cap is ~2·server_lr mid-run, not server_lr.
    server_momentum_skip_round0: bool = False,
                                   # FedAvgM: do NOT accumulate v at round 0. Measured on this
                                   # repo's data: the round-0 average is DESTRUCTIVE
                                   # (val(agg₀)=4.59/4.54 > ln(64)=4.16 uniform floor while the
                                   # clients sit at ≤3.5) — heavy-ball with no damping would
                                   # memorise exactly that direction, amplified up to 1/(1−β).
    val_mode: str = "legacy",      # "legacy" | "fixed" — see _val_loss_prior. Fixed masks +
                                   # fp32 val; prerequisite for small-τ regimes.
    val_every: int = 1,            # evaluate val every K rounds (aggregation still every
                                   # round). Patience counts EVAL rounds, not raw rounds:
                                   # at small τ a per-round val dominates wall-clock and a
                                   # round-counted patience collapses to a few optimizer
                                   # steps of staleness. 1 = today's behaviour.
    agg_penalty: bool = False,     # log, per eval round, the val of each client's OWN
                                   # end-of-round body (pre-aggregation) next to the val of
                                   # the broadcast average: penalty = val_post − val_pre.
                                   # THE number that decides whether FedProx-on-the-prior has
                                   # anything to fix (A2's histories suggest ~0). Costs one
                                   # extra val sweep per eval round (~2% of an epoch-round).
    snapshot_every: int = 0,       # >0: every K rounds save shared body + per-client local
                                   # heads to `snapshot_dir` (s2_snap_rNNNN.pt), plus
                                   # s2_last.pt before the best-round restore. Fuels the
                                   # trajectory-detection probe (val↔AUPRC round by round)
                                   # and the best-val-vs-last-round selection contrast.
    snapshot_dir=None,             # where snapshots land (the arm scratch dir). None with
                                   # snapshot_every>0 is refused: silent no-write is how
                                   # instrumented runs turn out to be uninstrumented.
    patience_rounds: int = 0,      # 0 = off. >0 = stop once the cohort val loss has not
                                   # improved for N consecutive EVALS (== rounds when
                                   # val_every=1, the historical contract), so a generous
                                   # --s2-rounds budget can be over-provisioned safely
                                   # (same contract as federated_stage1). Needs select_on_val.
    prior_prox_mu: float = 0.0,    # FedProx on the SHARED prior body: μ of the proximal
                                   # anchor toward the start-of-round broadcast body (params
                                   # only, local heads never anchored). 0 = off = plain
                                   # FedAvg, bit-identical update path.
    prior_prox_form: str = "decoupled",
                                   # "decoupled" (§7 primary: post-step contraction, bypasses
                                   # Adam's preconditioner) | "loss" (paper objective — under
                                   # AdamW it can be a silent no-op; watch prox_grad_ratio).
) -> tuple[list[Stage2ClientState], list[dict], float]:
    # FAIL-FAST, before any client is built: the two server optimizers are MUTUALLY
    # EXCLUSIVE (if/elif in the round loop). Setting both used to make fedadam win in
    # silence while the FedAvgM banner still printed — a log that lies about the mechanism.
    if server_opt == "fedadam" and server_momentum > 0.0:
        raise ValueError(f"server_opt='fedadam' and server_momentum={server_momentum} are "
                         "mutually exclusive (the round loop is if/elif: fedadam would win "
                         "silently and the FedAvgM banner would lie). Pick one.")
    if snapshot_every > 0 and snapshot_dir is None:
        raise ValueError("snapshot_every>0 with snapshot_dir=None: the snapshots would be "
                         "silently dropped and the 'instrumented' run would not be.")
    if prior_prox_form not in {"loss", "decoupled"}:
        raise ValueError(f"unknown prior_prox_form {prior_prox_form!r} (use 'loss' or 'decoupled')")
    # CO-LOCATE with the frozen stage1 encoder: Stage2System wraps s1c.model, so the
    # prior MUST land on the same GPU or the forward throws a device mismatch.
    devices = [next(c.model.parameters()).device for c in stage1_clients]
    device = devices[0]
    clients = [_build_stage2_client(c, d) for c, d in zip(stage1_clients, devices)]
    shared_keys = _prior_shared_keys(clients[0].s2.prior, local_prefixes)
    print(f"[fed-s2] prior shared keys={len(shared_keys)} "
          f"(local heads kept: {local_prefixes or '() — fully-shared prior (B-ablation, codebook FL kept)'})")
    if clients[0].s2._C == 1 and "channel_embedding" in local_prefixes:
        print("[fed-s2] NOTE C=1: channel_embedding is Embedding(1, d) — a constant offset, "
              "not a personalization head. The (B) arm rests on output_bias alone.")

    # Broadcast one common shared body so all clients start aligned.
    body0 = {k: clients[0].s2.prior.state_dict()[k].detach().clone() for k in shared_keys}
    for c in clients:
        c.s2.prior.load_state_dict(body0, strict=False)

    # FedProx on the shared prior body
    prox_param_names: list[str] = []
    if prior_prox_mu > 0:
        prox_param_names = _shared_param_names(clients[0].s2.prior, shared_keys)
        if not prox_param_names:
            # Same contract as federated_stage1's enc_param_names guard: a μ>0 run whose
            # anchored set is empty is FedAvg wearing a FedProx label.
            raise ValueError(
                f"prior_prox_mu={prior_prox_mu} but the shared prior body contains 0 "
                f"PARAMETERS ({len(shared_keys)} tensors) — the proximal anchor would not "
                "exist. Use --fed-enc-prior partial|shared.")
        for c in clients:
            c.prox_mu, c.prox_form = float(prior_prox_mu), prior_prox_form
        print(f"[fed-s2] FedProx-on-prior: mu={prior_prox_mu} form={prior_prox_form} — anchor "
              f"w^t = start-of-round broadcast body ({len(prox_param_names)} param tensors of "
              f"{len(shared_keys)} shared; local heads NOT anchored)")
        if prior_prox_form == "decoupled":
            lr0 = float(clients[0].opt.param_groups[0]["lr"])
            a = min(1.0, lr0 * prior_prox_mu)
            est_steps = (tau_steps if tau_steps > 0
                         else local_epochs * max(1, len(clients[0].s2_loader)))
            print(f"[fed-s2] decoupled contraction (1−lr·mu)={1 - a:.6f}/step ⇒ estimated "
                  f"prox_pull_frac ≈ {1 - (1 - a) ** est_steps:.1%} of the round drift removed "
                  f"(~{est_steps} steps/round); the measured value is history.prox_pull_frac.")
        else:
            print("[fed-s2] ⚠ loss-form prox under AdamW: the prox gradient goes through the "
                  "preconditioner and can be a silent no-op — prox_grad_ratio ≥ 1e-2 on LATE "
                  "rounds is the pre-registered evidence that μ bites.")

    # FedAvgM server state: momentum on the pseudo-gradient (x_t - aggregate). beta=0 -> plain FedAvg.
    # SIGN CONVENTION (do not "fix" to match FedAdam): here δ = x − agg (DESCENT form,
    # Hsu et al. 2019, heavy-ball WITHOUT (1−β) damping ⇒ steady-state step up to 1/(1−β)×δ);
    # FedAdam below uses Δ = agg − x (ASCENT toward the clients). Both broadcast x, not agg.
    _srv_x = {k: body0[k].float().cpu().clone() for k in shared_keys} if server_momentum > 0.0 else None
    _srv_v = {k: torch.zeros_like(v) for k, v in _srv_x.items()} if server_momentum > 0.0 else None
    if server_momentum > 0.0:
        print(f"[fed-s2] FedAvgM server momentum={server_momentum}"
              + (" (round-0 accumulation SKIPPED)" if server_momentum_skip_round0 else ""))
        if server_momentum > 0.5:
            print(f"[fed-s2] ⚠ FedAvgM β={server_momentum} with FULL local epochs: undamped "
                  f"heavy-ball amplifies the pseudo-gradient up to {1/(1-server_momentum):.0f}×, "
                  "and the round-0 average is measured DESTRUCTIVE on this repo's data — "
                  "consider β≤0.3 and/or server_momentum_skip_round0.")

    # FedAdam (FedOpt) server state: an Adam optimizer over the pseudo-gradient
    # Δ = (avg client body) − x. Recommended companion to small-τ FedSGD — the
    # adaptive per-coordinate step handles the sparse token-gradient signal that
    # plain server SGD mishandles. server_opt="sgd" leaves this None.
    _fa = None
    if server_opt == "fedadam":
        _fa = {
            "x": {k: body0[k].float().cpu().clone() for k in shared_keys},
            "m": {k: torch.zeros_like(body0[k].float().cpu()) for k in shared_keys},
            "v": {k: torch.zeros_like(body0[k].float().cpu()) for k in shared_keys},
            "b1": float(server_b1), "b2": float(server_b2), "eps": float(server_eps),
            "lr": float(server_lr),
        }
        print(f"[fed-s2] FedAdam server lr={server_lr} b1={server_b1} b2={server_b2} "
              f"eps={server_eps} (no bias correction; per-coordinate step caps ~2·lr mid-run)")
    if tau_steps > 0:
        print(f"[fed-s2] FedSGD regime: tau_steps={tau_steps} step(s)/round × {rounds} round(s) "
              f"= {tau_steps * rounds} total local steps/client")
        print(f"[fed-s2] NB τ regime: --local-epochs is IGNORED; the LOCAL heads "
              f"({', '.join(local_prefixes) or 'none'}) also take exactly {tau_steps} Adam "
              f"step(s)/round — head capacity is coupled to the round count.")
    if val_mode == "fixed":
        print("[fed-s2] val oracle FIXED: frozen per-client masks + fp32 (selection noise "
              "σ→0; legacy runs carried σ≈0.006-0.018 nats of mask-resampling noise)")
    if val_every > 1:
        print(f"[fed-s2] val cadence: every {val_every} rounds; patience counts EVALS "
              f"(effective step-patience = patience × val_every × steps/round)")

    history: list[dict] = []
    best_val, best_round, best_states = float("inf"), None, None
    # Round-level convergence stop, same contract as federated_stage1. Without it whatever
    # --s2-rounds happens to be IS the budget: there is no early stop, so the shared prior
    # body trains for exactly that many rounds and the truncation alarm can only report the
    # fact after the fact. With it, --s2-rounds becomes a ceiling that can be over-provisioned
    # safely -- which is the project's hard rule (train to convergence, never a fixed budget).
    stale = 0
    min_delta = float(getattr(base_cfg.training, "early_stopping_min_delta", 0.0))
    if patience_rounds > 0:
        print(f"[fed-s2] convergence stop ARMED: patience={patience_rounds} rounds, "
              f"min_delta={min_delta:g} (needs select_on_val: "
              f"{'ON' if select_on_val else 'OFF -> the stop will be DISABLED'})")
    # Local (non-shared) keys, for the snapshots: the heads that never cross the network.
    local_keys = [k for k in clients[0].s2.prior.state_dict().keys() if k not in set(shared_keys)]
    for r in range(rounds):
        # Eval cadence decided UP FRONT: val_pre (below) and val_post must share it, or the
        # aggregation penalty would compare numbers from different rounds.
        do_val = select_on_val and (r % max(1, val_every) == 0 or r == rounds - 1)
        if prior_prox_mu > 0:
            # w^t: every client holds the IDENTICAL broadcast body here (asserted at the end
            # of the previous round), so a per-client snapshot IS the global anchor — taken
            # per client so each anchor lives on its own device.
            for c in clients:
                c.prox_ref = _snapshot_prox_ref(c.s2.prior, prox_param_names)
        sds, weights, losses = [], [], []
        for ci, (c, d) in enumerate(zip(clients, devices)):
            torch.manual_seed(_round_seed(seed, r, ci, stage=2))
            losses.append(_local_train_prior_steps(c, tau_steps, d) if tau_steps > 0
                          else _local_train_prior(c, local_epochs, d))
            sds.append({k: c.s2.prior.state_dict()[k].detach().clone() for k in shared_keys})
            weights.append(c.n_windows)
        # PRE-aggregation val: each client scored with its OWN end-of-round drifted body
        # (the models still hold it — this must run BEFORE the broadcast overwrites them).
        val_pre = _mean_finite([_val_loss_prior(c, d, val_mode) for c, d in zip(clients, devices)]) \
            if (agg_penalty and do_val) else float("nan")
        agg = _fedavg_shared(sds, weights, shared_keys)
        if server_opt == "fedadam":                     # FedOpt: Adam over Δ = agg − x
            for k in shared_keys:
                delta = agg[k].float().cpu() - _fa["x"][k]
                _fa["m"][k] = _fa["b1"] * _fa["m"][k] + (1 - _fa["b1"]) * delta
                _fa["v"][k] = _fa["b2"] * _fa["v"][k] + (1 - _fa["b2"]) * delta * delta
                _fa["x"][k] = _fa["x"][k] + _fa["lr"] * _fa["m"][k] / (_fa["v"][k].sqrt() + _fa["eps"])
                agg[k] = _fa["x"][k].to(agg[k].dtype)
        elif server_momentum > 0.0:                     # FedAvgM: v = beta*v + (x - agg); x = x - v
            if r == 0 and server_momentum_skip_round0:
                # Install the plain average, accumulate nothing: the round-0 pseudo-gradient
                # points at a measured-destructive average (see the signature comment) and
                # undamped heavy-ball would replay it for ~1/(1−β) rounds.
                for k in shared_keys:
                    _srv_x[k] = agg[k].float().cpu()
            else:
                for k in shared_keys:
                    delta = _srv_x[k] - agg[k].float().cpu()
                    _srv_v[k] = server_momentum * _srv_v[k] + delta
                    _srv_x[k] = _srv_x[k] - _srv_v[k]
                    agg[k] = _srv_x[k].to(agg[k].dtype)
        for c in clients:
            c.s2.prior.load_state_dict(agg, strict=False)
        # CORRECTNESS: every client now holds the identical shared body.
        ref = clients[0].s2.prior.state_dict()
        for c in clients[1:]:
            sd = c.s2.prior.state_dict()
            for k in shared_keys:
                # `.cpu()` — clients may live on different GPUs and allclose would raise.
                assert torch.allclose(sd[k].cpu(), ref[k].cpu()), \
                    f"shared-body broadcast mismatch: {k}"

        # Scored AFTER the broadcast, so `val_loss` belongs to the aggregated body
        # paired with each client's own local head — which is exactly the artifact
        # this arm ships. Averaged uniformly over clients: one body per cluster.
        # `val_every`: non-eval rounds carry val=NaN and do NOT touch patience — at small τ
        # a per-round val dominates wall-clock and turns the stop into a noise-record process.
        val_loss = _mean_finite([_val_loss_prior(c, d, val_mode) for c, d in zip(clients, devices)]) \
            if do_val else float("nan")
        row = {"round": r, "mean_loss": sum(losses) / len(losses), "val_loss": val_loss}
        if agg_penalty and math.isfinite(val_pre):
            # penalty > 0 ⇒ the broadcast average scores WORSE than the clients' own drifted
            # bodies — the pathology FedProx-on-the-prior would exist to fix.
            row["val_pre_agg"] = val_pre
            row["agg_penalty"] = (val_loss - val_pre) if math.isfinite(val_loss) else float("nan")
        if prior_prox_mu > 0:
            row["prox_dist"] = _mean_finite([c.last_prox for c in clients])
            if prior_prox_form == "decoupled":
                row["prox_pull_frac"] = _mean_finite([c.last_prox_pull for c in clients])
            else:
                row["prox_grad_ratio"] = _mean_finite([c.last_prox_ratio for c in clients])
        history.append(row)
        print(f"[fed-s2] round {r}: prior_loss={row['mean_loss']:.4f}"
              + (f" val={val_loss:.4f}" if math.isfinite(val_loss) else "")
              + (f" val_pre={val_pre:.4f} agg_pen={row.get('agg_penalty', float('nan')):+.4f}"
                 if agg_penalty and math.isfinite(val_pre) else "")
              + ((f" prox_dist={row['prox_dist']:.3e}"
                  + (f" pull={row['prox_pull_frac']:.1%}" if prior_prox_form == "decoupled"
                     else f" ratio={row['prox_grad_ratio']:.3g}"))
                 if prior_prox_mu > 0 else ""))
        # ── trajectory snapshots (shared body once + per-client heads) ────────────────
        if snapshot_every > 0 and (r % snapshot_every == 0 or r == rounds - 1):
            snap = {"round": r, "val_loss": val_loss,
                    "body": {k: agg[k].detach().cpu() for k in shared_keys},
                    "heads": [{k: c.s2.prior.state_dict()[k].detach().cpu() for k in local_keys}
                              for c in clients],
                    "entities": [c.entity_id for c in clients]}
            os.makedirs(str(snapshot_dir), exist_ok=True)
            torch.save(snap, os.path.join(str(snapshot_dir), f"s2_snap_r{r:04d}.pt"))

        # Snapshot the FULL prior state (shared body + each client's local head): the
        # head co-adapts to the body, so rolling the body back to round k while leaving
        # a head trained to round R would ship a pair that never existed.
        improved = math.isfinite(val_loss) and val_loss < best_val - min_delta
        if math.isfinite(val_loss) and val_loss < best_val:
            best_val, best_round = val_loss, r
            best_states = [copy.deepcopy(c.s2.prior.state_dict()) for c in clients]
        # ── convergence stop ─────────────────────────────────────────────────
        if patience_rounds > 0 and math.isfinite(val_loss):
            stale = 0 if improved else stale + 1
            if stale >= patience_rounds:
                print(f"[fed-s2] CONVERGED: no val improvement > {min_delta:g} for {stale} "
                      f"rounds (best={best_val:.5f} at round {best_round}); stopping at "
                      f"round {r} of {rounds - 1}.")
                history[-1]["converged_stop"] = True
                break
        elif patience_rounds > 0 and do_val:
            # `do_val` guard: with val_every>1 the skipped rounds carry val=NaN by DESIGN —
            # warning there would print thousands of false alarms in a τ regime. This branch
            # now fires only when an eval was hidden and still came back non-finite.
            print("[fed-s2] patience_rounds set but val is not finite — is select_on_val on? "
                  "Convergence stop DISABLED for this run.")

    # ── TRUNCATION ALARM ────────────────────────────────────────────────────────
    # A run whose BEST round is its LAST round was still improving when the budget ran out:
    # it is truncated, not converged. That is not a nuance — measured 2026-07-24 on wsd c3,
    # a 30-round cb_only run left 4 of 6 entities with a near-random detector (VUS-PR
    # 0.03-0.10), and continuing to convergence took the cluster from 0.16 to 0.69. Two
    # independent 30-round runs both showed the broken value, so it reproduces. Every
    # federated result on disk from before that date carries this risk, and nothing in the
    # output said so. Now it does, loudly, in every run that ends this way.
    if math.isfinite(best_val) and best_round == rounds - 1 and rounds > 1:
        print("\n" + "!" * 78)
        # `rounds_done` belongs to federated_stage1, NOT here: this f-string used to raise
        # NameError and kill the job AFTER all training, in exactly the case the alarm exists
        # to report. Stage 2 has no resume path, so its round index is already absolute.
        print("!! TRUNCATED, NOT CONVERGED: the best validation round IS the last round "
              f"({best_round}).")
        print("!! The model was still improving when the round budget ran out, so this run "
              "reports an")
        print("!! UNDERTRAINED model. On wsd c3 that difference was VUS-PR 0.16 vs 0.69.")
        print("!! Fix: raise --s2-rounds so the run stops where it actually flattens.")
        print("!" * 78 + "\n", flush=True)
        history[-1]["truncated"] = True

    # The LAST-round state, saved BEFORE the best-round restore: the free selection-rule
    # contrast (best-on-val vs kept-last — on ucr_170 the val-argmin itself is the suspect:
    # es 0.060 vs kept-last 0.517). Only when snapshots are on: it is instrumentation.
    if snapshot_every > 0:
        torch.save({"round": len(history) - 1, "val_loss": history[-1]["val_loss"],
                    "states": [copy.deepcopy(c.s2.prior.state_dict()) for c in clients],
                    "entities": [c.entity_id for c in clients],
                    "best_round": best_round, "best_val": best_val},
                   os.path.join(str(snapshot_dir), "s2_last.pt"))
    if best_states is not None and best_round != rounds - 1:
        for c, sd in zip(clients, best_states):
            c.s2.prior.load_state_dict(sd)
        print(f"[fed-s2] restored round {best_round} (val={best_val:.4f}); "
              f"last round was {history[-1]['val_loss']:.4f}")
    if best_round is not None:
        history[best_round]["selected"] = True

    # Personalization invariant — asserted HERE so it holds in EVERY caller (the
    # eval harness discards the returned head_div). Cover *every* local tensor
    # (all keys matching local_prefixes), averaged over all client pairs — not
    # just output_bias — so an accidental sharing of channel_embedding can't pass
    # silently. local_prefixes=() ⇒ no local keys ⇒ head_div 0 and the
    # broadcast-identity assert above already proves the head is fully shared.
    local_keys = ([k for k in clients[0].s2.prior.state_dict() if k.startswith(local_prefixes)]
                  if local_prefixes else [])
    ref_sd = clients[0].s2.prior.state_dict()
    diffs = [float((c.s2.prior.state_dict()[k].cpu() - ref_sd[k].cpu()).abs().mean())
             for k in local_keys for c in clients[1:]]
    head_div = float(np.mean(diffs)) if diffs else 0.0
    if local_prefixes and len(clients) >= 2:
        assert head_div > 0, "personalized arm: local heads must DIFFER across clients (got 0)"
    return clients, history, head_div


# ═════════════════════════════════════════════════════════════════════════════
#   FedProto AT THE PRIOR LEVEL — federated_stage2_proto
# ═════════════════════════════════════════════════════════════════════════════
# The FedProto anchor on the *encoder* (federated_anchor) targets the tokenizer,
# which the suff-stat merge already solves. This lifts the prototype idea to the
# PRIOR: the shared object per round is the ensemble-consensus predictive
# distribution over tokens on a common probe (a function-space prototype), and
# each client's prior is pulled toward it by KL while training on its own data.
# NO weight averaging ever happens — the only thing communicated is the averaged
# predictive distribution (a prototype), exactly the FedProto communication model.
# Deploy = per-client personalized prior (the λ=0 null is `federated_cb_only`).

@torch.no_grad()
def _build_probe_tokens_pooled(clients: list[Stage2ClientState], probe_windows: int,
                               device: torch.device) -> torch.Tensor:
    """Pool train windows across clients, tokenize with the shared (merged, frozen)
    codebook. Returns (N, L) long tokens on CPU — the shared unlabeled probe."""
    ref = clients[0]
    per = max(1, probe_windows // max(1, len(clients)))
    toks = []
    for c in clients:
        got = 0
        for batch in c.data.train_loader:
            b = {**batch, "inputs": batch["inputs"].to(device, non_blocking=True)}
            t = ref.s2.tokens_from_batch(b)          # shared frozen codebook
            toks.append(t.cpu()); got += t.shape[0]
            if got >= per:
                break
    allt = torch.cat(toks, 0)
    if allt.shape[0] > probe_windows:
        allt = allt[torch.randperm(allt.shape[0])[:probe_windows]]
    return allt


def _fixed_probe_mask(probe_tokens: torch.Tensor, mask_ratio: float,
                      mask_token_id: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One fixed random mask per probe window (stable across rounds & clients so
    the consensus target is defined on consistent positions). Returns (masked, mask)."""
    N, L = probe_tokens.shape
    g = torch.Generator().manual_seed(int(seed))
    masked = probe_tokens.clone()
    mask = torch.zeros(N, L, dtype=torch.bool)
    k = max(1, int(round(mask_ratio * L)))
    for i in range(N):
        perm = torch.randperm(L, generator=g)[:k]
        mask[i, perm] = True
    masked[mask] = mask_token_id
    return masked, mask


@torch.no_grad()
def _consensus_prototype(clients: list[Stage2ClientState], probe_masked: torch.Tensor,
                         device: torch.device, chunk: int = 256) -> torch.Tensor:
    """Ensemble-mean predictive distribution over tokens on the masked probe:
    (N, L, K) on CPU. The prior-level prototype broadcast each round."""
    N, L = probe_masked.shape
    K = clients[0].s2.prior.codebook_size
    pbar = torch.zeros(N, L, K)
    for c in clients:
        c.s2.prior.eval()
        for i in range(0, N, chunk):
            pm = probe_masked[i:i + chunk].to(device)
            logits = c.s2.prior._logits(pm)
            pbar[i:i + chunk] += torch.softmax(logits.float(), dim=-1).cpu()
    pbar /= len(clients)
    return pbar


def _local_train_prior_proto(c: Stage2ClientState, n_epochs: int, device: torch.device,
                             probe_masked: torch.Tensor, probe_mask: torch.Tensor,
                             pbar: torch.Tensor | None, proto_weight: float, batch: int) -> float:
    """Local prior training (masked-CE on own data) + λ·KL(π̄ ‖ p_local) on a probe
    minibatch each step. pbar=None ⇒ warmup round (proto off)."""
    c.s2.stage1.eval()
    c.s2.prior.train()
    N = probe_masked.shape[0]
    total, count = 0.0, 0
    for _ in range(n_epochs):
        for tb in c.s2_loader:                 # batch_size_stage2, not the stage-1 loader
            tb = {**tb, "inputs": tb["inputs"].to(device, non_blocking=True)}
            with _amp():
                tokens = c.s2.tokens_from_batch(tb)
                c.s2._inform_latent_shape()
                out = c.s2.prior(tokens)
                loss = out.loss
                if pbar is not None and proto_weight > 0:
                    idx = torch.randint(0, N, (batch,))
                    pm = probe_masked[idx].to(device)
                    pmask = probe_mask[idx].to(device)
                    tgt = pbar[idx].to(device)
                    logits = c.s2.prior._logits(pm)
                    logq = torch.log_softmax(logits.float(), dim=-1)
                    kl = -(tgt * logq).sum(-1)[pmask].mean()
                    loss = loss + proto_weight * kl
            c.opt.zero_grad(set_to_none=True)
            c.scaler.scale(loss).backward()
            c.scaler.step(c.opt)
            c.scaler.update()
            lv = float(loss.detach())
            if math.isfinite(lv):
                total += lv; count += 1
    return total / count if count else float("nan")


def federated_stage2_proto(
    stage1_clients: list[ClientState],
    base_cfg: Config,
    rounds: int,
    local_epochs: int,
    *,
    seed: int = 7,
    proto_weight: float = 1.0,
    probe_windows: int = 1024,
    mask_ratio: float = 0.5,
    batch: int = 16,
) -> tuple[list[Stage2ClientState], list[dict], float]:
    devices = [next(c.model.parameters()).device for c in stage1_clients]
    device = devices[0]
    clients = [_build_stage2_client(c, d) for c, d in zip(stage1_clients, devices)]
    print(f"[fed-s2-proto] clients={len(clients)} proto_weight={proto_weight} "
          f"probe_windows={probe_windows} mask_ratio={mask_ratio}")
    probe_tokens = _build_probe_tokens_pooled(clients, probe_windows, device)
    mtid = clients[0].s2.prior.mask_token_id
    probe_masked, probe_mask = _fixed_probe_mask(probe_tokens, mask_ratio, mtid, seed * 13 + 1)
    print(f"[fed-s2-proto] probe={probe_masked.shape[0]} windows, seq_len={probe_masked.shape[1]}")

    history: list[dict] = []
    for r in range(rounds):
        # Consensus from the CURRENT priors (trained through r-1). Round 0 = warmup.
        pbar = _consensus_prototype(clients, probe_masked, device) if (r > 0 and proto_weight > 0) else None
        losses = []
        for ci, (c, d) in enumerate(zip(clients, devices)):
            torch.manual_seed(_round_seed(seed, r, ci, stage=2))
            losses.append(_local_train_prior_proto(
                c, local_epochs, d, probe_masked, probe_mask, pbar, proto_weight, batch))
        history.append({"round": r, "mean_loss": sum(losses) / len(losses)})
        print(f"[fed-s2-proto] round {r}: prior_loss={history[-1]['mean_loss']:.4f}"
              + (" (proto on)" if pbar is not None else " (warmup)"))

    # Personalization invariant: per-client priors must differ (no averaging done).
    ref_sd = clients[0].s2.prior.state_dict()
    if len(clients) >= 2:
        div = float((clients[1].s2.prior.state_dict()[next(iter(ref_sd))].cpu()
                     - ref_sd[next(iter(ref_sd))].cpu()).abs().mean())
        assert div > 0, "protoprior: per-client priors must differ (got identical)"
    return clients, history, 0.0


def main() -> int:
    force_utf8_stdout()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="toy_fed_uni")
    p.add_argument("--clients", type=str, default=None,
                   help="comma-separated entity ids; default = every entity on disk")
    p.add_argument("--cluster", type=str, default=None,
                   help="federate one machine-type cluster (needs clusters.json, e.g. toy_fed_uni). "
                        "Overrides --clients.")
    p.add_argument("--s1-rounds", type=int, default=3)
    p.add_argument("--s2-rounds", type=int, default=3)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=None, help="override batch size (memory budget)")
    # STEP 4 ablation knobs:
    p.add_argument("--anchor-weight", type=float, default=0.0,
                   help="FedProto encoder anchor weight on Stage-1 (0 = off, the default).")
    p.add_argument("--no-revive", action="store_true",
                   help="disable server-side dead-code revival.")
    p.add_argument("--dead-thr", type=int, default=None,
                   help="override the global dead-code count threshold for revival.")
    p.add_argument("--skip-stage2", action="store_true")
    args = p.parse_args()

    cfg = Config()
    cfg.dataset.name = args.dataset
    apply_dataset_overrides(cfg)          # metadata.json: period / metrics_tolerance
    apply_env_overrides(cfg)              # AMP / SCORE_MASK_CHUNK / DETECT_AMP
    if args.batch:
        cfg.dataset.batch_size_stage1 = args.batch
        cfg.dataset.batch_size_stage2 = args.batch
    client_entities = resolve_clients(cfg, args.clients, args.cluster)
    if not client_entities:
        raise SystemExit(f"no clients resolved for {cfg.dataset.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = cfg.quantizer.codebook_size

    # STAGE 1: federated codebook
    clients, global_cb, h1 = federated_stage1(
        cfg, client_entities, rounds=args.s1_rounds, local_epochs=args.local_epochs,
        revive_dead=not args.no_revive, anchor_weight=args.anchor_weight, dead_thr=args.dead_thr,
    )

    probe = _load_public_probe(cfg, cluster=args.cluster)
    probe_src = f"public probe ({args.cluster})" if args.cluster else "public probe"
    if probe is None:
        probe = next(iter(clients[0].data.val_loader))["inputs"]
        probe_src = "client-0 val (no public probe found)"
    # Each client's probe forward must run on that client's own device (they can differ
    # under FEDVQ_DEVICES); `device` here is only the server/print device.
    js = _mean_pairwise_js([_token_usage_hist(c.model, probe, K,
                                              next(c.model.parameters()).device)
                            for c in clients])

    print("\n=== STEP 2 summary (federated Stage 1) ===")
    print(f"loss {h1[0]['mean_loss']:.4f} -> {h1[-1]['mean_loss']:.4f}  "
          f"perplexity {h1[0]['perplexity']:.1f} -> {h1[-1]['perplexity']:.1f}/{K}")
    print(f"cross-client token-usage JS ({probe_src}): {js:.4f}")
    print("OK: codebook frozen locally + identical after every broadcast.")

    if args.skip_stage2:
        return 0

    # STAGE 2: federated apart-personalized prior
    s2_clients, h2, head_div = federated_stage2(
        clients, cfg, rounds=args.s2_rounds, local_epochs=args.local_epochs,
    )

    print("\n=== STEP 3 summary (federated Stage 2) ===")
    print(f"prior loss {h2[0]['mean_loss']:.4f} -> {h2[-1]['mean_loss']:.4f}")
    print(f"local-head divergence across clients (|Δ output_bias|): {head_div:.4f}  (>0 ⇒ personalized)")
    print("OK: shared body identical after every broadcast; channel_embedding + output_bias kept local.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
