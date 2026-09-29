"""STEP 5 — federated evaluation harness (RQ1: does federation help?).

Trains three arms on the synthetic federated benchmark and compares per-client
detection by REUSING the existing, validated detection pipeline (detect.detect):

  * local-only   — each client trains its full stack (stage1+stage2) ALONE.
  * centralized  — ONE model trained on the POOLED windows of all clients
                   (KEEPING per-entity scaler + per-entity threshold; the only
                   thing centralized is representation learning). The skyline.
  * federated    — Method A: FedProto global codebook + partially-personalized
                   prior (from pipeline/federated.py).

For every (arm, client) we save stage1+stage2 checkpoints and call
`detect.detect(...)`, then aggregate the report metrics MACRO across clients
(mean, std, worst-client). This is the first table of real numbers (RQ1).

Smoke:
    CUDA_VISIBLE_DEVICES=1 python pipeline/federated_eval.py \
        --arms local,centralized,federated --clients fed_0,fed_1,fed_2 \
        --s1-epochs 2 --s2-epochs 2 --s1-rounds 2 --s2-rounds 2 --batch 16
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))    # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))           # pipeline/

from torch.utils.data import ConcatDataset, Subset

from config import Config, apply_dataset_overrides, apply_env_overrides
from data import make_dataloaders, _build_loader
from stage1 import Stage1VQVAE, save_stage1_checkpoint, load_stage1
from stage2 import Stage2System, save_stage2_checkpoint, load_stage2, counterfactual
from utils import resolve_path, seed_everything, force_utf8_stdout
import detect
from federated import (
    federated_stage1, federated_stage2, federated_stage2_proto,
    resolve_clients, LOCAL_PRIOR_PREFIXES, _amp,
    _amp_dtype, _make_scaler, _build_client, _cycle,
)

# Detection metrics surfaced in the comparison table (higher is better).
# THRESHOLD-FREE only. `f1` and `affiliation_f1` are read off one arbitrary quantile
# threshold, so an arm can win or lose the table by where its score distribution
# happens to sit rather than by how well it ranks anomalies; they moved to the
# appendix block below. `pate_f1` is PATE's threshold-free (proximity-aware AUC-PR)
# form despite the name. detect() still computes all of them.
METRIC_KEYS = ["vus_pr", "auprc", "auroc", "pate_f1"]
THRESHOLDED_KEYS = ["affiliation_f1", "f1"]
# Counterfactual-fidelity-vs-x_clean metrics (the explainability thesis).
#   cf_repair_improvement ↑ : 1 - MAE(x_cf,x_clean)/MAE(x,x_clean) on ANOMALOUS cells
#                             (how much the CF moved the anomaly back to true normal)
#   cf_repair_ratio       ↓ : the raw ratio (0 = perfect recovery, 1 = no change)
#   cf_disturb_mae        ↓ : MAE(x_cf,x_clean) on CLEAN cells (non-disturbance)
CF_KEYS = ["cf_repair_improvement", "cf_repair_ratio", "cf_disturb_mae", "cf_n_windows"]

# Default server-side codebook-EMA decay γ for `federated_cb_only_ema`. The server EMA
# runs over ROUNDS (a handful), not the thousands of steps `cfg.quantizer.ema_decay`
# (0.99) was tuned for, so 0.99 there means "barely update per round". 0.8 gives an
# effective window of ~1/(1−γ)=5 rounds — enough to average out the noisy per-round
# k-FED M-step while still tracking the drifting encoder. Override with --cb-server-ema-decay.
_CB_SERVER_EMA_DEFAULT_DECAY = 0.8


# the harm registry
# `PAPER_ARMS` is the reporting table: two references plus the (codebook primitive ×
# prior sharing) factorial. Every cell of that factorial exists, which is what makes the
# two contributions readable as main effects instead of as a list of ablations:
#
#                     | codebook: suff-stat        | codebook: weight-FedAvg
#     prior LOCAL     | federated_cb_only(_ema)    | federated_fedavg_cb_only
#     prior SHARED    | federated_shared           | federated_fedavg_cb_sharedprior
#
# `federated` (partially-personalized prior) is the method, sitting between the two prior
# levels. The floor audit is why `_ema` is the (A) row and plain `cb_only` is the γ=0
# ablation: against a zero-parameter moving average, cb_only_ema wins by +0.137 (p=3e-5)
# while cb_only wins by +0.016 and is NOT distinguishable on the second extreme (p=0.209).
# See documentation/research/FLOOR_BASELINE.md §0.3 before promoting cb_only to a headline row.
PAPER_ARMS = [
    "local",                              # floor
    "centralized",                        # skyline
    "federated_cb_only_ema",              # (A): suff-stat + server EMA  -- the row that clears the floor
    "federated_cb_only",                  # (A) at gamma=0: the EMA ablation
    "federated_fedavg_cb_only",           # -(A): the wrong primitive, everything else matched
    "federated_shared",                   # -(B): prior fully shared
    "federated_fedavg_cb_sharedprior",    # both surfaces done the obvious way
    "federated",                          # the method: suff-stat + partially-personalized prior
]

# Everything else that dispatches. Kept working (sweeps, probes, the retired arms), but
# NOT part of the reporting table -- see documentation/operations/LAUNCH_RUNBOOK.md §5 for which are
# retired and why. `federated_enc_fedavg` is the one to run and report in TEXT, not as a
# row: against the floor it sits at p=0.992, i.e. indistinguishable from a moving average,
# which is a stronger answer to "did you try FedAvg on the network?" than a table line.
OTHER_ARMS = [
    "centralized_cap",
    "federated_anchor", "federated_align", "federated_protoprior",
    "federated_cb_only_ema_norevive", "federated_fedavg_cb",
    "federated_enc", "federated_enc_partial", "federated_enc_neck",
    "federated_enc_fedavg", "federated_enc_fedprox", "federated_enc_fedproto",
    "federated_enc_commoninit", "federated_enc_commoninit_cbshared",
    "federated_enc_commoninit_cblocal",
    "federated_enc_fedavg_cblocal", "federated_enc_fedprox_cblocal",
    "federated_fedavgm", "federated_fedsgd", "federated_fedsgd_fedenc",
    "federated_fedsgd_align", "federated_fedsgd_pooltok",
    "federated_pooltok_centralprior",
    "fedtok_centralprior",                # ⛔ DIAGNOSTIC (federation-illegal): central prior
                                          # on a frozen federated tokenizer — the ceiling of
                                          # every stage-2 aggregation upgrade. PROJECT_S2_GATES §2b.
    "centraltok_fedprior",                # ⛔ DIAGNOSTIC (federation-illegal): the INVERSE —
                                          # central tokenizer + A2's federated stage-2. Completes
                                          # the 2×2 attribution on ucr_170 (2026-08-08).
    "federated_fedavg_whole",             # DEPRECATED alias of federated_fedavg_cb_sharedprior
]

KNOWN_ARMS = frozenset(PAPER_ARMS) | frozenset(OTHER_ARMS)

# ─── which arms actually READ which CLI knob ─────────────────────────────────────
# The encoder-trio knobs are consumed by ONE branch of the dispatch chain (`federated_enc_*`);
# the other 24 arms never look at them. argparse accepts them anyway, launch.sh copies them
# into RUN.json's `extra_flags`, and the run's provenance then claims a treatment that was
# never applied — `--arms federated_cb_only --extra "--fedprox-mu 0.1"` was accepted and
# recorded verbatim. That is this repo's characteristic failure (wrong numbers, no error), so
# main() turns "no requested arm reads this flag" into a hard exit. Keys are argparse dests.
_ENC_TRIO_ARMS = ("federated_enc_fedavg", "federated_enc_fedprox", "federated_enc_fedproto",
                  "federated_enc_commoninit", "federated_enc_commoninit_cbshared",
                  "federated_enc_commoninit_cblocal", "federated_enc_fedavg_cblocal",
                  "federated_enc_fedprox_cblocal")
# μ/form reach the client only when algo == "fedprox" (`enc_prox_mu=… if algo == "fedprox"`).
_FEDPROX_ARMS = ("federated_enc_fedprox", "federated_enc_fedprox_cblocal")
# The prototype knobs need enc_proto_weight > 0 to do anything (federated.py: `enc_proto_on`,
# and `enc_avg` only consults enc_proto_fedavg for fedproto). The commoninit* nulls pin λ=0,
# so they are NOT owners: passing --fedproto-agg count there changes nothing.
_FEDPROTO_ARMS = ("federated_enc_fedproto",)
FLAG_OWNER_ARMS: dict[str, tuple[str, ...]] = {
    "fedprox_mu": _FEDPROX_ARMS,
    "fedprox_form": _FEDPROX_ARMS,
    "fedproto_weight": _FEDPROTO_ARMS,
    "fedproto_agg": _FEDPROTO_ARMS,
    "fedproto_code_weight": _FEDPROTO_ARMS,
    "fedproto_no_seed": _FEDPROTO_ARMS,
    "fedproto_fedavg": _FEDPROTO_ARMS,
    # scope/bn/prior are passed to train_federated by the trio branch ONLY: the probe arms
    # (federated_enc, _partial, _neck, fedsgd_fedenc) hardcode their own fed_encoder and
    # leave enc_bn at its default, so a --fed-enc-scope neck there is a no-op.
    "fed_enc_scope": _ENC_TRIO_ARMS,
    "fed_enc_bn": _ENC_TRIO_ARMS,
    "fed_enc_prior": _ENC_TRIO_ARMS,
    # …minus the four suffixed names, which PIN the codebook regime in the dispatch precisely
    # so it is legible from the arm id; there the flag is overridden, not honoured.
    "fed_enc_cb": ("federated_enc_fedavg", "federated_enc_fedprox",
                   "federated_enc_fedproto", "federated_enc_commoninit"),
    # Stage-2 server/regime knobs. Before 2026-08-07 the first four had NO entry here, so
    # `--tau-steps 8 --arms federated_cb_only` was accepted, silently ignored, and RUN.json
    # recorded it as applied — exactly the failure this dict exists to prevent.
    "tau_steps": _ENC_TRIO_ARMS + ("federated_fedsgd", "federated_fedsgd_pooltok",
                                   "federated_fedsgd_fedenc", "federated_fedsgd_align"),
    "server_opt": _ENC_TRIO_ARMS + ("federated_fedsgd", "federated_fedsgd_pooltok",
                                    "federated_fedsgd_fedenc", "federated_fedsgd_align"),
    "server_lr": _ENC_TRIO_ARMS + ("federated_fedsgd", "federated_fedsgd_pooltok",
                                   "federated_fedsgd_fedenc", "federated_fedsgd_align"),
    "server_momentum": ("federated_fedavgm",),
    "s2_server_momentum": _ENC_TRIO_ARMS,
    "s2_momentum_skip_round0": _ENC_TRIO_ARMS,
    # b1/b2/eps are forwarded ONLY by the enc-trio branch (the fedsgd probe branches predate
    # them and pass server_opt/server_lr alone) — owners restricted accordingly, or the
    # check would bless a flag those arms still ignore.
    "server_b1": _ENC_TRIO_ARMS,
    "server_b2": _ENC_TRIO_ARMS,
    "server_eps": _ENC_TRIO_ARMS,
    # NB --fed-s2-val is NOT listed: it is honoured everywhere stage-2 val exists (the
    # federated shared-body loop AND the per-client converged loop the baselines use).
    # centraltok_fedprior reads the val cadence and the EVAL-counted patience too (its
    # dispatch branch forwards them into federated_stage2 like the trio does).
    "fed_s2_val_every": _ENC_TRIO_ARMS + ("centraltok_fedprior",),
    "fed_s2_patience": _ENC_TRIO_ARMS + ("centraltok_fedprior",),
    "fed_s2_agg_penalty": _ENC_TRIO_ARMS,
    "fed_s2_snapshot_every": _ENC_TRIO_ARMS,
    # Forwarded only by the trio branch (enc_sched=args.fed_enc_sched): without this entry
    # `--fed-enc-sched cosine` on a baseline rerun was accepted, recorded in RUN.json and
    # silently ignored — the exact failure class this dict exists to refuse (audit 2026-08-08).
    "fed_enc_sched": _ENC_TRIO_ARMS,
    # FedProx-on-the-PRIOR (≠ fedprox_mu above, which anchors the stage-1 ENCODER and is
    # owned by the federated_enc_fedprox arms): forwarded by the enc-trio branch only.
    "fed_prior_prox_mu": _ENC_TRIO_ARMS,
    "fed_prior_prox_form": _ENC_TRIO_ARMS,
    "fedtokcp_stage1_root": ("fedtok_centralprior", "centraltok_fedprior"),
}


# building blocks (local / centralized training)

def _materialized_stage1(cfg: Config, example: torch.Tensor, device, collect_stats: bool) -> Stage1VQVAE:
    m = Stage1VQVAE(cfg)
    m.eval()
    with torch.no_grad():
        m(example.cpu())
    m.to(device)
    m.quantizer.collect_stats_only = collect_stats   # False ⇒ normal local EMA codebook (both VQ types)
    return m


# NOTE both trainers run their forward under the SAME `_amp()` guard — and the same
# `_make_scaler()` — as the federated local trainers (pipeline/federated.py). Without
# it the baselines would train in fp32 while the federated arms train in fp16/bf16,
# and the arms would not be iso-precision — an uncontrolled confound in the RQ1 table.
# `_amp()` is a no-op off CUDA and is disabled by FEDVQ_AMP=off, so every arm follows
# one knob. The scaler is a pass-through unless the dtype is fp16.
def _epoch_log(tag: str, stage: str, ep: int, n_epochs: int, tot: float, nb: int) -> None:
    # Convergence trace: first 2, every 5th, and last 3 epochs — enough to SEE the
    # plateau (a still-falling loss at the last epochs => undertrained, bump budget).
    if nb and (ep < 2 or ep >= n_epochs - 3 or (ep + 1) % 5 == 0):
        print(f"  [{stage}{(' ' + tag) if tag else ''}] ep {ep + 1}/{n_epochs} loss={tot / nb:.4f}", flush=True)


def _train_stage1(model: Stage1VQVAE, loader, n_epochs: int, lr: float, device, tag: str = "") -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=(torch.device(device).type == "cuda"))
    scaler = _make_scaler()
    model.train()
    for ep in range(n_epochs):
        tot, nb = 0.0, 0
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            with _amp():
                loss = model(x)["losses"]["loss"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            lv = float(loss.detach())
            if lv == lv:                       # skip NaN in the running mean
                tot += lv; nb += 1
        _epoch_log(tag, "s1", ep, n_epochs, tot, nb)


def _build_train_stage2(cfg: Config, stage1: Stage1VQVAE, loader, n_epochs: int,
                        lr: float, device, example: torch.Tensor, tag: str = "") -> Stage2System:
    loader = _restage2_loader(cfg, loader, shuffle=True)   # batch_size_stage2, not stage1's
    s2 = Stage2System(cfg, stage1)
    s2.prior.to(device)
    s2.materialize(example.to(device))
    s2.to(device)
    opt = torch.optim.AdamW(s2.prior.parameters(), lr=lr, fused=(torch.device(device).type == "cuda"))
    scaler = _make_scaler()
    s2.stage1.eval(); s2.prior.train()
    for ep in range(n_epochs):
        tot, nb = 0.0, 0
        for batch in loader:
            batch["inputs"] = batch["inputs"].to(device, non_blocking=True)
            with _amp():
                tokens = s2.tokens_from_batch(batch)
                s2._inform_latent_shape()
                loss = s2.prior(tokens).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            lv = float(loss.detach())
            if lv == lv:
                tot += lv; nb += 1
        _epoch_log(tag, "s2", ep, n_epochs, tot, nb)
    return s2


def _build_train_stage2_steps(cfg: Config, stage1: Stage1VQVAE, loader, n_steps: int,
                              lr: float, device, example: torch.Tensor, tag: str = "") -> Stage2System:
    """STEP-based pooled prior (vs the epoch-based `_build_train_stage2`). Used by Cell C so
    the pooled prior's optimizer-update budget matches the τ-FedSGD prior's round count
    exactly, on a frozen tokenizer. Same AMP/scaler path as every other trainer.

    The re-batch to `batch_size_stage2` does NOT change the step-matching: the τ-FedSGD
    arm this is matched against drew from the same stage-1-batched loader, so both sides
    were equally off. Both now move to the configured stage-2 batch together, keeping
    a "step" here equal to a "step" there."""
    loader = _restage2_loader(cfg, loader, shuffle=True)
    s2 = Stage2System(cfg, stage1)
    s2.prior.to(device)
    s2.materialize(example.to(device))
    s2.to(device)
    opt = torch.optim.AdamW(s2.prior.parameters(), lr=lr, fused=(torch.device(device).type == "cuda"))
    scaler = _make_scaler()
    s2.stage1.eval(); s2.prior.train()
    it = _cycle(loader)
    tot, nb = 0.0, 0
    for step in range(n_steps):
        batch = next(it)
        batch["inputs"] = batch["inputs"].to(device, non_blocking=True)
        with _amp():
            tokens = s2.tokens_from_batch(batch)
            s2._inform_latent_shape()
            loss = s2.prior(tokens).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        lv = float(loss.detach())
        if lv == lv:
            tot += lv; nb += 1
        if step < 2 or step >= n_steps - 2 or (step + 1) % 200 == 0:
            print(f"  [s2{(' ' + tag) if tag else ''}] step {step + 1}/{n_steps} "
                  f"loss={tot / max(nb, 1):.4f}", flush=True)
    return s2


# ─── "converged" protocol: the run.py training recipe, in-process ────────────
# `--protocol converged` swaps the fixed-epoch trainers above for the SAME recipe
# pipeline/stage1.py and pipeline/stage2.py run under run.py: warmup+cosine LR,
# an fp32 validation pass every epoch, step-based early-stopping patience, and
# best-on-val weight selection. Those two scripts are the trainers that produced
# the high-scoring runs; the fixed-epoch trainers above never build a val loader
# at all, so they can neither stop on convergence nor select a best checkpoint.
#
# Deliberately NOT a wrapper around stage1.main()/stage2.main(): the federated
# arms need live in-memory models to average weights mid-training, so this file
# keeps its own loops. The recipe is mirrored; the code path is not shared.
#
# AMP still goes through federated.py's `_amp()`/`_make_scaler()` rather than
# stage{1,2}.py's inline autocast: both resolve to the same dtype (fp16 below
# Ampere, bf16 from Ampere on -- see `_amp_dtype`), but routing through the
# federated helpers keeps every arm on ONE precision knob (FEDVQ_AMP), so the
# baselines stay iso-precision with the federated arms.
def _converged_budget(cfg: Config, batches_per_epoch: int, stage: int, tag: str = ""):
    """Resolve (max_steps, warmup_steps, patience_steps, max_epochs) as
    pipeline/stage{1,2}.py do — `min_epochs` RAISES the step cap so the LR schedule
    spans the guaranteed passes over the data — but CLAMPED at the configured ceiling.

    The clamp is what keeps `local` and `centralized` comparable. The floor scales with
    `batches_per_epoch`, which is ~K x larger on a pooled loader, so an unclamped floor
    hands the skyline arm a proportionally bigger budget than the baseline it is meant
    to be measured against. On the pooled UCR arm (5.3M windows) it reached 31x the
    stage-1 ceiling. Under the clamp every arm shares one ceiling and EARLY STOPPING —
    the intended binding stopper — decides where each actually halts.

    Harmless where the floor never binds anyway: on wsd/toy at the configured batch
    sizes, and on every single-client `local` run, floor < ceiling already.
    """
    max_steps = int(getattr(cfg.training, f"stage{stage}_max_steps"))
    min_epochs = int(getattr(cfg.training, f"stage{stage}_min_epochs", 0))
    if min_epochs > 0:
        floor = min_epochs * batches_per_epoch
        if floor > max_steps:
            print(f"  [s{stage}{(' ' + tag) if tag else ''}] min_epochs floor {floor} "
                  f"CLAMPED to the {max_steps}-step ceiling ({batches_per_epoch} "
                  f"batches/epoch); early stopping decides the real stop.", flush=True)
    warmup_steps = max(1, int(max_steps * cfg.training.warmup_rate))
    return (max_steps, warmup_steps,
            int(getattr(cfg.training, f"stage{stage}_patience_steps")),
            int(getattr(cfg.training, f"stage{stage}_max_epochs")))


def _cosine_lr(opt, max_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _snapshot(model) -> dict:
    """CPU copy of the weights = the in-memory analogue of stage{1,2}.py's
    `best.ckpt`. Detection must score the BEST-on-val weights, not whatever the
    last epoch happened to leave behind."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _converged_loop(*, step_fn, val_fn, params, cfg, stage: int,
                    batches_per_epoch: int, train_loader, val_loader,
                    model_for_state, tag: str, device) -> None:
    """Shared train/val/early-stop driver for both stages (they differ only in the
    per-batch forward and the module whose weights are snapshotted)."""
    max_steps, warmup_steps, patience_steps, max_epochs = _converged_budget(
        cfg, batches_per_epoch, stage, tag=tag)
    opt = torch.optim.AdamW(params, lr=cfg.training.lr,
                            fused=(torch.device(device).type == "cuda"))
    sched = _cosine_lr(opt, max_steps, warmup_steps)
    scaler = _make_scaler()
    label = f"s{stage}{(' ' + tag) if tag else ''}"
    print(f"  [{label}] converged protocol: max_steps={max_steps} "
          f"warmup={warmup_steps} patience={patience_steps} "
          f"({batches_per_epoch} batches/epoch)", flush=True)

    # `keep_last_weights` reproduces upstream's protocol: no checkpoint selection at
    # all, the weights left by the last step are the weights that get scored. The
    # snapshot is skipped entirely rather than taken-and-discarded — with no val
    # loader (see `pool_val_into_train`) `best_state` would never be refreshed, and
    # restoring it would silently load the INITIAL weights.
    keep_last = bool(getattr(cfg.training, "keep_last_weights", False))
    best_val, best_step = float("inf"), 0
    best_state = None if keep_last else _snapshot(model_for_state)
    if keep_last:
        print(f"  [{label}] keep_last_weights: no best-on-val selection "
              f"(upstream protocol — the final step's weights are kept)", flush=True)
    step, epoch, stop = 0, 0, False
    while not stop and epoch < max_epochs:
        tot, nb = 0.0, 0
        for batch in train_loader:
            with _amp():
                loss = step_fn(batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            prev_scale = scaler.get_scale()
            scaler.step(opt); scaler.update()
            # A skipped step means no weights moved: the LR schedule must not
            # advance either, or it desyncs from the optimiser under fp16.
            if scaler.get_scale() >= prev_scale:
                sched.step()
            lv = float(loss.detach())
            if lv == lv:
                tot += lv; nb += 1
            step += 1
            if step >= max_steps:
                stop = True
                break

        # Validation stays fp32 on purpose: it drives early stopping and weight
        # selection, so it must not depend on the training precision.
        ran_val, vtot, vnb = False, 0.0, 0
        if val_loader is not None and epoch % cfg.training.check_val_every_n_epoch == 0:
            ran_val = True
            with torch.no_grad():
                for batch in val_loader:
                    lv = float(val_fn(batch))
                    if lv == lv:
                        vtot += lv; vnb += 1

        tr = tot / max(nb, 1)
        va = vtot / vnb if vnb else float("nan")
        if epoch < 2 or (epoch + 1) % 5 == 0:
            print(f"  [{label}] ep {epoch} step {step} train={tr:.4f} val={va:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)

        if ran_val and vnb:
            if va < best_val - cfg.training.early_stopping_min_delta:
                best_val, best_step = va, step
                if not keep_last:
                    best_state = _snapshot(model_for_state)
            elif (cfg.training.early_stopping and step >= warmup_steps
                  and step - best_step >= patience_steps):
                # Gated on WARMUP, not on min_epochs: min_epochs raises max_steps to
                # span the LR schedule, so gating on it would disable early stopping.
                print(f"  [{label}] early stop @ep{epoch} step{step} "
                      f"(no val gain for {step - best_step} steps; "
                      f"best={best_val:.4f}@{best_step})", flush=True)
                stop = True
        epoch += 1

    if not keep_last:
        model_for_state.load_state_dict(best_state)      # restore BEST-on-val
    print(f"  [{label}] done: {epoch} epochs, {step} steps, "
          f"{'kept LAST weights' if keep_last else f'best val={best_val:.4f} @step {best_step}'}",
          flush=True)


def _train_stage1_converged(model: Stage1VQVAE, train_loader, val_loader,
                            cfg: Config, device, tag: str = "") -> None:
    def step_fn(batch):
        return model(batch["inputs"].to(device, non_blocking=True))["losses"]["loss"]

    def val_fn(batch):
        return model(batch["inputs"].to(device, non_blocking=True))["losses"]["loss"]

    model.train()
    _converged_loop(
        step_fn=lambda b: (model.train(), step_fn(b))[1],
        val_fn=lambda b: (model.eval(), val_fn(b))[1],
        params=model.parameters(), cfg=cfg, stage=1,
        batches_per_epoch=len(train_loader), train_loader=train_loader,
        val_loader=val_loader, model_for_state=model, tag=tag, device=device)


def _restage2_loader(cfg: Config, loader, shuffle: bool):
    """Re-batch a stage-1 loader at `batch_size_stage2`.

    `_loaders_for` builds every client bundle with `stage="stage1"`, and
    `_pooled_loader` likewise uses `batch_size_stage1` -- so every stage-2 trainer
    reachable from here inherited the STAGE-1 batch size (256) instead of
    `batch_size_stage2` (128), making the latter dead config. The federated arms had
    the identical defect via `Stage2ClientState.data.train_loader`; it is fixed there
    by `Stage2ClientState.s2_loader`. Rebuilding from the same underlying dataset puts
    every arm on the configured stage-2 batch -- the run.py recipe, and the OOM-safe one.
    """
    if loader is None:
        return None
    return _build_loader(loader.dataset, cfg.dataset.batch_size_stage2,
                         cfg.dataset.num_workers, shuffle=shuffle)


# --fed-s2-val, visible to every stage-2 val site. A mutable dict (like _RESUME_CTX), set
# once by main() after parse_args — threading it through the 18 train call sites would touch
# every arm's signature for a knob that defaults to no-op.
_S2_VAL_MODE: dict = {"mode": "legacy"}


def _build_train_stage2_converged(cfg: Config, stage1: Stage1VQVAE, train_loader,
                                  val_loader, device, example: torch.Tensor,
                                  tag: str = "") -> Stage2System:
    train_loader = _restage2_loader(cfg, train_loader, shuffle=True)
    val_loader = _restage2_loader(cfg, val_loader, shuffle=False)
    s2 = Stage2System(cfg, stage1)
    s2.prior.to(device)
    s2.materialize(example.to(device))
    s2.to(device)
    s2.stage1.eval()

    def _loss(batch):
        batch["inputs"] = batch["inputs"].to(device, non_blocking=True)
        tokens = s2.tokens_from_batch(batch)
        s2._inform_latent_shape()
        return s2.prior(tokens).loss

    def _val(batch):
        # --fed-s2-val fixed: freeze the MaskGIT masks the val forward draws, so repeated
        # evals of the same weights return the same number (legacy: σ≈0.006-0.018 nats of
        # mask-resampling noise vs min_delta=1e-4). fork_rng leaves the TRAINING stream
        # untouched — trajectories are byte-identical either way. The per-tag seed keeps
        # different clients' oracles decorrelated; same masks across evals is the point.
        s2.prior.eval()
        if _S2_VAL_MODE["mode"] != "fixed":
            return _loss(batch)
        devs = [device] if getattr(device, "type", "") == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(int.from_bytes(
                hashlib.blake2b(f"s2val:{tag}".encode(), digest_size=6).digest(), "big"))
            return _loss(batch)

    _converged_loop(
        step_fn=lambda b: (s2.prior.train(), _loss(b))[1],
        val_fn=_val,
        params=s2.prior.parameters(), cfg=cfg, stage=2,
        batches_per_epoch=len(train_loader), train_loader=train_loader,
        val_loader=val_loader, model_for_state=s2.prior, tag=tag, device=device)
    return s2


def _records_fingerprint(ds) -> tuple:
    """Source identity of a SlidingWindowDataset, INVARIANT to the per-client scaler.

    Used to detect splits that several clients were each handed their own copy of.
    Hashing the values does NOT work: on `ucr_split` the five val files on disk are
    byte-identical (verified: one sha1 across all five `val/ucr_001_p*.npy`), but
    `PerEntityScaler` fits on each client's own train shard, so by the time the
    records reach here the five copies carry five different affine transforms and
    hash five different ways.

    So fingerprint the RANKS instead. A per-channel z-score is strictly increasing,
    so it leaves the rank permutation untouched — same source under any client's
    scaler gives the same fingerprint, while a genuinely different split gives a
    different one. Exact integer comparison, no float tolerance to tune.
    """
    h = hashlib.sha1()
    for r in getattr(ds, "records", []):
        a = np.ascontiguousarray(r.X)
        h.update(str(a.shape).encode())
        for c in range(a.shape[1]):
            h.update(np.argsort(np.argsort(a[:, c], kind="stable"),
                                kind="stable").astype(np.int64).tobytes())
    return (len(ds), h.hexdigest())


def _pooled_loader(cfg: Config, entities: list[str], data_by_e: dict, cap: int = 0):
    """ONE loader over the CONCATENATION of every client's training windows.

    `cap>0` subsamples the pooled set to `cap` windows drawn DIVERSELY across all
    clients (random Subset). Used to isolate diversity/regularization from raw data
    volume: a small diverse budget vs a single client's own budget.

    The centralized skyline must see genuinely pooled, entity-MIXED batches. The
    previous implementation looped `for loader in loaders` inside each epoch, so
    every batch was entity-pure and each epoch was a round-robin — the VQ's EMA
    (effective window ~1/(1-decay) = 100 steps, vs 49 steps per entity) then
    tracked whichever entity trained last, biasing the final codebook by loop
    order. Windows are already per-entity standardised, so concatenating is sound.
    """
    parts = [data_by_e[e].train_dataset for e in entities]
    if getattr(cfg.dataset, "pool_val_into_train", False):
        # The pooled arm is the ONE place where the 10% val holdout is removable:
        # the union of the shards plus the val slice is the original UCR train
        # series, which is what upstream trains on. Folding it back closes the last
        # data-side deviation for `centralized` only (`train_local` and every
        # federated arm are untouched — no client owns the pooled val).
        if not getattr(cfg.training, "keep_last_weights", False):
            raise ValueError(
                "dataset.pool_val_into_train=True requires training.keep_last_weights=True: "
                "no val loader is built, so the best-on-val snapshot is never refreshed and "
                "restoring it would load the model's INITIAL weights."
            )
        # ⚠️ DEDUPLICATE. On `ucr_split` every client is served the SAME pre-built
        # val slice ("using pre-built val from disk (1 record(s))" × 5), so a naive
        # concat folds it in FIVE times: measured 15 465 = 5 × 3 093 val windows
        # against 29 465 train ones, i.e. 34% of the pooled set would have been five
        # copies of one tenth of the series. Harmless in `_pooled_val_loader` (a mean
        # over duplicates is the same mean), NOT harmless as training data.
        # See `_records_fingerprint` for why the dedup is rank-based: the five copies
        # are byte-identical ON DISK but arrive here under five different per-client
        # scalers, so a value hash reports them as five distinct splits.
        #
        # ⚠️ WHICH copy survives is arbitrary — the first client's, under ITS scaler.
        # On `ucr_001` the sd ratio across the five shards is 1.04, so the choice is
        # immaterial there; on a skewed series it would not be. The principled fix is
        # the pooled/federated scaler (validated 2026-08-03 to 1e-14 by
        # `scripts/fedscaler_train_probe.py`, still a probe rather than a code path).
        seen, val_parts = set(), []
        for e in entities:
            ds = getattr(data_by_e[e], "val_dataset", None)
            if ds is None or len(ds) == 0:
                continue
            key = _records_fingerprint(ds)
            if key in seen:
                continue
            seen.add(key)
            val_parts.append(ds)
        n_tr = sum(len(d) for d in parts)
        n_va = sum(len(d) for d in val_parts)
        parts = parts + val_parts
        print(f"[centralized] POOL VAL INTO TRAIN: {n_tr} + {n_va} = {n_tr + n_va} windows "
              f"({len(val_parts)} distinct val split(s) across {len(entities)} clients); "
              f"no validation loader, last-step weights kept")
        # HONEST RESIDUAL: this still does not equal upstream's window count. Each
        # shard is windowed independently, so the windows that would straddle a shard
        # boundary do not exist — on ucr_001 that is 32 558 windows here against the
        # 34 593 upstream gets from the contiguous 35 000-point series (94.1%). Closing
        # that would mean re-splicing the shards, i.e. undoing the federated split.
    ds = ConcatDataset(parts)
    if cap and cap < len(ds):
        idx = torch.randperm(len(ds))[:cap].tolist()   # seeded by seed_everything(seed) upstream
        ds = Subset(ds, idx)
        print(f"[centralized] POOL CAP: {cap} of {len(idx)} windows (diverse subsample across {len(entities)} clients)")
    return _build_loader(ds, cfg.dataset.batch_size_stage1, cfg.dataset.num_workers, shuffle=True)


def _pooled_val_loader(cfg: Config, entities: list[str], data_by_e: dict):
    """Validation twin of `_pooled_loader` — the pooled model must early-stop on the
    pooled val split, i.e. the same union of clients it trains on. Never capped and
    never shuffled: it is only ever consumed as a whole, in eval mode."""
    if getattr(cfg.dataset, "pool_val_into_train", False):
        return None                       # those windows are in the train loader now
    ds = ConcatDataset([data_by_e[e].val_dataset for e in entities])
    if len(ds) == 0:
        return None
    return _build_loader(ds, cfg.dataset.batch_size_stage1, cfg.dataset.num_workers, shuffle=False)


def _loaders_for(cfg: Config, entities: list[str]) -> dict[str, object]:
    out = {}
    for e in entities:
        c = copy.deepcopy(cfg); c.dataset.entity_id = e
        out[e] = make_dataloaders(c, stage="stage1")
    return out


# Arms mapped to per-client (stage1, stage2) checkpoints

def train_local(cfg, entities, data_by_e, s1_epochs, s2_epochs, device,
                protocol: str = "fixed") -> dict:
    """One model per client — literally one time series per model.

    `protocol="converged"` trains each client with the run.py recipe (val-driven
    early stopping, warmup+cosine, best-on-val weights), so `s1_epochs`/`s2_epochs`
    are IGNORED: the stop point comes from the client's own validation curve.
    """
    models = {}
    for e in entities:
        d = data_by_e[e]
        ex = next(iter(d.train_loader))["inputs"][:1]
        s1 = _materialized_stage1(cfg, ex, device, collect_stats=False)
        if protocol == "converged":
            _train_stage1_converged(s1, d.train_loader, d.val_loader, cfg, device, tag=e)
            s2 = _build_train_stage2_converged(cfg, s1, d.train_loader, d.val_loader,
                                               device, ex, tag=e)
        else:
            # STEP ASYMMETRY vs train_centralized: both take `s1_epochs`, neither is step-matched.
            # Here an epoch is ceil(n_e/batch) steps over THIS client; there it is ceil(sum_k n_k/batch)
            # over the pooled loader, so centralized gets ~K x the median client's optimizer steps and
            # the ratio grows with cluster size. "Epochs" is the matched unit; steps are NOT. The
            # step-matched reference is the Cell-C arm federated_pooltok_centralprior.
            _train_stage1(s1, d.train_loader, s1_epochs, cfg.training.lr, device, tag=e)
            s2 = _build_train_stage2(cfg, s1, d.train_loader, s2_epochs, cfg.training.lr, device, ex, tag=e)
        models[e] = (s1, s2)
        print(f"[local] {e} trained ({protocol})")
    return models


def train_centralized(cfg, entities, data_by_e, s1_epochs, s2_epochs, device, pool_cap: int = 0,
                      protocol: str = "fixed") -> dict:
    """The pooled skyline: the SAME model as `local`, same trainer, trained on the
    CONCATENATION of every client's windows.

    Under `protocol="converged"` it is also the same *protocol* as `local` — both
    stop on their own validation curve. That is what makes the two arms comparable:
    the old fixed-epoch pairing was epoch-matched but NOT step-matched (see below),
    so the pooled arm silently bought ~K x more optimizer steps.
    """
    loader = _pooled_loader(cfg, entities, data_by_e, cap=pool_cap)   # entity-MIXED batches
    ex = next(iter(loader))["inputs"][:1]
    s1 = _materialized_stage1(cfg, ex, device, collect_stats=False)
    if protocol == "converged":
        val_loader = _pooled_val_loader(cfg, entities, data_by_e)
        _train_stage1_converged(s1, loader, val_loader, cfg, device, tag="central")
        s2 = _build_train_stage2_converged(cfg, s1, loader, val_loader, device, ex, tag="central")
    else:
        # STEP ASYMMETRY vs train_local: the same `s1_epochs` buys ceil(sum_k n_k/batch) steps here
        # against ceil(n_e/batch) there -- ~K x more, tracking cluster size. This arm is EPOCH-matched
        # to `local`, NOT step-matched; a table that labels it "step-matched" means Cell C, not this.
        _train_stage1(s1, loader, s1_epochs, cfg.training.lr, device, tag="central")
        s2 = _build_train_stage2(cfg, s1, loader, s2_epochs, cfg.training.lr, device, ex, tag="central")
    print(f"[centralized] one model trained on {len(loader.dataset)} pooled windows "
          f"from {len(entities)} clients ({len(loader)} steps/epoch, {protocol})")
    return {e: (s1, s2) for e in entities}                                  # same model for every client


def train_fedtok_centralprior(cfg, entities, data_by_e, device, stage1_root: str,
                              cluster: str, seed: int) -> dict:
    """⛔ DIAGNOSTIC, federation-ILLEGAL — never a paper arm.

    The ceiling of EVERY stage-2 aggregation upgrade, measured in one cell: A2's stage-1
    encoders end the run IDENTICAL across clients (full FedAvg + pooled BN + shared
    codebook ⇒ the tokenization is client-independent), so training ONE prior centrally on
    the POOLED token stream, with the baselines' own converged recipe, bounds from above
    what any FedSGD-τ / FedAdam / FedAvgM / FedProx variant of the shared body can reach.
    If this does not beat A2's federated prior on the money series, the stage-2
    aggregation direction is dead by construction (documentation/research/PROJECT_S2_GATES.md §2b).

    Pools raw windows across clients (that is the point — it is an oracle), and detects
    with the single central prior on every client's test shard, `centralized`-style.
    """
    loader = _pooled_loader(cfg, entities, data_by_e)             # entity-MIXED batches
    ex = next(iter(loader))["inputs"][:1]
    src = Path(stage1_root) / cfg.dataset.name / cluster / f"seed{seed}"
    # The SOURCE arm dir: exactly one is expected under seed<k>/ for the A2 tag. Globbing
    # (not hardcoding the knob-suffixed name) keeps this working if the source tag ever
    # carries different suffix bits — but more than one match is refused, not "first wins".
    cands = sorted(d for d in src.glob("federated_enc_*") if d.is_dir())
    if len(cands) != 1:
        raise SystemExit(f"fedtok_centralprior: expected exactly ONE source arm dir under "
                         f"{src}, found {len(cands)}: {[c.name for c in cands]} — pass a "
                         f"--fedtokcp-stage1-root whose tag holds a single A2-family run.")
    ck = {e: cands[0] / e / "stage1.ckpt" for e in entities}
    missing = [str(p) for p in ck.values() if not p.exists()]
    if missing:
        raise SystemExit(f"fedtok_centralprior: missing stage1.ckpt(s): {missing}")
    # THE PREMISE, VERIFIED, NOT ASSUMED: the diagnostic only bounds stage-2 upgrades if
    # the TOKENIZATION really is client-independent — i.e. transform+encoder+quantizer
    # identical. The RECONSTRUCTION path is excluded on purpose: A2 never federates it (it
    # stays local by design), it plays no part in tokenization, and each client KEEPS its
    # own below so the s_local reconstruction score stays exactly A2's — the diagnostic
    # then isolates the prior. That path is decoder_2d/decoder_1d AND `refinement` (the
    # learned Linear(T,T) residual applied to the waveform AFTER the inverse STFT,
    # stage1.py:138-160 — downstream of the decoder, upstream of nothing token-related).
    # …and `num_batches_tracked`: the int64 BN step counter, LOCAL in every --fed-enc-bn
    # mode by documented design, and inert at eval time (eval-mode BN reads only
    # running_mean/running_var — which MUST match, and are checked).
    ref = torch.load(str(ck[entities[0]]), map_location="cpu")["state_dict"]
    tok_keys = [k for k in ref if not k.startswith(("decoder", "refinement"))
                and not k.endswith("num_batches_tracked")]
    for e in entities[1:]:
        other = torch.load(str(ck[e]), map_location="cpu")["state_dict"]
        diffs = [k for k in tok_keys if not torch.allclose(ref[k], other[k], atol=1e-6)]
        if diffs:
            raise SystemExit(f"fedtok_centralprior: TOKENIZER stage-1 of {e} differs from "
                             f"{entities[0]} on {len(diffs)}/{len(tok_keys)} non-decoder "
                             f"tensors (e.g. {diffs[:3]}) — the pooled-token premise fails, "
                             "the diagnostic would bound nothing.")
    print(f"[fedtokcp] tokenizer identity verified across {len(entities)} clients "
          f"({len(tok_keys)} non-decoder tensors, atol=1e-6; decoders stay per-client); "
          f"source={cands[0]}")
    # Per-client stage-1 (identical encoder/quantizer, OWN decoder), frozen.
    s1_by_e = {e: load_stage1(ck[e], cfg, ex, device) for e in entities}
    s1 = s1_by_e[entities[0]]
    val_loader = _pooled_val_loader(cfg, entities, data_by_e)
    # The baselines' EXACT converged recipe (warmup+cosine over stage2_max_steps, val
    # early-stop, best-on-val restore) — and the val oracle honours --fed-s2-val like
    # every other _build_train_stage2_converged caller. Tokens come from s1 (= p0), but
    # every client's tokens are identical by the check above.
    s2 = _build_train_stage2_converged(cfg, s1, loader, val_loader, device, ex, tag="fedtokcp")
    print(f"[fedtokcp] central prior trained on {len(loader.dataset)} pooled windows from "
          f"{len(entities)} clients, on the FROZEN federated tokenizer")
    # ONE central prior, each client's OWN stage-1 (own decoder for s_local): wrap the
    # trained prior weights around every client's s1.
    models = {}
    for e in entities:
        if e == entities[0]:
            models[e] = (s1, s2)
            continue
        s2_e = Stage2System(cfg, s1_by_e[e])
        s2_e.prior.to(device)
        s2_e.materialize(ex.to(device))
        s2_e.to(device)
        s2_e.prior.load_state_dict(s2.prior.state_dict())
        s2_e.stage1.eval(); s2_e.prior.eval()
        models[e] = (s1_by_e[e], s2_e)
    return models


def train_centraltok_fedprior(cfg, entities, data_by_e, device, stage1_root: str,
                              cluster: str, seed: int, *, s2_rounds: int, local_epochs: int,
                              select_on_val: bool, patience_rounds: int, val_mode: str,
                              val_every: int, out_history: dict | None = None) -> dict:
    """⛔ DIAGNOSTIC, federation-ILLEGAL — the INVERSE of train_fedtok_centralprior.

    Central TOKENIZER (the `centralized` arm's stage-1: ONE model trained on pooled
    windows) frozen and handed identically to every client; stage 2 then runs A2's EXACT
    federated loop (shared prior body FedAvg + local heads, epoch regime) on each client's
    OWN data shard. Completes the 2×2 attribution with {A2, fedtokcp, centralized}:

        tokenizer \\ prior |  federated   |  central
        federated (A2 s1) |  A2 = 0.086  |  fedtokcp = 0.202     (ucr_170, AUPRC)
        central   (c12d)  |  THIS ARM    |  centralized = 0.878

    If this cell lands near `centralized`, the federated stage-1 is the SOLE culprit on
    ucr_170; if it lands near fedtokcp, the federated stage-2 carries damage of its own.
    """
    src = Path(stage1_root) / cfg.dataset.name / cluster / f"seed{seed}"
    cands = sorted(d for d in src.glob("centralized*") if d.is_dir())
    if len(cands) != 1:
        raise SystemExit(f"centraltok_fedprior: expected exactly ONE centralized source arm "
                         f"dir under {src}, found {len(cands)}: {[c.name for c in cands]} — "
                         f"pass a --fedtokcp-stage1-root whose tag holds a single "
                         f"`centralized` run (e.g. artifacts/runs/zn_main/ckpt).")
    ck = {e: cands[0] / e / "stage1.ckpt" for e in entities}
    missing = [str(p) for p in ck.values() if not p.exists()]
    if missing:
        raise SystemExit(f"centraltok_fedprior: missing stage1.ckpt(s): {missing}")
    # THE PREMISE, VERIFIED: `centralized` trains ONE stage-1 and saves it per entity — the
    # five files must be the SAME model on EVERY tensor (decoder included: s_local must be
    # central too, or the contrast would smuggle per-client reconstruction paths in).
    ref = torch.load(str(ck[entities[0]]), map_location="cpu")["state_dict"]
    for e in entities[1:]:
        other = torch.load(str(ck[e]), map_location="cpu")["state_dict"]
        diffs = [k for k in ref if not torch.allclose(ref[k], other[k], atol=1e-6)]
        if diffs:
            raise SystemExit(f"centraltok_fedprior: centralized stage-1 of {e} differs from "
                             f"{entities[0]} on {len(diffs)}/{len(ref)} tensors "
                             f"(e.g. {diffs[:3]}) — the source is not one central model.")
    print(f"[ctfp] central tokenizer identity verified across {len(entities)} clients "
          f"(ALL tensors, atol=1e-6; source={cands[0]})")
    from types import SimpleNamespace
    clients = []
    for e in entities:
        ex = next(iter(data_by_e[e].train_loader))["inputs"][:1]
        s1 = load_stage1(ck[e], cfg, ex, device)
        # The duck-typed subset of ClientState that federated_stage2/_build_stage2_client
        # actually reads: entity_id, cfg, model, data.
        clients.append(SimpleNamespace(entity_id=e, cfg=cfg, model=s1, data=data_by_e[e]))
    print(f"[ctfp] stage 2: A2's federated loop (shared body FedAvg + local heads) on the "
          f"FROZEN central tokenizer — ceiling {s2_rounds} rounds, patience "
          f"{patience_rounds}, val={val_mode}")
    s2_clients, h2, _ = federated_stage2(clients, cfg, rounds=s2_rounds,
                                         local_epochs=local_epochs, seed=seed,
                                         local_prefixes=LOCAL_PRIOR_PREFIXES,
                                         select_on_val=select_on_val, val_mode=val_mode,
                                         val_every=val_every,
                                         patience_rounds=patience_rounds)
    if out_history is not None:
        out_history["cb_mode"] = "central-oracle"
        out_history["stage2"] = h2
    return {c.entity_id: (c.s2.stage1, c.s2) for c in s2_clients}


# Set by main() before each (arm, seed); read by `train_federated` when its own
# resume_from/resume_out are not given. Module-level rather than threaded through all 18
# train_federated call sites, so no other arm's signature or behaviour changes.
_RESUME_CTX: dict = {"from": None, "out": None}


def train_federated(cfg, entities, s1_rounds, s2_rounds, local_epochs, device, seed: int = 7,
                    local_prefixes: tuple[str, ...] = LOCAL_PRIOR_PREFIXES,
                    cb_merge: str = "suffstat", cb_server_ema_decay: float | None = None,
                    revive_dead: bool = True,
                    anchor_weight: float = 0.0,
                    fed_encoder: str = "off", enc_split_at: int = 2,
                    enc_fed_algo: str = "fedavg", enc_bn: str = "buffers_local",
                    enc_prox_mu: float = 0.0, enc_prox_form: str = "loss",
                    enc_proto_weight: float = 0.0, enc_proto_agg: str = "uniform",
                    enc_proto_code_weight: str = "uniform", enc_proto_seed_round0: bool = True,
                    enc_proto_fedavg: bool = False, enc_sched: str = "none",
                    resume_from=None, resume_out=None, patience_rounds: int = 0,
                    resume_rounds_done: int = 0,
                    fed_align: str = "off", align_weight: float = 0.0, align_cluster: str | None = None,
                    server_momentum: float = 0.0,
                    tau_steps: int = 0, server_opt: str = "sgd", server_lr: float = 0.01,
                    server_b1: float = 0.9, server_b2: float = 0.99, server_eps: float = 1e-3,
                    server_momentum_skip_round0: bool = False,
                    s2_val_mode: str = "legacy", s2_val_every: int = 1, s2_patience: int = 0,
                    s2_agg_penalty: bool = False, s2_snapshot_every: int = 0,
                    prior_prox_mu: float = 0.0, prior_prox_form: str = "decoupled",
                    proto_weight: float = 0.0, proto_probe_windows: int = 1024,
                    proto_mask_ratio: float = 0.5, protocol: str = "fixed",
                    out_history: dict | None = None) -> dict:
    # `protocol="converged"` gives the federated arm the SAME training regime that lifted
    # the local/centralized baselines, so its verdict is comparable to the risen local:
    #   * stage 1: select_on_val restores the best round (the round-based analogue of the
    #     baselines' best-on-val restore); over-provision rounds safely, the best is kept.
    #   * stage 2: if the prior is FULLY LOCAL (cb_only, local_prefixes==("",)), there is no
    #     cross-client aggregation, so the round loop was pure overhead — train each client's
    #     prior with the EXACT converged loop the baselines use (val early-stop, warmup+cosine,
    #     best-on-val). This makes cb_only's prior identical-protocol to `local`'s; the ONLY
    #     difference from `local` is then the shared codebook -> clean attribution.
    converged = (protocol == "converged")
    resume_from = resume_from if resume_from is not None else _RESUME_CTX["from"]
    # Every federated arm gets its per-round telemetry persisted, not only the encoder trio:
    # the trio's dispatch passes `out_history` explicitly, every other arm inherits the sink
    # from here. Without this, `federated_cb_only` runs (the probe, the ema sweep) produced no
    # fed_history.json and their curves survived only in stdout.
    out_history = out_history if out_history is not None else _RESUME_CTX.get("hist")
    patience_rounds = patience_rounds or int(_RESUME_CTX.get("patience") or 0)
    resume_out = resume_out if resume_out is not None else _RESUME_CTX["out"]
    if out_history is not None:
        # The codebook regime is recorded HERE, by the function that actually performs the
        # merge, and not by the dispatcher: main() only knows `--fed-enc-cb`, which exactly two
        # of the 32 arms read. Every other branch passes its own `cb_merge` ("fedavg" for the
        # naive-average arms) or takes this signature's "suffstat" default, so the old
        # `fed_hist["cb_mode"] = args.fed_enc_cb` could certify a merge that never happened —
        # e.g. `--arms federated_cb_only --fed-enc-cb local` wrote cb_mode="local" over a
        # suff-stat run. main() now only fills this key in if it is still missing.
        out_history["cb_mode"] = cb_merge
    # `out_history` is the ONLY way the per-round federated telemetry (loss, val, perplexity,
    # dead_frac, codebook drift, and the encoder-federation channels) survives this call:
    # the caller gets models back, and everything the rounds measured used to be dropped on
    # the floor. Optional so no existing call site changes behaviour.
    clients, _, s1_history = federated_stage1(
        cfg, entities, rounds=s1_rounds, local_epochs=local_epochs,
        seed=seed, merge=cb_merge, cb_server_ema_decay=cb_server_ema_decay,
        revive_dead=revive_dead, anchor_weight=anchor_weight,
        fed_encoder=fed_encoder, enc_split_at=enc_split_at,
        enc_fed_algo=enc_fed_algo, enc_bn=enc_bn,
        enc_prox_mu=enc_prox_mu, enc_prox_form=enc_prox_form,
        enc_proto_weight=enc_proto_weight, enc_proto_agg=enc_proto_agg,
        enc_proto_code_weight=enc_proto_code_weight,
        enc_proto_seed_round0=enc_proto_seed_round0,
        enc_proto_fedavg=enc_proto_fedavg, enc_sched=enc_sched,
        resume_from=resume_from, resume_out=resume_out,
        patience_rounds=patience_rounds,
        resume_rounds_done=resume_rounds_done or int(_RESUME_CTX.get("rounds_done") or 0),
        fed_align=fed_align, align_weight=align_weight,
        align_cluster=align_cluster, select_on_val=converged)
    if out_history is not None:
        out_history["stage1"] = s1_history
    prior_fully_local = (local_prefixes == ("",))
    # `prior_prox_mu == 0` in the fast-path condition: with a fully-local prior a μ>0 call
    # must NOT be silently converted to the per-client converged loop — falling through to
    # federated_stage2 makes it raise (empty anchored-param set), loudly.
    if converged and prior_fully_local and proto_weight == 0 and tau_steps == 0 \
            and prior_prox_mu == 0:
        out = {}
        for c in clients:
            d = next(next(iter(c.model.parameters())).device for _ in [0])
            ex = next(iter(c.data.train_loader))["inputs"][:1]
            s2 = _build_train_stage2_converged(cfg, c.model, c.data.train_loader,
                                               c.data.val_loader, d, ex, tag=c.entity_id)
            out[c.entity_id] = (c.model, s2)
            print(f"[fed-cb_only] {c.entity_id} prior trained (converged, "
                  f"{'LOCAL' if cb_merge == 'local' else 'shared'} codebook)")
        return out
    if proto_weight > 0:
        s2_clients, h2, _ = federated_stage2_proto(clients, cfg, rounds=s2_rounds, local_epochs=local_epochs,
                                                   seed=seed, proto_weight=proto_weight,
                                                   probe_windows=proto_probe_windows,
                                                   mask_ratio=proto_mask_ratio,
                                                   batch=cfg.dataset.batch_size_stage2)
    else:
        # (clients, HISTORY, head_div) — the middle element. Unpacking the third put the
        # scalar head-divergence into fed_history.json and silently dropped the rounds.
        s2_clients, h2, _ = federated_stage2(clients, cfg, rounds=s2_rounds, local_epochs=local_epochs, seed=seed,
                                             local_prefixes=local_prefixes, server_momentum=server_momentum,
                                             tau_steps=tau_steps, server_opt=server_opt, server_lr=server_lr,
                                             server_b1=server_b1, server_b2=server_b2,
                                             server_eps=server_eps,
                                             server_momentum_skip_round0=server_momentum_skip_round0,
                                             val_mode=s2_val_mode, val_every=s2_val_every,
                                             agg_penalty=s2_agg_penalty,
                                             snapshot_every=s2_snapshot_every,
                                             prior_prox_mu=prior_prox_mu,
                                             prior_prox_form=prior_prox_form,
                                             # the arm scratch dir — same place the resume
                                             # bundle lives, so snapshots stay with their run
                                             snapshot_dir=resume_out,
                                             select_on_val=converged,
                                             # Same ceiling-not-budget contract as stage 1: the
                                             # shared prior body has no other early stop, so
                                             # without this whatever --s2-rounds is IS the budget.
                                             # s2_patience (EVALS) overrides the shared knob so a
                                             # τ retune cannot silently change stage-1 stopping.
                                             patience_rounds=(s2_patience or patience_rounds))
    if out_history is not None:
        out_history["stage2"] = h2
    return {c.entity_id: (c.s2.stage1, c.s2) for c in s2_clients}


def train_fedsgd_pooltok(cfg, entities, data_by_e, s1_epochs, s2_rounds, local_epochs, device,
                         seed: int = 7, tau_steps: int = 1,
                         server_opt: str = "fedadam", server_lr: float = 0.01,
                         tok_cache_dir: "str | Path | None" = None) -> dict:
    """Cell A of the tokenizer×prior 2×2 — decouple the two axes of the wsd gap.

    A SINGLE POOLED tokenizer (trained on entity-MIXED batches, IDENTICAL code path to
    `train_centralized`'s stage1) is trained once and FROZEN; then the fully-shared
    prior (`local_prefixes=()`) is federated at τ optimizer STEPS/round with a FedAdam
    server — the exact `federated_fedsgd` prior regime — over each client's OWN token
    stream. Because every client wraps the *same* frozen pooled tokenizer, this holds
    the tokenizer fixed and varies only the prior:

        centralized − pooltok  = pure PRIOR-federation penalty (identical pooled tokenizer)
        pooltok     − fedsgd    = pure TOKENIZER penalty        (identical τ prior)

    τ=1 ⇒ each round is one distributed-SGD step from a common broadcast init, so the
    prior should recover the centralized prior — the sanity upper bound. The pooled
    tokenizer is cached per (seed, s1_epochs) so the τ sweep trains it ONCE, keeping
    the tokenizer bit-identical across τ (and cheap)."""
    # 1) POOLED tokenizer — cached across the τ sweep (τ-independent ⇒ same tokenizer).
    loader = _pooled_loader(cfg, entities, data_by_e)
    ex = next(iter(loader))["inputs"][:1]
    s1 = _materialized_stage1(cfg, ex, device, collect_stats=False)
    cache = (Path(tok_cache_dir) / f"pooltok_seed{seed}_s1e{s1_epochs}.pt") if tok_cache_dir else None
    if cache is not None and cache.exists():
        s1.load_state_dict(torch.load(cache, map_location=device))
        s1.to(device)
        print(f"[pooltok] loaded cached pooled tokenizer <- {cache}")
    else:
        _train_stage1(s1, loader, s1_epochs, cfg.training.lr, device, tag="pooltok")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(s1.state_dict(), cache)
            print(f"[pooltok] cached pooled tokenizer -> {cache}")
    s1.eval()
    print(f"[pooltok] frozen pooled tokenizer on {len(loader.dataset)} pooled windows "
          f"from {len(entities)} clients; federating shared prior at tau={tau_steps} "
          f"({s2_rounds} round(s), server={server_opt})")

    # 2) per-client stage1 states that all SHARE the one frozen pooled tokenizer but
    #    carry their OWN data → the prior federates over per-client token streams.
    #    (_build_client materializes a throwaway stage1 we immediately replace with s1.)
    stage1_clients = []
    for e in entities:
        c = _build_client(cfg, e, device, collect_stats=False)
        c.model = s1                                          # SHARE the pooled tokenizer
        stage1_clients.append(c)

    # 3) FedSGD/τ prior with the server optimizer — the identical federated_stage2 call
    #    federated_fedsgd uses, so `pooltok − fedsgd` isolates the tokenizer cleanly.
    s2_clients, _, _ = federated_stage2(stage1_clients, cfg, rounds=s2_rounds,
                                        local_epochs=local_epochs, seed=seed, local_prefixes=(),
                                        tau_steps=tau_steps, server_opt=server_opt, server_lr=server_lr)
    return {c.entity_id: (c.s2.stage1, c.s2) for c in s2_clients}


def train_pooltok_centralprior(cfg, entities, data_by_e, s1_epochs, prior_steps, device,
                               seed: int = 7, batch: int = 16,
                               tok_cache_dir: "str | Path | None" = None) -> dict:
    """Cell C of the tokenizer×prior 2×2 — isolate the PRIOR term cleanly.

    The SAME cached POOLED tokenizer as `federated_fedsgd_pooltok` (bit-identical, loaded from
    cache), FROZEN, with a POOLED (centralized) prior trained STEP-based at the same
    optimizer-update budget (`prior_steps`) and the same effective per-update batch
    (K clients × `batch`) as the τ=1 FedSGD prior. So:

        C − pooltok = pure PRIOR-FEDERATION penalty (identical tokenizer AND identical budget)

    This is the one comparison that removes the budget/batch confound between `centralized`
    (batch 128, epoch-based) and the FedSGD sweep. C ≈ pooltok ⇒ the prior federates losslessly
    and the local→centralized gap was a budget artifact; C ≫ pooltok ⇒ a real pooled-prior
    advantage the FedSGD prior cannot capture."""
    loader = _pooled_loader(cfg, entities, data_by_e)
    ex = next(iter(loader))["inputs"][:1]
    s1 = _materialized_stage1(cfg, ex, device, collect_stats=False)
    cache = (Path(tok_cache_dir) / f"pooltok_seed{seed}_s1e{s1_epochs}.pt") if tok_cache_dir else None
    if cache is not None and cache.exists():
        s1.load_state_dict(torch.load(cache, map_location=device))
        s1.to(device)
        print(f"[centralprior] loaded cached pooled tokenizer <- {cache}")
    else:
        _train_stage1(s1, loader, s1_epochs, cfg.training.lr, device, tag="pooltok")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(s1.state_dict(), cache)
            print(f"[centralprior] cached pooled tokenizer -> {cache}")
    s1.eval()

    # Pooled prior at effective batch = K×`batch`, matching the τ=1 FedSGD per-update batch
    # (one `batch`-sized minibatch per client, aggregated) and update count (`prior_steps`).
    K = len(entities)
    ds = ConcatDataset([data_by_e[e].train_dataset for e in entities])
    ploader = _build_loader(ds, K * batch, cfg.dataset.num_workers, shuffle=True)
    pex = next(iter(ploader))["inputs"][:1]
    print(f"[centralprior] frozen pooled tokenizer; POOLED prior {prior_steps} steps "
          f"@ effective batch {K}×{batch}={K * batch} on {len(ds)} pooled windows")
    s2 = _build_train_stage2_steps(cfg, s1, ploader, prior_steps, cfg.training.lr, device, pex, tag="central")
    return {e: (s1, s2) for e in entities}


# per-client detection via the existing pipeline

def save_client_ckpts(cfg, entity, stage1, s2, scratch: Path) -> tuple[Path, Path]:
    c = copy.deepcopy(cfg); c.dataset.entity_id = entity
    d = scratch / entity; d.mkdir(parents=True, exist_ok=True)
    s1p, s2p = d / "stage1.ckpt", d / "stage2.ckpt"
    save_stage1_checkpoint(s1p, stage1, c, 0, 0)
    save_stage2_checkpoint(s2p, s2, c, 0, 0)
    return s1p, s2p


def eval_from_ckpts(cfg, entity, s1p: Path, s2p: Path, scratch: Path) -> dict:
    c = copy.deepcopy(cfg); c.dataset.entity_id = entity
    d = scratch / entity; d.mkdir(parents=True, exist_ok=True)
    report = detect.detect(c, stage1_ckpt=s1p, stage2_ckpt=s2p, output_dir=d)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _aggregate(reports, keys: list[str]) -> dict[str, dict]:
    """Mean / std / worst (min) for each metric, pooled over an iterable of
    per-client report dicts (across clients AND seeds)."""
    rlist = list(reports)
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rlist
                if isinstance(r.get(k), (int, float)) and np.isfinite(r.get(k))]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "worst": float(np.min(vals)), "n": len(vals)}
    return out


@torch.no_grad()
def cf_fidelity_from_ckpts(cfg, entity, s1p: Path, s2p: Path, device,
                           score_quantile: float = 0.9, max_windows: int = 24) -> dict:
    """Counterfactual quality vs the TRUE clean test signal (x_clean).

    Reloads the trained model, generates counterfactuals for the test windows
    that overlap an injected anomaly, decodes them, and compares against
    test_clean using the per-channel-per-timestep deviation mask. Everything is
    done in the per-entity z-scored space the model was trained in.

    Requires a ground-truth clean signal, so it is defined only for the synthetic
    builds. On a real dataset (wsd_fed) `test_clean/` does not exist — there is no
    counterfactual to score against — and every CF key comes back NaN, exactly as
    it already does when a client's anomalies all sit at the series boundary."""
    root = resolve_path(cfg.paths.raw_data) / cfg.dataset.name
    if not (root / "test_clean").exists() or not (root / "test_mask").exists():
        return {k: float("nan") for k in CF_KEYS}
    train = np.load(root / "train" / f"{entity}.npy").astype(np.float32)
    test = np.load(root / "test" / f"{entity}.npy").astype(np.float32)
    clean = np.load(root / "test_clean" / f"{entity}.npy").astype(np.float32)
    mask = np.load(root / "test_mask" / f"{entity}.npy").astype(bool)        # (T, C)

    mu, sd = train.mean(0, keepdims=True), train.std(0, keepdims=True) + 1e-8  # per_entity_standard
    xs = (test - mu) / sd
    cs = (clean - mu) / sd
    W = cfg.dataset.window_length
    T = xs.shape[0]

    cand = list(range(0, max(1, T - W + 1), max(1, W // 2)))
    sel = [s for s in cand if mask[s:s + W].any()]
    if not sel:                                                  # anomalies only near the boundary
        anom_t = np.where(mask.any(axis=1))[0]
        sel = sorted({int(np.clip(t - W // 2, 0, T - W)) for t in anom_t})
    sel = sel[:max_windows]
    if not sel:
        return {k: float("nan") for k in CF_KEYS}

    c = copy.deepcopy(cfg); c.dataset.entity_id = entity
    example = torch.tensor(xs[:W].T, dtype=torch.float32)[None]   # (1, C, W) CPU for materialize
    s2 = load_stage2(s2p, c, stage1_ckpt=s1p, stage1_example_inputs=example, device=device)

    xb = np.stack([xs[s:s + W].T for s in sel])                  # (n, C, W)
    out = counterfactual(s2, torch.tensor(xb, dtype=torch.float32, device=device),
                         score_quantile=score_quantile, greedy=True)
    cf = out["x_cf"].detach().cpu().numpy()                      # (n, C, W)

    repair_num = repair_den = disturb_sum = disturb_cnt = 0.0
    for i, s in enumerate(sel):
        cw = cs[s:s + W].T                                       # (C, W) scaled clean
        xw = xs[s:s + W].T                                       # (C, W) scaled test
        mw = mask[s:s + W].T                                     # (C, W) bool
        a, cl = mw, ~mw
        repair_num += np.abs(cf[i] - cw)[a].sum()
        repair_den += np.abs(xw - cw)[a].sum()                  # anomaly magnitude
        disturb_sum += np.abs(cf[i] - cw)[cl].sum()
        disturb_cnt += cl.sum()

    del s2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ratio = float(repair_num / repair_den) if repair_den > 1e-8 else float("nan")
    return {
        "cf_repair_ratio": ratio,
        "cf_repair_improvement": (1.0 - ratio) if np.isfinite(ratio) else float("nan"),
        "cf_disturb_mae": float(disturb_sum / disturb_cnt) if disturb_cnt > 0 else float("nan"),
        "cf_n_windows": float(len(sel)),
    }


def _git_provenance() -> dict:
    """Stamp the code state into the run so the camera-ready table ties to a commit."""
    import subprocess
    root = str(Path(__file__).resolve().parent.parent)
    try:
        sha = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _arm_tag(arm: str, fed: dict | None) -> str:
    """Artefact directory name for one (arm, knob-setting). Plain `arm` when it has no
    swept knobs, so every existing arm's path is unchanged."""
    if not fed:
        return arm
    bits = []
    if "mu" in fed:
        bits.append(f"mu{fed['mu']:g}" + ("" if fed.get("form") == "loss" else "_dec"))
    if "lambda" in fed and fed.get("lambda"):
        bits.append(f"lam{fed['lambda']:g}_{fed.get('agg', 'uniform')}")
        if fed.get("code_weight") == "count":
            bits.append("cwcount")
        if fed.get("hybrid_fedavg"):
            bits.append("hybrid")
        if not fed.get("seed_round0", True):
            bits.append("noseed")
    if fed.get("scope") not in (None, "full"):
        # split_at only means anything for 'partial', but two 'partial' runs that differ
        # ONLY by it would otherwise share a directory and silently overwrite each other.
        bits.append(str(fed["scope"])
                    + (f"{fed.get('enc_split_at')}" if fed["scope"] == "partial" else ""))
    if fed.get("bn") not in (None, "buffers_local"):
        bits.append(f"bn-{fed['bn']}")
    if fed.get("sched") not in (None, "none"):
        bits.append(f"sched-{fed['sched']}")
    if fed.get("cb") not in (None, "suffstat"):
        bits.append(f"cb-{fed['cb']}")
    if fed.get("prior") not in (None, "local"):
        bits.append(f"prior-{fed['prior']}")
    # ── stage-2 server bits (2026-08-07), terminal and in canonical order ───────────────
    # Emitted only when non-default, so every existing directory name is unchanged. Without
    # these, two stage-2 variants of the same enc knobs would SHARE
    # .../seed0/federated_enc_fedavg_bn-shared_prior-partial/ — mixed checkpoints,
    # knobs.json overwritten by the last writer, and _fed_resume.pt adopted cross-variant.
    if fed.get("tau"):
        bits.append(f"tau{fed['tau']:g}")
    if fed.get("srv"):
        bits.append(f"srv-{fed['srv']}_slr{fed['slr']:g}")
    if fed.get("sm"):
        bits.append(f"sm{fed['sm']:g}" + ("_skip0" if fed.get("sm_skip0") else ""))
    if fed.get("pmu"):   # FedProx-on-prior (port to come); reserved here so the order is fixed
        bits.append(f"pmu{fed['pmu']:g}" + ("" if fed.get("pform") == "loss" else "_dec"))
    if fed.get("K"):     # codebook capacity override (--codebook-size): without this bit a
        bits.append(f"K{fed['K']}")   # K=128 cell shares plain A2's dir NAME and any script
                                      # keyed on (fingerprint, cluster, arm-dir) pools them
                                      # as comparable — the incomplete-fingerprint trap.
    return "_".join([arm, *bits]) if bits else arm


def _structural_echo(cfg: Config, stage1, s2, fed: dict | None = None, models: dict | None = None) -> dict:
    """Knobs REQUESTED (cfg) vs REALIZED (MEASURED off the built tensors, never cfg).

    cfg.encoder.dim=64 sat in cfg for the whole life of the fork while the conv body was
    hardcoded to 4, so a cfg-only echo records the intent and certifies the lie. `requested`
    != `realized` is the only machine-checkable evidence that a knob actually threaded --
    a `--width-base 16` run that still reports encoder_n_params=5,256 is the false null, caught.
    Known-good: wb4 -> body 4/8/16, encoder 5,256, stage1 85,546; wb16 -> 16/32/64, 74,568, 352,666.

    Crash-proof by construction: every field is guarded, so a wrong introspection path degrades
    to None rather than killing a run whose training already completed.
    """
    def _try(fn):
        try:
            return fn()
        except Exception:
            return None
    enc = getattr(stage1, "encoder", None)
    prior = getattr(s2, "prior", None)
    realized = {
        "encoder_conv_out_channels": _try(lambda: [int(m.out_channels) for m in enc.modules()
                                                   if isinstance(m, torch.nn.Conv2d)]),
        "encoder_n_params": _try(lambda: sum(p.numel() for p in enc.parameters())),
        "stage1_n_params": _try(lambda: sum(p.numel() for p in stage1.parameters())),
        "prior_n_params": _try(lambda: sum(p.numel() for p in prior.parameters())),
    }
    if fed and models:
        # The machine-checkable proof that the arm did what its name says: EXACTLY 0 for
        # any arm that weight-averages the encoder, > 0 for one that does not (fedproto,
        # commoninit). A fedproto row reporting 0 here would be a mislabelled FedAvg.
        # Added only for arms that federate an encoder, so every other arm's knobs.json
        # keeps exactly the fields it had.
        from federated import _encoder_shared_keys       # local: avoids an import cycle at module load

        def _delta():
            keys = _encoder_shared_keys(stage1, fed.get("scope", "full"),
                                        int(fed.get("enc_split_at", 2)),
                                        fed.get("bn", "buffers_local"))
            sds = [m[0].state_dict() for m in
                   {id(v[0]): v for v in models.values()}.values()]   # one entry per DISTINCT model
            if len(sds) < 2:
                return 0.0
            ref = sds[0]
            return max(float((sd[k].detach().float().cpu() - ref[k].detach().float().cpu())
                             .abs().max()) for sd in sds[1:] for k in keys)
        realized["enc_shared_tensors"] = _try(lambda: len(
            _encoder_shared_keys(stage1, fed.get("scope", "full"),
                                 int(fed.get("enc_split_at", 2)), fed.get("bn", "buffers_local"))))
        realized["enc_max_cross_client_delta"] = _try(_delta)
    return {
        "requested": {
            "encoder.width_base": getattr(cfg.encoder, "width_base", None),
            "quantizer.token_embedding_dim": cfg.quantizer.token_embedding_dim,
            "quantizer.codebook_size": cfg.quantizer.codebook_size,
            "quantizer.name": cfg.quantizer.name,
            "prior.embed_dim": cfg.prior.embed_dim,
            "prior.hidden_dim": cfg.prior.hidden_dim,
            "prior.heads": cfg.prior.heads,
            "prior.dropout": cfg.prior.dropout,
            "dataset.window_length": cfg.dataset.window_length,
        },
        "realized": realized,
        "amp_dtype": (str(_amp_dtype()).replace("torch.", "") if _amp_dtype() is not None else "fp32"),
        # Federation knobs that live in argv rather than cfg (e.g. the encoder-federation
        # trio's μ / λ / aggregation). None for arms that have none.
        **({"federation": fed} if fed else {}),
    }


def main() -> int:
    force_utf8_stdout()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="toy_fed_uni")
    p.add_argument("--clients", type=str, default=None,
                   help="comma-separated entity ids; default = every entity on disk")
    p.add_argument("--eval-clients", type=str, default=None,
                   help="comma-separated subset of the TRAINED clients to score. Default "
                        "= score all of them. Use when a pooled arm trains on far more "
                        "series than have a matched `local` counterpart (e.g. centralized "
                        "over all 248 UCR series, scored only on the 3 that also have a "
                        "local model): each skipped client saves a full detection pass.")
    p.add_argument("--cluster", type=str, default=None,
                   help="run the comparison on ONE machine-type cluster (needs clusters.json, "
                        "e.g. toy_fed_uni). Overrides --clients; artifacts land under "
                        "<out-dir>/<cluster>/. This is the intended unit of federation for the "
                        "univariate benchmark.")
    p.add_argument("--arms", type=str, default="local,centralized,federated")
    p.add_argument("--s1-epochs", type=int, default=2)      # local/centralized stage1
    p.add_argument("--s2-epochs", type=int, default=2)      # local/centralized stage2
    p.add_argument("--protocol", choices=["fixed", "converged"], default="fixed",
                   help="Training recipe for the local/centralized BASELINES. "
                        "'fixed' (default, unchanged) = --s1-epochs/--s2-epochs, no "
                        "validation. 'converged' = the run.py recipe: warmup+cosine LR, "
                        "fp32 val pass every epoch, step-based early stopping, best-on-val "
                        "weights; the epoch flags are then ignored. NOTE the federated "
                        "arms still train on a FIXED round budget either way — see the "
                        "warning printed when both are selected.")
    p.add_argument("--s1-rounds", type=int, default=2)      # federated stage1
    p.add_argument("--s2-rounds", type=int, default=2)      # federated stage2
    p.add_argument("--local-epochs", type=int, default=1)   # federated local epochs/round
    p.add_argument("--cb-server-ema-decay", type=float, default=None,
                   help=f"server-side codebook EMA decay γ for the 'federated_cb_only_ema' arm. "
                        f"Default (unset) = {_CB_SERVER_EMA_DEFAULT_DECAY} (round-granularity rate; "
                        f"NOT cfg.quantizer.ema_decay=0.99, a within-step rate). γ=0 reproduces the "
                        f"plain 'federated_cb_only' merge.")
    p.add_argument("--anchor-weight", type=float, default=1.0,
                   help="FedProto encoder-anchor lambda for the 'federated_anchor' arm (0 = off).")
    p.add_argument("--align-weight", type=float, default=0.5,
                   help="B1 probe-alignment lambda for the 'federated_align' arm (0 = off).")
    p.add_argument("--pool-cap", type=int, default=0,
                   help="centralized_cap: cap the pooled training set to N diverse windows (0 = no cap).")
    p.add_argument("--proto-weight", type=float, default=1.0,
                   help="federated_protoprior: λ on the prior-prototype KL consensus term.")
    p.add_argument("--proto-probe-windows", type=int, default=1024,
                   help="federated_protoprior: shared probe size for the consensus prototype.")
    p.add_argument("--tau-steps", type=int, default=0,
                   help="federated_fedsgd: optimizer STEPS per round (τ). 0 = epoch-based (other arms).")
    p.add_argument("--server-opt", type=str, default="sgd", choices=["sgd", "fedadam"],
                   help="federated_fedsgd server optimizer over the pseudo-gradient.")
    p.add_argument("--server-lr", type=float, default=0.01, help="FedAdam server step size η.")
    p.add_argument("--server-momentum", type=float, default=0.9,
                   help="FedAvgM server momentum for the 'federated_fedavgm' arm (0 = plain FedAvg).")
    # ── stage-2 knobs for the ENC-TRIO arms (the A2 family) ─────────────────────────────
    # Dedicated flags with NULL defaults, on purpose: --server-momentum above defaults to
    # 0.9 because it was written for federated_fedavgm — forwarding it blindly from the trio
    # branch would silently turn every plain A2 rerun into FedAvgM. tau/server-opt/server-lr
    # are shared with the fedsgd arms (their argparse defaults ARE the function defaults, so
    # reuse is safe); only the momentum needs its own name.
    p.add_argument("--s2-server-momentum", type=float, default=0.0,
                   help="FedAvgM for the enc-trio stage-2 prior body (0 = off = plain FedAvg, "
                        "today's behaviour). Mutually exclusive with --server-opt fedadam. "
                        "β>0.5 with full local epochs is warned against (undamped heavy-ball).")
    p.add_argument("--s2-momentum-skip-round0", action="store_true",
                   help="FedAvgM: skip momentum accumulation at round 0 (the round-0 average "
                        "is measured destructive: val(agg0) above the uniform floor).")
    p.add_argument("--server-b1", type=float, default=0.9,
                   help="FedAdam β1 (was hardcoded; default unchanged).")
    p.add_argument("--server-b2", type=float, default=0.99,
                   help="FedAdam β2 (was hardcoded; default unchanged).")
    p.add_argument("--server-eps", type=float, default=1e-3,
                   help="FedAdam ε (was hardcoded; default unchanged — NB large: damps early steps).")
    p.add_argument("--fed-s2-val", type=str, default="legacy", choices=["legacy", "fixed"],
                   help="stage-2 val oracle. 'legacy' (default, every run on disk) = random "
                        "MaskGIT masks + autocast: σ≈0.006-0.018 nats/eval. 'fixed' = frozen "
                        "per-client masks (fork_rng: training stream untouched) + fp32. "
                        "PREREQUISITE for any small-τ run; applies to the federated shared-body "
                        "val AND to the per-client converged loop (baselines/cb_only priors).")
    p.add_argument("--fed-s2-val-every", type=int, default=1,
                   help="evaluate the stage-2 cohort val every K rounds (aggregation still every "
                        "round); patience counts EVALS. 1 = today's behaviour. Use with small τ "
                        "(e.g. τ=16→K=2, τ=1→K=16) or per-round val dominates wall-clock.")
    p.add_argument("--fed-s2-agg-penalty", action="store_true",
                   help="log per eval round the val of each client's OWN end-of-round body "
                        "next to the broadcast average's: agg_penalty = val_post − val_pre. "
                        "The number that decides FedProx-on-the-prior.")
    p.add_argument("--fed-s2-snapshot-every", type=int, default=0,
                   help=">0: save shared body + per-client heads every K rounds into the arm "
                        "dir (s2_snap_rNNNN.pt) plus s2_last.pt before the best-round restore "
                        "— fuels the trajectory-detection probe and the best-vs-last contrast.")
    p.add_argument("--fedtokcp-stage1-root", type=str, default=None,
                   help="fedtok_centralprior ONLY: ckpt root of the SOURCE federated run "
                        "(e.g. artifacts/runs/zn_a2/ckpt) whose frozen stage-1 tokenizer the "
                        "central prior is trained on. The arm dir under it must hold one "
                        "<entity>/stage1.ckpt per client.")
    p.add_argument("--fed-s2-patience", type=int, default=0,
                   help="separate patience for the stage-2 shared-body loop, in EVALS "
                        "(0 = inherit --fed-patience-rounds, today's behaviour). Needed because "
                        "one --fed-patience-rounds feeds BOTH stages: retuning it for a τ regime "
                        "would silently change stage-1 stopping too.")
    p.add_argument("--fed-prior-prox-mu", type=float, default=0.0,
                   help="FedProx on the SHARED stage-2 prior body (enc-trio arms): μ of the "
                        "proximal anchor toward the start-of-round broadcast body; params only, "
                        "local heads never anchored. 0 = off (plain FedAvg, bit-identical). "
                        "NB declared completeness row: the measured agg_penalty is NEGATIVE "
                        "late in training, so there is no pathology for μ to fix.")
    p.add_argument("--fed-prior-prox-form", type=str, default="decoupled",
                   choices=["decoupled", "loss"],
                   help="'decoupled' (primary): post-step contraction w ← w + lr·μ·(w^t−w), "
                        "bypasses AdamW's preconditioner — diagnostic prox_pull_frac. "
                        "'loss': the FedProx paper objective — under AdamW the prox gradient "
                        "is preconditioned and can be a silent no-op; diagnostic prox_grad_ratio.")
    p.add_argument("--batch", type=int, default=None,
                   help="override BOTH stage batch sizes with one value. Default (unset) "
                        "keeps cfg.dataset.batch_size_stage1/stage2 (256/128) -- i.e. the "
                        "run.py recipe. This used to default to 16 and clobber the config "
                        "unconditionally, which broke local-vs-pooled step parity: at "
                        "batch 16 the `min_epochs` floor raises the pooled arm's step cap "
                        "3.4x above a local client's, so the skyline outspent the baseline.")
    # Point 1: shared-codebook CAPACITY sweep
    p.add_argument("--codebook-size", type=int, default=0,
                   help="override cfg.quantizer.codebook_size (0 = config default 64). Capacity test.")
    p.add_argument("--token-dim", type=int, default=0,
                   help="override cfg.quantizer.token_embedding_dim (= encoder/decoder latent dim; "
                        "0 = config default 4). Capacity test.")
    # ── B1: architecture knobs that were UNREACHABLE from this CLI ───────────
    # Sentinel is None, not 0: `--prior-dropout 0` is a LEGITIMATE setting (dropout off),
    # so the falsy-0 sentinel used by --token-dim above would silently ignore it.
    p.add_argument("--width-base", type=int, default=None,
                   help="override cfg.encoder.width_base, the encoder/decoder conv body base width "
                        "(None = config default 4 -> body 4/8/16; 16 -> 16/32/64 = upstream).")
    p.add_argument("--prior-embed-dim", type=int, default=None,
                   help="override cfg.prior.embed_dim AND cfg.prior.hidden_dim together "
                        "(None = config default 128). Must be divisible by cfg.prior.heads.")
    p.add_argument("--prior-hidden-dim", type=int, default=None,
                   help="override cfg.prior.hidden_dim alone (FFN width = 4*hidden_dim). Applied "
                        "AFTER --prior-embed-dim, so pass both to decouple d_model from the FFN.")
    p.add_argument("--prior-dropout", type=float, default=None,
                   help="override cfg.prior.dropout (None = config default 0.2; 0.0 = disable).")
    # ── High-frequency-precision levers (each no-op at its default; orthogonal) ──
    p.add_argument("--spec-weight", type=float, default=None,
                   help="Lever 1: weight of the STFT-domain loss in the Stage-1 backprop total "
                        "(None = config default 0.0 = off). Adds a high-frequency-aware objective.")
    p.add_argument("--downsampled-width", type=int, default=None,
                   help="Lever 2: cfg.encoder.downsampled_width, the latent time resolution "
                        "(None = config default 32; higher = less temporal compression).")
    p.add_argument("--n-fft", type=int, default=None,
                   help="Lever 3: cfg.transform.n_fft, STFT window (None = config default 4 -> 3 "
                        "freq bins; higher = finer frequency resolution).")
    p.add_argument("--refine-mode", type=str, default=None, choices=["linear", "conv"],
                   help="Lever 4: RefinementHead architecture (None = config default 'linear').")
    # Point 2: residual (multi-stage) shared VQ
    p.add_argument("--quantizer", type=str, default="",
                   help="override cfg.quantizer.name; 'residual_shared_codebook_per_channel_vq' "
                        "enables Residual-VQ (coarse+fine shared codebooks). '' = config default.")
    p.add_argument("--n-stages", type=int, default=0,
                   help="Residual-VQ: number of stages (0 = config default 2). Effective capacity "
                        "= K^n_stages cells with n_stages*K codewords.")
    # Point 3: alignment-neck sweep
    p.add_argument("--enc-split-at", type=int, default=2,
                   help="federated_enc_partial: keep the first N encoder blocks (raw-signal front-end) "
                        "LOCAL, share the rest + the codebook-facing proj (the alignment neck). "
                        "Higher N = less sharing (N=11 ≈ share only the last block).")
    # encoder-federation trio: federated_enc_{fedavg,fedprox,fedproto}
    p.add_argument("--fed-enc-scope", type=str, default="full",
                   choices=["full", "partial", "neck"],
                   help="WHICH encoder tensors the trio federates. 'full' (default) = the whole "
                        "encoder; 'partial'/'neck' reuse the alignment-neck split (--enc-split-at).")
    p.add_argument("--fed-enc-bn", type=str, default="buffers_local",
                   choices=["buffers_local", "fedbn", "shared"],
                   help="BatchNorm regime. 'buffers_local' (default) shares the AFFINE parameters "
                        "(gamma/beta) and keeps only the running statistics local — this is NOT "
                        "FedBN, despite what an earlier version of this help text said. 'fedbn' "
                        "keeps the WHOLE BN layer local (gamma, beta and statistics), which is what "
                        "FedBN (Li et al., ICLR 2021) actually specifies. 'shared' federates the "
                        "statistics too, POOLED by the law of total variance rather than averaged. "
                        "`num_batches_tracked` stays local in every mode (int64 counter).")
    p.add_argument("--fed-enc-cb", type=str, default="suffstat",
                   choices=["suffstat", "local", "union_recluster", "fedavg"],
                   help="the VQ CODEBOOK for the trio. 'suffstat' (default) federates it by the "
                        "Prop.1 sufficient-statistic merge; 'local' never federates it at all — "
                        "each client k-means-seeds and EMA-updates its own dictionary exactly as "
                        "the `local` baseline does, so ONLY the encoder crosses the network. "
                        "'union_recluster' = k-FED-style server step: per-client centroids → "
                        "union → farthest-point + weighted Lloyd — preserves minority-client "
                        "motifs the count-pooled M-step dilutes (the ucr_170 fix candidate). "
                        "'fedavg' = the dictionary MOVES locally (EMA + dead-code expiry) and "
                        "the server weight-averages it — the primitive of federated_fedavg_cb_only, "
                        "opened to the trio on 2026-08-08 because at LOCAL encoder it is worth ×20 "
                        "over suffstat on ucr_170 (0.338 vs 0.018) and had never been combined with "
                        "a FEDERATED encoder: every failed 170 fix held the primitive at suffstat. "
                        "Refused on multi-stage RVQ (federates stage 0 only — see federated.py). "
                        "Incompatible with federated_enc_fedproto, whose classes ARE the shared "
                        "codebook indices (see the error message for why).")
    p.add_argument("--fed-patience-rounds", type=int, default=0,
                   help="stop the federated stage-1 loop once the cohort validation loss has "
                        "not improved for N consecutive rounds (0 = off, every arm on disk). "
                        "Needs --protocol converged, which is what turns the per-round "
                        "validation on. Lets a generous --s1-rounds be requested safely: the "
                        "run halts where it actually flattens and logs CONVERGED.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="CONTINUE a finished federated run instead of restarting it. Point at "
                        "the arm directory of that run (…/seed<N>/<arm-tag>/, the one holding "
                        "one sub-directory per client). --s1-rounds then means ADDITIONAL "
                        "rounds. Model + full VQ state always restore; the optimizer/scaler "
                        "restore only if that run wrote `_fed_resume.pt` (runs from this "
                        "version onward do). Single (arm, seed) only — see the guard.")
    p.add_argument("--resume-rounds-done", type=int, default=0,
                   help="for a WEIGHTS-ONLY --resume-from (a source run that predates the resume "
                        "bundle): how many rounds that run completed. Only affects LABELLING — "
                        "round indices in the log and in fed_history.json continue from there "
                        "instead of restarting at 0, so a continuation cannot be misread as a "
                        "fresh run. Ignored when the source has a full bundle (it knows).")
    p.add_argument("--fed-enc-sched", type=str, default="none", choices=["none", "cosine"],
                   help="stage-1 LR schedule for the trio. 'none' (default) = constant LR, what "
                        "every federated arm on disk used — keeps the trio comparable with "
                        "federated_cb_only. 'cosine' = warmup+cosine over the full budget, the "
                        "recipe `local`/`centralized` get — use it when the comparison is against "
                        "`local`, so the gap measures federation and not the training recipe. "
                        "Pick one per table; never mix them in one table.")
    p.add_argument("--fed-enc-prior", type=str, default="local",
                   choices=["local", "partial", "shared"],
                   help="stage-2 prior pairing for the trio. 'local' (default) keeps the ENTIRE prior "
                        "per-client so stage 1 is the only federated surface -> clean attribution; "
                        "'partial' = the main method's personalization; 'shared' = FedAvg the whole prior.")
    p.add_argument("--fedprox-mu", type=float, default=0.01,
                   help="FedProx μ: local objective gains (μ/2)‖w − w^t‖² over the shared encoder "
                        "parameters. μ=0 reduces exactly to FedAvg. NOT comparable to a FedProx-paper μ "
                        "(the local optimizer here is a persistent AdamW, not SGD) — sweep it.")
    p.add_argument("--fedprox-form", type=str, default="loss", choices=["loss", "decoupled"],
                   help="'loss' = the FedProx paper's objective (the prox gradient goes through "
                        "AdamW's preconditioner, so small μ can be a NO-OP — watch prox_grad_ratio). "
                        "'decoupled' = AdamW-style post-step contraction w += lr·μ·(w^t − w). NB the "
                        "lr: the fraction of drift removed per step is lr·μ, NOT μ (an earlier "
                        "version of this help said μ, wrong by 3 orders of magnitude at our lr). "
                        "At lr=1e-3, μ=0.1 over 5 steps removes 5e-4 of the drift, not 41%%; μ=O(10) "
                        "is what gives the anchor authority over a round. `prox_pull_frac` in the "
                        "round log is the measured 1−(1−lr·μ)^steps — read it, do not assume μ.")
    p.add_argument("--fedproto-weight", type=float, default=1.0,
                   help="FedProto λ: weight of the per-code prototype pull mean_k‖μ_k(batch) − p̄_k‖². "
                        "λ=0 leaves the encoder unfederated (no weight averaging either) — that is the "
                        "arm's own null, NOT `local` (the codebook is still federated).")
    p.add_argument("--fedproto-agg", type=str, default="uniform", choices=["uniform", "count"],
                   help="prototype aggregation. 'uniform' (default) = one vote per client, a target "
                        "that DIFFERS from the merged codebook. 'count' = paper-faithful weighting, "
                        "which here equals the Prop.1 codebook — the degeneracy ablation.")
    p.add_argument("--fedproto-code-weight", type=str, default="uniform",
                   choices=["uniform", "count"],
                   help="how the per-code errors are weighted IN THE LOSS (independent of "
                        "--fedproto-agg, which sets the target). 'uniform' = one weight per code "
                        "(pure between-class, but amplifies rare codes — and rare codes are the "
                        "anomaly signal); 'count' = token-frequency weighting, commitment's own.")
    p.add_argument("--fedproto-no-seed", action="store_true",
                   help="do NOT seed round-0 prototypes from the broadcast codebook. Round 0 then "
                        "has no prototype term at all, so the encoder is federated in rounds-1 of "
                        "the rounds its siblings get. Ablation only.")
    p.add_argument("--fedproto-fedavg", action="store_true",
                   help="hybrid ablation: prototype pull AND weight averaging (FedProto is "
                        "prototype-ONLY by definition; this measures whether they compose).")
    # ── Paper-faithful window: TimeVQVAE-AD Algorithm 1 uses T = 2 x period ──────
    # arXiv 2311.12550v5 defines the rolling window from the SERIES' period, not as a
    # fixed hyperparameter ("Define a period length P of x*", then "x in R^T, T = 2P").
    # config.py fixes window_length=128 for every dataset (the period override was
    # removed for wsd_fed, where 2P=2880 would have dropped 31 clients to 18). On UCR
    # that removes the paper's protocol: measured over the archive, the median series
    # sees 128/period = 0.77 CYCLES where the paper prescribes 2.0, and on ZERO of the
    # 250 series is 128 the right value.
    p.add_argument("--window-length", type=int, default=None,
                   help="override cfg.dataset.window_length (None = config default 128). "
                        "Pass 2*period to reproduce the paper's T=2P. NB this also rescales "
                        "the detection stride (eval_stride_rate=0.1 of the window, which IS "
                        "the paper's rule) and, unless --metrics-tolerance is given, the "
                        "VUS/PATE/top-k buffer (window//2) -- pin that to keep tables "
                        "comparable across window settings.")
    p.add_argument("--window-normalization", choices=["none", "zscore"], default=None,
                   help="override cfg.dataset.window_normalization. 'zscore' = per-window "
                        "z-score at window extraction, which is what the TimeVQVAE-AD paper "
                        "prescribes (S5.2 and A.1: 'each input window is z-normalized', in the "
                        "same sentence as the T=2P rule) and what upstream does in BOTH training "
                        "(preprocessing/preprocess.py:149) and detection "
                        "(evaluation/__init__.py:119). Measured on ucr_001/ucr_043 arm "
                        "centralized: AUPRC 0.646->0.946 and 0.515->0.868. It also makes the "
                        "per-entity scaler irrelevant (a per-window z-score is invariant to any "
                        "preceding affine map), which removes the shard-dependence artefact. "
                        "NB: NOT honoured by scripts/floor_eval.py, and it attenuates 2 of the 3 "
                        "toy anomaly families (amp_burst to 21%%, level_shift to 49%%) -- it is a "
                        "UCR/paper-fidelity knob, not a global default.")
    p.add_argument("--metrics-tolerance", type=int, default=None,
                   help="override cfg.evaluation.paper_metrics_tolerance, the VUS/PATE buffer "
                        "AND the top-k hit radius. Applied AFTER apply_dataset_overrides, so it "
                        "beats both metadata.json and the window-derived default. REQUIRED "
                        "alongside --window-length for a clean A/B: otherwise the window moves "
                        "the metric's own yardstick and the two effects cannot be separated.")
    p.add_argument("--seeds", type=str, default="0", help="comma-separated seeds; pooled for mean±std.")
    p.add_argument("--out-json", type=str, default=None, help="write the summary dict to this path.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="root for per-client checkpoints + detect reports/plots "
                        "(default: artifacts/fed_eval/<dataset>).")
    args = p.parse_args()
    # One knob, every stage-2 val site: the federated shared-body loop reads it via the
    # explicit s2_val_mode param; the per-client converged loop (baselines, cb_only-family
    # priors, A1) reads this module global inside _build_train_stage2_converged.
    _S2_VAL_MODE["mode"] = args.fed_s2_val
    if args.fed_s2_val != "legacy":
        print(f"[config] stage-2 val oracle: {args.fed_s2_val} (frozen masks + fp32 on the "
              "federated path; frozen masks on the per-client converged path)")

    cfg = Config()
    cfg.dataset.name = args.dataset
    if args.batch:                  # unset ⇒ keep the config's 256/128 (the run.py recipe)
        cfg.dataset.batch_size_stage1 = args.batch
        cfg.dataset.batch_size_stage2 = args.batch
    # data.make_dataloaders honours DEBUG_NUM_WORKERS on its own, but `_pooled_loader`
    # and `_restage2_loader` build loaders straight from `cfg.dataset.num_workers`, so
    # without this the env var would silently cover only part of the run. Folding it
    # into cfg here makes ONE knob govern every loader this process creates.
    _nw = os.environ.get("DEBUG_NUM_WORKERS")
    if _nw:
        cfg.dataset.num_workers = int(_nw)
        print(f"[config] num_workers={cfg.dataset.num_workers} (env DEBUG_NUM_WORKERS)")
    # We build Config() directly (dataset.name comes from argv), so the metadata.json
    # overrides load_config() would have applied must be applied explicitly. This used
    # to be a hand-copy of the tolerance rule only, which ignored `metrics_tolerance`.
    apply_dataset_overrides(cfg)
    apply_env_overrides(cfg)   # AMP / SCORE_MASK_CHUNK / DETECT_AMP were silently dropped here
    # Point 1 — override VQ capacity (K, D) for the shared-codebook capacity sweep.
    # token_embedding_dim wires encoder/quantizer/decoder together; codebook_size wires
    # the quantizer + the stage-2 prior vocab. The suff-stat merge is K-agnostic.
    if args.codebook_size:
        cfg.quantizer.codebook_size = int(args.codebook_size)
    if args.token_dim:
        cfg.quantizer.token_embedding_dim = int(args.token_dim)
    # B1 — architecture knobs. embed_dim is d_model; the FFN is 4*hidden_dim
    # (model/prior.py), so --prior-embed-dim drags hidden_dim to keep the shipped 4x ratio;
    # pass --prior-hidden-dim after it to decouple.
    if args.width_base is not None:
        cfg.encoder.width_base = int(args.width_base)
    if args.prior_embed_dim is not None:
        cfg.prior.embed_dim = int(args.prior_embed_dim)
        cfg.prior.hidden_dim = int(args.prior_embed_dim)
    if args.prior_hidden_dim is not None:
        cfg.prior.hidden_dim = int(args.prior_hidden_dim)
    if args.prior_dropout is not None:
        cfg.prior.dropout = float(args.prior_dropout)
    # nn.MultiheadAttention asserts this deep into training; fail here, naming the knob.
    if cfg.prior.embed_dim % cfg.prior.heads:
        raise SystemExit(f"--prior-embed-dim {cfg.prior.embed_dim} is not divisible by "
                         f"cfg.prior.heads={cfg.prior.heads}")
    # High-frequency-precision levers (orthogonal; each no-op unless passed).
    if args.spec_weight is not None:
        cfg.decoder.spec_weight = float(args.spec_weight)
    if args.downsampled_width is not None:
        cfg.encoder.downsampled_width = int(args.downsampled_width)
    if args.n_fft is not None:
        cfg.transform.n_fft = int(args.n_fft)
    if args.refine_mode is not None:
        cfg.decoder.refine_mode = str(args.refine_mode)
    # Point 2 — Residual-VQ selection.
    if args.quantizer:
        cfg.quantizer.name = args.quantizer
    if args.n_stages:
        cfg.quantizer.n_residual_stages = int(args.n_stages)
    # Paper-faithful window (T = 2*period). Set BEFORE the tolerance below, because the
    # window-derived tolerance default is window_length//2 and we want an explicit
    # --metrics-tolerance to win over it.
    if args.window_length is not None:
        w = int(args.window_length)
        if w < 8:
            raise SystemExit(f"--window-length {w} is too small (STFT + 4 downsampling "
                             f"stages need room); use >= 8.")
        cfg.dataset.window_length = w
        print(f"[config] window_length -> {w} "
              f"(detection stride becomes {max(1, round(cfg.dataset.eval_stride_rate * w))})")
    if args.window_normalization is not None:
        prev = cfg.dataset.window_normalization
        cfg.dataset.window_normalization = args.window_normalization
        print(f"[config] window_normalization: {prev!r} -> {args.window_normalization!r}"
              + (" (paper: 'each input window is z-normalized', S5.2/A.1)"
                 if args.window_normalization == "zscore" else ""))
        if args.window_normalization == "zscore":
            # Loud, because two things silently keep living in the OLD space and would make
            # any table that mixes them a cross-representation comparison:
            print("[config] NB with zscore: scripts/floor_eval.py does NOT apply it (the "
                  "deep-vs-floor comparison becomes cross-space), and the public probe used "
                  "by federated_align/token-JS is normalised separately -- check both before "
                  "reporting anything that uses them.")
    # AFTER apply_dataset_overrides (called above), so an explicit tolerance beats both
    # metadata.json's `metrics_tolerance` and the window//2 fallback.
    if args.metrics_tolerance is not None:
        cfg.evaluation.paper_metrics_tolerance = int(args.metrics_tolerance)
        print(f"[config] paper_metrics_tolerance PINNED to {args.metrics_tolerance} "
              f"(VUS/PATE buffer + top-k radius); window//2 would have been "
              f"{cfg.dataset.window_length // 2}")
    elif args.window_length is not None:
        print(f"[config] WARNING: --window-length without --metrics-tolerance. The buffer "
              f"follows the window ({cfg.dataset.window_length // 2}), so this run is NOT "
              f"comparable to a different-window run on the same metric.")
    entities = resolve_clients(cfg, args.clients, args.cluster)
    # Train on `entities`, SCORE only `eval_entities`. Decouples the two because a
    # pooled body can be worth training on hundreds of series while only a handful
    # have a comparable `local` counterpart to be matched against -- scoring the rest
    # costs a full detection pass each and answers nothing. Default: score everything.
    eval_entities = entities
    if args.eval_clients:
        want = [e.strip() for e in args.eval_clients.split(",") if e.strip()]
        unknown = [e for e in want if e not in entities]
        if unknown:
            raise SystemExit(f"--eval-clients not in the trained set: {unknown}")
        eval_entities = want
        print(f"[eval-scope] training on {len(entities)} client(s), "
              f"scoring {len(eval_entities)}: {', '.join(eval_entities)}")
    if not entities:
        raise SystemExit(f"no clients resolved for {cfg.dataset.name}")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    # Validate the arm names BEFORE anything expensive. The dispatch chain raises
    # `unknown arm` only after the config echo, the loaders and (for multi-arm runs) any
    # earlier arm has already trained — so a typo in a 438-cluster sweep used to burn a
    # data-loading pass per job before failing. Fail here instead, with the list.
    _unknown = [a for a in arms if a not in KNOWN_ARMS]
    if _unknown:
        raise SystemExit(f"unknown arm(s) {_unknown}\nknown arms:\n  "
                         + "\n  ".join(sorted(KNOWN_ARMS)))
    # ── a knob that NO requested arm reads is an error, never a shrug ─────────────────
    # See FLAG_OWNER_ARMS. An ignored flag is not harmless here: launch.sh records it in
    # RUN.json's `extra_flags`, so the run's own provenance certifies a treatment that the
    # dispatch chain never applied, and the resulting row is unfalsifiable after the fact.
    # Hence SystemExit rather than a warning — a warning scrolls past in a 438-job sweep.
    #
    # "Explicitly set" = differs from the PARSER's default, read back with `p.get_default`.
    # No sentinel defaults, on purpose: they would change what `args.<knob>` holds for every
    # downstream read, and this file must not move a single number. The other candidate,
    # re-parsing an empty argv, needs the parser to have no required options and re-runs the
    # type= callbacks for nothing. The trade-off of the default-comparison is one FALSE
    # NEGATIVE — passing a flag's own default value explicitly is not flagged — and that is
    # the behaviour the runners need: every script here puts cohort-wide knobs such as
    # `--fed-enc-prior local` (the default) in a COMMON string handed to EVERY arm, `local`
    # and `centralized` included, and none of those runs is affected by it.
    _ignored = [(dest, owners) for dest, owners in FLAG_OWNER_ARMS.items()
                if getattr(args, dest) != p.get_default(dest)
                and not any(a in owners for a in arms)]
    if _ignored:
        raise SystemExit(
            "flag(s) set that NONE of the requested arms read — this run would have ignored "
            "them in silence while RUN.json recorded them as applied:\n"
            + "\n".join(f"  --{d.replace('_', '-')} {getattr(args, d)!r}  is honoured only by: "
                        + ", ".join(o) for d, o in _ignored)
            + f"\nrequested arms: {', '.join(arms)}\n"
            "Drop the flag, or run it against an arm that honours it.")
    # A HALF-converged table is worse than a fixed-budget one. Stage 1 honours
    # `--protocol converged` for EVERY federated arm (select_on_val + patience); what can
    # still differ is STAGE 2. An arm whose prior is fully local takes the per-client
    # converged prior path. An arm with a SHARED prior body trains that body in rounds --
    # and those rounds early-stop too, via `federated_stage2(patience_rounds=...)`, fed from
    # `--fed-patience-rounds` through _RESUME_CTX (see train_federated below).
    #
    # ⚠ CORRETTO 2026-08-01. This block used to warn UNCONDITIONALLY that a shared prior
    # body "trains on a FIXED round budget [and] cannot early-stop without a redesign", so
    # any 'federated < local' read off the table was a budget artifact. The redesign exists
    # and the text was never updated. Measured on ucr001_v1/ucr011_v1: all 12 shared-prior
    # stage-2 histories end at `best + 6` -- the patience-6 signature -- against a cap of
    # --s2-rounds 300, one of them stopping at r10. The logs print
    # "[fed-s2] convergence stop ARMED: patience=6" for every one. The warning was calling
    # valid results invalid, which is the expensive direction to be wrong in.
    #
    # So: warn only when stage-2 patience is genuinely OFF, which is the case the original
    # text described.
    if args.protocol == "converged" and not args.fed_patience_rounds:
        # Converged END-TO-END: stage 1 by select_on_val + patience, stage 2 by the same
        # per-client converged loop the baselines use (reached iff the prior is fully local).
        _CONVERGED_FED = {"federated_cb_only", "federated_cb_only_ema",
                          "federated_cb_only_ema_norevive",
                          # The matched twin of cb_only: same fully-local prior, only the
                          # codebook primitive differs, so it takes the same converged path.
                          "federated_fedavg_cb_only"}
        # The encoder-federation trio inherits cb_only's converged treatment ONLY when its
        # prior is fully local (--fed-enc-prior local): that is the configuration in which
        # `train_federated` takes the per-client converged prior path. With a partial/shared
        # prior the shared body trains in rounds -- which is fine WITH patience, and is why
        # this whole block is now gated on --fed-patience-rounds being 0.
        if args.fed_enc_prior == "local":
            _CONVERGED_FED |= {"federated_enc_fedavg", "federated_enc_fedprox",
                               "federated_enc_fedproto", "federated_enc_commoninit",
                               "federated_enc_commoninit_cbshared",
                               "federated_enc_commoninit_cblocal",
                               "federated_enc_fedavg_cblocal", "federated_enc_fedprox_cblocal"}
        # centraltok_fedprior trains a shared prior body in rounds exactly like the
        # shared-body federated arms — same budget trap, same warning (audit 2026-08-08).
        _fed_arms = [a for a in arms
                     if (a.startswith("federated") or a == "centraltok_fedprior")
                     and a not in _CONVERGED_FED]
        if _fed_arms:
            print("\n" + "!" * 78)
            print("!! WARNING: these arms are converged at STAGE 1 but NOT at stage 2.")
            print(f"!! Shared prior body, and --fed-patience-rounds is 0, so that body runs "
                  f"the FULL --s2-rounds={args.s2_rounds}: {', '.join(_fed_arms)}")
            print("!! Baselines trained to convergence vs a prior capped by the budget is "
                  "NOT a fair")
            print("!! comparison; any 'federated < local' read off this table would be a "
                  "budget artifact.")
            print("!! FIX: pass --fed-patience-rounds N (stage 2 early-stops too). Otherwise "
                  "run the")
            print("!! baselines and the federated arms in SEPARATE tables, or use "
                  "--protocol fixed.")
            print("!" * 78 + "\n", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scratch = (resolve_path(args.out_dir) if args.out_dir
               else resolve_path("artifacts/fed_eval") / args.dataset)
    # Separate roots per cluster: the same dataset hosts several independent
    # federations, and they must not overwrite each other's checkpoints.
    if args.cluster:
        scratch = scratch / args.cluster
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"[fed_eval] dataset={cfg.dataset.name} "
          f"{'cluster=' + args.cluster + ' ' if args.cluster else ''}"
          f"clients={entities}")

    data_by_e = _loaders_for(cfg, entities)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    pooled = {arm: [] for arm in arms}            # per-client reports pooled across seeds

    for seed in seeds:
        for arm in arms:
            print(f"\n===== TRAIN arm={arm} seed={seed} =====")
            seed_everything(seed)                  # comparable init across arms within a seed
            fed_echo = None                        # set by arms whose knobs are NOT in cfg
            fed_hist: dict = {}                    # per-round federated telemetry sink
            # A resume bundle belongs to ONE (arm, seed); loading it into a different arm or
            # seed would silently mix optimizer states across conditions.
            if args.resume_from and (len(arms) > 1 or len(seeds) > 1):
                raise SystemExit("--resume-from takes a single (arm, seed): the checkpoints and "
                                 "optimizer state it restores belong to one specific run. "
                                 f"Got --arms {','.join(arms)} --seeds {args.seeds}.")
            # The resume bundle must land NEXT TO the checkpoints it complements, so the
            # destination is resolved before training rather than after it. `_arm_tag` needs
            # `fed_echo`, which only the trio sets, so the pre-training value is the plain arm
            # name; the post-training `sdir` below is the authoritative one and they agree for
            # every arm that can be resumed (the trio's tag is added by the same call).
            # Written to a provisional directory because the authoritative `sdir` depends on
            # `_arm_tag(arm, fed_echo)`, and fed_echo is only known after the arm branch runs;
            # the file is moved next to its checkpoints as soon as sdir exists (below).
            #
            # That provisional directory MUST be private to this (arm, process). It used to be
            # one shared `seed<N>/_resume_tmp` for everybody, and launch.sh dispatches one
            # process per (dataset, cluster, arm) into the SAME --out-dir, two arms of a cluster
            # starting ~3 s apart: whoever reached the `prov.replace(sdir/…)` below first walked
            # off with the OTHER arm's bundle, and the load-time guards (federated.py) compare
            # only `entities` and `merge`, which are identical across arms, so the swap passed in
            # silence. It happened on disk: artifacts/_archive_20260729/ucrsplit/ckpt/ucr_200/
            # seed0/local/_fed_resume.pt holds {"merge": "suffstat", "rounds_done": 300} while
            # the `local` arm never calls train_federated at all. `_arm_tag(arm, None)` is the
            # plain arm name (fed_echo, hence the μ/λ bits, exists only after the branch runs),
            # so the PID separates two sweep points of the SAME arm too, and also stops a
            # crashed run's stale bundle from being adopted by the next process — the orphan
            # directory keeps the arm in its name, which is more forensics than `_resume_tmp`
            # ever gave. `sdir` below stays the authoritative destination.
            _RESUME_CTX["from"] = args.resume_from
            _RESUME_CTX["patience"] = args.fed_patience_rounds
            _RESUME_CTX["rounds_done"] = args.resume_rounds_done
            _RESUME_CTX["hist"] = fed_hist
            _RESUME_CTX["out"] = (scratch / f"seed{seed}"
                                  / f"_resume_tmp_{_arm_tag(arm, None)}_{os.getpid()}")
            # Codebook regime REQUESTED for the encoder trio: the flag, unless one of the
            # commoninit_cb{shared,local} / *_cblocal arms pins it below. It is the dispatch
            # value only — NOT a record of what ran. Every other arm hardcodes its own
            # `cb_merge` in its train_federated call (or takes the "suffstat" default) and never
            # reads this variable, so echoing it into fed_history.json used to publish
            # cb_mode="local" for a `federated_cb_only` run that had merged by suff-stats. The
            # truth is written by train_federated itself, which is the function that performs
            # the merge; this stays only as the fallback for a caller that bypasses it.
            cb_regime = args.fed_enc_cb
            if arm == "local":
                models = train_local(cfg, entities, data_by_e, args.s1_epochs, args.s2_epochs, device,
                                     protocol=args.protocol)
            elif arm == "centralized":
                models = train_centralized(cfg, entities, data_by_e, args.s1_epochs, args.s2_epochs, device,
                                           protocol=args.protocol)
            elif arm == "fedtok_centralprior":
                # ⛔ DIAGNOSTIC — see train_fedtok_centralprior's docstring. Needs the source
                # federated run's ckpt root; refuses to guess.
                if not args.fedtokcp_stage1_root:
                    raise SystemExit("fedtok_centralprior needs --fedtokcp-stage1-root "
                                     "(e.g. artifacts/runs/zn_a2/ckpt)")
                if not args.cluster:
                    raise SystemExit("fedtok_centralprior is per-cluster: pass --cluster")
                models = train_fedtok_centralprior(cfg, entities, data_by_e, device,
                                                   args.fedtokcp_stage1_root, args.cluster,
                                                   seed)
            elif arm == "centraltok_fedprior":
                # ⛔ DIAGNOSTIC (inverse of fedtok_centralprior) — see the docstring. The
                # stage-1 root must point at a tag holding the CENTRALIZED source run.
                if not args.fedtokcp_stage1_root:
                    raise SystemExit("centraltok_fedprior needs --fedtokcp-stage1-root "
                                     "(e.g. artifacts/runs/zn_main/ckpt — the CENTRALIZED source)")
                if not args.cluster:
                    raise SystemExit("centraltok_fedprior is per-cluster: pass --cluster")
                models = train_centraltok_fedprior(
                    cfg, entities, data_by_e, device, args.fedtokcp_stage1_root,
                    args.cluster, seed, s2_rounds=args.s2_rounds,
                    local_epochs=args.local_epochs,
                    select_on_val=(args.protocol == "converged"),
                    # same inheritance rule as the trio: the dedicated EVAL-counted knob
                    # wins, else the shared round-patience feeds stage 2 as it always has.
                    patience_rounds=(args.fed_s2_patience or args.fed_patience_rounds),
                    val_mode=args.fed_s2_val, val_every=args.fed_s2_val_every,
                    out_history=fed_hist)
            elif arm == "centralized_cap":
                # DIVERSITY vs QUANTITY probe: centralized on a small DIVERSE subsample
                # (--pool-cap windows drawn across ALL clients) at the SAME data budget as
                # one local client. > local ⇒ the centralized win is regularization/diversity,
                # not raw volume (confirms "local overfits its narrow normal").
                models = train_centralized(cfg, entities, data_by_e, args.s1_epochs, args.s2_epochs,
                                           device, pool_cap=args.pool_cap, protocol=args.protocol)
            elif arm == "federated":
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed, protocol=args.protocol)
            elif arm == "federated_shared":
                # (B)-ablation: FedAvg the WHOLE prior (no local head), with the SAME
                # Prop.1 codebook federation as 'federated' held fixed. Measures ONLY
                # the value of partial personalization. NOT a "naive FL" baseline — it
                # keeps contribution (A), the sufficient-statistic codebook merge.
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed, local_prefixes=(), protocol=args.protocol)
            elif arm == "federated_fedavg_cb":
                # (A)-ablation: naive weight-FedAvg of the codebook (vs the Prop.1
                # sufficient-statistic merge), with the partially-personalized prior
                # held at the main-method setting. Isolates the value of the merge.
                #
                # ⚠ NOT the clean (A) contrast — the prior is federated here too, and prior
                # weight-federation collapses on wsd (0.011-0.041, RESEARCH_LEDGER), so the
                # merge effect is read through a much larger effect that has nothing to do
                # with it. Use `federated_fedavg_cb_only` for the matched contrast.
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed, cb_merge="fedavg",
                                         protocol=args.protocol)
            elif arm == "federated_fedavg_cb_only":
                # THE CLEAN (A) ABLATION — the matched twin of `federated_cb_only`.
                #
                # Everything is byte-identical to `federated_cb_only` (encoder local, decoder
                # local, ENTIRE prior local via local_prefixes=("",), same rounds, same seeds,
                # same converged protocol) EXCEPT the codebook aggregation primitive:
                #
                #   federated_cb_only        e_j = Σm_j / smoothed(Σn_j)   <- Prop. 1, exact
                #                            pooled k-means M-step; codebook FROZEN locally.
                #   federated_fedavg_cb_only e_j = Σ (n_k/n) · e_j^k       <- naive weight
                #                            average; codebook moves locally via EMA.
                #
                # This cell did not exist: every route to a weight-averaged codebook also
                # federated the prior, so contribution (A) had never been measured against a
                # matched control. It is the cell that makes the (codebook primitive × prior
                # sharing) factorial complete, and it is the row that answers "why not just
                # average the codebooks?" on the axis the claim is made.
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed,
                                         cb_merge="fedavg", local_prefixes=("",),
                                         protocol=args.protocol)
            elif arm == "federated_anchor":
                # ABLATION (c): the full recipe (A+B) PLUS the FedProto encoder
                # anchor (lambda>0), which adds an extra pull of local encoders
                # toward the frozen broadcast codebook. Tests whether the anchor
                # raises cross-client index coherence (tok-agree) and at what cost
                # to detection. lambda set by --anchor-weight (0 reduces to plain
                # 'federated'). Codebook merge + partial personalization unchanged.
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed,
                                         anchor_weight=args.anchor_weight, protocol=args.protocol)
            elif arm in ("federated_fedavg_cb_sharedprior", "federated_fedavg_whole"):
                # Drops BOTH contributions: weight-FedAvg the codebook (no (A) suff-stat
                # merge) AND FedAvg the whole prior (no (B) partial personalization). The
                # gap to 'federated' is the value of the recipe (A+B) together.
                #
                # ⚠ RENAMED 2026-07-29. The old name was `federated_fedavg_whole` and the
                # comment here suspicion it was the "federate everything the obvious way"
                # baseline. It is NOT, and the call below proves it: `fed_encoder` is left
                # at its default "off", so the ENCODER and the DECODER are never federated —
                # 100% of the stage-1 network stays local and only the dictionary and the
                # prior cross the network. "whole" referred to the whole PRIOR, not the
                # whole model, which is exactly the misreading the name invited.
                #
                # The real "federate everything" strawman does not exist in this repo:
                # `_encoder_shared_keys` (pipeline/federated.py) filters on
                # `k.startswith("encoder.")`, so no arm can select the decoder. If that row
                # is wanted, the prefix tuple has to become a parameter first.
                #
                # The old name still dispatches so a sweep already in flight does not die,
                # but it is reported loudly and must not appear in a table.
                if arm == "federated_fedavg_whole":
                    print("\n" + "!" * 78)
                    print("!! DEPRECATED ARM NAME: 'federated_fedavg_whole' -> "
                          "'federated_fedavg_cb_sharedprior'.")
                    print("!! The old name reads as 'FedAvg the whole model'. It is not: the "
                          "encoder and the")
                    print("!! decoder are NEVER federated by this arm (fed_encoder stays "
                          "'off'). Only the")
                    print("!! codebook (naive weight-average) and the prior (fully shared) "
                          "are.")
                    print("!! Re-run under the new name before putting this row in a table.")
                    print("!" * 78 + "\n", flush=True)
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed,
                                         cb_merge="fedavg", local_prefixes=(), protocol=args.protocol)
            elif arm == "federated_cb_only":
                # ISOLATES contribution (A) ALONE: Prop.1 sufficient-statistic codebook
                # federation with the ENTIRE prior kept LOCAL — local_prefixes=("",) makes
                # str.startswith(("",)) True for every key => 0 shared prior keys. Contrast
                # vs 'local' = value of federating ONLY the codebook; contrast vs 'federated'
                # = value of ALSO FedAvg-ing the prior body. Disentangles the two surfaces.
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed, local_prefixes=("",),
                                         protocol=args.protocol)
            elif arm == "federated_cb_only_ema":
                # federated_cb_only PLUS a server-side codebook EMA: the server keeps a
                # running EMA of the AGGREGATE suff-stats across rounds (C←γC+(1−γ)N,
                # S←γS+(1−γ)M) and broadcasts e_j=S_j/smoothed(C_j) with C→ema_cluster_size,
                # S→ema_embed_sum. Encoder/decoder/prior stay fully local (local_prefixes=("",)),
                # exactly like cb_only. γ from --cb-server-ema-decay (default 0.8, a round-scale
                # rate; γ=0 collapses to federated_cb_only). Tests server-memory vs memoryless merge.
                gamma = (args.cb_server_ema_decay if args.cb_server_ema_decay is not None
                         else _CB_SERVER_EMA_DEFAULT_DECAY)
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed, local_prefixes=("",),
                                         cb_server_ema_decay=gamma, protocol=args.protocol)
            elif arm == "federated_cb_only_ema_norevive":
                # A-ABLATION of federated_cb_only_ema: server EMA ON but dead-code REVIVAL OFF.
                # Hypothesis: the EMA memory already preserves a transiently-unused code (its
                # accumulated S,C restore it next round via e_j=S/smoothed(C)), so noise-revival
                # is redundant and may even fight the memory. cb_only (revive on, no memory) vs
                # cb_only_ema (revive on, memory) vs this (no revive, memory) isolates whether
                # the memory REPLACES revival. Same γ default as the parent arm.
                gamma = (args.cb_server_ema_decay if args.cb_server_ema_decay is not None
                         else _CB_SERVER_EMA_DEFAULT_DECAY)
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed, local_prefixes=("",),
                                         cb_server_ema_decay=gamma, revive_dead=False,
                                         protocol=args.protocol)
            elif arm == "federated_enc":
                # PROBE: federate the ENCODER too (whole stack) — align the tokenizer so the
                # shared codebook is coherent. Decoder stays local, codebook = suffstat. Tests
                # whether representation misalignment is what makes FL lose to local (and whether
                # Prop.1 resurrects once its comparable-assignment precondition holds).
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed, fed_encoder="full", protocol=args.protocol)
            elif arm == "federated_enc_partial":
                # PROBE variant: share the LATE encoder blocks + ProjectBlock (codebook-facing
                # map), keep the first `--enc-split-at` blocks (raw-signal front-end) LOCAL — align
                # the representation while personalizing the input adapter. Sweep --enc-split-at
                # for the alignment-neck curve (Point 3).
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed,
                                         fed_encoder="partial", enc_split_at=args.enc_split_at, protocol=args.protocol)
            elif arm == "federated_enc_neck":
                # Point 3 (purest neck): share ONLY the codebook-facing projection (A_G),
                # keep ALL encoder blocks local (H_k). Minimal shared alignment map.
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed,
                                         fed_encoder="neck", protocol=args.protocol)
            elif arm in ("federated_enc_fedavg", "federated_enc_fedprox",
                         "federated_enc_fedproto", "federated_enc_commoninit",
                         "federated_enc_commoninit_cbshared",
                         "federated_enc_commoninit_cblocal",
                         # ADDITIONS, not replacements: the cb-local twins of the two
                         # weight-space algorithms. `federated_enc_fedproto_cblocal` is
                         # deliberately ABSENT — FedProto aggregates one prototype per
                         # codebook index, which only means something while index k denotes
                         # the same thing on every client. That is a property of the method,
                         # not a gap in this repo, and the paper should say so.
                         "federated_enc_fedavg_cblocal",
                         "federated_enc_fedprox_cblocal"):
                # THE THREE-WAY ENCODER-FEDERATION
                # Same everything (suff-stat codebook merge, whole encoder shared, decoder
                # local, stage-2 prior per --fed-enc-prior, same rounds/seeds/budget); the
                # ONLY difference is HOW the encoder is federated:
                #
                #   federated_enc_fedavg   weight-space: server averages the encoder
                #   federated_enc_fedprox  weight-space + local prox anchor (--fedprox-mu)
                #   federated_enc_fedproto function-space: NO weight averaging, per-code
                #                          prototype pull (--fedproto-weight/-agg)
                #   federated_enc_commoninit_cbshared  common encoder init + the codebook
                #                          STILL FEDERATED (Prop.1 merge, broadcast every
                #                          round); the encoder is never synchronised again.
                #   federated_enc_commoninit_cblocal   common encoder init and NOTHING else
                #                          crosses the network: per-client k-means codebook,
                #                          exactly the `local` baseline's quantizer.
                #
                # ⚠ `federated_enc_commoninit` (no suffix) is the LEGACY name and resolves to
                # _cbshared. It was described in this file and in FED_ENCODER_ALGOS.md as
                # "the null: common init, then every client trains alone" — which is WRONG,
                # and the weights prove it: on ucr_162 the 5 clients' codebooks are
                # BIT-IDENTICAL (max|Δ| = 0.0) while their encoders diverge (max|Δ| = 0.17).
                # The codebook merge at federated.py:1458 is NOT gated on enc_fed_algo, so it
                # runs for every arm in this branch. The un-suffixed arm therefore bundles TWO
                # treatments — shared init AND a federated dictionary — and cannot isolate
                # either. Kept working because a sweep may be mid-flight under the old name;
                # use the explicit names for anything new.
                #
                # The one-axis null the old name promised is _cblocal: `local` plus a shared
                # starting point, nothing else. Only against THAT can "FedAvg helps" be told
                # apart from "a shared starting point helps".
                #
                # Default prior = FULLY LOCAL (--fed-enc-prior local), so stage 1 is the only
                # thing federated and the contrast against `local`/`federated_cb_only` is a
                # clean attribution to the encoder-federation algorithm. `partial`/`shared`
                # reproduce the main-method / naive-FL prior pairings for the same three arms.
                # `commoninit` IS fedproto with λ=0 and no averaging: the encoder is
                # broadcast once and never synchronised again.
                # Strip the codebook-regime suffix BEFORE reading the algorithm off the name,
                # or `federated_enc_fedavg_cblocal`.rsplit("_")[1] resolves to "cblocal" and
                # train_federated raises on an unknown algo.
                base_arm = arm
                for _sfx in ("_cbshared", "_cblocal"):
                    if base_arm.endswith(_sfx):
                        base_arm = base_arm[: -len(_sfx)]
                is_null = "commoninit" in base_arm
                algo = "fedproto" if is_null else base_arm.rsplit("_", 1)[1]
                lam = 0.0 if is_null else args.fedproto_weight
                # The null variants pin the codebook regime themselves; every other arm takes
                # it from --fed-enc-cb. Pinning is the point: the whole reason these two names
                # exist is that the regime must be legible from the arm id alone, not inferred
                # from a flag that defaults to "suffstat" three files away.
                cb_regime = args.fed_enc_cb
                if arm.endswith("_cbshared"):
                    cb_regime = "suffstat"
                elif arm.endswith("_cblocal"):
                    cb_regime = "local"
                    # federated.py refuses a local codebook for fedproto only when
                    # enc_proto_weight > 0; the null runs at lambda=0, so nothing to aggregate
                    # by code id and the guard correctly does not fire.
                # --fedproto-fedavg would otherwise turn the NULL control into plain FedAvg
                # while the banner still says "never synchronised" — a silently corrupted
                # control row is worse than no control row.
                hybrid = bool(args.fedproto_fedavg) and not is_null
                if cb_regime == "local" and args.fed_enc_prior != "local":
                    raise SystemExit(
                        f"--fed-enc-cb local with --fed-enc-prior {args.fed_enc_prior} is not "
                        "meaningful. With per-client dictionaries the token ids are "
                        "client-specific, and the prior's `token_embedding` is an "
                        "nn.Embedding indexed BY CODE ID (model/prior.py): FedAvg-ing it "
                        "averages embeddings of unrelated symbols — the same error the "
                        "fedproto guard refuses, one level up. Use --fed-enc-prior local.")
                prefixes = {"local": ("",), "partial": LOCAL_PRIOR_PREFIXES,
                            "shared": ()}[args.fed_enc_prior]
                # internship-2 knob guards (2026-08-07)
                # With a fully-LOCAL prior the shared-key set is EMPTY: every stage-2 server
                # knob would be a federation no-op dressed as a treatment (and tau>0 also
                # evicts the run from the per-client converged path). Refuse loudly.
                _s2_knobs_set = (args.tau_steps > 0 or args.server_opt != "sgd"
                                 or args.s2_server_momentum > 0
                                 or args.fed_s2_val_every > 1 or args.fed_s2_patience > 0
                                 or args.fed_prior_prox_mu > 0)
                if args.fed_enc_prior == "local" and _s2_knobs_set:
                    raise SystemExit(
                        "stage-2 server knobs (--tau-steps/--server-opt/--s2-server-momentum/"
                        "--fed-s2-val-every/--fed-s2-patience/--fed-prior-prox-mu) with "
                        "--fed-enc-prior local: the prior is fully per-client, there is NO "
                        "shared body to aggregate (nor an anchor to pull toward) — the knobs "
                        "would be silently meaningless. Use --fed-enc-prior partial|shared.")
                if args.server_opt == "fedadam" and args.s2_server_momentum > 0:
                    raise SystemExit(
                        "--server-opt fedadam and --s2-server-momentum are mutually exclusive "
                        "(if/elif in federated_stage2: fedadam wins silently and the FedAvgM "
                        "banner lies). Pick one.")
                # Self-documenting run: these knobs live in argv, NOT in cfg, so without this
                # echo a knobs.json could not distinguish μ=0.01 from μ=1 after the fact.
                # Stage-2 keys emitted ONLY when non-default so every existing dir/knobs.json
                # is byte-identical.
                fed_echo = {"algo": algo, "scope": args.fed_enc_scope, "bn": args.fed_enc_bn,
                            "prior": args.fed_enc_prior, "cb": cb_regime,
                            "sched": args.fed_enc_sched,
                            "enc_split_at": args.enc_split_at,
                            **({"mu": args.fedprox_mu, "form": args.fedprox_form}
                               if algo == "fedprox" else {}),
                            **({"lambda": lam, "agg": args.fedproto_agg,
                                "code_weight": args.fedproto_code_weight,
                                "seed_round0": not args.fedproto_no_seed,
                                "hybrid_fedavg": hybrid}
                               if algo == "fedproto" else {}),
                            **({"tau": args.tau_steps} if args.tau_steps > 0 else {}),
                            **({"srv": args.server_opt, "slr": args.server_lr,
                                "sb1": args.server_b1, "sb2": args.server_b2,
                                "seps": args.server_eps}
                               if args.server_opt != "sgd" else {}),
                            **({"sm": args.s2_server_momentum,
                                "sm_skip0": bool(args.s2_momentum_skip_round0)}
                               if args.s2_server_momentum > 0 else {}),
                            **({"s2val": args.fed_s2_val} if args.fed_s2_val != "legacy" else {}),
                            **({"s2val_every": args.fed_s2_val_every}
                               if args.fed_s2_val_every > 1 else {}),
                            **({"s2pat": args.fed_s2_patience} if args.fed_s2_patience > 0 else {}),
                            **({"s2pen": True} if args.fed_s2_agg_penalty else {}),
                            **({"s2snap": args.fed_s2_snapshot_every}
                               if args.fed_s2_snapshot_every > 0 else {}),
                            **({"pmu": args.fed_prior_prox_mu,
                                "pform": args.fed_prior_prox_form}
                               if args.fed_prior_prox_mu > 0 else {}),
                            **({"K": int(args.codebook_size)}
                               if args.codebook_size else {})}
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed,
                                         local_prefixes=prefixes, cb_merge=cb_regime,
                                         fed_encoder=args.fed_enc_scope, enc_split_at=args.enc_split_at,
                                         enc_fed_algo=algo, enc_bn=args.fed_enc_bn,
                                         enc_prox_mu=(args.fedprox_mu if algo == "fedprox" else 0.0),
                                         enc_prox_form=args.fedprox_form,
                                         enc_proto_weight=(lam if algo == "fedproto" else 0.0),
                                         enc_proto_agg=args.fedproto_agg,
                                         enc_proto_code_weight=args.fedproto_code_weight,
                                         enc_proto_seed_round0=(not args.fedproto_no_seed),
                                         enc_proto_fedavg=hybrid, enc_sched=args.fed_enc_sched,
                                         # stage-2 server knobs: dedicated flags with NULL
                                         # defaults (NEVER args.server_momentum here — its
                                         # argparse default is 0.9, owned by federated_fedavgm).
                                         tau_steps=args.tau_steps, server_opt=args.server_opt,
                                         server_lr=args.server_lr, server_b1=args.server_b1,
                                         server_b2=args.server_b2, server_eps=args.server_eps,
                                         server_momentum=args.s2_server_momentum,
                                         server_momentum_skip_round0=bool(args.s2_momentum_skip_round0),
                                         s2_val_mode=args.fed_s2_val,
                                         s2_val_every=args.fed_s2_val_every,
                                         s2_patience=args.fed_s2_patience,
                                         s2_agg_penalty=bool(args.fed_s2_agg_penalty),
                                         s2_snapshot_every=args.fed_s2_snapshot_every,
                                         prior_prox_mu=args.fed_prior_prox_mu,
                                         prior_prox_form=args.fed_prior_prox_form,
                                         protocol=args.protocol, out_history=fed_hist)
            elif arm == "federated_fedavgm":
                # ①: FedAvgM (server momentum on the prior pseudo-gradient) + small tau (run with
                # --local-epochs 1 and more --s2-rounds). Rides the proven federated->centralized
                # path; output_bias/channel_embedding stay LOCAL (fixes the calibration smear).
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed,
                                         server_momentum=args.server_momentum, protocol=args.protocol)
            elif arm == "federated_protoprior":
                # FedProto AT THE PRIOR LEVEL: frozen suff-stat tokenizer; per-client prior
                # trained on OWN data + λ·KL toward the ensemble-consensus predictive
                # distribution on a shared probe (a function-space prototype). No weight
                # averaging. Deploy = personalized per-client prior. Null (λ=0) = cb_only.
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed,
                                         proto_weight=args.proto_weight,
                                         proto_probe_windows=args.proto_probe_windows, protocol=args.protocol)
            elif arm == "federated_fedsgd":
                # #2 (R1-Really): frozen suff-stat tokenizer; federate the FULLY-SHARED prior
                # (local_prefixes=()) at τ optimizer STEPS/round instead of epochs, with an
                # adaptive server optimizer (FedAdam). τ=1 ≈ centralized SGD; the sweep over
                # τ maps the knee where step-scale FedSGD still beats `local` on VUS-PR. The
                # one regime where weight-averaging the prior is NOT monotone-harmful.
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed, local_prefixes=(),
                                         tau_steps=args.tau_steps, server_opt=args.server_opt,
                                         server_lr=args.server_lr, protocol=args.protocol)
            elif arm == "federated_fedsgd_pooltok":
                # Cell A of the tokenizer×prior 2×2: POOLED (centralized) tokenizer frozen,
                # fully-shared prior federated at τ steps/round. Splits the wsd gap
                # (centralized vs fedsgd) into a prior term and a tokenizer term. τ=1 ⇒
                # prior ≈ centralized-SGD on an IDENTICAL pooled tokenizer. Tokenizer is
                # cached (τ-independent) so the sweep trains it once per (seed, s1-epochs).
                models = train_fedsgd_pooltok(cfg, entities, data_by_e, args.s1_epochs,
                                              args.s2_rounds, args.local_epochs, device, seed=seed,
                                              tau_steps=args.tau_steps, server_opt=args.server_opt,
                                              server_lr=args.server_lr,
                                              tok_cache_dir=(resolve_path("artifacts/fed_eval/_pooltok_cache")
                                                             / args.dataset / (args.cluster or "all")))
            elif arm == "federated_pooltok_centralprior":
                # Cell C: SAME cached pooled tokenizer as pooltok, frozen, POOLED prior at the
                # SAME update budget (--s2-rounds steps) and effective batch (K×--batch) as the
                # τ=1 FedSGD prior. C − pooltok = pure prior-federation penalty (identical
                # tokenizer AND budget). Removes the batch/budget confound vs `centralized`.
                models = train_pooltok_centralprior(cfg, entities, data_by_e, args.s1_epochs,
                                                    args.s2_rounds, device, seed=seed,
                                                    batch=(args.batch or cfg.dataset.batch_size_stage2),
                                                    tok_cache_dir=(resolve_path("artifacts/fed_eval/_pooltok_cache")
                                                                   / args.dataset / (args.cluster or "all")))
            elif arm == "federated_fedsgd_fedenc":
                # DEPLOYABLE tokenizer federation #1: FedAvg the WHOLE encoder (fed_encoder="full")
                # so the tokenizer is genuinely federated (no data pooling), paired with the same
                # fully-shared τ FedSGD prior as pooltok. If it reaches ≈ pooltok/local it recovers
                # the tokenizer gain WITHOUT pooling — the thesis-supporting method.
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed, local_prefixes=(),
                                         tau_steps=args.tau_steps, server_opt=args.server_opt,
                                         server_lr=args.server_lr, fed_encoder="full", protocol=args.protocol)
            elif arm == "federated_fedsgd_align":
                # DEPLOYABLE tokenizer federation #2: encoders stay LOCAL, aligned functionally on
                # a shared public probe (permutation-free, fed_align="probe_mse"), paired with the
                # same fully-shared τ FedSGD prior. The alignment alternative to weight-FedAvg.
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed, local_prefixes=(),
                                         tau_steps=args.tau_steps, server_opt=args.server_opt,
                                         server_lr=args.server_lr, fed_align="probe_mse",
                                         align_weight=args.align_weight, align_cluster=args.cluster, protocol=args.protocol)
            elif arm == "federated_align":
                # B1: encoders stay LOCAL; align them functionally on a SHARED public probe
                # (broadcast consensus encoding, per-round MSE pull) instead of averaging weights.
                # Permutation-free tokenizer alignment; codebook = suffstat.
                models = train_federated(cfg, entities, args.s1_rounds, args.s2_rounds,
                                         args.local_epochs, device, seed=seed,
                                         fed_align="probe_mse", align_weight=args.align_weight,
                                         align_cluster=args.cluster, protocol=args.protocol)
            else:
                raise ValueError(f"unknown arm {arm!r}")

            # Save checkpoints, FREE the GPU, then evaluate (memory-safe on shared GPUs).
            # The directory carries the KNOBS, not just the arm: a μ or λ sweep runs the same
            # arm name many times, and an un-tagged path silently overwrote every sweep point
            # with the last one (checkpoints AND knobs.json).
            sdir = scratch / f"seed{seed}" / _arm_tag(arm, fed_echo)
            prov = Path(_RESUME_CTX["out"]) / "_fed_resume.pt" if _RESUME_CTX["out"] else None
            if prov is not None and prov.exists():
                sdir.mkdir(parents=True, exist_ok=True)
                prov.replace(sdir / "_fed_resume.pt")     # same filesystem: atomic rename
                prov.parent.rmdir() if not any(prov.parent.iterdir()) else None
            # Checkpoint only what will be scored: `centralized` hands back the SAME
            # model object for every client, so writing one copy per trained entity
            # would be N identical files (248 of them on the pooled UCR arm).
            ckpts = {e: save_client_ckpts(cfg, e, models[e][0], models[e][1], sdir)
                     for e in eval_entities}
            # B1 safety: measure the built model (params/conv widths), NOT cfg -- the only way to
            # prove a --width-base / --prior-* knob actually threaded. See _structural_echo.
            _first = entities[0]
            echo = _structural_echo(cfg, models[_first][0], models[_first][1], fed=fed_echo,
                                    models=models)
            print(f"[echo] {arm} seed{seed} realized: {echo['realized']}")
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / "knobs.json").write_text(json.dumps(echo, indent=2), encoding="utf-8")
            if fed_hist:
                # perplexity/dead_frac/cb_drift are single-dictionary occupancy statistics under
                # 'suffstat' and CROSS-dictionary spread statistics under 'local'. Same JSON keys,
                # different meaning: record the mode so a table cannot silently mix them.
                # `setdefault`, because train_federated already wrote the regime it MERGED WITH;
                # overwriting it with the CLI's intent is exactly the bug that made a suff-stat
                # cb_only run declare cb_mode="local". This line now only covers a future arm
                # that fills fed_hist without going through train_federated.
                fed_hist.setdefault("cb_mode", cb_regime)
                # Per-round federated telemetry. Without this file the round-level evidence
                # (drift, prox_grad_ratio, proto_agg_gap, perplexity, dead_frac) exists only
                # in the stdout log, which is not what the aggregation scripts read.
                # `allow_nan=False` + a None-sanitiser: NaN is the normal value for a
                # telemetry channel that does not apply to this arm, and bare `NaN` tokens
                # are not valid JSON for a strict reader.
                def _clean(o):
                    if isinstance(o, float):
                        return o if np.isfinite(o) else None
                    if isinstance(o, dict):
                        return {k: _clean(v) for k, v in o.items()}
                    if isinstance(o, list):
                        return [_clean(v) for v in o]
                    return o
                (sdir / "fed_history.json").write_text(
                    json.dumps(_clean(fed_hist), indent=2, allow_nan=False), encoding="utf-8")
            del models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"===== EVAL arm={arm} seed={seed} =====")
            for e in eval_entities:
                rep = eval_from_ckpts(cfg, e, ckpts[e][0], ckpts[e][1], sdir)               # detection
                rep.update(cf_fidelity_from_ckpts(cfg, e, ckpts[e][0], ckpts[e][1], device))  # CF vs x_clean
                rep["_seed"], rep["_entity"] = seed, e
                rep["_knobs"] = echo
                pooled[arm].append(rep)
                print(f"[{arm} s{seed}] {e}: vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                      f"cf={rep.get('cf_repair_improvement', float('nan')):.3f}")

    summary = {arm: _aggregate(pooled[arm], METRIC_KEYS + THRESHOLDED_KEYS + CF_KEYS) for arm in arms}

    # Raw per-(arm, seed, client) metric records — the substrate for a CORRECT
    # two-level aggregation (per-seed macro-mean, THEN across-seed dispersion) and
    # for paired significance tests. The in-process `summary`/_aggregate pools all
    # (client, seed) reports into one flat list, so its std confounds client and
    # seed variance and its "worst" is a single min-of-min point — fine as a quick
    # console glance, WRONG for a camera-ready error bar. scripts/fed_aggregate.py
    # consumes these records instead.
    records = []
    for arm in arms:
        for rep in pooled[arm]:
            rec = {"_arm": arm, "_seed": rep.get("_seed"), "_entity": rep.get("_entity")}
            for k in METRIC_KEYS + THRESHOLDED_KEYS + CF_KEYS:
                v = rep.get(k)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    rec[k] = float(v)
            records.append(rec)

    # PERSIST BEFORE PRINTING. The tables below are pure presentation; the run's
    # value is in `records`. Writing first means no formatting/encoding failure can
    # destroy hours of training that already completed.
    if args.out_json:
        meta = {"dataset": args.dataset, "cluster": args.cluster,
                "clients": entities, "seeds": seeds,
                "s1_epochs": args.s1_epochs, "s2_epochs": args.s2_epochs,
                "protocol": args.protocol,   # 'converged' => the epoch counts were IGNORED
                                             # for local/centralized (val-driven stop instead)
                "s1_rounds": args.s1_rounds, "s2_rounds": args.s2_rounds,
                "local_epochs": args.local_epochs,
                # effective, not requested: --batch unset means "whatever config says"
                "batch_stage1": cfg.dataset.batch_size_stage1,
                "batch_stage2": cfg.dataset.batch_size_stage2,
                # Was `"bf16": bool`. Now records the dtype actually used, since `auto`
                # resolves to fp16 below Ampere and bf16 from Ampere on.
                "amp_dtype": (str(_amp_dtype()).replace("torch.", "")
                              if _amp_dtype() is not None else "fp32"),
                **_git_provenance()}
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(
            {"meta": meta, "summary": summary, "records": records}, indent=2), encoding="utf-8")
        print(f"\n[out] wrote summary + {len(records)} raw records -> {args.out_json}")

    # ── comparison tables ─────────────────────────────────────────────────────
    def _print_table(title: str, keys: list[str], note: str) -> None:
        print(f"\n================ {title} ================")
        kk = [k for k in keys if any(k in summary[a] for a in arms)]
        w = max(13, max(len(a) for a in arms) + 1)
        print("arm".ljust(w) + "".join(f"{k:>22}" for k in kk))
        for arm in arms:
            row = arm.ljust(w)
            for k in kk:
                s = summary[arm].get(k)
                row += (f"{s['mean']:.3f}±{s['std']:.3f}(min{s['worst']:.3f})".rjust(22)) if s else " " * 22
            print(row)
        print(note)

    _print_table("RQ1: detection, THRESHOLD-FREE (macro over clients)", METRIC_KEYS,
                 f"(higher = better; min = worst-client; VUS/PATE buffer = "
                 f"{cfg.evaluation.paper_metrics_tolerance})")
    _print_table("RQ1b: threshold-dependent (appendix)", THRESHOLDED_KEYS,
                 f"(read off the q={cfg.threshold.q} quantile of TRAIN scores — "
                 f"not the comparison surface)")
    _print_table("RQ4: counterfactual fidelity vs x_clean", CF_KEYS,
                 "(cf_repair_improvement ↑ better; cf_repair_ratio ↓, cf_disturb_mae ↓; cf_n_windows = #anomalous windows)"
                 "\n(all-NaN on real datasets: no ground-truth clean signal exists)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
