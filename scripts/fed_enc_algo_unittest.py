#!/usr/bin/env python3
"""Unit tests for the STAGE-1 ENCODER federation trio (FedAvg / FedProx / FedProto).

CPU-only, no data, no GPU, seconds to run:

    python3 scripts/fed_enc_algo_unittest.py

What it pins down — in order of how badly a silent break would hurt:

 1. NON-REGRESSION. With `export_proto_tokens` False (every pre-existing arm) the VQ
    forward is bit-identical and stashes nothing. The trio must be free when unused.
 2. FEDPROTO DEGENERACY, the trap `anchor_weight` already fell into (see the CAVEAT in
    model/vector_quantizer.py): count-weighted prototypes ARE the Prop.1 codebook, and
    uniform-weighted ones are NOT. Both directions are asserted, because the arm's whole
    claim to be a distinct mechanism rests on it.
 3. FEDPROTO FORM. The prototype term is the BETWEEN-class part of the commitment loss:
    inflating within-class spread at fixed class means leaves it EXACTLY unchanged while
    the commitment loss grows. That is what makes it not-a-commitment-weight-sweep.
 4. FEDPROX. ‖w − w^t‖² is exact, zero at snapshot time, and grows as the client drifts.
 5. KEY SELECTION. The three BN regimes are genuinely different: 'buffers_local' shares the
    affine gamma/beta (so it is NOT FedBN), 'fedbn' keeps the whole BN layer local (that IS
    FedBN, Li et al. ICLR 2021), 'shared' federates the statistics too. `num_batches_tracked`
    is excluded in every mode (int64 counter — an averaged float truncates on copy-back).
"""
from __future__ import annotations

import sys
from pathlib import Path

import math

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from model.vector_quantizer import SharedVectorQuantizer
from pipeline.federated import (
    _aggregate_prototypes, _encoder_drift, _encoder_shared_keys, _flat_all, _pool_encoder_bn,
    _prox_grad_ratio, _prox_term, _proto_term, _shared_param_names, _snapshot_prox_ref,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


def approx(a, b, tol=1e-5) -> bool:
    return bool(torch.allclose(torch.as_tensor(a).float(), torch.as_tensor(b).float(),
                               rtol=tol, atol=tol))


# ── 1. non-regression: the export hook is inert by default ────────────────────

def test_export_hook_inert():
    print("\n[1] export_proto_tokens=False leaves the VQ forward untouched")
    torch.manual_seed(0)
    vq = SharedVectorQuantizer(token_embedding_dim=4, codebook_size=8)
    vq.initialized.fill_(True)
    vq.collect_stats_only = True
    vq.train()
    # requires_grad on the input, so "the stash carries gradient" is a real assertion
    # below rather than a property of the fixture.
    x = torch.randn(3, 4, 5, requires_grad=True)   # (B, D, L) — the VQ flattens to (B, L, D)

    out_off = vq(x)
    check("nothing stashed when the flag is off",
          vq._last_tokens is None and vq._last_indices is None)

    vq.reset_round_stats()
    vq.export_proto_tokens = True
    torch.manual_seed(0)
    out_on = vq(x)
    check("forward output identical with the flag on",
          approx(out_off.loss, out_on.loss) and torch.equal(out_off.indices, out_on.indices),
          f"loss {float(out_off.loss):.6f} vs {float(out_on.loss):.6f}")
    # `requires_grad` is not enough — a detached-but-leaf tensor would satisfy it. The
    # stash must be an autograd NODE, or the whole FedProto arm silently stops reaching
    # the encoder while every shape assertion still passes.
    check("tokens stashed carry gradient and align with indices",
          vq._last_tokens is not None
          and vq._last_tokens.grad_fn is not None
          and vq._last_tokens.shape[:2] == vq._last_indices.shape
          and vq._last_tokens.shape[-1] == 4,
          f"grad_fn={type(vq._last_tokens.grad_fn).__name__ if vq._last_tokens is not None and vq._last_tokens.grad_fn else None}")

    vq.eval()
    vq._last_tokens = vq._last_indices = None
    vq(x)
    check("eval() stashes nothing even with the flag on", vq._last_tokens is None)

    check("the flag never enters state_dict",
          not any("export" in k or "_last" in k for k in vq.state_dict()))


# ── 2. prototype aggregation: the degeneracy analysis, both directions ────────

def test_prototype_aggregation():
    print("\n[2] prototype aggregation vs the Prop.1 codebook")
    torch.manual_seed(1)
    K, D, J = 6, 3, 4
    # Deliberately UNEQUAL client counts — the heterogeneity that separates the two
    # aggregations. With equal counts they coincide and the test would be vacuous.
    counts = [torch.tensor([10., 0., 4., 7., 1., 0.]),
              torch.tensor([1., 5., 0., 7., 0., 0.]),
              torch.tensor([30., 2., 9., 7., 0., 0.]),
              torch.tensor([0., 0., 1., 7., 0., 0.])]
    sums = [torch.randn(K, D) * c.unsqueeze(1) for c in counts]

    p_count, mask_c, gap_c = _aggregate_prototypes(counts, sums, "count")
    N = torch.stack(counts).sum(0)
    M = torch.stack(sums).sum(0)
    pooled = M / N.clamp_min(1.0).unsqueeze(1)
    live = N > 0
    check("count-weighted prototype == pooled mean (= the merged codebook)",
          approx(p_count[live], pooled[live]))

    # The Laplace-smoothed codebook the server actually broadcasts differs only by the
    # smoothing term — assert they are CLOSE, which is what 'degenerate target' means.
    e = SharedVectorQuantizer.codebook_from_stats(N, M, eps=1e-5)
    rel = ((p_count[live] - e[live]).norm() / e[live].norm().clamp_min(1e-12))
    check("count-weighted prototype ~= the Laplace-smoothed codebook", float(rel) < 1e-2,
          f"relative gap {float(rel):.2e}")

    p_uni, mask_u, gap_u = _aggregate_prototypes(counts, sums, "uniform")
    check("uniform-weighted prototype DIFFERS from the codebook (non-degenerate target)",
          not approx(p_uni[live], p_count[live], tol=1e-3),
          f"‖Δ‖={float((p_uni[live] - p_count[live]).norm()):.4f}")
    # The degeneracy witness must be reported identically whichever target is selected —
    # it is a property of the cohort, not of the choice, and a near-zero gap is the
    # pre-registered kill signal for the whole arm.
    check("proto_agg_gap is aggregation-independent and matches the manual ratio",
          approx(gap_c, gap_u, tol=1e-6)
          and approx(gap_u, float((p_uni[live] - p_count[live]).norm() / p_count[live].norm())),
          f"gap={gap_u:.4f}")

    # Code 3 is used by every client with the SAME count 7 → the two aggregations must
    # agree there. This proves the difference above is the weighting, not a bug.
    check("they agree on a code with equal counts across clients",
          approx(p_uni[3], p_count[3]))

    check("mask marks exactly the codes some client used",
          torch.equal(mask_c, live) and torch.equal(mask_u, live),
          f"live={live.tolist()}")

    # Uniform must ignore clients that did not use the code (not average in a zero).
    manual = torch.stack([s[2] / c[2] for c, s in zip(counts, sums) if c[2] > 0]).mean(0)
    check("uniform averages only over the clients that used the code", approx(p_uni[2], manual))

    try:
        _aggregate_prototypes(counts, sums, "bogus")
        check("unknown aggregation raises", False)
    except ValueError:
        check("unknown aggregation raises", True)


# ── 3. the prototype LOSS is the between-class term, not a commitment rescale ──

class _FakeClient:
    """Minimal ClientState stand-in: `_proto_term` only touches these five fields."""

    def __init__(self, vqs, protos, masks, weight=1.0, code_weight="uniform"):
        self._vqs, self.protos, self.proto_mask, self.proto_weight = vqs, protos, masks, weight
        self.proto_code_weight = code_weight

    @property
    def vqs(self):
        return self._vqs


def test_proto_loss_form():
    print("\n[3] the prototype term is BETWEEN-class only")
    torch.manual_seed(2)
    K, D = 4, 3
    vq = SharedVectorQuantizer(token_embedding_dim=D, codebook_size=K)
    proto = torch.randn(K, D)
    mask = torch.tensor([True, True, True, False])

    # Two token sets with IDENTICAL per-code means but different within-code spread.
    # The means are deliberately OFFSET from the prototypes (off) so the term is far from
    # zero: with means sitting exactly on the prototypes, "spread does not change it" would
    # hold trivially for any implementation that returns 0, including a broken one.
    off = torch.tensor([0.1, -0.2, 0.3])
    idx = torch.tensor([[0, 0, 1, 1, 2, 2]])
    base = torch.stack([proto[0] + off[0] + 0.3, proto[0] + off[0] - 0.3,
                        proto[1] + off[1] + 0.1, proto[1] + off[1] - 0.1,
                        proto[2] + off[2] + 0.5, proto[2] + off[2] - 0.5]).unsqueeze(0)
    spread = torch.stack([proto[0] + off[0] + 3.0, proto[0] + off[0] - 3.0,
                          proto[1] + off[1] + 2.0, proto[1] + off[1] - 2.0,
                          proto[2] + off[2] + 4.0, proto[2] + off[2] - 4.0]).unsqueeze(0)
    expect_base = float((off ** 2).mean())        # mean over codes of ‖offset‖²/D

    def term(tokens):
        vq._last_tokens, vq._last_indices = tokens.clone().requires_grad_(True), idx
        c = _FakeClient([vq], [proto], [mask])
        return _proto_term(c)

    t_base, t_spread = term(base).detach(), term(spread).detach()
    check("the term equals the exact between-class value (non-zero, so the test can fail)",
          approx(t_base, expect_base, tol=1e-5) and float(t_base) > 1e-3,
          f"{float(t_base):.6f} vs {expect_base:.6f}")
    check("within-class spread does NOT change the prototype term",
          approx(t_base, t_spread, tol=1e-6),
          f"{float(t_base):.6f} vs {float(t_spread):.6f}")

    # ...whereas the commitment loss (per-sample) explodes with the same spread. This is
    # the exact contrast that `anchor_weight` fails (it IS the commitment loss rescaled).
    com_base = ((base[0] - proto[idx[0]]) ** 2).mean()
    com_spread = ((spread[0] - proto[idx[0]]) ** 2).mean()
    check("the commitment loss DOES change with the same spread",
          float(com_spread) > 10 * float(com_base),
          f"{float(com_base):.4f} -> {float(com_spread):.4f}")

    # Shifting class 0's mean by +1 changes ONLY that class's contribution, from
    # off[0]² to (off[0]+1)², averaged over the 3 contributing classes.
    shifted = base.clone(); shifted[0, 0:2] += 1.0
    t_shift = term(shifted)
    expected = expect_base + ((off[0] + 1.0) ** 2 - off[0] ** 2).item() / 3.0
    check("shifting one class mean moves the term by the exact between-class amount",
          approx(t_shift, expected, tol=1e-5), f"{float(t_shift):.6f} vs {expected:.6f}")

    # Codes without a prototype are excluded, not pulled toward zero.
    idx_dead = torch.tensor([[3, 3, 0, 0]])
    tok_dead = torch.stack([torch.full((D,), 9.0), torch.full((D,), 9.0),
                            proto[0], proto[0]]).unsqueeze(0)
    vq._last_tokens, vq._last_indices = tok_dead.requires_grad_(True), idx_dead
    t_dead = _proto_term(_FakeClient([vq], [proto], [mask]))
    check("masked codes contribute nothing", approx(t_dead, 0.0, tol=1e-6),
          f"{float(t_dead):.3e}")

    # Gradient reaches the tokens (i.e. the whole encoder) ...
    tok = base.clone().requires_grad_(True)
    vq._last_tokens, vq._last_indices = tok, idx
    _proto_term(_FakeClient([vq], [proto], [mask])).backward()
    check("gradient flows back to the encoder tokens",
          tok.grad is not None and float(tok.grad.abs().sum()) > 0)
    # ... as the SHARED per-class direction 2(mu_k - p_k)/n_k, equal within a class.
    g = tok.grad[0]
    check("within a class every sample gets the same gradient (centroid pull, not residual)",
          approx(g[0], g[1], tol=1e-6) and approx(g[2], g[3], tol=1e-6))

    check("the stash is always released after use",
          vq._last_tokens is None and vq._last_indices is None)

    # No prototypes yet (round 0 with seeding disabled) -> no term, and no leaked graph.
    vq._last_tokens, vq._last_indices = base.clone(), idx
    check("round 0 (no prototypes) returns None",
          _proto_term(_FakeClient([vq], None, None)) is None
          and vq._last_tokens is None)

    # code_weight='count' re-weights by token frequency without changing the form.
    idx_skew = torch.tensor([[0, 0, 0, 0, 0, 1]])           # code 0 x5, code 1 x1
    tok_skew = torch.cat([proto[0].expand(5, D), (proto[1] + 1.0).unsqueeze(0)]).unsqueeze(0)
    vq._last_tokens, vq._last_indices = tok_skew.clone(), idx_skew
    t_uni = float(_proto_term(_FakeClient([vq], [proto], [mask])).detach())
    vq._last_tokens, vq._last_indices = tok_skew.clone(), idx_skew
    t_cnt = float(_proto_term(_FakeClient([vq], [proto], [mask], code_weight="count")).detach())
    # uniform: (0 + 1)/2 = 0.5 ; count: (5*0 + 1*1)/6 = 1/6
    check("code_weight uniform vs count re-weight the same per-code errors",
          approx(t_uni, 0.5, tol=1e-5) and approx(t_cnt, 1.0 / 6.0, tol=1e-5),
          f"uniform={t_uni:.4f} count={t_cnt:.4f}")

    # A Residual-VQ would silently contribute gradient-dead stages: assert the guard.
    vq2 = SharedVectorQuantizer(token_embedding_dim=D, codebook_size=K)
    for v in (vq, vq2):
        v._last_tokens, v._last_indices = base.clone(), idx
    try:
        _proto_term(_FakeClient([vq, vq2], [proto, proto], [mask, mask]))
        check("more than one contributing RVQ stage is rejected", False)
    except AssertionError:
        check("more than one contributing RVQ stage is rejected", True)


# 4FedProx proximal term

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(2, 3, 3), nn.BatchNorm2d(3))
        self.decoder = nn.Linear(4, 4)          # must never be touched by the encoder keys


def test_prox_term():
    print("\n[4] FedProx proximal term")
    torch.manual_seed(3)
    m = _Net()
    keys = _encoder_shared_keys(m, "full")
    names = _shared_param_names(m, keys)
    check("shared params are encoder-only", all(n.startswith("encoder.") for n in names)
          and len(names) > 0, f"{names}")

    ref = _snapshot_prox_ref(m, names)
    params = dict(m.named_parameters())
    pairs = [(params[k], r) for k, r in ref.items()]
    check("prox term is 0 at snapshot time", approx(_prox_term(pairs), 0.0, tol=1e-9))

    with torch.no_grad():
        m.encoder[0].weight += 0.1
    n_el = m.encoder[0].weight.numel()
    check("prox term equals the exact squared distance",
          approx(_prox_term(pairs), 0.01 * n_el),
          f"{float(_prox_term(pairs)):.6f} vs {0.01 * n_el:.6f}")

    with torch.no_grad():
        m.decoder.weight += 5.0
    check("the decoder does not enter the prox term", approx(_prox_term(pairs), 0.01 * n_el))

    g = torch.autograd.grad(_prox_term(pairs), m.encoder[0].weight)[0]
    check("prox gradient is 2(w - w^t)", approx(g, 0.2 * torch.ones_like(g)))

    try:
        _prox_term([])
        check("an empty pair list raises instead of returning None", False)
    except ValueError:
        check("an empty pair list raises instead of returning None", True)

    # prox_grad_ratio must recover ‖μ(w−w^t)‖/‖∇L_task‖ from the SUMMED gradient.
    mu = 0.5
    for p, r in pairs:
        p.grad = torch.zeros_like(p)
    w, r0 = pairs[0]
    fake_data_grad = torch.full_like(w, 0.3)
    w.grad = fake_data_grad + mu * (w.detach() - r0)      # what backward would leave behind
    ratio = _prox_grad_ratio(pairs, mu, "loss")
    expect = float((mu * (w.detach() - r0)).norm() / fake_data_grad.norm())
    check("prox_grad_ratio subtracts the analytic prox part ('loss' form)",
          approx(ratio, expect, tol=1e-4), f"{ratio:.4f} vs {expect:.4f}")
    # In the decoupled form the prox never enters the objective, so the stored gradient
    # already IS the data gradient and nothing may be subtracted from it.
    w.grad = fake_data_grad.clone()
    ratio_d = _prox_grad_ratio(pairs, mu, "decoupled")
    check("in the 'decoupled' form the stored gradient IS the data gradient",
          approx(ratio_d, expect, tol=1e-4), f"{ratio_d:.4f} vs {expect:.4f}")


# 5. shared-key selection + drift

def test_key_selection_and_drift():
    print("\n[5] shared-key selection and encoder drift")
    m = _Net()
    local_bn = _encoder_shared_keys(m, "full", bn="buffers_local")
    shared_bn = _encoder_shared_keys(m, "full", bn="shared")
    fedbn = _encoder_shared_keys(m, "full", bn="fedbn")
    check("default keeps running stats local",
          not any(k.endswith(("running_mean", "running_var")) for k in local_bn))
    check("bn='shared' adds the running stats",
          sum(k.endswith(("running_mean", "running_var")) for k in shared_bn) == 2)
    # The distinction the citation rests on: the default SHARES gamma/beta, FedBN does not.
    bn_affine = {"encoder.1.weight", "encoder.1.bias"}
    check("default SHARES the BN affine gamma/beta -> it is NOT FedBN",
          bn_affine <= set(local_bn), f"{sorted(bn_affine & set(local_bn))}")
    check("bn='fedbn' keeps the WHOLE BN layer local (affine AND stats)",
          not (bn_affine & set(fedbn))
          and not any(k.endswith(("running_mean", "running_var")) for k in fedbn)
          and len(fedbn) == len(local_bn) - 2,
          f"fedbn={len(fedbn)} vs default={len(local_bn)}")
    check("num_batches_tracked is excluded in EVERY mode",
          not any(k.endswith("num_batches_tracked") for k in local_bn + shared_bn + fedbn))
    check("no decoder key ever leaks in",
          not any(k.startswith("decoder") for k in shared_bn))

    class _C:
        def __init__(self, model):
            self.model = model

    names = _shared_param_names(m, local_bn)
    a, b = _Net(), _Net()
    b.load_state_dict(a.state_dict())
    d0 = _encoder_drift([_C(a), _C(b)], names)
    check("identical clients have zero drift", approx(d0["drift"], 0.0, tol=1e-9))
    # `prev` is the POST-aggregation snapshot of the previous round — captured by _flat_all,
    # not returned by _encoder_drift, so that enc_step is local movement only.
    prev = _flat_all([_C(a), _C(b)], names)
    with torch.no_grad():
        b.encoder[0].weight += 1.0
    n_el = m.encoder[0].weight.numel()
    d1 = _encoder_drift([_C(a), _C(b)], names, prev=prev)
    # Two EQUAL-sized clients: each sits ‖Δ‖/2 from the mean, so mean distance = ‖Δ‖/2.
    check("drift is the mean distance to the client mean",
          approx(d1["drift"], 0.5 * n_el ** 0.5, tol=1e-4), f"{d1['drift']:.4f}")
    # ...but the centroid must follow FedAvg's data-size weights, not a plain mean. THREE
    # clients are needed to see it: with two, the mean distance is 0.5‖Δ‖ for ANY weighting
    # (the two distances always sum to ‖Δ‖), which is exactly how an unweighted centroid
    # could hide here unnoticed.
    c3 = _Net(); c3.load_state_dict(a.state_dict())
    trio = [_C(a), _C(b), _C(c3)]                       # a, a+Δ, a
    d_un = _encoder_drift(trio, names)["drift"]
    d_w = _encoder_drift(trio, names, weights=[1.0, 3.0, 1.0])["drift"]
    check("the centroid is data-size weighted, like _fedavg_encoder",
          approx(d_un, (4 / 9) * n_el ** 0.5, tol=1e-4)
          and approx(d_w, (8 / 15) * n_el ** 0.5, tol=1e-4),
          f"unweighted={d_un:.4f} (exp {(4/9)*n_el**0.5:.4f}), "
          f"weighted={d_w:.4f} (exp {(8/15)*n_el**0.5:.4f})")
    # `enc_step` is per-client movement since last round — the cross-arm-comparable channel.
    check("enc_step is the mean per-client movement since the previous round's END",
          approx(d1["step"], 0.5 * n_el ** 0.5, tol=1e-4), f"{d1['step']:.4f}")
    check("drift_rel is scale-free", d1["drift_rel"] > 0 and math.isfinite(d1['drift_rel']))


def test_bn_pooling():
    print("\n[6] BN statistics are POOLED, not linearly averaged")
    a, b = _Net(), _Net()
    bn_keys = [k for k in _encoder_shared_keys(a, "full", bn="shared")
               if k.endswith(("running_mean", "running_var"))]
    with torch.no_grad():
        a.encoder[1].running_mean.fill_(0.0); a.encoder[1].running_var.fill_(1.0)
        b.encoder[1].running_mean.fill_(4.0); b.encoder[1].running_var.fill_(1.0)

    class _C:
        def __init__(self, model):
            self.model = model

    _pool_encoder_bn([_C(a), _C(b)], [torch.device("cpu")] * 2, [1.0, 1.0], bn_keys)
    # Law of total variance: within=1, between=(0-2)^2/2+(4-2)^2/2=4 -> pooled var = 5.
    # A linear average would have said 1, understating the spread 5x.
    check("pooled mean is the weighted mean", approx(a.encoder[1].running_mean, 2.0))
    check("pooled var = within + between (NOT the linear average)",
          approx(a.encoder[1].running_var, 5.0),
          f"{float(a.encoder[1].running_var[0]):.4f} (linear average would be 1.0)")
    check("both clients receive the same pooled statistics",
          approx(a.encoder[1].running_var, b.encoder[1].running_var))
    check("num_batches_tracked is untouched",
          int(a.encoder[1].num_batches_tracked) == 0)


# ── 7. merge='local': the codebook must be the `local` baseline's, per client ──────

def test_local_codebook_semantics():
    print("\n[7] merge='local' — per-client dictionaries, seeded from data")
    import copy as _copy
    from pipeline.federated import _build_client   # noqa: F401  (import-time check only)

    # The two switches that define the mode, asserted on the quantizer directly rather than
    # through a full training run (which needs GPU + data).
    fed_vq = SharedVectorQuantizer(token_embedding_dim=3, codebook_size=4)
    fed_vq.collect_stats_only = True
    fed_vq.initialized.fill_(True)                 # what a FEDERATED client gets
    loc_vq = SharedVectorQuantizer(token_embedding_dim=3, codebook_size=4)
    loc_vq.collect_stats_only = False
    loc_vq.initialized.fill_(False)                # what merge='local' gives (= the baseline)

    check("federated client: codebook frozen, k-means disabled",
          fed_vq.collect_stats_only and bool(fed_vq.initialized.item()))
    check("local client: codebook live, k-means ENABLED",
          not loc_vq.collect_stats_only and not bool(loc_vq.initialized.item()))

    # Train one step each on the SAME data and check the dictionaries behave differently.
    torch.manual_seed(11)
    x = torch.randn(8, 3, 16)                       # (B, D, L), plenty of vectors for k-means
    before_fed = fed_vq.codebook.weight.detach().clone()
    before_loc = loc_vq.codebook.weight.detach().clone()
    fed_vq.train(); loc_vq.train()
    fed_vq(x); loc_vq(x)
    check("federated: local training leaves the codebook UNTOUCHED",
          torch.equal(fed_vq.codebook.weight.detach(), before_fed))
    check("local: the first step k-means-seeds the codebook from the data",
          not torch.equal(loc_vq.codebook.weight.detach(), before_loc)
          and bool(loc_vq.initialized.item()))

    # Two 'clients' with DIFFERENT data must end up with DIFFERENT dictionaries — the
    # property that makes index-wise prototype aggregation meaningless (hence the refusal).
    a = SharedVectorQuantizer(token_embedding_dim=3, codebook_size=4)
    b = SharedVectorQuantizer(token_embedding_dim=3, codebook_size=4)
    for v in (a, b):
        v.collect_stats_only = False; v.initialized.fill_(False); v.train()
    torch.manual_seed(12); a(torch.randn(8, 3, 16))
    torch.manual_seed(13); b(torch.randn(8, 3, 16) + 5.0)
    gap = float((a.codebook.weight - b.codebook.weight).abs().max())
    check("two clients' local dictionaries DIVERGE (why fedproto refuses this mode)",
          gap > 1e-3, f"max |A-B| = {gap:.3f}")


# ── 8. THE PROOF: canonical FedProto degenerates to the commitment loss here ──────

def test_canonical_fedproto_is_the_commitment_loss():
    """The single most important check in this file.

    FedProto's OFFICIAL implementation (yuetan031/FedProto, `update_weights_het`) is

        proto_new[i] = global_protos[label_i]          # per SAMPLE
        loss2 = nn.MSELoss()(proto_new, protos)        # mean reduction
        loss  = loss1 + args.ld * loss2

    i.e. `mean_i ||z_i - pbar_{y_i}||^2`, and the server aggregation (Eq. 6) is
    COUNT-weighted. Transplant that literally into this codebase, where the "class" is the
    codebook index, and:

        pbar_k = sum_j m_j^k / sum_j n_j^k = e_k          (the Prop.1 merged codebook)
        => canonical term = mean_i ||z_i - e_{k(i)}||^2 = THE COMMITMENT LOSS, exactly.

    So canonical FedProto here is algebraically identical to raising `commitment_weight` by
    lambda — the very degeneracy model/vector_quantizer.py documents for `anchor_weight`.
    That is WHY this repo uses the between-class (per-class-mean) form with uniform
    aggregation, and it is the fact any write-up has to state.
    """
    print("\n[8] canonical FedProto (per-sample + count-agg) == the commitment loss")
    torch.manual_seed(21)
    K, D, J, N = 5, 4, 3, 40

    # Per-client round statistics, as the VQ accumulates them.
    z = torch.randn(N, D)
    idx = torch.randint(0, K, (N,))
    split = torch.randint(0, J, (N,))                    # which client each token came from
    counts, sums = [], []
    for j in range(J):
        m = split == j
        c = torch.zeros(K).index_add(0, idx[m], torch.ones(int(m.sum())))
        v = torch.zeros(K, D).index_add(0, idx[m], z[m])
        counts.append(c); sums.append(v)

    # Server: count-weighted prototypes (Eq. 6) and the Prop.1 merged codebook.
    p_count, live, _ = _aggregate_prototypes(counts, sums, "count")
    N_tot = torch.stack(counts).sum(0)
    M_tot = torch.stack(sums).sum(0)
    e = SharedVectorQuantizer.codebook_from_stats(N_tot, M_tot, eps=1e-5)

    # THE canonical FedProto term, written exactly as the official code writes it.
    proto_new = p_count[idx]                              # per-sample target
    canonical = torch.nn.functional.mse_loss(z, proto_new)
    # THE quantizer's own commitment loss, written exactly as vector_quantizer.py writes it.
    commitment = torch.nn.functional.mse_loss(z, e[idx].detach())

    rel = abs(float(canonical) - float(commitment)) / max(float(commitment), 1e-12)
    check("canonical FedProto term == the VQ commitment loss (rel. err < 1e-4)",
          rel < 1e-4, f"canonical={float(canonical):.6f} commitment={float(commitment):.6f} "
                      f"rel={rel:.2e} (differ only by Laplace smoothing)")

    # ...and OUR form is provably NOT that number, which is the whole point.
    vq = SharedVectorQuantizer(token_embedding_dim=D, codebook_size=K)
    p_uni, live_u, gap = _aggregate_prototypes(counts, sums, "uniform")
    vq._last_tokens, vq._last_indices = z.unsqueeze(0).clone(), idx.unsqueeze(0)
    ours = float(_proto_term(_FakeClient([vq], [p_uni], [live_u])).detach())
    check("our between-class form differs from the canonical/commitment value",
          abs(ours - float(commitment)) / float(commitment) > 0.05,
          f"ours={ours:.6f} vs commitment={float(commitment):.6f}, agg_gap={gap:.1%}")


def main() -> int:
    print("=== encoder-federation trio (FedAvg / FedProx / FedProto) unit tests ===")
    test_export_hook_inert()
    test_prototype_aggregation()
    test_proto_loss_form()
    test_prox_term()
    test_key_selection_and_drift()
    test_bn_pooling()
    test_local_codebook_semantics()
    test_canonical_fedproto_is_the_commitment_loss()
    print("\n" + ("ALL TESTS PASSED" if not FAILURES
                  else f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES)))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
