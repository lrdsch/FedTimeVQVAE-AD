"""STEP 1 unit test: federated codebook = exact pooled k-means M-step.

Verifies the load-bearing claim (Proposition 1) and the blocker fix:

  (1) `collect_stats_only` does NOT mutate the codebook (the in-place EMA
      overwrite / k-means / dead-code paths are disabled in federated mode).
  (2) Aggregating clients' sufficient statistics reproduces the EXACT pooled
      centroid: for every code j with N_j>0,  (Σ_k m_j^k)/(Σ_k n_j^k)  equals the
      mean of all latent vectors assigned to j across all clients (computed
      independently by a second code path).
  (3) `merge_round_stats` returns N=Σn, M=Σm and finite codebook.
  (4) The broadcast→collect→merge→set_codebook round loop is stable (no NaN).

Run:  python scripts/fed_codebook_unittest.py
Exit code 0 and "ALL PASS" => the federated codebook primitive is correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.vector_quantizer import SharedVectorQuantizer
from utils import force_utf8_stdout

force_utf8_stdout()     # the PASS banners print 'Σ' — see utils.force_utf8_stdout

ATOL = 1e-4


def _client_latent(B: int, D: int, L: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, D, L, generator=g) * 1.5            # (B, D, L)


def _assign(x: torch.Tensor, codebook: torch.Tensor):
    """Independent assignment path: flatten (B,D,L)->(N,D), nearest code."""
    D = codebook.shape[1]
    flat = x.permute(0, 2, 1).reshape(-1, D)                  # (N, D)
    idx = torch.cdist(flat, codebook).argmin(dim=-1)          # (N,)
    return flat, idx


def main() -> int:
    torch.manual_seed(0)
    K, D, L = 16, 8, 50

    vq = SharedVectorQuantizer(token_embedding_dim=D, codebook_size=K,
                               threshold_ema_dead_code=0)
    vq.train()
    vq.collect_stats_only = True

    fixed = torch.randn(K, D)
    vq.set_codebook(fixed)                                    # broadcast a known codebook
    cb_before = vq.codebook.weight.detach().clone()

    x1 = _client_latent(4, D, L, seed=11)
    x2 = _client_latent(3, D, L, seed=22)

    # Client 1
    vq.reset_round_stats(); vq(x1); n1, m1 = vq.pull_round_stats()
    # Client 2 (same frozen broadcast codebook)
    vq.reset_round_stats(); vq(x2); n2, m2 = vq.pull_round_stats()

    # (1) codebook must be untouched by collect_stats_only forwards
    assert torch.allclose(vq.codebook.weight, cb_before), \
        "FAIL (1): codebook mutated under collect_stats_only"
    print("PASS (1): collect_stats_only leaves the codebook frozen")

    # (2) Proposition 1: merged centroid == independently-pooled centroid
    f1, i1 = _assign(x1, fixed)
    f2, i2 = _assign(x2, fixed)
    flat_all = torch.cat([f1, f2], dim=0)
    idx_all = torch.cat([i1, i2], dim=0)
    N, M = n1 + n2, m1 + m2

    checked = 0
    for j in range(K):
        mask = idx_all == j
        cnt = int(mask.sum())
        assert abs(cnt - int(N[j])) == 0, \
            f"FAIL (2): count mismatch code {j}: indep={cnt} vq={int(N[j])}"
        if cnt > 0:
            pooled = flat_all[mask].mean(dim=0)
            merged = M[j] / N[j]
            assert torch.allclose(pooled, merged, atol=ATOL), \
                f"FAIL (2): centroid mismatch code {j} (max|Δ|={(pooled-merged).abs().max():.2e})"
            checked += 1
    print(f"PASS (2): Σm/Σn == pooled centroid for all {checked} live codes (Prop. 1)")

    # (3) server merge helper
    weight, N2, M2 = SharedVectorQuantizer.merge_round_stats([n1, n2], [m1, m2])
    assert torch.allclose(N2, N) and torch.allclose(M2, M), "FAIL (3): merge sums wrong"
    assert torch.isfinite(weight).all(), "FAIL (3): non-finite codebook from merge"
    live = N > 5
    assert torch.allclose(weight[live], M[live] / N[live].unsqueeze(1), atol=1e-3), \
        "FAIL (3): smoothed weight deviates from M/N on well-supported codes"
    print("PASS (3): merge_round_stats sums correct; smoothing negligible on live codes")

    # (4) round loop stability
    cb = fixed.clone()
    for r in range(3):
        vq.set_codebook(cb)
        vq.reset_round_stats(); vq(x1); na, ma = vq.pull_round_stats()
        vq.reset_round_stats(); vq(x2); nb, mb = vq.pull_round_stats()
        cb, _, _ = SharedVectorQuantizer.merge_round_stats([na, nb], [ma, mb])
        assert torch.isfinite(cb).all(), f"FAIL (4): non-finite codebook at round {r}"
    print("PASS (4): broadcast->collect->merge->set_codebook loop stable over 3 rounds")

    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
