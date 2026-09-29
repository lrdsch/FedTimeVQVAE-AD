"""Unit test for the server-side codebook EMA (`federated_cb_only_ema` arm).

`federated_cb_only_ema` is `federated_cb_only` PLUS a server that keeps a running
EMA of the AGGREGATE per-code sufficient statistics across rounds:

    C_j^t = γ·C_j^(t-1) + (1-γ)·N_j ,   S_j^t = γ·S_j^(t-1) + (1-γ)·M_j
    e_j^(t+1) = S_j^t / smoothed(C_j^t)

where N_j = Σ_s n_j^s, M_j = Σ_s m_j^s are this round's stats summed over clients.
C, S start at zero; each round the server broadcasts the new codebook plus C (as
ema_cluster_size) and S (as ema_embed_sum). Encoder/decoder/prior stay fully local.

Verifies the three load-bearing claims:

  (1) EMA MEMORY: with γ>0 the accumulators carry state between two rounds — round
      t's broadcast depends on round t-1's stats, not only round t's.
  (2) γ=0 EQUIVALENCE: at γ=0 the server-EMA merge is bit-identical to the plain
      per-round `federated_cb_only` merge (C=N, S=M ⇒ e_j = M_j / smoothed(N_j)).
  (3) BROADCAST IDENTITY: after `set_codebook`, every client holds the identical
      codebook AND the identical (C, S) EMA state.
  (4) BIAS-CORRECTED DEAD-CODE TEST: the cold-start damping C≈(1−γ^{t+1})·N must not
      make a healthy code trip the fixed integer dead-code threshold; Ĉ=C/(1−γ^{t+1})
      flags only the truly-dead codes (== cb_only's N<threshold at round 0).

Run:  python scripts/fed_cb_server_ema_unittest.py
Exit code 0 and "ALL PASS" => the server codebook EMA is correct.
(pytest-discoverable too: the checks are `test_*` functions.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from model.vector_quantizer import SharedVectorQuantizer  # noqa: E402
from federated import _server_merge                        # noqa: E402
from utils import force_utf8_stdout                        # noqa: E402

EPS = 1e-5
THRESH = 2


def _client_stats(K: int, D: int, seed: int, n_clients: int = 2):
    """A list of per-client (count, sum) tensors — the message each client uploads.
    Counts are kept well above THRESH so dead-code revival never fires and the
    algebra under test is exercised cleanly."""
    g = torch.Generator().manual_seed(seed)
    counts, sums = [], []
    for _ in range(n_clients):
        counts.append(torch.randint(THRESH + 1, 40, (K,), generator=g).float())
        sums.append(torch.randn(K, D, generator=g) * 3.0)
    return counts, sums


# (1) EMA memory between two rounds

def test_ema_memory():
    K, D, gamma = 16, 8, 0.9
    ema = {"C": torch.zeros(K), "S": torch.zeros(K, D)}

    c0, s0 = _client_stats(K, D, seed=11)
    N0 = torch.stack(c0).sum(0); M0 = torch.stack(s0).sum(0)
    c1, s1 = _client_stats(K, D, seed=22)                    # DIFFERENT stats round 1
    N1 = torch.stack(c1).sum(0); M1 = torch.stack(s1).sum(0)

    # Round 0: C ← (1-γ)·N0 (from zero), S ← (1-γ)·M0.
    w0, Cb0, Sb0, _ = _server_merge(c0, s0, EPS, THRESH, revive=False,
                                    ema_state=ema, ema_decay=gamma)
    assert torch.allclose(ema["C"], (1 - gamma) * N0), "round-0 C != (1-γ)·N0"
    assert torch.allclose(ema["S"], (1 - gamma) * M0), "round-0 S != (1-γ)·M0"
    assert torch.allclose(Cb0, ema["C"]) and torch.allclose(Sb0, ema["S"]), \
        "round-0 broadcast (C,S) != accumulator"

    C_prev, S_prev = ema["C"].clone(), ema["S"].clone()
    # Round 1: C ← γ·C_prev + (1-γ)·N1 — must remember round 0.
    w1, Cb1, Sb1, _ = _server_merge(c1, s1, EPS, THRESH, revive=False,
                                    ema_state=ema, ema_decay=gamma)
    assert torch.allclose(ema["C"], gamma * C_prev + (1 - gamma) * N1), \
        "round-1 C != γ·C_prev + (1-γ)·N1 (recurrence broken)"
    assert torch.allclose(ema["S"], gamma * S_prev + (1 - gamma) * M1), \
        "round-1 S != γ·S_prev + (1-γ)·M1 (recurrence broken)"

    # MEMORY: round 1's broadcast must differ from a memoryless (round-1-only) merge.
    memoryless_C = (1 - gamma) * N1
    trace = (ema["C"] - memoryless_C).abs().max()
    assert trace > 1e-4, f"no cross-round memory: |C1 - (1-γ)N1|={trace:.2e} (γ={gamma})"

    # The broadcast codebook is derived from the SMOOTHED accumulators.
    assert torch.allclose(w1, SharedVectorQuantizer.codebook_from_stats(ema["C"], ema["S"], EPS)), \
        "round-1 codebook != S/smoothed(C)"
    assert torch.allclose(Cb1, ema["C"]) and torch.allclose(Sb1, ema["S"]), \
        "round-1 broadcast (C,S) != accumulator"
    print(f"PASS (1): EMA remembers round 0 into round 1 (|C1-(1-γ)N1|={trace:.3f}); "
          "recurrence C←γC+(1-γ)N, S←γS+(1-γ)M exact; e_j=S/smoothed(C)")


# ── (2) γ=0 equivalence with the current per-round merge ──────────────────────

def test_gamma0_equivalence():
    K, D = 20, 8
    counts, sums = _client_stats(K, D, seed=7, n_clients=3)

    # Reference: the plain federated_cb_only merge (no server EMA).
    w_ref, N_ref, M_ref, dead_ref = _server_merge(counts, sums, EPS, THRESH, revive=False)

    # γ=0 server-EMA merge from zero accumulators.
    ema = {"C": torch.zeros(K), "S": torch.zeros(K, D)}
    w_ema, C_ema, S_ema, dead_ema = _server_merge(counts, sums, EPS, THRESH, revive=False,
                                                  ema_state=ema, ema_decay=0.0)

    assert torch.equal(w_ema, w_ref), "γ=0 codebook != plain merge codebook"
    assert torch.equal(C_ema, N_ref), "γ=0 broadcast C != N (aggregate round count)"
    assert torch.equal(S_ema, M_ref), "γ=0 broadcast S != M (aggregate round sum)"
    assert dead_ema == dead_ref, "γ=0 dead-code count differs from plain merge"
    # And the accumulators are exactly this round's aggregate stats.
    assert torch.equal(ema["C"], N_ref) and torch.equal(ema["S"], M_ref), \
        "γ=0 accumulators != (N, M)"
    print("PASS (2): γ=0 server-EMA merge is BIT-IDENTICAL to federated_cb_only "
          "(codebook, C=N, S=M, dead-count all equal)")


# ── (3) codebook identical on all clients after the broadcast ─────────────────

def test_broadcast_identical():
    K, D, n_clients = 16, 8, 4

    # Clients start with DIFFERENT random codebooks — the broadcast must overwrite them.
    clients = []
    for i in range(n_clients):
        vq = SharedVectorQuantizer(token_embedding_dim=D, codebook_size=K,
                                   threshold_ema_dead_code=0)
        vq.train(); vq.collect_stats_only = True
        torch.manual_seed(100 + i)
        vq.codebook.weight.data.normal_()                # diverge each client's codebook
        clients.append(vq)
    assert not torch.equal(clients[0].codebook.weight, clients[1].codebook.weight), \
        "test setup: clients should start with different codebooks"

    # Server computes the round's global codebook + EMA state (γ>0), then broadcasts.
    counts, sums = _client_stats(K, D, seed=5, n_clients=n_clients)
    ema = {"C": torch.zeros(K), "S": torch.zeros(K, D)}
    weight, C, S, _ = _server_merge(counts, sums, EPS, THRESH, revive=False,
                                    ema_state=ema, ema_decay=0.5)
    for vq in clients:                                   # same call the orchestrator makes
        vq.set_codebook(weight, ema_cluster_size=C, ema_embed_sum=S)

    ref = clients[0]
    for i, vq in enumerate(clients[1:], start=1):
        assert torch.equal(vq.codebook.weight, ref.codebook.weight), f"client {i} codebook differs"
        assert torch.equal(vq.ema_cluster_size, ref.ema_cluster_size), f"client {i} C differs"
        assert torch.equal(vq.ema_embed_sum, ref.ema_embed_sum), f"client {i} S differs"
    # …and every client holds exactly what the server broadcast.
    assert torch.equal(ref.codebook.weight, weight), "codebook != server weight"
    assert torch.equal(ref.ema_cluster_size, C), "ema_cluster_size != C"
    assert torch.equal(ref.ema_embed_sum, S), "ema_embed_sum != S"
    print(f"PASS (3): all {n_clients} clients hold the identical codebook + (C, S) "
          "after broadcast (C→ema_cluster_size, S→ema_embed_sum)")


# (4) bias-corrected dead-code detection under cold-start damping

def test_dead_detection_biascorrected():
    K, D, gamma = 8, 4, 0.8
    N = torch.full((K,), 50.0)
    N[0] = 0.0                       # code 0: genuinely dead
    N[1] = 5.0                       # code 1: small but healthy (5 > THRESH=2)
    counts, sums = [N.clone()], [torch.randn(K, D) * 3.0]
    ema = {"C": torch.zeros(K), "S": torch.zeros(K, D)}

    _, _, _, n_dead = _server_merge(counts, sums, EPS, THRESH, revive=False,
                                    ema_state=ema, ema_decay=gamma)

    # Round 0: raw C = (1-γ)·N is cold-start damped, so a naive `C < THRESH` kills any
    # code with N < THRESH/(1-γ) = 10 — i.e. BOTH code 0 AND the healthy code 1.
    naive_dead = int((ema["C"] < THRESH).sum())
    assert naive_dead == 2, f"setup: raw-C should over-flag (got {naive_dead}, want 2)"
    # Ĉ = C / (1-γ^1) = N ⇒ only the truly-dead code 0 is flagged — the fix.
    assert n_dead == 1, f"bias-corrected dead count should be 1 (code 0 only), got {n_dead}"
    assert int((N < THRESH).sum()) == n_dead, \
        "bias-corrected round-0 dead-set must equal cb_only's N<threshold"
    print(f"PASS (4): bias-corrected dead-detection flags only truly-dead codes "
          f"(n_dead={n_dead}; naive raw-C would wrongly flag {naive_dead})")


def main() -> int:
    force_utf8_stdout()          # PASS banners print 'γ', 'Σ' — see utils.force_utf8_stdout
    test_ema_memory()
    test_gamma0_equivalence()
    test_broadcast_identical()
    test_dead_detection_biascorrected()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
