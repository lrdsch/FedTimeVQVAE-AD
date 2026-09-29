"""Regression test for val-based round selection in the federated orchestrator.

`federated_stage1` / `federated_stage2` run every round of the budget (so the
gradient-step count stays identical across arms) and use each client's held-out
`val` only to choose WHICH round's weights to keep. The interesting branch is the
one that fires when the best round is NOT the last — and on a healthy smoke run
the val loss decreases monotonically, so that branch never executes and can rot
silently.

Here the val loss is scripted to dip at round 1 (worse, BEST, worse), forcing the
restore. We then assert the invariants the restore must preserve:

  * stage 1: the returned `global_cb` IS the round-1 codebook, and every client
    holds it (a restore that reloaded client states but left `global_cb` at the
    last round would silently ship a codebook no client has).
  * stage 2: the shared prior body is still identical across clients, and the
    local heads still differ — the full prior state is snapshotted, because a head
    trained to round R paired with a body rolled back to round 1 is a combination
    that never existed during training.

    python scripts/fed_val_selection_unittest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from config import Config, apply_dataset_overrides  # noqa: E402
import pipeline.federated as F  # noqa: E402

DATASET = "toy_fed_uni"
CLIENTS = ["uni_00", "uni_01"]           # two clients of one synthetic cluster
# (generate the data first: python scripts/build_toy_fed_uni.py)
S1_VALS = [3.0, 1.0, 5.0]
S2_VALS = [30.0, 10.0, 50.0]
BEST_ROUND = 1


def _scripted(values: list[float], n_clients: int):
    """Return a `_val_loss_*` stand-in yielding values[round], called once per client."""
    state = {"n": 0}

    # *args: the two real probes do not share a signature — `_val_loss_prior` grew a third
    # positional (`val_mode`) — and a stub pinned to two arguments makes this test fail with a
    # TypeError inside the orchestrator instead of testing anything.
    def fn(*_args, **_kw) -> float:
        v = values[state["n"] // n_clients]
        state["n"] += 1
        return v
    return fn


def main() -> int:
    cfg = Config()
    cfg.dataset.name = DATASET
    apply_dataset_overrides(cfg)
    cfg.dataset.window_stride = 128                     # test speed only
    cfg.dataset.batch_size_stage1 = cfg.dataset.batch_size_stage2 = 64

    F._val_loss_stage1 = _scripted(S1_VALS, len(CLIENTS))
    F._val_loss_prior = _scripted(S2_VALS, len(CLIENTS))

    # ⚠ select_on_val=True is REQUIRED and is not a detail of the test: it is a keyword
    # argument that defaults to False in both stages, and with it off no round is ever marked
    # `selected`, so the restore branch this file exists to pin is never reached and every
    # assertion below passes vacuously or fails for the wrong reason. The real launcher sets
    # it from the protocol (`federated_eval.py`: select_on_val=(args.protocol == "converged")),
    # which is the condition every reported cell runs under.
    clients, global_cb, h1 = F.federated_stage1(cfg, CLIENTS, rounds=len(S1_VALS),
                                                local_epochs=1, select_on_val=True)
    selected = [r["round"] for r in h1 if r.get("selected")]
    assert selected == [BEST_ROUND], f"stage1 selected {selected}, expected [{BEST_ROUND}]"
    assert all("val_loss" in r for r in h1), "stage1 history must record val_loss per round"
    for c in clients:
        assert torch.allclose(c.vq.codebook.weight, global_cb), \
            "after restore, global_cb must equal every client's codebook"

    s2_clients, h2, head_div = F.federated_stage2(clients, cfg, rounds=len(S2_VALS),
                                                  local_epochs=1, select_on_val=True)
    selected = [r["round"] for r in h2 if r.get("selected")]
    assert selected == [BEST_ROUND], f"stage2 selected {selected}, expected [{BEST_ROUND}]"
    assert head_div > 0, f"local heads must stay personalized after restore (head_div={head_div})"

    shared = F._prior_shared_keys(s2_clients[0].s2.prior, F.LOCAL_PRIOR_PREFIXES)
    ref = s2_clients[0].s2.prior.state_dict()
    for c in s2_clients[1:]:
        sd = c.s2.prior.state_dict()
        for k in shared:
            assert torch.allclose(sd[k], ref[k]), f"shared body diverged after restore: {k}"

    print(f"OK: restore fired at round {BEST_ROUND} in both stages; "
          f"codebook broadcast intact, shared body intact, head_div={head_div:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
