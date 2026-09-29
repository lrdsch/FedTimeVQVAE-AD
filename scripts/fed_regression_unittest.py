#!/usr/bin/env python3
"""Regression guards for the federated launch path. CPU-only, no data, seconds to run:

    python3 scripts/fed_regression_unittest.py

Every check here corresponds to a defect that WAS live in this repo on 2026-07-29 and cost
either a wrong number or a wasted sweep. They are cheap to re-break (a new arm, a copied
call site, a "harmless" refactor of the seed schedule) and expensive to notice, so they are
pinned rather than remembered.

  1. FEDAVG MATH. The weighted average is n_k-weighted, all clients end identical, nothing
     outside the shared key set moves, and BN running statistics never enter the linear mean.
  2. SEED SCHEDULE. The old `seed*1000 + round*100 + client` made (seed 7, round 10) and
     (seed 8, round 0) the SAME stream, so multi-seed replicates shared their batch ordering
     and the seed-to-seed variance -- the only thing multi-seed estimates -- was understated.
  3. NO FREE NAMES in federated_stage2. Its truncation alarm referenced `rounds_done`, a
     local of federated_stage1: NameError at the end of training, in exactly the case the
     alarm exists to report.
  4. PROTOCOL PLUMBING. 14 of 19 `train_federated` call sites omitted `protocol=`, so
     `--protocol converged` never reached them: select_on_val off => --fed-patience-rounds
     silently dead => the full 300x10 budget with no stop and no best-on-val restore.
  5. ARM REGISTRY == DISPATCH, and the paper table is exactly what the launcher expands.

Complements scripts/fed_enc_algo_unittest.py (the encoder trio's own semantics).
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from pipeline.federated import (_client_weights, _encoder_shared_keys, _fedavg_encoder,
                                _fedavg_shared, _pool_encoder_bn, _round_seed)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


class _Tiny(nn.Module):
    def __init__(self, v: float):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(1, 2, 3), nn.BatchNorm1d(2))
        self.decoder_2d = nn.Conv1d(2, 1, 3)
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(v)
            self.encoder[1].running_mean.fill_(v)
            self.encoder[1].running_var.fill_(v ** 2)


class _Client:
    def __init__(self, v: float, n: int):
        self.model, self.entity_id = _Tiny(v), f"c{v}"
        self.data = type("D", (), {"train_dataset": list(range(n))})()


CPU = [torch.device("cpu")] * 2


# ── 1. FedAvg math ────────────────────────────────────────────────────────────

def test_fedavg_math() -> None:
    print("\n[1] FedAvg is the exact n_k-weighted average")
    cs = [_Client(1.0, 10), _Client(3.0, 30)]
    keys = _encoder_shared_keys(cs[0].model, "full", 2, "buffers_local")
    check("client weights are the training-set sizes", _client_weights(cs) == [10.0, 30.0])
    _fedavg_encoder(cs, CPU, keys)
    got = float(cs[0].model.encoder[0].weight.detach().flatten()[0])
    check("weighted by n_k, not uniform", abs(got - 2.5) < 1e-6,
          f"got={got} expected=2.5 (uniform would be 2.0)")
    check("every client holds the identical aggregate",
          all(torch.equal(cs[0].model.state_dict()[k], cs[1].model.state_dict()[k])
              for k in keys))
    check("the decoder is untouched",
          float(cs[0].model.decoder_2d.weight.detach().flatten()[0]) == 1.0
          and float(cs[1].model.decoder_2d.weight.detach().flatten()[0]) == 3.0)

    cs2 = [_Client(1.0, 10), _Client(3.0, 30)]
    shared = _encoder_shared_keys(cs2[0].model, "full", 2, "shared")
    _fedavg_encoder(cs2, CPU, shared)
    rv = float(cs2[0].model.encoder[1].running_var.flatten()[0])
    check("running_var never enters the LINEAR average", rv == 1.0,
          f"running_var={rv}; a linear mean would be 7.0")
    _pool_encoder_bn(cs2, CPU, _client_weights(cs2), shared)
    exp = (10 * (1 + 1) + 30 * (9 + 9)) / 40 - 2.5 ** 2       # within + between
    rv2 = float(cs2[0].model.encoder[1].running_var.flatten()[0])
    check("_pool_encoder_bn gives the law-of-total-variance value", abs(rv2 - exp) < 1e-5,
          f"got={rv2} expected={exp}")

    agg = _fedavg_shared([{"w": torch.full((3,), 1.0)}, {"w": torch.full((3,), 3.0)}],
                         [10, 30], ["w"])
    check("the stage-2 prior body averages the same way", abs(float(agg["w"][0]) - 2.5) < 1e-6)
    agg_h = _fedavg_shared([{"w": torch.full((3,), 1.0, dtype=torch.float16)},
                            {"w": torch.full((3,), 3.0, dtype=torch.float16)}], [10, 30], ["w"])
    check("the aggregate keeps the source dtype", agg_h["w"].dtype == torch.float16)


# ── 2. seed schedule ──────────────────────────────────────────────────────────

def test_round_seed() -> None:
    print("\n[2] the per-round seed schedule has no cross-seed aliasing")
    seen: dict[int, tuple] = {}
    coll = 0
    for stage in (1, 2):
        for seed in range(20):
            for r in range(400):
                for ci in range(40):
                    v = _round_seed(seed, r, ci, stage)
                    if v in seen:
                        coll += 1
                    seen[v] = (stage, seed, r, ci)
    check("no collisions over 20 seeds x 400 rounds x 40 clients x 2 stages", coll == 0,
          f"{len(seen)} combinations, {coll} collisions")
    check("the OLD schedule did alias (this is what was fixed)",
          7 * 1000 + 10 * 100 + 0 == 8 * 1000 + 0 * 100 + 0)
    check("the new one does not", _round_seed(7, 10, 0) != _round_seed(8, 0, 0))
    check("stage 1 and stage 2 are separate streams",
          _round_seed(7, 0, 0, 1) != _round_seed(7, 0, 0, 2))
    check("deterministic across calls", _round_seed(7, 10, 3) == _round_seed(7, 10, 3))
    check("accepted by torch.manual_seed", torch.manual_seed(max(seen)) is not None)


# 3-5. static guards on the source

def _free_names(fn: ast.FunctionDef, tree: ast.Module) -> list[str]:
    assigned = {t.id for n in ast.walk(fn) for t in ast.walk(n)
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
    assigned |= {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    # Nested defs and lambdas bind their own names: `def _fedavg(get)` inside
    # federated_stage1 binds BOTH `_fedavg` and `get`, and without this they read as free.
    for n in ast.walk(fn):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = n.args
            assigned |= {x.arg for x in a.args + a.kwonlyargs + a.posonlyargs}
            assigned |= {x.arg for x in (a.vararg, a.kwarg) if x}
            if not isinstance(n, ast.Lambda):
                assigned.add(n.name)
        elif isinstance(n, (ast.comprehension,)):
            assigned |= {t.id for t in ast.walk(n.target) if isinstance(t, ast.Name)}
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    module = {t.id for n in tree.body if isinstance(n, ast.Assign)
              for t in ast.walk(n) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
    module |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    module |= {a.asname or a.name.split(".")[0] for n in tree.body
               if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    return sorted(used - assigned - module - set(dir(builtins)))


def test_no_free_names() -> None:
    print("\n[3] federated_stage2 has no undefined names (the truncation-alarm NameError)")
    tree = ast.parse((REPO / "pipeline" / "federated.py").read_text())
    for name in ("federated_stage1", "federated_stage2", "federated_stage2_proto"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        free = _free_names(fn, tree)
        check(f"{name}: no free names", not free, f"free: {free}" if free else "")


def test_protocol_plumbed() -> None:
    print("\n[4] every train_federated call site forwards --protocol")
    src = (REPO / "pipeline" / "federated_eval.py").read_text()
    tree = ast.parse(src)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "train_federated"]
    missing = [ast.unparse(c)[:60] for c in calls
               if "protocol" not in {k.arg for k in c.keywords}]
    check(f"all {len(calls)} call sites pass protocol=", not missing,
          f"missing on {len(missing)}: {missing}" if missing else f"{len(calls)} sites")


def test_arm_registry() -> None:
    print("\n[5] the arm registry matches the dispatch, and the paper table dispatches")
    src = (REPO / "pipeline" / "federated_eval.py").read_text()
    tree = ast.parse(src)
    g = {n.targets[0].id: n.value for n in tree.body
         if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    paper = [e.value for e in g["PAPER_ARMS"].elts]
    known = set(paper) | {e.value for e in g["OTHER_ARMS"].elts}
    disp: set[str] = set()
    for n in ast.walk(tree):
        if not isinstance(n, ast.If):
            continue
        for c in ast.walk(n.test):
            if isinstance(c, ast.Compare) and isinstance(c.left, ast.Name) and c.left.id == "arm":
                for cmp in c.comparators:
                    if isinstance(cmp, ast.Constant):
                        disp.add(cmp.value)
                    elif isinstance(cmp, (ast.Tuple, ast.List)):
                        disp |= {e.value for e in cmp.elts if isinstance(e, ast.Constant)}
    check("registry == dispatch", known == disp,
          f"registry-only={sorted(known - disp)} dispatch-only={sorted(disp - known)}")
    check("every paper arm dispatches", set(paper) <= disp)
    # The (codebook primitive x prior sharing) factorial must be COMPLETE, or the two
    # contributions stop being readable as main effects. This is the cell that was missing.
    for cell in ("federated_cb_only", "federated_fedavg_cb_only",
                 "federated_shared", "federated_fedavg_cb_sharedprior"):
        check(f"factorial cell present: {cell}", cell in paper)
    check("the (A) headline row is the EMA one (cb_only ties the floor, p=0.209)",
          paper.index("federated_cb_only_ema") < paper.index("federated_cb_only"))


def test_rvq_guard() -> None:
    print("\n[6] merge='fedavg' refuses a multi-stage Residual-VQ instead of half-federating it")
    src = (REPO / "pipeline" / "federated.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "federated_stage1")
    # The guard must sit BEFORE the round loop, or a run would train first and raise after.
    raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)
              and "STAGE 0 ONLY" in ast.unparse(n)]
    check("the guard exists and raises", len(raises) == 1)
    loop = next(n for n in fn.body if isinstance(n, ast.For)
                and isinstance(n.iter, ast.Call) and ast.unparse(n.iter).startswith("range(rounds"))
    check("it fires before the round loop, not after training",
          bool(raises) and raises[0].lineno < loop.lineno,
          f"guard at line {raises[0].lineno}, loop at {loop.lineno}" if raises else "")
    # The suff-stat path must keep looping every stage — that is what makes the guard the
    # right fix rather than a limitation of the whole file.
    check("merge='suffstat' still merges every stage (loops c.vqs)",
          "for s, v in enumerate(c.vqs)" in src)


def test_launcher_guards() -> None:
    """[G] The launcher must not be able to mint a run that LOOKS reportable but is not.

    Both defects here were live on 2026-07-30 and both produce plausible-looking artifacts:

      * a TVQ_SMOKE run wrote a RUN.json with a valid `cohort_fingerprint` and a note
        asserting "Comparable to any other tag with the same cohort_fingerprint". Every
        number in it is meaningless (budgets collapsed to 40 steps), so the manifest was
        actively lying about the one property the cohort system exists to guarantee.
      * `launch_all_fa.sh` had no guard at all. The FA suite is SHELVED (mixture_eval:174
        tokenizes every client with client-0's encoder) but that was recorded only in the
        ledger, and `mixture_eval.py`'s SystemExit sits on a DIFFERENT entry point.
    """
    print("\n[G] launcher cannot mint a run that looks reportable but is not")
    launch = (REPO / "scripts" / "launch.sh").read_text()
    check("launch.sh reads TVQ_SMOKE", 'TVQ_SMOKE' in launch)
    # RUN.json used to be emitted by a shell heredoc, so these two checks matched the literal
    # `"smoke": $SMOKE_FLAG` / `"note": "$SMOKE_NOTE"`. It is now serialised by json.dumps with
    # the values travelling as RJ_* environment variables (a tag or an --extra carrying a quote
    # emitted a file that was not JSON at all). The INVARIANT is unchanged and is what we pin:
    # the field exists, and the note is fed from a variable set differently in the two branches
    # of the TVQ_SMOKE test -- never a hardcoded string.
    check("RUN.json carries a `smoke` field", '"smoke": e["RJ_SMOKE"]' in launch)
    check("the comparability note is conditional, not hardcoded",
          '"note": e["RJ_NOTE"]' in launch and 'RJ_NOTE="$SMOKE_NOTE"' in launch
          and launch.count("SMOKE_NOTE=") >= 2)
    check("RUN.json is serialised by json, not by shell interpolation",
          "json.dump" in launch)
    check("the smoke note denies comparability",
          "comparable to NOTHING" in launch)

    smoke = (REPO / "scripts" / "smoke_arms.sh").read_text()
    # Every arm the runbook tells you to LAUNCH must also be an arm the smoke exercises,
    # or the smoke green-lights a table it has never run.
    for arm in ("federated_enc_fedavg", "federated_enc_fedprox"):
        check(f"smoke_arms covers the text arm {arm}", arm in smoke)
    # fedprox's CLI default is mu=0.01; every claim is at the paper value 0.1, and
    # federated.py warns below it. Smoking the default would green-light a config nobody runs.
    check("smoke_arms runs fedprox at the paper mu, not the CLI default",
          "--fedprox-mu 0.1" in smoke)
    check("ARMS=<one arm> is honoured (text arms appended only when unset)",
          "_ARMS_FROM_ENV" in smoke)
    # The trio of §5.2 is one `--arms` edit away from a full-budget launch, and until it
    # entered the smoke NOTHING in this repo had ever executed the fedproto branch (per-code
    # prototypes, the live-assignment mask, the stage-0-only Residual-VQ restriction, the
    # round-0 seeding from the broadcast codebook). A crash there used to surface hours in.
    for case in ("federated_enc_fedproto", "federated_enc_fedproto@count",
                 "federated_enc_commoninit"):
        check(f"smoke_arms covers {case}", case in smoke)

    # Same family, one level up: --extra is appended LAST to the job command and argparse
    # keeps the last occurrence, so `--extra "--metrics-tolerance 64"` silently replaces what
    # the cohort pinned while RUN.json goes on declaring that fingerprint -- a run wearing a
    # certificate of comparability it does not have.
    check("--extra cannot override a cohort-pinned axis", "PINNED_FLAGS" in launch)
    for flag in ("--window-length", "--metrics-tolerance", "--seeds", "--out-dir"):
        check(f"  {flag} is pinned", flag in launch.split("PINNED_FLAGS=(")[1][:300])
    check("--dry leaves NOTHING on disk (no mkdir either)",
          '[[ $DRY -eq 0 ]] && mkdir -p "$RUNDIR" "$LOGDIR"' in launch)
    # An unknown --engine used to fall through to the deep dispatcher while SKIPPING the
    # arm-name validation (gated on == "deep"): `--engine dep` launched 428 unchecked jobs.
    check("--engine is validated against a spelled-out list", "ENGINES=(" in launch)


def main() -> int:
    test_fedavg_math()
    test_round_seed()
    test_no_free_names()
    test_protocol_plumbed()
    test_arm_registry()
    test_rvq_guard()
    test_launcher_guards()
    print("\n" + ("ALL TESTS PASSED" if not FAILURES
                  else f"{len(FAILURES)} FAILED:\n  " + "\n  ".join(FAILURES)))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
