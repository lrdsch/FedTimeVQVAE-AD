#!/usr/bin/env python3
"""cohort.py — pin WHICH data a run covers, so runs and ablations are comparable.

The failure this exists to prevent: every comparison in this repo has at some point been
broken by an unpinned axis — a different tree, a different window, a different tolerance, a
different cluster set — and the damage was always found late, by hand. A *cohort* is a file
that pins all of them at once:

    datasets  ->  the exact cluster list, the window rule, the metric tolerance, the seeds

Every launch names a cohort. Two runs on the same cohort are matched by construction: same
series, same clients, same window, same yardstick. An ablation is the SAME cohort with
different knobs, so "did the knob move it" is answerable without re-deriving the cohort.

    python scripts/cohort.py datasets                      # what exists on disk
    python scripts/cohort.py new full --datasets all
    python scripts/cohort.py new probe --datasets wsd_fed,ucr_split_w2p --clusters 4
    python scripts/cohort.py new ucr10 --datasets ucr_split_w2p --clusters ucr_001,ucr_005
    python scripts/cohort.py new paper --datasets ucr_split --clusters-file scripts/ucr_split_clusters.txt
    python scripts/cohort.py show full
    python scripts/cohort.py verify full                   # every cluster still resolves?
    python scripts/cohort.py jobs full --arms local,centralized --tag run1
"""
from __future__ import annotations

import argparse
import hashlib
import io
import contextlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COHORTS = REPO / "cohorts"
sys.path[:0] = [str(REPO), str(REPO / "pipeline"), str(REPO / "scripts")]

# Pretraining pools, not federations: one cluster holding every series, so `local` vs
# `federated` has no meaning there. Excluded from `--datasets all` on purpose; name them
# explicitly if you really want them.
NOT_FEDERATED = {"ucr_ad", "ucr_pool"}


def discover() -> dict[str, dict]:
    """Every dataset on disk, with the facts a launch needs."""
    out: dict[str, dict] = {}
    for meta_p in sorted((REPO / "data" / "raw").glob("*/metadata.json")):
        ds = meta_p.parent.name
        try:
            m = json.loads(meta_p.read_text())
        except Exception:
            continue
        clusters = m.get("clusters")
        if not isinstance(clusters, dict) or not clusters:
            continue
        n_ent = (sum(len(v) for v in clusters.values())
                 if isinstance(next(iter(clusters.values())), list)
                 else sum(int(v) for v in clusters.values()))
        out[ds] = {
            "clusters": sorted(clusters),
            "n_clusters": len(clusters),
            "n_entities": int(n_ent),
            "window_mode": m.get("window_mode", "fixed"),
            "windows": m.get("windows") or {},
            "metrics_tolerance": m.get("metrics_tolerance"),
            "federated": ds not in NOT_FEDERATED,
            "note": m.get("note", ""),
        }
    return out


def effective_cfg(ds: str, info: dict) -> tuple[int | None, int]:
    """(window or None if per-series, metrics_tolerance) that a launch must PIN.

    Tolerance is never left to the config default when the window moves: the VUS/PATE buffer
    and the top-k radius default to window//2, so a per-series window would silently give
    every series a different yardstick and no cross-series aggregate would mean anything.
    """
    import config as C
    cfg = C.Config()
    cfg.dataset.name = ds
    with contextlib.redirect_stdout(io.StringIO()):
        C.apply_dataset_overrides(cfg)
        C.apply_env_overrides(cfg)
    tol = info["metrics_tolerance"] or cfg.evaluation.paper_metrics_tolerance
    win = None if info["window_mode"] == "2p" else cfg.dataset.window_length
    return win, int(tol)


def build(name: str, datasets: list[str], clusters_spec: str | None,
          clusters_file: Path | None, seeds: list[int], max_window: int = 0) -> dict:
    disc = discover()
    unknown = [d for d in datasets if d not in disc]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; available: {sorted(disc)}")

    picked: list[str] | None = None
    if clusters_file:
        picked = [l.strip() for l in clusters_file.read_text().split("\n") if l.strip()]
    elif clusters_spec and clusters_spec != "all":
        if clusters_spec.isdigit():
            picked = None                      # a COUNT, resolved per dataset below
        else:
            picked = [c.strip() for c in clusters_spec.split(",") if c.strip()]

    entry: dict[str, dict] = {}
    for ds in datasets:
        info = disc[ds]
        avail = info["clusters"]
        if picked is not None:
            keep = [c for c in picked if c in avail]
            missing = [c for c in picked if c not in avail]
            if missing and len(datasets) == 1:
                raise SystemExit(f"{ds}: unknown cluster(s) {missing[:8]}"
                                 f"{' ...' if len(missing) > 8 else ''}")
        elif clusters_spec and clusters_spec.isdigit():
            keep = avail[: int(clusters_spec)]
        else:
            keep = list(avail)
        if not keep:
            raise SystemExit(f"{ds}: no clusters selected")
        win, tol = effective_cfg(ds, info)
        if max_window and info["window_mode"] == "2p":
            # Drop the per-series windows that do not fit, and RECORD the fact here rather
            # than letting the hardware decide it implicitly. On ucr_split_w2p the cost is a
            # staircase, not a curve — measured peak GPU for 5 clients at batch 64:
            #   W<=1024  3.00 GB (170/180 clusters)   W=1782  6.87 GB   W=3028  16.79 GB
            # so 10 clusters cost more than the other 170 combined, and at SLOTS_PER_GPU=7
            # the largest three alone exceed a 48 GB card. The scientific reason is the
            # better one though: at W=3028 the model is 85.7M parameters against 0.1M at
            # W=128, and aggregating detection metrics over models three orders of magnitude
            # apart in capacity is hard to interpret whatever the GPU can hold.
            wins = info["windows"]
            dropped = sorted((c for c in keep if int(wins.get(c, 0)) > max_window),
                             key=lambda c: -int(wins[c]))
            keep = [c for c in keep if c not in set(dropped)]
            if not keep:
                raise SystemExit(f"{ds}: --max-window {max_window} excluded every cluster")
            if dropped:
                print(f"  {ds}: --max-window {max_window} drops {len(dropped)} cluster(s) "
                      f"(largest {', '.join(f'{c}:W={wins[c]}' for c in dropped[:5])}"
                      f"{' ...' if len(dropped) > 5 else ''}) -> {len(keep)} kept")
        entry[ds] = {
            "clusters": keep,
            "n_clusters": len(keep),
            "window_mode": info["window_mode"],
            "window": win,                       # None => per-series, see `windows`
            "windows": ({c: info["windows"][c] for c in keep if c in info["windows"]}
                        if info["window_mode"] == "2p" else {}),
            "metrics_tolerance": tol,
            "federated": info["federated"],
        }
        if info["window_mode"] == "2p":
            miss = [c for c in keep if c not in entry[ds]["windows"]]
            if miss:
                raise SystemExit(f"{ds}: no per-series window for {miss[:5]} — rebuild the dataset")

    payload = {
        "name": name,
        "max_window": int(max_window) or None,
        "datasets": entry,
        "seeds": seeds,
        "totals": {"datasets": len(entry),
                   "clusters": sum(v["n_clusters"] for v in entry.values())},
    }
    # Fingerprint over the pinned content only, so the same selection always hashes the same
    # and a run can prove which cohort it belongs to.
    payload["fingerprint"] = hashlib.sha256(
        json.dumps({k: payload[k] for k in ("datasets", "seeds")},
                   sort_keys=True).encode()).hexdigest()[:16]
    return payload


def load(name: str) -> dict:
    p = COHORTS / f"{name}.json"
    if not p.exists():
        have = sorted(x.stem for x in COHORTS.glob("*.json")) if COHORTS.exists() else []
        raise SystemExit(f"no cohort {name!r}; have: {have or '(none)'}")
    return json.loads(p.read_text())


def cmd_datasets(_args) -> int:
    disc = discover()
    print(f"{'dataset':24s} {'cluster':>8s} {'entities':>8s} {'window':>10s} {'tol':>5s}  note")
    for ds, i in disc.items():
        win, tol = effective_cfg(ds, i)
        w = "per-serie" if win is None else str(win)
        tag = "" if i["federated"] else "  <- pretraining pool, NOT federated"
        print(f"{ds:24s} {i['n_clusters']:8d} {i['n_entities']:8d} {w:>10s} {tol:5d}{tag}")
    print(f"\n'--datasets to the federated {sum(1 for i in disc.values() if i['federated'])} "
          f"(esclude {sorted(NOT_FEDERATED)})")
    return 0


def cmd_new(args) -> int:
    disc = discover()
    ds = ([d for d, i in disc.items() if i["federated"]] if args.datasets == "all"
          else [d.strip() for d in args.datasets.split(",") if d.strip()])
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    payload = build(args.name, ds, args.clusters, args.clusters_file, seeds,
                    max_window=args.max_window)
    COHORTS.mkdir(exist_ok=True)
    p = COHORTS / f"{args.name}.json"
    if p.exists() and not args.overwrite:
        raise SystemExit(f"{p} exists; pass --overwrite")
    p.write_text(json.dumps(payload, indent=2))
    print(f"scritta {p}  fingerprint={payload['fingerprint']}")
    cmd_show(argparse.Namespace(name=args.name))
    return 0


def cmd_show(args) -> int:
    c = load(args.name)
    print(f"{c['name']} fingerprint={c['fingerprint']} seeds={c['seeds']}")
    print(f"{'dataset':24s} {'cluster':>8s} {'window':>12s} {'tol':>5s}")
    for ds, v in c["datasets"].items():
        w = "per-serie" if v["window"] is None else str(v["window"])
        print(f"{ds:24s} {v['n_clusters']:8d} {w:>12s} {v['metrics_tolerance']:5d}")
    print(f"{'TOTAL':24s} {c['totals']['clusters']:8d}")
    return 0


def cmd_verify(args) -> int:
    from federated import resolve_clients
    import config as C
    c = load(args.name)
    bad = 0
    for ds, v in c["datasets"].items():
        cfg = C.Config(); cfg.dataset.name = ds
        with contextlib.redirect_stdout(io.StringIO()):
            C.apply_dataset_overrides(cfg); C.apply_env_overrides(cfg)
        n_ent = 0
        for cl in v["clusters"]:
            try:
                n_ent += len(list(resolve_clients(cfg, None, cl)))
            except Exception as e:
                print(f"  ✗ {ds}/{cl}: {e}"); bad += 1
        print(f"  ✓ {ds:24s} {v['n_clusters']:4d} cluster -> {n_ent:5d}")
    print("OK" if not bad else f"{bad} CLUSTER ROTTI")
    return 1 if bad else 0


def cmd_jobs(args) -> int:
    """One line per job: dataset cluster arm window tolerance out_json.

    The launcher consumes this; keeping the enumeration here means the shell never decides
    which data a run covers.
    """
    c = load(args.name)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    root = REPO / "artifacts" / "runs" / args.tag
    for ds, v in c["datasets"].items():
        for cl in v["clusters"]:
            win = v["window"] if v["window"] is not None else v["windows"][cl]
            for arm in arms:
                out = root / ds / f"{cl}__{arm}.json"
                print(f"{ds} {cl} {arm} {win} {v['metrics_tolerance']} {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datasets").set_defaults(fn=cmd_datasets)

    n = sub.add_parser("new"); n.set_defaults(fn=cmd_new)
    n.add_argument("name")
    n.add_argument("--datasets", default="all", help="'all' (the federates) or separate list from commas")
    n.add_argument("--clusters", default="all",
                   help="'all' | a NUMBER (the first N per dataset) | separate list from commas")
    n.add_argument("--clusters-file", type=Path, default=None, help="a cluster per line")
    n.add_argument("--seeds", default="0")
    n.add_argument("--max-window", type=int, default=0,
                   help="drop per-series-window clusters above this W (ucr_split_w2p only). "
                        "1024 keeps 170/180 at 3.0 GB/job; the 10 excluded cost more than "
                        "all the rest combined and reach 85.7M parameters. 0 = keep all.")
    n.add_argument("--overwrite", action="store_true")

    for name, fn in (("show", cmd_show), ("verify", cmd_verify)):
        s = sub.add_parser(name); s.add_argument("name"); s.set_defaults(fn=fn)

    j = sub.add_parser("jobs"); j.set_defaults(fn=cmd_jobs)
    j.add_argument("name"); j.add_argument("--arms", required=True); j.add_argument("--tag", required=True)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
