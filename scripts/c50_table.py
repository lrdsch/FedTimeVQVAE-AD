#!/usr/bin/env python3
"""c50_table.py — Table IV: the pre-registered confirmation sample, recomputed from disk.

    python scripts/c50_table.py                  # print the table and check it against the paper
    python scripts/c50_table.py --json out.json  # ... and dump the per-series values

The analysis was fixed before any of these numbers existed (docs/PREREGISTRATION.md):

  * PRIMARY endpoint   the archive's own accuracy at tolerance 64 -- one predicted location per
                       series, correct if within 64 timesteps of the labelled anomaly. Read from
                       `paper_top1_acc_at_64` in each client's report.json and averaged over the
                       five clients of the series.
  * SECONDARY          AUPRC of the per-timestep score, read from `summary.<arm>.auprc.mean`.
  * TEST               two-sided sign test across series, ties EXCLUDED from the test and
                       reported separately. Ties are the story here, not a nuisance: the arms
                       agree about whether the single predicted location is found on 31 to 37 of
                       the 48 series, so the effective sample size is 11 to 17, not 48.
  * UNIT               the series. The five clients are shards of one signal, not replicates,
                       so every contrast is paired by series.

The series set is the INTERSECTION of the three arms, which is the pre-registered rule: a series
with an incomplete arm is discarded from all of them. That is what takes 50 down to 48 --
`ucr_190` and go and go `ucr_240` aborted in the federated arm on non-finite entries in the aggregated
codebook.

The mean is printed next to the median because on the primary endpoint the median is zero
wherever ties dominate, which is everywhere.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "artifacts", "runs")
DATASET = "ucr_split_w2p"
TOL = 64
SEED = 0

# contrast label -> (tag_a, arm_a, tag_b, arm_b): the table reports a - b
CONTRASTS = [
    ("(g) - local",         ("c50_a2", "federated_enc_fedavg"), ("c50_local", "local")),
    ("centralized - local", ("c50_central", "centralized"),     ("c50_local", "local")),
    ("(g) - centralized",   ("c50_a2", "federated_enc_fedavg"), ("c50_central", "centralized")),
]

# What Table IV of the paper PRINTS, as printed: (W, L, T, median, mean, p) per endpoint.
# The floats are strings on purpose. A printed number carries its own tolerance -- "0.06" is
# any value that rounds to 0.06 at two decimals, "0.332" any value that rounds to 0.332 at
# three -- and the table does not use the same precision in every cell. Comparing against a
# float with one hardcoded epsilon flags a cell that is printed correctly (p = 0.0595 -> 0.06).
PAPER = {
    "(g) - local":         {"top1": (11, 6, 31, "0.000", "+0.083", "0.332"),
                            "auprc": (None, None, None, "+0.027", "+0.073", "0.029")},
    "centralized - local": {"top1": (14, 5, 29, "0.000", "+0.142", "0.064"),
                            "auprc": (None, None, None, "+0.053", "+0.141", "0.002")},
    "(g) - centralized":   {"top1": (3, 8, 37, "0.000", "-0.058", "0.227"),
                            "auprc": (None, None, None, "-0.009", "-0.067", "0.06")},
}


def matches(computed: float, printed: str) -> bool:
    """Does `computed` round to what the table prints, at the precision it prints it?"""
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    fmt = f"{computed:+.{decimals}f}" if printed[0] in "+-" else f"{computed:.{decimals}f}"
    return fmt == printed


def auprc(tag: str, arm: str) -> dict[str, float]:
    """series -> AUPRC averaged over the clients, exactly as the pipeline wrote it."""
    out = {}
    for f in glob.glob(os.path.join(RUNS, tag, DATASET, f"*__{arm}.json")):
        series = os.path.basename(f)[: -len(f"__{arm}.json")]
        with open(f) as fh:
            d = json.load(fh)
        m = d.get("summary", {}).get(arm, {})
        if "auprc" in m:
            out[series] = m["auprc"]["mean"]
    return out


def top1(tag: str) -> dict[str, float]:
    """series -> mean over clients of paper_top1_acc_at_64 (0 or 1 per client)."""
    out: dict[str, float] = {}
    pat = os.path.join(RUNS, tag, "ckpt", DATASET, "*", f"seed{SEED}", "*", "*", "report.json")
    per: dict[str, list[float]] = {}
    for f in glob.glob(pat):
        series = f.split(os.sep)[-5]
        with open(f) as fh:
            d = json.load(fh)
        k = [x for x in d if x.startswith(f"paper_top1_acc_at_{TOL}")]
        if k:
            per.setdefault(series, []).append(d[k[0]])
    for s, v in per.items():
        out[s] = statistics.mean(v)
    return out


def sign_p(w: int, l: int) -> float:
    """Two-sided sign test, ties excluded. The pre-registered test."""
    n = w + l
    if n == 0:
        return float("nan")
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def contrast(a: dict[str, float], b: dict[str, float], series: list[str]):
    d = [a[s] - b[s] for s in series]
    w = sum(1 for x in d if x > 0)
    l = sum(1 for x in d if x < 0)
    return {"W": w, "L": l, "T": len(d) - w - l, "median": statistics.median(d),
            "mean": statistics.mean(d), "p": sign_p(w, l), "diffs": d}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="dump the per-series values here")
    args = ap.parse_args()

    tags = {t for _, (t, _), (u, _) in CONTRASTS for t in (t, u)}
    A = {(t, arm): auprc(t, arm) for _, (t, arm), _ in CONTRASTS}
    A.update({(t, arm): auprc(t, arm) for _, _, (t, arm) in CONTRASTS})
    T1 = {t: top1(t) for t in tags}

    missing = [t for t in tags if not T1[t]]
    if missing:
        sys.exit(f"no report.json under artifacts/runs/{{{','.join(missing)}}}/ckpt -- "
                 "the result bundle is incomplete or this is not the release tree")

    # the pre-registered rule: a series is analysed only where EVERY arm is complete
    series = sorted(set.intersection(*[set(v) for v in A.values()],
                                     *[set(v) for v in T1.values()]))
    drawn = 50
    print(f"confirmation sample: {len(series)} of {drawn} series analysed "
          f"({drawn - len(series)} discarded from all arms by the pre-registered rule)")
    print(f"paired by series · sign test, two-sided, ties excluded · seed {SEED} · tolerance {TOL}\n")

    hdr = (f"{'contrast':<22}| {'accuracy@64 (primary)':^38}| {'AUPRC (secondary)':^30}")
    print(hdr)
    print(f"{'':<22}| {'W/L/T':>10} {'med':>7} {'mean':>8} {'p':>8} | "
          f"{'med':>8} {'mean':>8} {'p':>8}")
    print("-" * len(hdr))

    bad, dump = 0, {}
    for label, (ta, aa), (tb, ab) in CONTRASTS:
        c1 = contrast(T1[ta], T1[tb], series)
        c2 = contrast(A[(ta, aa)], A[(tb, ab)], series)
        wlt = f"{c1['W']}/{c1['L']}/{c1['T']}"
        print(f"{label:<22}| {wlt:>10} {c1['median']:>7.3f} {c1['mean']:>+8.3f} {c1['p']:>8.3f} | "
              f"{c2['median']:>+8.3f} {c2['mean']:>+8.3f} {c2['p']:>8.3f}")
        dump[label] = {"top1": {k: v for k, v in c1.items() if k != "diffs"},
                       "auprc": {k: v for k, v in c2.items() if k != "diffs"},
                       "per_series": {s: {"top1": T1[ta][s] - T1[tb][s],
                                          "auprc": A[(ta, aa)][s] - A[(tb, ab)][s]}
                                      for s in series}}

        # ---- check against the printed table -------------------------------------------
        want = PAPER[label]
        for endpoint, got in (("top1", c1), ("auprc", c2)):
            W, L, T, med, mean, p = want[endpoint]
            for name, g, w in (("W", got["W"], W), ("L", got["L"], L), ("T", got["T"], T),
                               ("median", got["median"], med), ("mean", got["mean"], mean),
                               ("p", got["p"], p)):
                if w is None:
                    continue
                ok = (g == w) if isinstance(w, int) else matches(g, w)
                if not ok:
                    bad += 1
                    print(f"    [FAIL] {endpoint}.{name}: computed {g} · in the paper {w}")

    print()
    if bad:
        print(f"⚠️  {bad} value(s) do NOT match Table IV of the paper.")
    else:
        print("every value matches Table IV of the paper.")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"series": series, "tolerance": TOL, "seed": SEED,
                       "contrasts": dump, "checks_failed": bad}, fh, indent=2)
        print(f"wrote {args.json}")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
