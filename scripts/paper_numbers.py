#!/usr/bin/env python3.10
"""paper_numbers.py — recalculate and VERIFICAte any number of documentation/paper/paper.tex
that was not already read by a pipeline report.

    $PY scripts/paper_numbers.py # Verifies, exit 1 if something does not return
    $PY scripts/paper_numbers.py --json evidence/paper_numbers.json

Because it exists. The paper numbers have two different backgrounds and the difference counts:

  (1) LETTI from the pipeline II/III/V tables like out of `summary.<arm>.auprc.mean` in the
      json of `artifacts/runs/<tag>/ucr_split_w2p/<serie>__<arm>.json` and from
      `paper_top1_acc_at_64` in `report.json`. Those do not recalculate: they read.

  (2) AGGREGATI over the pipeline — vinte/perse counts, median, sign test, range, and
      ALL fusion in function-space. These were calculated by hand, once,
      and they were not reproducible by any script. This file fills that hole: every
      assertion below is a paper sentence, with the value that's written to us.

 ⁇  The top-1 of the fusion does NOT come from a report: the pipeline does not write it, because the
fusion is not a trained arm. We recalculate it with the `detect._paper_metrics` rule
(argmax of the profile, hit if by `TOL` from a positive timestep). The validation
authorizes to use it: recalculated per-client on 5 `local` must reproduce ESALY the
Table II `local` column. If that check fails, every merger number here is
suspicious and the script comes out 1.

 ⁇  Declared asymmetry: fusion produces A number per cluster, the arm trained
MEDIA on the client 5. The fusion-vs-(g) comparison is therefore "unexplained artifact against
average client, which is the right reading, but it's not a unit tie. On the top-1 the point
does not arise: (g) is 0 or 1 on all and 10 development series (no disconnecting client).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics
import sys

import numpy as np
from sklearn.metrics import average_precision_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "artifacts", "runs")
DATASET = "ucr_split_w2p"
TOL = 64
SEED = 0

DEV = ["ucr_011", "ucr_014", "ucr_043", "ucr_170",
       "ucr_001", "ucr_083", "ucr_086", "ucr_082", "ucr_222", "ucr_229"]
DISC = ["ucr_011", "ucr_014", "ucr_043", "ucr_170"]

# harm of the paper -> (tag, harm name on disk)
G = ("zn_a2", "federated_enc_fedavg")            # (g) = A2
A1 = ("zn_a1", "federated_enc_fedavg")           # (f)
LOCAL = ("zn_main", "local")                     # (a)
CENTR = ("zn_main", "centralized")               # (b)
LOCAL_OT = ("zn_ot", "local")                    # baseline sovrallenata
CB_SUF = ("zn_main", "federated_cb_only")        # (c)
CB_FA = ("zn_main", "federated_fedavg_cb_only")  # (d)


# reading pipeline
def auprc_table() -> dict[tuple[str, str], dict[str, float]]:
    """(tag, arm) -> {series: mean AUPRC across clients}, as the paper reads it."""
    out: dict[tuple[str, str], dict[str, float]] = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(RUNS, "*", DATASET, "*__*.json")):
        tag = f.split(os.sep)[-3]
        series = os.path.basename(f)[:-5].split("__", 1)[0]
        with open(f) as fh:
            d = json.load(fh)
        for arm, m in d.get("summary", {}).items():
            if "auprc" in m:
                out[(tag, arm)][series] = m["auprc"]["mean"]
    return out


def top1_from_reports(tag: str, series: str) -> float | None:
    """Average per-client of `paper_top1_acc_at_64`, the accuracy@64 column of the paper."""
    vals = []
    pat = os.path.join(RUNS, tag, "ckpt", DATASET, series, f"seed{SEED}", "*", "*", "report.json")
    for f in glob.glob(pat):
        with open(f) as fh:
            d = json.load(fh)
        k = [x for x in d if x.startswith(f"paper_top1_acc_at_{TOL}")]
        if k:
            vals.append(d[k[0]])
    return statistics.mean(vals) if vals else None


# ---------------------------------------------------------------- fusione
def load_cluster(tag: str, arm: str, series: str):
    """Test profiles of the client 5. Same recipe for scripts/fusion_probe.py."""
    pat = os.path.join(RUNS, tag, "ckpt", DATASET, series, f"seed{SEED}", arm,
                       f"{series}_p*", "scores.npz")
    paths = sorted(glob.glob(pat))
    if not paths:
        return None, None
    S, y = [], None
    for p in paths:
        z = np.load(p)
        S.append(z["test_scores"])
        y = z["test_labels"]
    return np.asarray(S), (y > 0).astype(int)


def zscore(S: np.ndarray) -> np.ndarray:
    sd = S.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (S - S.mean(axis=1, keepdims=True)) / sd


def top1_hit(score: np.ndarray, pos: np.ndarray, tol: int = TOL) -> float:
    """track._paper_metrics, k=1: argmax, hit within tol by a positive."""
    return float(np.min(np.abs(int(np.argmax(score)) - pos)) <= tol)


def fusion_row(tag: str, arm: str, series: str) -> dict | None:
    S, y = load_cluster(tag, arm, series)
    if S is None or y is None or y.sum() == 0:
        return None
    pos = np.flatnonzero(y)
    Z = zscore(S)
    fused = Z.mean(axis=0)
    per = [float(average_precision_score(y, s)) for s in S]
    corr = np.corrcoef(Z)
    iu = np.triu_indices(len(S), 1)
    return {
        "series": series,
        "n_client": len(S),
        "fusion_auprc": float(average_precision_score(y, fused)),
        "per_client_auprc_mean": float(np.mean(per)),
        "best_client_auprc": float(np.max(per)),
        "fusion_top1": top1_hit(fused, pos),
        "per_client_top1_mean": float(np.mean([top1_hit(s, pos) for s in S])),
        "corr_mean": float(corr[iu].mean()),
    }


# ---------------------------------------------------------------- statistica
def sign_p(w: int, l: int) -> float:
    """Sign test at DUE code, excluding draws. It is the Convention of Table III."""
    n = w + l
    if n == 0:
        return float("nan")
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wl(diffs) -> tuple[int, int, int]:
    d = list(diffs)
    w = sum(1 for x in d if x > 0)
    l = sum(1 for x in d if x < 0)
    return w, l, len(d) - w - l


# verification
class Check:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str, str]] = []

    def eq(self, label: str, got, want, tol=0.0, section="") -> None:
        if isinstance(got, float) and isinstance(want, float):
            ok = abs(got - want) <= tol
            g, w = f"{got:.4f}", f"{want:.4f}"
        else:
            ok = got == want
            g, w = str(got), str(want)
        self.rows.append((ok, section, label, f"calculated {g} · in {w} paper"))

    def report(self) -> int:
        bad = 0
        sec = None
        for ok, section, label, detail in self.rows:
            if section != sec:
                print(f"\n--- {section}")
                sec = section
            print(f"  [{'OK ' if ok else 'FAIL'}] {label}: {detail}")
            bad += (not ok)
        print(f"\n{len(self.rows) - bad}/{len(self.rows)} verificati" +
              ("" if not bad else f"  ⚠️  {bad} NON TORNANO"))
        return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="write here the recalculated raw values")
    args = ap.parse_args()

    A = auprc_table()
    for key in (G, A1, LOCAL, CENTR, CB_SUF, CB_FA):
        missing = [s for s in DEV if s not in A.get(key, {})]
        if missing:
            sys.exit(f"missing series for {key}: {missing}")

    c = Check()

    # --- gate: our top-1 must play the `local` column of Table II ------
    fus = {s: fusion_row(*LOCAL, s) for s in DEV}
    paper_local_top1 = {"ucr_011": 0.40, "ucr_014": 0.80, "ucr_043": 0.20, "ucr_170": 0.20,
                        "ucr_001": 1.00, "ucr_083": 1.00, "ucr_086": 1.00,
                        "ucr_082": 0.00, "ucr_222": 0.00, "ucr_229": 0.00}
    gate_ok = True
    for s in DEV:
        got = fus[s]["per_client_top1_mean"]
        c.eq(f"top-1 per-client {s}", got, paper_local_top1[s], 1e-9,
             "GATE — our top-1 rule reproduces the `local` column (Tab. II)")
        gate_ok &= abs(got - paper_local_top1[s]) < 1e-9

    # VI-A: minimum of two-tail sign test on 4 series
    c.eq("minimum two-sided p-value, n=4", sign_p(4, 0), 0.125, 1e-9,
         "VI-A — «its minimum attainable value is 0.125»")

    # --- VI-B: centralized against the BEST client ------------------------------------------------------------
    best_local = {s: fusion_row(*LOCAL, s)["best_client_auprc"] for s in DEV}
    d_best = [A[CENTR][s] - best_local[s] for s in DEV]
    w, l, _ = wl(d_best)
    c.eq("vinte", w, 7, section="VI-B — «exceeds it on 7 of 10 series in AUPRC (p=0.34)»")
    c.eq("p (two-sided)", sign_p(w, l), 0.34, 0.005, "VI-B — «exceeds it on 7 of 10 series in AUPRC (p=0.34)»")

    # --- VI-C: (g) versus local, and against the overlying baseline -----------------------------
    d_g_loc = [A[G][s] - A[LOCAL][s] for s in DEV]
    w, l, _ = wl(d_g_loc)
    c.eq("vinte", w, 9, section="VI-C — 'improves 9 of 10 series (p=0.021)' and medians")
    c.eq("p", sign_p(w, l), 0.021, 0.001, "VI-C — 'improves 9 of 10 series (p=0.021)' and medians")
    c.eq("median vs baseline patience", statistics.median(d_g_loc), 0.021, 0.0005,
         "VI-C — 'improves 9 of 10 series (p=0.021)' and medians")
    d_g_ot = [A[G][s] - A[LOCAL_OT][s] for s in DEV if s in A[LOCAL_OT]]
    w_ot, l_ot, _ = wl(d_g_ot)
    c.eq("mediana vs baseline sovrallenata", statistics.median(d_g_ot), 0.022, 0.0005,
         "VI-C — 'improves 9 of 10 series (p=0.021)' and medians")
    c.eq("Count vs Overlay", (w_ot, l_ot), (6, 4),
         section="VI-C — 'improves 9 of 10 series (p=0.021)' and medians")

    # «the gains concentrated on three (+0.42,+0.46,+0.30) and six of them below 0.036»:
    # of the 9 earnings take off the Tre grandi, the remaining six must stand under 0.036.
    gains = sorted((x for x in d_g_loc if x > 0), reverse=True)
    S036 = "VI-C — «six of them below 0.036»"
    c.eq("guadagni positivi", len(gains), 9, section=S036)
    c.eq("the three largest gains", [round(x, 2) for x in gains[:3]], [0.46, 0.42, 0.30], section=S036)
    c.eq("the six remaining", len(gains[3:]), 6, section=S036)
    c.eq("the greatest of the six", max(gains[3:]), 0.0353, 0.0005, S036)
    c.eq("all six under 0.036", all(x < 0.036 for x in gains[3:]), True, section=S036)

    # VI-D: codebook rule, (c) against (d)
    d_cd = [A[CB_SUF][s] - A[CB_FA][s] for s in DEV]
    w, l, _ = wl(d_cd)
    c.eq("vinte/perse", (w, l), (5, 5), section="VI-D — «(c)-(d) run from -0.946 to +0.292»")
    c.eq("minimo", min(d_cd), -0.946, 0.001, "VI-D — «(c)-(d) run from -0.946 to +0.292»")
    c.eq("massimo", max(d_cd), 0.292, 0.001, "VI-D — «(c)-(d) run from -0.946 to +0.292»")

    # --- VI-E: fusione in function-space -----------------------------------------------
    S = "VI-E — merger: counts, correlations, head to head with (g)"
    c.eq("ucr_170 fusione", fus["ucr_170"]["fusion_auprc"], 0.698, 0.0005, S)
    c.eq("ucr_170 miglior client", fus["ucr_170"]["best_client_auprc"], 0.588, 0.0005, S)
    c.eq("ucr_170 (g)", A[G]["ucr_170"], 0.086, 0.0005, S)

    d_mean = [fus[s]["fusion_auprc"] - fus[s]["per_client_auprc_mean"] for s in DEV]
    w, l, _ = wl(d_mean)
    c.eq("beats the average per-client", w, 7, section=S)
    c.eq("median gain", statistics.median(d_mean), 0.027, 0.0005, S)
    d_bst = [fus[s]["fusion_auprc"] - fus[s]["best_client_auprc"] for s in DEV]
    c.eq("beats the BEST client", wl(d_bst)[0], 1, section=S)

    corrs = [fus[s]["corr_mean"] for s in DEV]
    c.eq("corr su ucr_170", fus["ucr_170"]["corr_mean"], 0.56, 0.005, S)
    c.eq("ucr_170 is the minimum", min(corrs), fus["ucr_170"]["corr_mean"], 1e-12, S)
    c.eq("median correlations", statistics.median(corrs), 0.86, 0.005, S)

    d_fg = [fus[s]["fusion_auprc"] - A[G][s] for s in DEV]
    w, l, _ = wl(d_fg)
    c.eq("fusione vs (g): vinte/perse", (w, l), (3, 7), section=S)
    c.eq("fusione vs (g): mediana", statistics.median(d_fg), -0.006, 0.0005, S)
    for s, want in [("ucr_043", -0.53), ("ucr_011", -0.39), ("ucr_014", -0.23)]:
        c.eq(f"fusion vs (g) on {s}", fus[s]["fusion_auprc"] - A[G][s], want, 0.005, S)
    # the other defeated 4 are negligible: the paper says "the GRANDI are on these three"
    other = [fus[s]["fusion_auprc"] - A[G][s] for s in DEV
             if s not in ("ucr_043", "ucr_011", "ucr_014") and fus[s]["fusion_auprc"] < A[G][s]]
    c.eq("the other defeats are under 0.02", max(abs(x) for x in other) < 0.02, True, section=S)

    g_top1 = {s: top1_from_reports(G[0], s) for s in DEV}
    c.eq("(g) is binary on all 10 (comparable unit)",
         all(g_top1[s] in (0.0, 1.0) for s in DEV), True, section=S)
    d_t1 = [fus[s]["fusion_top1"] - g_top1[s] for s in DEV]
    c.eq("top-1 fusione vs (g): V/S/P", wl(d_t1), (1, 1, 8), section=S)

    # --- VI-C: the rhythm of the stage 2. The reference is the CTRL, not (g) ----------------
    # If this contrast is attached to zn_a2 instead of zn_a2s2_ctrl numbers
    # They would change: it's the reason why the paper explicitly appoints the reference.
    ST = "VI-C — tau64 against frozen oracle reference (NON versus (g))"
    TAU, CTRL, LEP = (("zn_a2s2_tau64", "federated_enc_fedavg"),
                      ("zn_a2s2_ctrl", "federated_enc_fedavg"),
                      ("zn_a2s2_lep", "federated_enc_fedavg"))
    d_tau = [A[TAU][s] - A[CTRL][s] for s in DEV if s in A.get(TAU, {}) and s in A.get(CTRL, {})]
    w, l, _ = wl(d_tau)
    c.eq("tau64 vs ctrl: vinte/perse", (w, l), (4, 6), section=ST)
    c.eq("tau64 vs ctrl: mediana", statistics.median(d_tau), -0.001, 0.0005, ST)
    for s, want in [("ucr_011", -0.198), ("ucr_043", -0.245)]:
        if s in A.get(LEP, {}):
            c.eq(f"lep (24 up 011, 31 up 043) vs ctrl on {s}", A[LEP][s] - A[CTRL][s], want, 0.001, ST)

    # --- VI-C: (h) on the confirmation sample, where the ctrl does NOT exist -----------------------
    SC = "VI-C — (h) on the 48 of confirmation: there is an oracle travelling with tau"
    c50a2 = {s: v for s, v in A.get(("c50_a2", "federated_enc_fedavg"), {}).items()}
    c50t = {s: v for s, v in A.get(("c50_tau64", "federated_enc_fedavg"), {}).items()}
    common = sorted(set(c50a2) & set(c50t))
    c.eq("serie appaiate", len(common), 48, section=SC)
    d_c50 = [c50t[s] - c50a2[s] for s in common]
    c.eq("AUPRC vinte/perse", wl(d_c50)[:2], (25, 23), section=SC)
    c.eq("AUPRC p", sign_p(*wl(d_c50)[:2]), 0.89, 0.005, SC)
    t_a2 = {s: top1_from_reports("c50_a2", s) for s in common}
    t_t = {s: top1_from_reports("c50_tau64", s) for s in common}
    d_t = [t_t[s] - t_a2[s] for s in common]
    c.eq("top-1 V/S/P", wl(d_t), (2, 4, 42), section=SC)
    c.eq("top-1 p", sign_p(*wl(d_t)[:2]), 0.69, 0.005, SC)

    # Table II: ucr_082 It's on the floor.
    v82 = {"local": A[LOCAL]["ucr_082"], "(g)": A[G]["ucr_082"],
           "centr": A[CENTR]["ucr_082"]}
    c.eq("all under 1e-4", max(v82.values()) < 1e-4, True,
         section="Tab. II — «on ucr_082 every arm is at floor ... fifth decimal»")
    c.eq("(g) beats local to the fifth digit", (A[G]["ucr_082"] - A[LOCAL]["ucr_082"]) > 0, True,
         section="Tab. II — «on ucr_082 every arm is at floor ... fifth decimal»")

    bad = c.report()
    if not gate_ok:
        print("\n ⁇  GATE FALLITO: the top-1 rule does not reproduce the `local` column. "
              "Any number of mergers above must be considered invalid.")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        payload = {
            "dataset": DATASET, "seed": SEED, "tolerance": TOL,
            "series": DEV, "discriminating": DISC,
            "auprc": {f"{t}:{a}": {s: A[(t, a)][s] for s in DEV}
                      for (t, a) in (G, A1, LOCAL, CENTR, CB_SUF, CB_FA)},
            "auprc_local_overtrained": {s: A[LOCAL_OT][s] for s in DEV if s in A[LOCAL_OT]},
            "fusion": {s: fus[s] for s in DEV},
            "top1_g": g_top1,
            "checks_failed": bad,
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwritten {args.json}")

    sys.exit(1 if (bad or not gate_ok) else 0)


if __name__ == "__main__":
    main()
