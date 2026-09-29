#!/usr/bin/env python3
"""Fusion into FUNCTION-SPACE of clients of a cluster, and diversity of errors.

Because it exists (2026-08-08). On ucr_170 the average score profiles of 5 models
`local` from AUPRC 0,698 versus 0,303 of the average per-client -- and above the MIGLIOR client
single (0,588). The reason, measured: with local encoder the false positives of the clients are
IDIOSINCRATICI (average current between 0,558 profiles; top-peaks fall in different points) while
True anomaly is common, so mediating erases FPs. With the mediated encoder (A2) the 5
models are the same model -- corr 0,981, same ranking of peaks on 5/5 -- and the
Merger gain is exactly +0,000.

It follows that each cell that changes QUANTO encoder shares should read on DUE numbers, not
one: the average per-client (the standard metric of cells) and the fusion. The correlation
between the profiles is the mechanism variable that binds them.

 ⁇  The gain depends on the Merger Regulation: on ucr_170/local the average from 0,698 and the
trimmed 0,690, but rank-mean 0,277 (under average per-client) and median 0,363. Living.
in the AMPIEZZA of the peaks, not in the order -- must be declared when it is reported.

 ⁇  On the ucr_split_w2p dataset, TEST is identical to clients (they are SHARD DI TRAIN a
to be disjoined): the merger is therefore a merger of MODELLI on the same data, and remains
federation-legal (no data crosses the network, only modelli/score). The analysis unit
remains the CLUSTER: the fusion produces A number, the average per-client another -- is not the
Same amount and they don't have to be compared to the per-cell noise thresholds.

Uso:
    python3.10 scripts/fusion_probe.py --run artifacts/runs/zn_main --series ucr_170
    python3.10 scripts/fusion_probe.py --run artifacts/runs/zn_170_neck --series ucr_170 \
        --baseline artifacts/runs/zn_a2
    python3.10 scripts/fusion_probe.py --run artifacts/runs/zn_main --all-series --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def _load_cluster(arm_dir: str, series: str):
    """Test profiles of clients of a cluster. -> (S [n_client, T], y [T]) or (None, None)."""
    paths = sorted(glob.glob(os.path.join(arm_dir, f"{series}_p*", "scores.npz")))
    if not paths:
        return None, None
    scores, labels = [], None
    for p in paths:
        z = np.load(p)
        scores.append(z["test_scores"])
        labels = z["test_labels"]
    lens = {len(s) for s in scores}
    if len(lens) != 1:
        raise SystemExit(f"client with different test lengths in {arm_dir}: {lens}")
    return np.asarray(scores), (labels > 0).astype(int)


def _z(S: np.ndarray) -> np.ndarray:
    sd = S.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (S - S.mean(axis=1, keepdims=True)) / sd


def analyse(arm_dir: str, series: str) -> dict | None:
    S, y = _load_cluster(arm_dir, series)
    if S is None or y is None or y.sum() == 0:
        return None
    per = [float(average_precision_score(y, s)) for s in S]
    Z = _z(S)
    R = np.asarray([rankdata(s) / len(s) for s in S])
    trimmed = np.sort(Z, axis=0)[1:-1].mean(axis=0) if len(S) > 2 else Z.mean(axis=0)
    fus = {
        "mean_z": float(average_precision_score(y, Z.mean(axis=0))),
        "median_z": float(average_precision_score(y, np.median(Z, axis=0))),
        "trimmed_z": float(average_precision_score(y, trimmed)),
        "mean_rank": float(average_precision_score(y, R.mean(axis=0))),
    }
    corr = np.corrcoef(Z)
    iu = np.triu_indices(len(S), 1)
    return {
        "arm_dir": arm_dir,
        "series": series,
        "n_client": len(S),
        "per_client_auprc": per,
        "mean_auprc": float(np.mean(per)),
        "best_auprc": float(np.max(per)),
        "fusion": fus,
        "gain_vs_mean": fus["mean_z"] - float(np.mean(per)),
        "gain_vs_best": fus["mean_z"] - float(np.max(per)),
        "corr_mean": float(corr[iu].mean()) if len(S) > 1 else float("nan"),
        "corr_min": float(corr[iu].min()) if len(S) > 1 else float("nan"),
    }


def _arm_dirs(run: str, series: str, dataset: str, seed: int) -> list[str]:
    root = os.path.join(run, "ckpt", dataset, series, f"seed{seed}")
    return sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--run", required=True, help="artifacts/runs/<tag>")
    ap_.add_argument("--series", default=None, help="es. ucr_170")
    ap_.add_argument("--all-series", action="store_true")
    ap_.add_argument("--dataset", default="ucr_split_w2p")
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--baseline", default=None, help="other run-dir to compare")
    ap_.add_argument("--json", default=None, help="Write the raw results here")
    a = ap_.parse_args()

    if not a.series and not a.all_series:
        ap_.error("serve --series o --all-series")
    if a.all_series:
        root = os.path.join(a.run, "ckpt", a.dataset)
        series = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "ucr_*")))
    else:
        series = [a.series]

    rows = []
    for run in filter(None, [a.run, a.baseline]):
        for s in series:
            for d in _arm_dirs(run, s, a.dataset, a.seed):
                r = analyse(d, s)
                if r:
                    r["run"] = run
                    rows.append(r)

    if not rows:
        raise SystemExit("no cluster with scores.npz found (run/serie/seed right?)")

    hdr = f"{'run':<26}{'arm':<44}{'series':<9}{'mean':>7}{'best':>7}{'FUSED':>7}{'d.med':>7}{'d.best':>8}{'corr':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{os.path.basename(r['run']):<26}{os.path.basename(r['arm_dir'])[:43]:<44}"
              f"{r['series']:<9}{r['mean_auprc']:>7.3f}{r['best_auprc']:>7.3f}"
              f"{r['fusion']['mean_z']:>7.3f}{r['gain_vs_mean']:>+7.3f}"
              f"{r['gain_vs_best']:>+8.3f}{r['corr_mean']:>7.3f}")

    print("\nregole di fusion (AUPRC) — the average is the one shown above:")
    for r in rows:
        f = r["fusion"]
        print(f"  {os.path.basename(r['arm_dir'])[:43]:<44}{r['series']:<9}"
              f"media {f['mean_z']:.3f} · trimmed {f['trimmed_z']:.3f} · "
              f"mediana {f['median_z']:.3f} · rank {f['mean_rank']:.3f}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwritten {a.json}")


if __name__ == "__main__":
    main()
