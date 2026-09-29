#!/usr/bin/env python3.10
"""Explainable sampling as in the original paper: the mask chooses the MODELLO.

`cf_quality.py` masks the ETICHETTATE positions, and is the right choice there: the
Three metrics are comparisons between arm, and the mask must be identical
between arm because Δ and G remain the same quantity. But that's not what they do
authors, and a figure showing the pipeline «as it turns» can not depend on
labels that do not exist at inference.

Rule replicated by §4.4 and A.2 of the `documentation/references/2311.12550v5.pdf`:

  «we first compute the threshold as n-th quantile of a_final computed with a
   TRAINING dataset [...] we detect timesteps as anomalous if the anomaly scores
   a_final of the test dataset are above the threshold. We mask s for those
   anomalous timesteps ACROSS THE FREQUENCY DIMENSION, and perform p(s|s_M)»

  «there is a possibility of completely masking s [...] the prior model ends up
   performing UNCONDITIONAL sampling due to the absence of contextual
   information. To ensure a minimum context [...] we limit the masking rate to a
   MAXIMUM OF 90% of the temporal dimension»

Here the threshold should not be adjusted: it is already `report.json["threshold"]`, calculated from
`_fit_threshold_paper` (quantile per-τ on the train, sum on the τ, average on the
frequencies) — the same as `detect` uses for `predictions`.

A latent column is masked if its timestep interval contains
at least one timestep above threshold; if the columns exceed the roof of the 90%,
the lowest average score. No label comes in.

`--draws` independent extracts of the same mask: the sampling is
Stochastic, and the gang between the draws is the nin-model that
`cf_quality` already uses as a client-agreement denominator.

Just CPU.

    python3.10 scripts/cf_explainable.py --series ucr_011 --client 0 \
        --starts 1498,1680,1862,2044 --draws 5 \
        --out evidence/cf_quality/arrays/explainable_A2_ucr_011.npz
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from config import Config                                  # noqa: E402
from data import make_dataloaders                          # noqa: E402
from stage1 import load_stage1                             # noqa: E402
from stage2 import counterfactual, load_stage2             # noqa: E402

torch.set_num_threads(2)

BUILD = "ucr_split_w2p"
MAX_MASKING_RATE = 0.90        # A.2: «maximum of 90% of the temporal dimension»


def _fill(obj, src: dict) -> None:
    for f in dataclasses.fields(obj):
        if f.name not in src:
            continue
        cur, val = getattr(obj, f.name), src[f.name]
        if dataclasses.is_dataclass(cur) and isinstance(val, dict):
            _fill(cur, val)
        elif isinstance(cur, tuple) and isinstance(val, list):
            setattr(obj, f.name, tuple(val))
        else:
            setattr(obj, f.name, val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="ucr_011")
    ap.add_argument("--tag", default="zn_a2")
    ap.add_argument("--arm", default="federated_enc_fedavg_bn-shared_prior-partial")
    ap.add_argument("--client", type=int, default=0)
    ap.add_argument("--starts", required=True,
                    help="beginning of the windows, in coordinates of the TEST, separated by comma")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = (REPO / "artifacts" / "runs" / a.tag / "ckpt" / BUILD / a.series / "seed0" /
         a.arm / f"{a.series}_p{a.client}")
    assert (d / "stage2.ckpt").exists(), f"manca {d}"

    z = np.load(d / "scores.npz")
    test_scores = z["test_scores"].astype(float)
    thr = float(json.loads((d / "report.json").read_text())["threshold"])
    flagged_ts = test_scores > thr
    print(f"threshold (train-fitted, paper rule) = {thr:.2f} ·  "
          f"test timestep above threshold = {flagged_ts.mean():.3f}")

    blob = torch.load(d / "stage1.ckpt", map_location="cpu", weights_only=False)
    cfg = Config()
    _fill(cfg, blob["cfg_dict"])
    cfg.dataset.num_workers = 0
    assert cfg.dataset.window_normalization == "zscore", cfg.dataset.window_normalization
    data = make_dataloaders(cfg, stage="eval")
    ds = data.test_dataset

    probe = ds[0]["inputs"][None].float()
    s1 = load_stage1(str(d / "stage1.ckpt"), cfg, probe, device=torch.device("cpu"))
    with torch.no_grad():
        _, _, sp = s1.encode_tokens(probe)
    F_, W_lat = int(sp[0]), int(sp[1])
    del s1
    T = int(ds.window_length)
    edges = np.linspace(0, T, W_lat + 1).round().astype(int)
    max_cols = int(np.floor(MAX_MASKING_RATE * W_lat))
    print(f"window {T} timestep · latex grid {F_}×{W_lat} · "
          f"tetto {max_cols}/{W_lat} colonne ({MAX_MASKING_RATE:.0%})")

    by_start = {int(y.start): i for i, y in enumerate(ds.indices)}
    wanted = [int(s) for s in a.starts.split(",")]

    xs, cols_all, starts_all, capped_all = [], [], [], []
    for s in wanted:
        assert s in by_start, f"no test window starts at {s}"
        item = ds[by_start[s]]
        # Masked column if its range contains timestep over threshold
        col_flag = np.array([flagged_ts[s + edges[c]:s + edges[c + 1]].any()
                             for c in range(W_lat)])
        col_score = np.array([test_scores[s + edges[c]:s + edges[c + 1]].mean()
                              for c in range(W_lat)])
        capped = False
        if col_flag.sum() > max_cols:                    # roof of 90%
            keep = np.argsort(-np.where(col_flag, col_score, -np.inf))[:max_cols]
            new = np.zeros_like(col_flag)
            new[keep] = True
            col_flag, capped = new, True
        if not col_flag.any():
            print(f"  {s} window: no column above threshold, jumped")
            continue
        print(f"  {s} window: {col_flag.sum():>2}/{W_lat} masked columns"
              + ("   <- tetto applicato" if capped else ""))
        xs.append(item["inputs"].float())
        cols_all.append(col_flag)
        starts_all.append(s)
        capped_all.append(capped)

    assert xs, "no window to rewrite"
    x = torch.stack(xs)
    B, C = x.shape[0], x.shape[1]
    mask = torch.zeros((B, C, F_, W_lat), dtype=torch.bool)
    for b, cf in enumerate(cols_all):
        mask[b, :, :, torch.from_numpy(np.flatnonzero(cf))] = True

    system = load_stage2(str(d / "stage2.ckpt"), cfg, str(d / "stage1.ckpt"),
                         probe, device="cpu")
    torch.manual_seed(a.seed)
    out = counterfactual(system, x, token_mask=mask, n_samples=a.draws, greedy=False)
    cf = out["x_cf"].numpy().reshape(B, a.draws, C, T).transpose(1, 0, 2, 3)

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outp, x=x.numpy(), cf=cf.astype(np.float32),
        starts=np.array(starts_all), cols=np.array(cols_all),
        capped=np.array(capped_all), threshold=np.array(thr),
        kind=np.array("explainable"), client=np.array(a.client),
        steps=np.array(int(system.prior.steps)),
    )
    print(f"-> {outp} ({a.draws} drawings × {B} windows, "
          f"decodifica iterativa in {system.prior.steps} passi)")


if __name__ == "__main__":
    main()
