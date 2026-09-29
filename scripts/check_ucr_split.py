#!/usr/bin/env python3
"""check_ucr_split.py — re-verify a built `ucr_split` against the raw UCR archive.

`build_ucr_split.py` asserts the partition property on the arrays it holds in
memory; this script asserts it on the bytes that actually landed on disk, reading
the source .txt back independently. Checks, per series:

  1. concat(train shards, in partition order) + val  ==  the original train segment
     (exact equality — no overlap, no dropped sample, order preserved)
  2. shard lengths match the requested share percentages
  3. test  ==  values[train_len:] verbatim, and is IDENTICAL across the cluster
  4. test_label ==  the archive's anomaly window, identical across the cluster
  5. val is identical across the cluster and disjoint from every shard
  6. clusters.json / metadata.json agree with the files on disk

Usage:
    python scripts/check_ucr_split.py
    python scripts/check_ucr_split.py --root data/raw/ucr_split_dev
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.build_ucr_split import (  # noqa: E402
    DEFAULT_UCR_DIR, UCR_NAME_RE, anomaly_label, read_ucr_values,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "data" / "raw" / "ucr_split")
    ap.add_argument("--ucr-dir", type=Path, default=DEFAULT_UCR_DIR)
    args = ap.parse_args()

    root = args.root
    meta = json.loads((root / "metadata.json").read_text())
    clusters = json.loads((root / "clusters.json").read_text())
    by_eid = {e["entity_id"]: e for e in meta["entities"]}
    shares = meta["protocol"]["shares_pct"]
    val_pct = meta["protocol"]["val_pct"]
    train_cap = meta["protocol"]["train_cap"]
    test_cap = meta["protocol"]["test_cap"]

    # index the archive by the 3-digit id in the filename
    src: dict[str, Path] = {}
    for p in sorted(Path(args.ucr_dir).glob("*.txt")):
        m = UCR_NAME_RE.match(p.name)
        if m:
            src[f"ucr_{m.group(1)}"] = p

    on_disk = sorted(p.stem for p in (root / "train").glob("*.npy"))
    listed = sorted(e for members in clusters.values() for e in members)
    errs: list[str] = []
    if on_disk != listed:
        errs.append(f"train/*.npy ({len(on_disk)}) != clusters.json members ({len(listed)})")
    if sorted(by_eid) != on_disk:
        errs.append("metadata.json entities != train/*.npy on disk")

    for series, members in sorted(clusters.items()):
        p = src[series]
        m = UCR_NAME_RE.match(p.name)
        train_len, a0, a1 = int(m.group(3)), int(m.group(4)), int(m.group(5))
        values = read_ucr_values(p)
        train_full = values[: train_len if train_cap is None else min(train_len, train_cap)]
        test_ref = values[train_len:] if test_cap is None else values[train_len: train_len + test_cap]
        label_ref = anomaly_label(test_ref.shape[0], train_len, a0, a1)

        if len(members) != len(shares):
            errs.append(f"{series}: {len(members)} clients, expected {len(shares)}")
            continue

        shards = [np.load(root / "train" / f"{e}.npy")[:, 0] for e in members]
        vals = [np.load(root / "val" / f"{e}.npy")[:, 0] for e in members]
        tests = [np.load(root / "test" / f"{e}.npy")[:, 0] for e in members]
        labels = [np.load(root / "test_label" / f"{e}.npy") for e in members]

        # (5) val/test/label shared verbatim across the cluster
        for name, arrs in (("val", vals), ("test", tests), ("test_label", labels)):
            if not all(np.array_equal(arrs[0], a) for a in arrs[1:]):
                errs.append(f"{series}: {name} differs across clients (must be shared)")

        # (1) exact partition of the original train
        rebuilt = np.concatenate(shards + [vals[0]])
        if rebuilt.shape != train_full.shape or not np.array_equal(rebuilt, train_full):
            errs.append(f"{series}: shards+val ({rebuilt.shape[0]}) != original train "
                        f"({train_full.shape[0]}) — NOT a partition")

        # (2) share percentages, and (5) the offsets recorded in metadata
        T = int(train_full.shape[0])
        off = 0
        for e, sh, share in zip(members, shards, shares):
            md = by_eid[e]
            if (md["train_start"], md["train_stop"]) != (off, off + sh.shape[0]):
                errs.append(f"{e}: metadata offsets {md['train_start']}:{md['train_stop']} "
                            f"!= actual {off}:{off + sh.shape[0]}")
            if not np.array_equal(sh, train_full[off: off + sh.shape[0]]):
                errs.append(f"{e}: shard bytes != train_full[{off}:{off + sh.shape[0]}]")
            got = 100.0 * sh.shape[0] / T
            if abs(got - share) > 0.5:                 # integer cut-points, <=1 sample slack
                errs.append(f"{e}: holds {got:.2f}% of train, requested {share}%")
            off += sh.shape[0]
        got_val = 100.0 * vals[0].shape[0] / T
        if abs(got_val - val_pct) > 0.5:
            errs.append(f"{series}: val is {got_val:.2f}% of train, requested {val_pct}%")

        # (3)(4) test untouched
        if not np.array_equal(tests[0], test_ref):
            errs.append(f"{series}: test != values[{train_len}:] — the test was modified")
        if not np.array_equal(labels[0], label_ref):
            errs.append(f"{series}: test_label != archive anomaly window")
        if labels[0].sum() == 0:
            errs.append(f"{series}: test carries no anomaly point")

    n_ent = len(listed)
    if errs:
        print(f"[check_ucr_split] FAIL — {len(errs)} problem(s) over "
              f"{len(clusters)} series / {n_ent} entities:")
        for e in errs[:40]:
            print("   -", e)
        if len(errs) > 40:
            print(f"   ... and {len(errs) - 40} more")
        return 1
    print(f"[check_ucr_split] OK — {len(clusters)} series x {len(shares)} clients "
          f"= {n_ent} entities")
    print(f"[check_ucr_split]   train is an exact partition ({'/'.join(map(str, shares))}% "
          f"+ {val_pct:g}% val) of every series' original train")
    print(f"[check_ucr_split]   test == values[train_len:] verbatim and shared within each cluster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
