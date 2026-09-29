"""
=============================================================================
  list_entities.py — print the entity ids available on disk for a dataset.
=============================================================================

Thin CLI over `data.list_entities()`. The unified launcher (`run.py`) calls
this to expand `--entities all` into the real per-entity list, so the entity
universe lives in exactly one place (the dataset loaders) and never drifts from
what training actually reads.

Usage:
    python scripts/list_entities.py --dataset smap
    python scripts/list_entities.py --dataset toy_basic_channel_anomalies_32k

Prints one entity id per line on stdout (nothing else), so shell can do:
    mapfile -t ENTS < <(python scripts/list_entities.py --dataset "$DS")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from data import list_entities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="DATASET_NAME, e.g. smap / msl / toy_point_channel_anomalies_32k")
    args = parser.parse_args()

    cfg = load_config()
    cfg.dataset.name = args.dataset
    for entity in list_entities(cfg):
        print(entity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
