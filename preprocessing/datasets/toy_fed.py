"""Loader for the federated benchmarks — synthetic (``toy_fed*``) and real (``wsd_fed*``).

All of them share one on-disk layout, so one loader serves them (see the dispatch
in ``data.load_records``):

  * ``toy_fed`` / ``toy_fed_full`` / ``toy_fed_t<N>`` — multivariate, fixed C=8,
    clients ``fed_0..``, anomaly mix {level_shift, amp_burst, corr_break}.
    Built by ``scripts/build_toy_fed.py``.
  * ``toy_fed_uni`` / ``toy_fed_uni_full`` — UNIVARIATE (C=1), clients ``uni_00..``
    partitioned into 6 machine-type clusters, anomaly mix {level_shift,
    amp_burst, period_break}. Built by ``scripts/build_toy_fed_uni.py``, which
    also writes ``clusters.json`` and per-cluster public probes.
  * ``wsd_fed`` — UNIVARIATE (C=1), 31 real WSD KPIs in 4 learned clusters, carved
    by ``dataset-federated/scripts/build_frozen.py`` and materialized into this
    layout by ``scripts/build_wsd_fed.py``. Protocol: ``dataset-federated/MODEL.md``.

Same on-disk layout as the toy_*_channel_anomalies family. The synthetic builds add
``test_clean/`` and ``test_mask/`` (ground-truth clean signal + per-channel deviation
mask) for the explainability evaluation; a real build has no clean counterfactual to
compare against, so those directories are absent and the CF metrics are skipped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from data import (
    RecordMetadata,
    TimeSeriesRecord,
    _select_entity,
    y_channel_from_events_csv,
)
from utils import resolve_path


def _load_series(path: Path) -> np.ndarray:
    """Load a series as (T, C) float32, promoting a bare (T,) univariate array.

    The builders always write (T, 1) at C=1, but a hand-rolled or externally
    supplied univariate array would otherwise fail deep in the pipeline —
    `TimeSeriesRecord` rejects a 1-D X, `y_channel_from_events_csv` does
    `x.shape[1]`, and `federated_eval`'s test_mask indexing silently mis-broadcasts.
    Promote here, once, where the shape contract is entered.
    """
    x = np.load(path).astype(np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"{path}: expected (T,) or (T, C), got {x.shape}")
    return x


def _builder_for(name: str) -> str:
    if name.startswith("wsd_fed"):
        return "build_wsd_fed.py"
    if name.startswith("ucr_split"):
        return "build_ucr_split.py"
    if name.startswith(("ucr_pool", "ucr_ad")):
        return "build_ucr_pool.py"
    if name.startswith("toy_fed_uni"):
        return "build_toy_fed_uni.py"
    return "build_toy_fed.py"


def load_fed_records(cfg, split: str) -> list[TimeSeriesRecord]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")

    root = resolve_path(cfg.paths.raw_data) / cfg.dataset.name
    if not (root / "train").exists():
        raise FileNotFoundError(
            f"{cfg.dataset.name} missing at {root}. "
            f"Run `python scripts/{_builder_for(cfg.dataset.name)} --output-dir {root}`."
        )

    available = sorted(p.stem for p in (root / "train").glob("*.npy"))
    entity = _select_entity(available, cfg.dataset.entity_id)

    def _md(n_channels: int) -> RecordMetadata:
        return RecordMetadata(
            dataset=cfg.dataset.name, entity_id=entity,
            feature_names=[f"feature_{i}" for i in range(n_channels)],
        )

    if split == "train":
        x = _load_series(root / "train" / f"{entity}.npy")
        return [TimeSeriesRecord(X=x, metadata=_md(x.shape[1]))]
    if split == "val":
        path = root / "val" / f"{entity}.npy"
        if not path.exists():
            return []
        x = _load_series(path)
        return [TimeSeriesRecord(X=x, metadata=_md(x.shape[1]))]
    x = _load_series(root / "test" / f"{entity}.npy")
    y = np.load(root / "test_label" / f"{entity}.npy").astype(np.int64)
    # y_channel is per-channel ground truth; at C=1 it is just `y` reshaped, so
    # every channel-attribution metric is trivially perfect. metrics_core /
    # detect / per_entity_eval now refuse to score it (see `channel_localization_at_k`).
    # Real builds (wsd_fed) ship no events.csv — the helper returns None and the
    # record degrades to timestamp-only labels, which is all C=1 can support anyway.
    y_ch = y_channel_from_events_csv(root / "events.csv", entity, x.shape[0], x.shape[1])
    return [TimeSeriesRecord(X=x, y=y, y_channel=y_ch, metadata=_md(x.shape[1]))]


# `data.load_records` dispatched here under the old name for the whole toy_fed
# family; keep it resolvable for any out-of-tree caller.
load_toy_fed_records = load_fed_records


def load_clusters(cfg) -> dict[str, list[str]]:
    """`{cluster_name: [entity_id, ...]}` for a clustered build, else `{}`.

    Used by `federated_eval.py --cluster` to scope a federation to one machine type.
    """
    import json
    path = resolve_path(cfg.paths.raw_data) / cfg.dataset.name / "clusters.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
