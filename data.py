"""
=============================================================================
  Data loading, windowing, scaling, and token cache.
=============================================================================

Everything needed to feed stage 1 / stage 2 / evaluation in one place:
  * `TimeSeriesRecord`       — container (X [T, C], y [T], metadata).
  * `load_smap_records()`    — reads the labelled SMAP .npy + labels.csv layout.
  * `load_ciss_records()`    — reads the 4 daily CISS 2019.A1 CSVs.
  * `load_toy_records()`     — reads the synthetic toy_multivariate dataset.
  * `load_wesad_records()`   — reads WESAD chest signals (15 subjects).
  * `PerEntityScaler`        — per-(entity, channel) z-score; fit on TRAIN only,
                                applied in-place to train/val/test BEFORE windowing.
  * `SlidingWindowDataset`   — Torch Dataset emitting raw windows from records
                                (records are pre-normalised by `PerEntityScaler`).
  * `make_dataloaders()`     — builds all three datasets + loaders in one pass.
  * `PrecomputedTokenDataModule` — Stage 2 uses this to skip the stage 1 forward pass.
  * `encode_and_cache_tokens()`  — produced after stage 1 training.
  * `split_train_val()`      — time-respecting train/val split (no shuffle).

Two datasets are implemented; dispatch via `cfg.dataset.name`. Add
more `load_*_records()` functions and extend `load_records()` when needed.

Expected raw layout:
  data/raw/smap_msl/
    labeled_anomalies.csv          (columns: chan_id, spacecraft, anomaly_sequences, ...)
    train/<entity>.npy             (shape [T, C])
    test/<entity>.npy              (shape [T, C])

  data/raw/OneDrive_1_17-03-2026/CISS/CISS 2019/CISS2019.A1/
    27th Aug 2019/CSV/27 August.csv          (~34k rows @ 1Hz, 80 columns + t_stamp)
    28th Aug 2019/CSV/28 August.csv
    29th Aug 2019/CSV/29 August.csv
    30th Aug 2019/CSV/30 August.csv
    CISS2019CombinedDataFile v3.xlsx         (sheet `list of attacks` → 187 launches)

  data/raw/toy_multivariate/                 (produced by scripts/create_toy_dataset.py)
    train/<entity>.npy                       (T, 15) — canonical train split
    val/<entity>.npy                         (T, 15) — canonical val split (used directly)
    train_with_val/<entity>.npy              (T, 15) — concatenation, NOT loaded by the pipeline
    test/<entity>.npy                        (T, 15) — single labelled anomaly per entity
    test_label/<entity>.npy                  (T,)    — int64 0/1
    labels.csv                               — anomaly metadata (informational, not loaded)

  data/raw/WESAD/
    S<id>/S<id>.pkl                          — pickled dict {signal:{chest:{...}}, label, subject}
                                                15 subjects (S2..S17, no S12)
                                                chest signals: ACC(3) + ECG + EMG + EDA + Temp + Resp = 8 chan @ 700 Hz
                                                labels: 0=transient, 1=baseline, 2=TSST=stress (anomaly target),
                                                3=amusement, 4=meditation, 5/6/7=ignored
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config
from utils import resolve_path


# ─── Records & scaling ───────────────────────────────────────────────────────

@dataclass
class RecordMetadata:
    dataset: str
    entity_id: str
    feature_names: list[str] = field(default_factory=list)


@dataclass
class TimeSeriesRecord:
    X: np.ndarray                      # (T, C)
    y: np.ndarray | None = None        # (T,) int labels (only populated on test split)
    y_channel: np.ndarray | None = None  # (T, C) int labels — per-channel anomaly mask; only some datasets surface this
    metadata: RecordMetadata | None = None

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError(f"X must be (T, C); got {self.X.shape}")
        if self.y is not None and self.y.shape[0] != self.X.shape[0]:
            raise ValueError("y must share the leading time dimension with X")
        if self.y_channel is not None and self.y_channel.shape != self.X.shape:
            raise ValueError(
                f"y_channel must match X shape (T, C); got {self.y_channel.shape} vs {self.X.shape}"
            )


class PerEntityScaler:
    """Per-(entity, channel) standard scaler.

    Fit once on TRAIN records (after the val split is peeled off), then apply
    in-place to train / val / test BEFORE windowing. Each entity gets its own
    (mean, std) vector of shape (C,), so channels and entities never share
    statistics. Test/val are transformed with the train statistics — they are
    never re-fitted.
    """

    def __init__(self, min_std: float = 1e-4):
        self.min_std = float(min_std)
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, records: list[TimeSeriesRecord]) -> "PerEntityScaler":
        """Compute per-entity stats. Multiple records sharing the same entity_id are
        concatenated before the moments are taken — this lets a single physical entity
        be split into several `TimeSeriesRecord`s (e.g. CISS 2019: one record per
        no-attack interval) without inventing fake entity ids."""
        by_entity: dict[str, list[np.ndarray]] = {}
        for r in records:
            if r.metadata is None:
                raise ValueError("PerEntityScaler requires record.metadata.entity_id")
            by_entity.setdefault(r.metadata.entity_id, []).append(r.X)
        for entity, arrays in by_entity.items():
            big = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
            mean = big.mean(axis=0).astype(np.float32)              # (C,)
            std = big.std(axis=0).astype(np.float32)                # (C,)
            std = np.maximum(std, self.min_std)
            self.stats[entity] = (mean, std)
        return self

    def transform(self, records: list[TimeSeriesRecord]) -> None:
        """Standardise `record.X` in-place using fitted (mean, std)."""
        for r in records:
            if r.metadata is None:
                raise ValueError("PerEntityScaler requires record.metadata.entity_id")
            stats = self.stats.get(r.metadata.entity_id)
            if stats is None:
                raise KeyError(
                    f"no fitted stats for entity {r.metadata.entity_id!r}; "
                    f"fit on train before transforming val/test"
                )
            mean, std = stats
            r.X = ((r.X - mean) / std).astype(np.float32)


# Split (time-respecting)

def split_train_val(
    records: list[TimeSeriesRecord],
    validation_fraction: float,
    *,
    min_train_length: int = 1,
    min_val_length: int = 1,
) -> tuple[list[TimeSeriesRecord], list[TimeSeriesRecord]]:
    """Peel the last `validation_fraction` of each record off as validation.

    Shuffling is *never* done — this respects temporal order so the validation
    set doesn't accidentally contain the training distribution.
    """
    if not records or validation_fraction <= 0:
        return list(records), []

    train_out: list[TimeSeriesRecord] = []
    val_out: list[TimeSeriesRecord] = []
    for r in records:
        n = r.X.shape[0]
        desired = max(min_val_length, int(round(n * validation_fraction)))
        max_val = n - max(1, min_train_length)
        if max_val < min_val_length:
            train_out.append(r)
            continue
        val_len = min(desired, max_val)
        split = n - val_len
        if split < min_train_length or val_len < min_val_length:
            train_out.append(r)
            continue
        train_out.append(TimeSeriesRecord(
            X=r.X[:split].copy(),
            y_channel=None if r.y_channel is None else r.y_channel[:split].copy(),
            metadata=r.metadata,
        ))
        val_out.append(TimeSeriesRecord(
            X=r.X[split:].copy(),
            y=None if r.y is None else r.y[split:].copy(),
            y_channel=None if r.y_channel is None else r.y_channel[split:].copy(),
            metadata=r.metadata,
        ))
    return train_out, val_out


# ─── SMAP loader ─────────────────────────────────────────────────────────────

def _labels_from_segments(length: int, segments_literal: str) -> np.ndarray:
    labels = np.zeros((length,), dtype=np.int64)
    for start, stop in ast.literal_eval(segments_literal):
        labels[int(start): int(stop) + 1] = 1
    return labels


def _select_entity(available: list[str], requested: str) -> str:
    """Validate that `requested` is a single, known entity id and return it.

    Multi-entity selection is intentionally NOT supported anywhere in this
    pipeline. Training always operates on exactly one time series at a time:
    one model per series, scaler fitted on that series' train segment,
    applied to the same series' val and test segments. Use the unified launcher
    `run.py` to loop over entities and produce one full pipeline (and one model)
    per series."""
    if requested == "all" or "," in requested or requested == "*":
        raise ValueError(
            f"Multi-entity training is not supported. cfg.dataset.entity_id "
            f"must be a single entity id, got {requested!r}. "
            f"Available: {sorted(available)}"
        )
    if requested not in available:
        raise ValueError(
            f"Unknown entity {requested!r}. Available: {sorted(available)}"
        )
    return requested


# Tags that select the whole entity universe of a dataset (pooled training).
POOLED_ENTITY_TAGS: frozenset[str] = frozenset({"all", "pooled", "*"})


def _select_entities(available: list[str], requested: str) -> list[str]:
    """Resolve ``requested`` into a LIST of entity ids — the multi-entity sibling
    of :func:`_select_entity`, used by loaders that support pooled (all-entity)
    training.

    Resolution rules:
      * a pooled tag (``"all"`` / ``"pooled"`` / ``"*"``) → every available entity,
        de-duplicated and sorted (deterministic order);
      * a comma-separated list (``"A-1,A-2"``) → exactly those ids (validated);
      * a single id (``"A-1"``) → ``["A-1"]`` (validated).

    The single-id path returns a 1-element list, so a loader that loops over the
    result behaves **bit-identically** to the old single-entity path when a plain
    id is requested. Each returned id still maps to its own ``TimeSeriesRecord``
    tagged with that id, so ``PerEntityScaler`` keeps standardising per entity.
    """
    uniq = sorted(dict.fromkeys(str(e) for e in available))
    if requested in POOLED_ENTITY_TAGS:
        if not uniq:
            raise ValueError("Pooled selection requested but no entities available.")
        return uniq
    if "," in requested:
        ids = [e.strip() for e in requested.split(",") if e.strip()]
        unknown = [e for e in ids if e not in uniq]
        if unknown:
            raise ValueError(f"Unknown entities {unknown}. Available: {uniq}")
        return list(dict.fromkeys(ids))
    if requested not in uniq:
        raise ValueError(f"Unknown entity {requested!r}. Available: {uniq}")
    return [requested]


def y_channel_from_events_csv(
    csv_path: Path, entity_id: str, T: int, n_channels: int,
) -> np.ndarray | None:
    """Materialize per-channel labels `(T, n_channels)` int64 from `events.csv`.

    Expected schema (toy_*_channel_anomalies family):
      entity_id, test_anomaly_start, test_anomaly_stop_exclusive,
      anomaly_channels (JSON list of int)

    Returns:
      np.ndarray of shape (T, n_channels), dtype int64, with 1 at
      [start:stop, ch] for every (event, ch) pair belonging to `entity_id`.

      `None` when the CSV is missing or has no rows for `entity_id` —
      lets callers degrade gracefully to timestamp-only labels.
    """
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df[df["entity_id"] == entity_id]
    if df.empty:
        return None
    y_ch = np.zeros((T, n_channels), dtype=np.int64)
    for _, row in df.iterrows():
        start = int(row["test_anomaly_start"])
        stop = int(row["test_anomaly_stop_exclusive"])
        chs = json.loads(row["anomaly_channels"])
        for c in chs:
            if 0 <= int(c) < n_channels:
                y_ch[start: stop, int(c)] = 1
    return y_ch


def list_entities(cfg: Config) -> list[str]:
    """Enumerate the entity ids available on disk for `cfg.dataset.name`.

    Single source of truth for the run launchers (`run_*/train.{sh,ps1}`):
    instead of hard-coding 55 SMAP / 27 MSL ids in shell, they call
    `python scripts/list_entities.py --dataset <name>` which delegates here.
    Mirrors exactly the discovery logic each loader uses to build its own
    `available` list, so the launcher and the loader can never disagree.

    Every live dataset (`toy_fed*`, `wsd_fed*`, `ucr_pool*`, `ucr_ad*`) shares one
    on-disk layout: a `train/<entity>.npy` per entity.
    """
    name = cfg.dataset.name
    train_dir = resolve_path(cfg.paths.raw_data) / name / "train"
    if train_dir.is_dir():
        return sorted(p.stem for p in train_dir.glob("*.npy"))
    raise ValueError(
        f"list_entities: don't know how to enumerate entities for dataset {name!r} "
        f"(no {train_dir} with *.npy)."
    )


def load_records(cfg: Config, split: str) -> list[TimeSeriesRecord]:
    """Dispatcher. `split` is 'train' | 'val' | 'test'. Datasets without a
    pre-built val on disk return [] for split='val' — `make_dataloaders` /
    `load_scaled_records` then fall back to peeling val from train via
    `validation_fraction`. Extend here if you add another dataset."""
    # One on-disk layout serves every federated benchmark: per-entity .npy under
    # train/ val/ test/ test_label/ plus clusters.json. `toy_fed*` is synthetic,
    # `wsd_fed*` is the frozen real WSD build (scripts/build_wsd_fed.py),
    # `ucr_pool*` is the public UCR PRETRAINING corpus (train-capped, test is a
    # vestigial stub), while `ucr_ad*` is the same archive built with the NATIVE UCR
    # protocol — full series, train=[0:trainEnd], test=[trainEnd:end], real labels —
    # i.e. the one to use for anomaly DETECTION. Both via scripts/build_ucr_pool.py.
    # `ucr_split*` re-cuts that same native protocol into a QUANTITY-SKEW federation:
    # one cluster per series, its clients holding disjoint 10/10/20/20/30% slices of
    # that series' train, sharing its val and test. Via scripts/build_ucr_split.py.
    if cfg.dataset.name.startswith(("toy_fed", "wsd_fed", "ucr_pool", "ucr_ad", "ucr_split")):
        from preprocessing.datasets.toy_fed import load_fed_records
        return load_fed_records(cfg, split)
    raise ValueError(f"Unsupported dataset: {cfg.dataset.name!r}")


# ─── Windowing ───────────────────────────────────────────────────────────────

@dataclass
class _WindowIndex:
    record_index: int
    start: int
    stop: int


class SlidingWindowDataset(Dataset):
    """Emits windows of shape (C, window_length) from pre-normalised records.

    Records are expected to already be standardised (e.g. by `PerEntityScaler`)
    before being passed here. The optional `window_normalization` argument
    layers a per-window z-norm on top — paper-style — for experiments that want
    each input window self-normalised regardless of the record-level scaler.
    """

    def __init__(
        self,
        records: list[TimeSeriesRecord],
        window_length: int,
        stride: int,
        window_normalization: str = "none",
    ):
        if window_normalization not in {"none", "zscore"}:
            raise ValueError(
                f"window_normalization must be 'none' or 'zscore', got {window_normalization!r}"
            )
        self.records = records
        self.window_length = window_length
        self.stride = stride
        self.window_normalization = window_normalization
        self.indices: list[_WindowIndex] = []
        for ri, record in enumerate(records):
            T = record.X.shape[0]
            if T < window_length:
                continue
            for start in range(0, T - window_length + 1, stride):
                self.indices.append(_WindowIndex(ri, start, start + window_length))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        wi = self.indices[idx]
        record = self.records[wi.record_index]

        x_np = record.X[wi.start: wi.stop]                  # (window, C)
        x = torch.from_numpy(np.asarray(x_np, dtype=np.float32).T)  # (C, window)

        if self.window_normalization == "zscore":
            mu = x.mean(dim=-1, keepdim=True)
            sigma = x.std(dim=-1, keepdim=True).clamp_min(1e-4)
            x = (x - mu) / sigma

        if record.y is None:
            labels = torch.full((self.window_length,), -1, dtype=torch.long)
        else:
            labels = torch.from_numpy(np.asarray(record.y[wi.start: wi.stop], dtype=np.int64))

        md = {
            "dataset": "unknown" if record.metadata is None else record.metadata.dataset,
            "entity_id": "unknown" if record.metadata is None else record.metadata.entity_id,
            "record_index": wi.record_index,
            "window_start": wi.start,
            "window_stop": wi.stop,
        }
        return {
            "inputs": x,
            "labels": labels,
            "metadata": md,
        }


# ─── Dataloaders bundle ──────────────────────────────────────────────────────

@dataclass
class Dataloaders:
    """Flat container holding train/val/test records, datasets, and loaders."""
    train_records: list[TimeSeriesRecord]
    val_records: list[TimeSeriesRecord]
    test_records: list[TimeSeriesRecord]
    train_dataset: SlidingWindowDataset
    val_dataset: SlidingWindowDataset
    test_dataset: SlidingWindowDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def _build_loader(dataset: SlidingWindowDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        # Pinning speeds up the H2D copy (and is what makes `non_blocking=True` in the
        # training loops actually overlap). It works in the main process too, so it is
        # NOT gated on num_workers > 0 as it used to be.
        "pin_memory": torch.cuda.is_available(),
        "shuffle": shuffle,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4          # keep the input pipeline ahead of the GPU
    return DataLoader(dataset, **kwargs)


def load_scaled_records(
    cfg: Config,
) -> tuple[list[TimeSeriesRecord], list[TimeSeriesRecord], list[TimeSeriesRecord]]:
    """Load train/val/test records and apply `PerEntityScaler` (fit on train).

    Val resolution: prefer the dataset's pre-built val (`load_records(cfg, "val")`)
    when it returns a non-empty list; otherwise peel val from train via
    `validation_fraction`. This is the same logic used by `make_dataloaders`.
    """
    train_records = load_records(cfg, "train")
    val_records = load_records(cfg, "val")
    test_records = load_records(cfg, "test")

    if val_records:
        print(f"[data] using pre-built val from disk ({len(val_records)} record(s)); "
              f"validation_fraction={cfg.dataset.validation_fraction} is ignored.")
    elif cfg.dataset.validation_fraction > 0:
        train_records, val_records = split_train_val(
            train_records, cfg.dataset.validation_fraction,
            min_train_length=cfg.dataset.window_length,
            min_val_length=cfg.dataset.window_length,
        )

    if cfg.dataset.scaling != "none":
        scaler = PerEntityScaler().fit(train_records)
        scaler.transform(train_records)
        if val_records:
            scaler.transform(val_records)
        scaler.transform(test_records)

    return train_records, val_records, test_records


def make_dataloaders(cfg: Config, stage: str = "stage1") -> Dataloaders:
    """Load records → split train/val → fit per-entity scaler on train →
    apply to train/val/test → build 3 datasets + 3 DataLoaders."""
    window_length = cfg.dataset.window_length
    stride = cfg.dataset.window_stride
    num_workers = int(os.environ.get("DEBUG_NUM_WORKERS", cfg.dataset.num_workers))
    batch_size = {
        "stage1": cfg.dataset.batch_size_stage1,
        "stage2": cfg.dataset.batch_size_stage2,
        "eval":   cfg.dataset.batch_size_eval,
    }.get(stage, cfg.dataset.batch_size_eval)

    train_records = load_records(cfg, "train")
    val_records = load_records(cfg, "val")
    test_records = load_records(cfg, "test")

    if val_records:
        print(f"[data] using pre-built val from disk ({len(val_records)} record(s)); "
              f"validation_fraction={cfg.dataset.validation_fraction} is ignored.")
    elif cfg.dataset.validation_fraction > 0:
        train_records, val_records = split_train_val(
            train_records, cfg.dataset.validation_fraction,
            min_train_length=window_length,
            min_val_length=window_length,
        )

    if not val_records:
        # Never fall back to test for model selection — copy train instead,
        # and warn loudly so the user knows selection metrics are optimistic.
        print("[data] WARNING: validation split is empty — using a copy of train. "
              "Early-stopping metrics will be optimistic.")
        from copy import deepcopy
        val_records = deepcopy(train_records)

    # Fit on TRAIN only, then apply train/val/test in-place BEFORE windowing.
    if cfg.dataset.scaling != "none":
        scaler = PerEntityScaler().fit(train_records)
        scaler.transform(train_records)
        scaler.transform(val_records)
        scaler.transform(test_records)

    win_norm = cfg.dataset.window_normalization
    train_dataset = SlidingWindowDataset(train_records, window_length, stride, win_norm)
    val_dataset   = SlidingWindowDataset(val_records,   window_length, stride, win_norm)
    test_dataset  = SlidingWindowDataset(test_records,  window_length, stride, win_norm)

    return Dataloaders(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=_build_loader(train_dataset, batch_size, num_workers, shuffle=True),
        val_loader=_build_loader(val_dataset,     batch_size, num_workers, shuffle=False),
        test_loader=_build_loader(test_dataset,   batch_size, num_workers, shuffle=False),
    )


# Token cache for internship 2

CACHE_FILENAME = "token_cache.pt"


def _tokens_config_hash(cfg: Config) -> str:
    """Deterministic hash of the subset of config that affects tokenisation.

    Stage 2 uses this to detect a stale cache (e.g. the user retrained stage 1
    with a different quantiser but forgot to refresh). `training` is included
    because changes to lr / max_steps / patience produce different stage-1
    weights and therefore different tokens, even with the same architecture.
    """
    from dataclasses import asdict
    _training = asdict(cfg.training)
    # `amp` is a training-SPEED flag: fp16 and fp32 produce ~identical argmax token
    # indices, so flipping it must NOT invalidate a cache produced by stage 1. Popping
    # it also keeps this hash byte-identical to the pre-AMP one, so existing caches on
    # disk stay valid. Stage-1 knobs (lr, warmup, stage1_*) stay in — they DO change the
    # stage-1 weights and therefore the tokens.
    for _k in ("amp",):
        _training.pop(_k, None)
    subset = {
        "dataset":   asdict(cfg.dataset),
        "transform": asdict(cfg.transform),
        "encoder":   asdict(cfg.encoder),
        "quantizer": asdict(cfg.quantizer),
        "decoder":   asdict(cfg.decoder),
        "training":  _training,
        "seed":      cfg.seed,
    }
    # Knobs added AFTER caches were already written to disk. Every one of them does
    # change the stage-1 weights (hence the tokens) when flipped, so they must be in
    # the hash — but including them unconditionally would rewrite the hash of every
    # run that never touches them and invalidate the whole cache population for a
    # value that is not actually different. So: hash them only when they are OFF the
    # default. A default-valued field is dropped, reproducing the pre-existing hash
    # byte-for-byte; any other value changes it, which is exactly the invalidation we
    # want. Extend this dict, never the `subset` above, when adding such a knob.
    _POST_HOC_DEFAULTS = {
        "quantizer": {"kmeans_init": True},
        "training":  {"keep_last_weights": False},
        "dataset":   {"pool_val_into_train": False},
    }
    for _sec, _fields in _POST_HOC_DEFAULTS.items():
        for _k, _default in _fields.items():
            if subset[_sec].get(_k, _default) == _default:
                subset[_sec].pop(_k, None)
    raw = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@torch.no_grad()
def encode_and_cache_tokens(
    stage1: Any,                                         # Stage1VQVAE (plain nn.Module)
    train_loader: DataLoader,
    val_loader: DataLoader,
    cache_path: Path,
    cfg: Config,
    device: str | torch.device = "cpu",
) -> Path:
    """Encode every train/val window through the frozen stage 1 and save tokens.

    Produces a `.pt` with keys: train_tokens (N_train, seq_len), val_tokens, and
    the latent spatial shape so stage 2 can reshape for 2D-pos priors.
    """
    stage1 = stage1.to(device)
    stage1.eval()

    def _encode(loader: DataLoader) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for batch in loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            _, indices, _ = stage1.encode_tokens(inputs)
            if indices.ndim == 2:
                flat = indices
            else:
                flat = indices.reshape(indices.shape[0], -1)
            chunks.append(flat.cpu().long())
        return torch.cat(chunks, dim=0) if chunks else torch.empty(0, dtype=torch.long)

    print("[data] encoding train tokens...")
    train_tokens = _encode(train_loader)
    print(f"       → {train_tokens.shape}")
    print("[data] encoding val tokens...")
    val_tokens = _encode(val_loader)
    print(f"       → {val_tokens.shape}")

    # Capture latent spatial shape from one forward pass — needed by 2D-pos priors.
    sample_batch = next(iter(train_loader))
    sample_inputs = sample_batch["inputs"][:1].to(device)
    _, _, latent_spatial = stage1.encode_tokens(sample_inputs)

    # Tokens are codebook indices (≤ cb_size + mask, typically ≤ 64). Storing them as
    # int64 wastes 4-8× disk AND RAM. Downcast to the smallest safe integer dtype;
    # consumers cast per-batch via `.long()` (pipeline/stage2.py:176,159) and
    # nn.Embedding would reject a wrong dtype loudly, so this cannot fail silently.
    def _compact(t: torch.Tensor) -> torch.Tensor:
        if t.numel() == 0:
            return t.to(torch.int16)
        return t.to(torch.uint8 if int(t.max()) <= 255 else torch.int16)

    payload = {
        "train_tokens": _compact(train_tokens),
        "val_tokens":   _compact(val_tokens),
        "token_shape_info": {
            "seq_len":        int(train_tokens.shape[1]) if train_tokens.ndim > 1 else 0,
            "latent_spatial": tuple(int(s) for s in latent_spatial),
        },
        "config_hash": _tokens_config_hash(cfg),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a kill mid-save must not leave a truncated cache that the loader
    # reads as valid (or that crashes stage2 with no fallback).
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)
    print(f"[data] token cache saved to {cache_path}")
    return cache_path


class _TokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor):
        if tokens.ndim != 2:
            raise ValueError(f"Token cache must be (N, seq_len); got {tokens.shape}")
        self.tokens = tokens

    def __len__(self) -> int: return self.tokens.shape[0]
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: return {"tokens": self.tokens[idx]}


class PrecomputedTokenDataModule:
    """Serves the saved tokens — zero stage 1 forward passes during stage 2.

    Plain container; call `.setup("fit")` once after construction.
    """

    def __init__(self, cache_path: Path | str, cfg: Config):
        self.cache_path = Path(cache_path)
        self.cfg = cfg
        self.batch_size = cfg.dataset.batch_size_stage2
        self.num_workers = int(os.environ.get("DEBUG_NUM_WORKERS", cfg.dataset.num_workers))
        self.pin_memory = self.num_workers > 0
        self.train_dataset: _TokenDataset | None = None
        self.val_dataset: _TokenDataset | None = None
        self.token_shape_info: dict[str, Any] = {}

    def setup(self, stage: str | None = None) -> None:
        payload = torch.load(self.cache_path, map_location="cpu")
        expected = _tokens_config_hash(self.cfg)
        stored = payload.get("config_hash")
        if stored and stored != expected:
            # Hard failure: silently using mismatched tokens trains stage 2 on
            # the wrong codebook. Delete the stale cache and rerun stage 1.
            raise RuntimeError(
                f"Token cache hash mismatch at {self.cache_path}: "
                f"stored={stored}, current={expected}. The cache was produced "
                f"by a stage-1 run with different dataset/transform/encoder/"
                f"quantizer/decoder/training/seed than the current cfg. Delete "
                f"the cache or rerun stage 1 to refresh it."
            )
        self.train_dataset = _TokenDataset(payload["train_tokens"])
        self.val_dataset = _TokenDataset(payload["val_tokens"])
        self.token_shape_info = payload.get("token_shape_info", {})
        print(f"[data] loaded token cache: {len(self.train_dataset)} train, "
              f"{len(self.val_dataset)} val (seq_len={self.token_shape_info.get('seq_len', -1)})")

    def _loader(self, dataset: _TokenDataset, shuffle: bool) -> DataLoader:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "shuffle": shuffle,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader: return self._loader(self.train_dataset, True)
    def val_dataloader(self)   -> DataLoader: return self._loader(self.val_dataset, False)
