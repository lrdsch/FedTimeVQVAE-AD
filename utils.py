"""
=============================================================================
  Utils — small helpers used by every stage.
=============================================================================

  * `project_root()` / `resolve_path()` — turn relative paths into absolute.
  * `run_dir_for()` / `checkpoint_dir_for()` — canonical output layout for a Config.
  * `seed_everything()` — seed python/numpy/torch.
  * `save_loss_plots()` — parse Lightning's metrics.csv into loss curves.
  * `configure_logging()` — one-line info logging.

Nothing fancy. No registries, no portable-path rehydration.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np

from config import Config, run_name


# ─── Console ─────────────────────────────────────────────────────────────────

def force_utf8_stdout() -> None:
    """Make stdout/stderr UTF-8 so a print can't kill a finished run.

    Summary tables and test banners print '±', '↑', 'Σ', 'Δ'. On Windows, Python's
    stdout falls back to the locale encoding (cp1252) under ANY redirect or
    non-interactive shell, and those characters raise UnicodeEncodeError —
    typically *after* the expensive work has already completed. The logs under
    logs/ are already UTF-8, so this also keeps new logs byte-compatible with
    them. `errors="replace"` is the last resort for streams that refuse to
    reconfigure (e.g. a wrapped/captured stream)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


# ─── Paths ───────────────────────────────────────────────────────────────────

def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else project_root() / p


def resolve_device() -> "torch.device":
    """The device this process should train/score on.

    OPT-IN multi-GPU, default-off: with `TVQ_DEVICE` unset this returns exactly what
    the hardcoded `torch.device("cuda" if torch.cuda.is_available() else "cpu")` used
    to return, so behaviour is unchanged.

    Set `TVQ_DEVICE=cuda:1` (or `cpu`) to pin a DIRECT script run to one GPU. This is
    for direct `python pipeline/stage1.py` invocations only — when launched through
    `run.py`, each subprocess is already pinned via `CUDA_VISIBLE_DEVICES`, which
    REMAPS device indices, so a `TVQ_DEVICE=cuda:1` inside a subprocess masked to a
    single GPU would be out of range. run.py deliberately does not set it.
    """
    import os
    import torch
    spec = os.environ.get("TVQ_DEVICE", "").strip()
    if spec:
        dev = torch.device(spec)
        if dev.type == "cuda" and not torch.cuda.is_available():
            print(f"[utils] TVQ_DEVICE={spec!r} but CUDA is unavailable — falling back to cpu.")
            return torch.device("cpu")
        if dev.type == "cuda" and dev.index is not None and dev.index >= torch.cuda.device_count():
            raise ValueError(
                f"TVQ_DEVICE={spec!r} but only {torch.cuda.device_count()} CUDA device(s) "
                f"are visible (CUDA_VISIBLE_DEVICES remaps indices)."
            )
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dir_for(cfg: Config, stage: str) -> Path:
    """`<runs>/<stage>/<dataset>/<entity>/<transform>/.../<seed>/`"""
    return resolve_path(cfg.paths.runs) / stage / run_name(cfg)


def checkpoint_dir_for(cfg: Config, stage: str) -> Path:
    return run_dir_for(cfg, stage) / "checkpoints"


def best_checkpoint(cfg: Config, stage: str) -> Path:
    """Return `best.ckpt` if it exists, else fall back to `last.ckpt`.

    `best.ckpt` is only written once a validation epoch has reported
    `val/loss`. Short runs (max_steps < steps_per_epoch, or
    check_val_every_n_epoch > epochs reached) may finish before any
    validation epoch completes, leaving only `last.ckpt`.
    """
    ckpt_dir = checkpoint_dir_for(cfg, stage)
    best = ckpt_dir / "best.ckpt"
    if best.exists():
        return best
    return ckpt_dir / "last.ckpt"


def token_cache_path(cfg: Config) -> Path:
    return run_dir_for(cfg, "stage1") / "token_cache.pt"


# ─── Seeding / logging ───────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def configure_logging(level: int = logging.INFO) -> None:
    # Windows console defaults to cp1252 which can't encode Unicode arrows
    # / em-dashes used throughout our print statements. Force UTF-8 so a
    # benign log line never crashes a long training run.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ─── Loss-curve plotting from Lightning's metrics.csv ────────────────────────

def save_loss_plots(metrics_csv: Path, output_dir: Path, stage_label: str) -> None:
    """Parse the Lightning CSV logger output and save train/val loss curves.

    Two passes are produced for every plot:
      * full range (`*.png`)
      * second half only (`*_late.png`, epoch ≥ max_epoch / 2) so the late-stage
        convergence isn't dwarfed visually by the early transient.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    if not metrics_csv.exists():
        return
    df = pd.read_csv(metrics_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    _render_loss_plots(df, output_dir, stage_label, suffix="")

    # Second-half pass: zoom in on the tail of training where the curves often
    # flatten. Only render if there are enough epochs for the cut to be useful.
    if "epoch" not in df.columns:
        return
    epochs = df["epoch"].dropna()
    if epochs.empty:
        return
    max_epoch = int(epochs.max())
    cutoff = max_epoch // 2
    df_late = df[df["epoch"] >= cutoff]
    if len(df_late) >= 2 and cutoff > 0:
        _render_loss_plots(
            df_late, output_dir,
            f"{stage_label} (epoch ≥ {cutoff})",
            suffix="_late",
        )


def _render_loss_plots(df, output_dir: Path, stage_label: str, suffix: str) -> None:
    """All the actual plotting, parameterised by DataFrame and filename suffix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _series(col_name: str) -> tuple[list, list]:
        if col_name not in df.columns:
            return [], []
        sub = df[["epoch", col_name]].dropna()
        return sub["epoch"].tolist(), sub[col_name].tolist()

    train_x, train_y = _series("train/loss")
    val_x, val_y = _series("val/loss")

    fig, ax = plt.subplots(figsize=(10, 5))
    if train_y:
        ax.plot(train_x, train_y, label="train", color="steelblue")
    if val_y:
        ax.plot(val_x, val_y, label="val", color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{stage_label} — loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"loss{suffix}.png", dpi=150)
    plt.close(fig)

    if train_y or val_y:
        fig, ax = plt.subplots(figsize=(10, 5))
        if train_y:
            ax.plot(train_x, train_y, label="train", color="steelblue")
        if val_y:
            ax.plot(val_x, val_y, label="val", color="darkorange")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log)")
        ax.set_title(f"{stage_label} — loss (log y)")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"loss_logy{suffix}.png", dpi=150)
        plt.close(fig)

    def _loss_components() -> list[str]:
        seen: set[str] = set()
        for col in df.columns:
            for prefix in ("train/", "val/"):
                if col.startswith(prefix) and not col.endswith("_step"):
                    name = col[len(prefix):]
                    if "loss" in name.lower():
                        seen.add(name)
        return sorted(seen, key=_loss_sort_key)

    components = _loss_components()
    if components:
        def _render_stacked(log_y: bool, filename: str) -> None:
            n = len(components)
            fig, axes = plt.subplots(
                n, 1, figsize=(11, max(3, 1.8 * n)),
                sharex=True, sharey=True,
            )
            axes = axes if n > 1 else [axes]
            for ax, name in zip(axes, components):
                tx, ty = _series(f"train/{name}")
                vx, vy = _series(f"val/{name}")
                if ty: ax.plot(tx, ty, label="train", color="steelblue", linewidth=1.0)
                if vy: ax.plot(vx, vy, label="val", color="darkorange", linewidth=1.0)
                if log_y: ax.set_yscale("log")
                ax.set_ylabel(name, fontsize=8)
                ax.grid(alpha=0.3, which="both" if log_y else "major")
            axes[0].legend(loc="upper right", fontsize=8)
            axes[-1].set_xlabel("Epoch")
            log_tag = " (log y)" if log_y else ""
            fig.suptitle(f"{stage_label} — train vs validation losses, stacked{log_tag}", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(output_dir / filename, dpi=150)
            plt.close(fig)

        _render_stacked(log_y=False, filename=f"train_validation_losses_stacked{suffix}.png")
        _render_stacked(log_y=True,  filename=f"train_validation_losses_stacked_logy{suffix}.png")

    main_losses = ["loss", "loss_time", "loss_spec", "loss_vq"]
    gap_components = [
        n for n in main_losses
        if f"train/{n}" in df.columns and f"val/{n}" in df.columns
    ]
    if gap_components:
        n = len(gap_components)
        fig, axes = plt.subplots(
            n, 1, figsize=(11, max(3, 1.8 * n)),
            sharex=True, sharey=True,
        )
        axes = axes if n > 1 else [axes]
        for ax, name in zip(axes, gap_components):
            merged = (
                df[["epoch", f"train/{name}", f"val/{name}"]]
                .dropna()
                .sort_values("epoch")
            )
            if merged.empty:
                continue
            gap = merged[f"val/{name}"].to_numpy() - merged[f"train/{name}"].to_numpy()
            ax.plot(merged["epoch"], gap, color="firebrick", linewidth=1.0)
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.7)
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_ylabel(f"{name}\n(val − train)", fontsize=8)
            ax.grid(alpha=0.3, which="both")
        axes[-1].set_xlabel("Epoch")
        fig.suptitle(
            f"{stage_label} — validation minus train, stacked (symlog y)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output_dir / f"train_validation_loss_gap_stacked_logy{suffix}.png", dpi=150)
        plt.close(fig)


# Ordering convention for the per-component stacked loss plots:
#   1. headline `loss` (total, only one row)
#   2. main components: loss_time, loss_spec, loss_vq
#   3. backward-compat aliases: reconstruction_loss, quantizer_loss
#   4. worst-channel summary
#   5. per-channel time/spec losses, sorted numerically (ch0, ch1, …)
#   6. codebook commitment / weighted commitment, sorted by codebook index
#   7. everything else, alphabetical
import re as _re

_FIXED_LOSS_ORDER: dict[str, tuple[int, int]] = {
    "loss":                (0, 0),
    "loss_time":           (0, 1),
    "loss_spec":           (0, 2),
    "loss_vq":             (0, 3),
    "reconstruction_loss": (0, 4),
    "quantizer_loss":      (0, 5),
    "loss_time_worst_ch":  (0, 6),
}


def _loss_sort_key(name: str) -> tuple:
    if name in _FIXED_LOSS_ORDER:
        tier, idx = _FIXED_LOSS_ORDER[name]
        return (tier, idx, 0, name)
    m = _re.match(r"^(loss_time|loss_spec)_ch(\d+)$", name)
    if m:
        family_idx = {"loss_time": 0, "loss_spec": 1}[m.group(1)]
        return (1, family_idx, int(m.group(2)), name)
    m = _re.match(r"^codebook_(\d+)_(.+)$", name)
    if m:
        return (2, int(m.group(1)), 0, m.group(2))
    return (3, 0, 0, name)


# GPU peak telemetry (ported from -Real)
def log_gpu_peak(stage: str, batch_size: int, cfg: Config | None = None,
                 reset: bool = False) -> None:
    """Record peak CUDA memory for a training/eval pass to stdout + logs/vram.csv.

    Call ONCE at the end of a stage (max_memory_allocated is a running peak over
    the whole process). The CSV row carries the batch size actually used so the
    profiles in config.load_config() can be re-tuned afterwards: read
    `logs/vram.csv`, see how close `util_pct` is to 100%, raise/lower the batch.
    No-op (and never raises) when CUDA is unavailable. `reset=True` clears the
    peak afterwards so a later stage in the same process measures independently.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        peak = torch.cuda.max_memory_allocated() / 2**30
        resv = torch.cuda.max_memory_reserved() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        gpu = torch.cuda.get_device_name(0)
        ds = getattr(getattr(cfg, "dataset", None), "name", "?")
        ent = getattr(getattr(cfg, "dataset", None), "entity_id", "?")
        util = 100.0 * resv / total if total else float("nan")
        print(f"[vram] {stage} dataset={ds} entity={ent} batch={batch_size} "
              f"peak_alloc={peak:.2f}G reserved={resv:.2f}G total={total:.1f}G "
              f"util={util:.0f}% gpu={gpu}", flush=True)
        csv_path = project_root() / "logs" / "vram.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8") as fh:
            if new:
                fh.write("stage,dataset,entity,batch_size,peak_alloc_gib,"
                         "reserved_gib,total_gib,util_pct,gpu\n")
            fh.write(f"{stage},{ds},{ent},{batch_size},{peak:.3f},{resv:.3f},"
                     f"{total:.2f},{util:.1f},{gpu}\n")
        if reset:
            torch.cuda.reset_peak_memory_stats()
    except Exception as exc:   # telemetry must never break a training run
        print(f"[vram] logging skipped: {exc}", flush=True)


# Crash-safe resumable training: helpers
# Design: documentation/operations/resume-and-telemetry-design.md. These helpers are
# byte-identical across all four repos. They are pure infrastructure (atomic IO,
# RNG snapshot/restore, checkpoint payload) and are NO-OPs unless the stage's
# main() opts into resume. Nothing here touches the GPU at import time.
import io as _io
import os as _os
import hashlib as _hashlib
import uuid as _uuid
from dataclasses import asdict as _asdict

CKPT_FORMAT_VERSION = 2

# Keys a v2 resumable checkpoint must carry (see _ckpt_complete / build_payload).
_REQUIRED_CKPT_KEYS = frozenset({
    "format_version", "stage", "state_dict", "optimizer", "scheduler",
    "step", "epoch", "best_val", "best_step", "rng",
    "loader_fingerprint", "csv_rows",
})


def _fsync_dir(d) -> None:
    """fsync a directory so a rename into it is durable across power loss."""
    fd = _os.open(str(d), _os.O_DIRECTORY)
    try:
        _os.fsync(fd)
    finally:
        _os.close(fd)


def atomic_torch_save(payload: dict, path) -> None:
    """Crash-safe, integrity-checked torch.save: serialise body, embed a sha256,
    write to a unique temp, fsync, os.replace (atomic same-dir rename), fsync dir.
    A power loss never destroys the previously committed `path`; a torn body is
    detectable on load via the digest."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    buf = _io.BytesIO()
    torch.save(payload, buf)                      # body WITHOUT the digest
    body = buf.getvalue()
    digest = _hashlib.sha256(body).hexdigest()

    buf2 = _io.BytesIO()
    torch.save({"__body__": body, "__sha256__": digest}, buf2)
    blob = buf2.getvalue()

    tmp = path.with_name(f"{path.name}.{_os.getpid()}.{_uuid.uuid4().hex}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(blob)
        fh.flush()
        _os.fsync(fh.fileno())
    _os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_text(path, text: str) -> None:
    """Atomic, durable text write (metrics.csv etc.): temp -> fsync -> replace -> dir fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{_os.getpid()}.{_uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
        fh.flush()
        _os.fsync(fh.fileno())
    _os.replace(tmp, path)
    _fsync_dir(path.parent)


def _ckpt_complete(ck) -> bool:
    return (isinstance(ck, dict)
            and ck.get("format_version") == CKPT_FORMAT_VERSION
            and _REQUIRED_CKPT_KEYS <= set(ck.keys()))


def _load_one(p):
    """Load + verify one checkpoint slot. Returns the payload dict or None if the
    file is missing, torn (zip/keys), fails its sha256, or is an old format."""
    import torch
    p = Path(p)
    if not p.exists():
        return None
    try:
        wrapper = torch.load(str(p), map_location="cpu", weights_only=False)
        body = wrapper["__body__"]
        want = wrapper["__sha256__"]
    except Exception:
        return None
    if _hashlib.sha256(body).hexdigest() != want:
        return None
    try:
        ck = torch.load(_io.BytesIO(body), map_location="cpu", weights_only=False)
    except Exception:
        return None
    return ck if _ckpt_complete(ck) else None


# ── Ping-pong double buffer with an atomic CURRENT pointer ────────────────────
def _read_pointer(ckpt_dir):
    p = Path(ckpt_dir) / "CURRENT"
    try:
        v = p.read_text().strip()
        return v if v in ("0", "1") else None
    except Exception:
        return None


def _write_pointer(ckpt_dir, val: str) -> None:
    p = Path(ckpt_dir) / "CURRENT"
    tmp = p.with_name(f"CURRENT.{_os.getpid()}.{_uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(val)
        fh.flush()
        _os.fsync(fh.fileno())
    _os.replace(tmp, p)
    _fsync_dir(Path(ckpt_dir))


def save_resumable(ckpt_dir, payload: dict) -> None:
    """Write `payload` to the slot NOT currently pointed to, fully commit it, then
    flip the CURRENT pointer. At every instant >=1 complete, checksum-valid
    checkpoint exists, so any crash leaves a recoverable state."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cur = _read_pointer(ckpt_dir)
    nxt = "1" if cur == "0" else "0"
    atomic_torch_save(payload, ckpt_dir / f"ckpt_{nxt}.pt")
    _write_pointer(ckpt_dir, nxt)


def load_resumable(ckpt_dir):
    """Return the newest valid resumable checkpoint payload, or None. Prefers the
    pointed slot, then falls back to the other slot."""
    ckpt_dir = Path(ckpt_dir)
    cur = _read_pointer(ckpt_dir)
    order = [cur, ("1" if cur == "0" else "0")] if cur in ("0", "1") else ["0", "1"]
    for s in order:
        ck = _load_one(ckpt_dir / f"ckpt_{s}.pt")
        if ck is not None:
            return ck
    return None


# ── RNG snapshot / restore (active training device only; see design §2.1) ─────
def snapshot_rng(device) -> dict:
    import random
    import numpy as np
    import torch
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if getattr(device, "type", None) == "cuda":
        rng["torch_cuda"] = torch.cuda.get_rng_state(device)
    return rng


def restore_rng(rng: dict, device) -> None:
    import random
    import numpy as np
    import torch
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if getattr(device, "type", None) == "cuda" and rng.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(rng["torch_cuda"], device)


# ── Payload + fingerprint ─────────────────────────────────────────────────────
def build_payload(*, stage, model, cfg, opt, sched, step, epoch,
                  best_val, best_step, rng, fingerprint, csv_rows, extra=None):
    payload = {
        "format_version": CKPT_FORMAT_VERSION,
        "stage": stage,
        "cfg_dict": _asdict(cfg),
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "best_step": int(best_step),
        "rng": rng,
        "loader_fingerprint": dict(fingerprint),
        "csv_rows": [dict(r) for r in csv_rows],
    }
    if extra:
        payload.update(extra)
    return payload


def build_fingerprint(cfg, *, batches_per_epoch, max_steps, warmup_steps,
                      batch_size, min_epochs, patience_steps, device,
                      deterministic, extra=None):
    """Pure function of cfg + run shape. Asserted (never restored) on resume so any
    config/data/env drift hard-fails instead of silently diverging (design §2.2)."""
    import torch
    tr = cfg.training
    fp = {
        "batches_per_epoch": int(batches_per_epoch),
        "max_steps": int(max_steps),
        "warmup_steps": int(warmup_steps),
        "batch_size": int(batch_size),
        "min_epochs": int(min_epochs),
        "patience_steps": int(patience_steps),
        "early_stopping": bool(getattr(tr, "early_stopping", False)),
        "early_stopping_min_delta": float(getattr(tr, "early_stopping_min_delta", 0.0)),
        "check_val_every_n_epoch": int(getattr(tr, "check_val_every_n_epoch", 1)),
        "lr": float(getattr(tr, "lr", 0.0)),
        "weight_decay": float(getattr(tr, "weight_decay", 0.0)),
        "seed": int(getattr(cfg, "seed", 0)),
        "deterministic": bool(deterministic),
        "matmul_precision": "highest" if deterministic else "high",
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        "device_name": (torch.cuda.get_device_name(device)
                        if getattr(device, "type", None) == "cuda" else "cpu"),
        "device_count": (torch.cuda.device_count() if torch.cuda.is_available() else 0),
    }
    if extra:
        fp.update(extra)
    return fp


def _assert_fingerprint(have: dict, want: dict, *, deterministic: bool) -> None:
    """Compare a checkpoint's fingerprint with the current run's. Hard-error under
    determinism (equivalence cannot be honored on drift); warn under Tier P."""
    diffs = []
    for k in sorted(set(have) | set(want)):
        if have.get(k) != want.get(k):
            diffs.append(f"    {k}: checkpoint={have.get(k)!r}  current={want.get(k)!r}")
    if diffs:
        msg = ("[resume] fingerprint mismatch — refusing to resume into a different "
               "configuration/environment:\n" + "\n".join(diffs))
        if deterministic:
            raise RuntimeError(msg)
        print(msg + "\n[resume] continuing anyway (Tier P, equivalence NOT guaranteed).")


def apply_determinism(deterministic: bool, strict_bitexact: bool = False) -> None:
    """Set the determinism tier (design §4.2). deterministic=False keeps today's
    Tier-P behavior (cudnn.benchmark=True) byte-for-byte."""
    import torch
    if deterministic:
        _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # must precede CUDA init
        torch.set_float32_matmul_precision("highest")                  # TF32 off
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=not strict_bitexact)
        except Exception as exc:
            print(f"[determinism] use_deterministic_algorithms unavailable: {exc}")
    else:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True


def resume_flags():
    """Read the opt-in flags from env. Returns (resume, deterministic, strict_bitexact)."""
    resume = _os.environ.get("RESUME") == "1"
    strict = _os.environ.get("STRICT_BITEXACT") == "1"
    det = _os.environ.get("DETERMINISTIC") == "1" or strict or resume
    if _os.environ.get("ALLOW_NONDETERMINISTIC_RESUME") == "1":
        det = _os.environ.get("DETERMINISTIC") == "1" or strict
    return resume, det, strict
# end crash-safe resumable training helpers
