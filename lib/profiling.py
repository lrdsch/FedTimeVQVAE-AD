# %%
"""
=============================================================================
  profiling.py — opt-in torch.profiler instrumentation for the WHOLE pipeline.
=============================================================================

OFF by default. When the env var TVQ_PROFILE is unset (or 0/false), `profile_run`
returns a zero-cost no-op handle and changes NOTHING — normal runs are byte-for-
byte identical and pay no overhead.

Turn it on with `TVQ_PROFILE=1` (the launcher `run.py --profile` sets it once for
every subprocess in a sweep). When on, each script writes, per (dataset, entity):

  artifacts/profiling/<dataset>/<entity>/<tag>/
    trace.json        Chrome/Perfetto trace  → open in chrome://tracing or
                      https://ui.perfetto.dev
    key_averages.txt  operator table, sorted by self-CUDA time (self-CPU if no GPU)
    summary.json      one-line roll-up for this script run
  artifacts/profiling/summary.csv             cumulative roll-up, one row per run
                                              → the cross-pipeline bottleneck view

Two modes, picked by the caller:
  * region  (sampled=False) — profile everything between start and finish. Used to
              wrap the whole entry call of the non-loop scripts (detect, quality_*,
              compare_aggregations, cf_eval).
  * sampled (sampled=True)  — torch.profiler step schedule; call `handle.step()`
              once per training iteration. Used by stage1/stage2 to capture a
              small representative window of steps instead of the whole run.

Usage — non-loop scripts (region mode, exception-safe via `with`):

    from lib.profiling import profile_run
    with profile_run("detect"):
        detect()

Usage — training loops (sampled mode):

    from lib.profiling import profile_run
    prof = profile_run("stage1", sampled=True); prof.start()
    while ...:
        for batch in loader:
            ...
            prof.step()        # advance one profiler step (no-op when disabled)
    prof.finish()              # stop + write trace/table (no-op when disabled)

Env knobs (all optional; only read when enabled):
  TVQ_PROFILE          1/true/on to enable (anything else = off)
  TVQ_PROFILE_DIR      output root                         (default artifacts/profiling)
  TVQ_PROFILE_WAIT     sampled: steps skipped before warmup (default 5)
  TVQ_PROFILE_WARMUP   sampled: warmup steps                (default 3)
  TVQ_PROFILE_ACTIVE   sampled: recorded steps              (default 10)
  TVQ_PROFILE_SHAPES   record input shapes 1/0              (default 1)
  TVQ_PROFILE_MEMORY   profile tensor memory 1/0            (default 0)
  TVQ_PROFILE_STACK    record python stacks 1/0             (default 0)
  TVQ_PROFILE_ROWS     rows in key_averages table           (default 30)

Standalone helper — no torch import unless actually profiling, so importing this
module is free even on machines without a GPU.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path


# ─── env helpers ─────────────────────────────────────────────────────────────

def profiling_enabled() -> bool:
    return os.environ.get("TVQ_PROFILE", "").strip().lower() in {"1", "true", "on", "yes"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "on", "yes"}


def _profile_root() -> Path:
    return Path(os.environ.get("TVQ_PROFILE_DIR", "").strip() or "artifacts/profiling")


def _dataset_entity() -> tuple[str, str]:
    """(dataset, entity) for output paths — taken from the env vars run.py sets
    on each subprocess; falls back to 'manual' for direct invocation."""
    ds = os.environ.get("DATASET_NAME", "").strip() or "manual"
    ent = (os.environ.get("DATASET_ENTITY", "").strip() or "manual").replace("/", "_").replace("\\", "_")
    return ds, ent


def _out_dir(tag: str) -> Path:
    ds, ent = _dataset_entity()
    d = _profile_root() / ds / ent / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─── no-op handle (returned when profiling is disabled) ─────────────────────

class _NoopRun:
    """Zero-cost handle. Every method is a harmless no-op."""
    enabled = False

    def start(self) -> "_NoopRun":
        return self

    def step(self) -> None:
        pass

    def finish(self) -> None:
        pass

    def __enter__(self) -> "_NoopRun":
        return self

    def __exit__(self, *exc) -> bool:
        return False


# real handle (torch.profiler-backed)

class _ProfileRun:
    """Wraps torch.profiler.profile. torch is imported lazily in start()."""
    enabled = True

    def __init__(self, tag: str, sampled: bool):
        self.tag = tag
        self.sampled = sampled
        self._prof = None
        self._cuda = False
        self._t0 = 0.0

    def start(self) -> "_ProfileRun":
        import torch
        from torch.profiler import ProfilerActivity, profile, schedule

        self._cuda = torch.cuda.is_available()
        activities = [ProfilerActivity.CPU]
        if self._cuda:
            activities.append(ProfilerActivity.CUDA)

        sched = None
        if self.sampled:
            sched = schedule(
                wait=_env_int("TVQ_PROFILE_WAIT", 5),
                warmup=_env_int("TVQ_PROFILE_WARMUP", 3),
                active=_env_int("TVQ_PROFILE_ACTIVE", 10),
                repeat=1,
            )

        self._prof = profile(
            activities=activities,
            schedule=sched,
            record_shapes=_env_flag("TVQ_PROFILE_SHAPES", True),
            profile_memory=_env_flag("TVQ_PROFILE_MEMORY", False),
            with_stack=_env_flag("TVQ_PROFILE_STACK", False),
        )
        self._t0 = time.perf_counter()
        self._prof.__enter__()
        mode = "sampled" if self.sampled else "region"
        print(f"[profile] {self.tag}: torch.profiler ON (mode={mode}, cuda={self._cuda})")
        return self

    def step(self) -> None:
        # Only meaningful in sampled mode; region mode never schedules.
        if self.sampled and self._prof is not None:
            self._prof.step()

    def finish(self) -> None:
        if self._prof is None:
            return
        wall = time.perf_counter() - self._t0
        try:
            self._prof.__exit__(None, None, None)
        except Exception as exc:  # never let profiling teardown kill the run
            print(f"[profile] {self.tag}: profiler stop failed ({exc!r}); skipping outputs.")
            self._prof = None
            return
        try:
            self._write_outputs(wall)
        except Exception as exc:
            print(f"[profile] {self.tag}: writing outputs failed ({exc!r}).")
        self._prof = None

    # context-manager sugar so non-loop callers can `with profile_run(...):`
    def __enter__(self) -> "_ProfileRun":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish()
        return False  # propagate any exception from the wrapped block

    # ── output writers ──────────────────────────────────────────────────────

    @staticmethod
    def _self_cpu_us(evt) -> float:
        return float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0)

    @staticmethod
    def _self_cuda_us(evt) -> float:
        # torch renamed self_cuda_time_total -> self_device_time_total; support both.
        return float(getattr(evt, "self_device_time_total",
                             getattr(evt, "self_cuda_time_total", 0.0)) or 0.0)

    def _write_outputs(self, wall: float) -> None:
        out = _out_dir(self.tag)
        rows = _env_int("TVQ_PROFILE_ROWS", 30)
        ka = self._prof.key_averages()

        self_cpu_ms = sum(self._self_cpu_us(e) for e in ka) / 1000.0
        self_cuda_ms = sum(self._self_cuda_us(e) for e in ka) / 1000.0
        # Rank by GPU kernel time only if any was actually captured. On machines
        # where CUPTI can't load (kineto warns "CUDA profiler activities will be
        # missing") the device column is all-zero, so CPU time is the useful key;
        # same for genuinely CPU-bound scripts (e.g. compare_aggregations).
        use_cuda_metric = self_cuda_ms > 0.0
        sort_key = "self_cuda_time_total" if use_cuda_metric else "self_cpu_time_total"
        try:
            table = ka.table(sort_by=sort_key, row_limit=rows)
        except Exception:
            table = ka.table(sort_by="self_cpu_time_total", row_limit=rows)
        (out / "key_averages.txt").write_text(table, encoding="utf-8")

        try:
            self._prof.export_chrome_trace(str(out / "trace.json"))
        except Exception as exc:
            print(f"[profile] {self.tag}: chrome trace export failed ({exc!r}).")

        pick = self._self_cuda_us if use_cuda_metric else self._self_cpu_us
        top = max(ka, key=pick, default=None)
        top_op = top.key if top is not None else ""
        top_op_ms = (pick(top) / 1000.0) if top is not None else 0.0

        ds, ent = _dataset_entity()
        summary = {
            "tag": self.tag,
            "dataset": ds,
            "entity": ent,
            "mode": "sampled" if self.sampled else "region",
            "cuda": self._cuda,
            "cuda_kernels_captured": use_cuda_metric,
            "wall_sec": round(wall, 3),
            "self_cpu_ms": round(self_cpu_ms, 3),
            "self_cuda_ms": round(self_cuda_ms, 3),
            "top_op": top_op,
            "top_op_ms": round(top_op_ms, 3),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _append_summary_row(summary)
        print(f"[profile] {self.tag}: wrote {out}  "
              f"(self_cpu={self_cpu_ms:.0f}ms self_cuda={self_cuda_ms:.0f}ms wall={wall:.1f}s, "
              f"top={top_op})")


def _append_summary_row(summary: dict) -> None:
    """Best-effort append to the cumulative summary.csv. Each run also writes an
    authoritative summary.json in its own dir, so a clobbered row here (possible
    under high concurrency) never loses data."""
    root = _profile_root()
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "summary.csv"
    fields = ["timestamp", "tag", "dataset", "entity", "mode", "cuda",
              "cuda_kernels_captured", "wall_sec", "self_cpu_ms", "self_cuda_ms",
              "top_op", "top_op_ms"]
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **summary}
    write_header = not csv_path.exists()
    try:
        with csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        print(f"[profile] WARNING: could not append summary.csv ({exc!r}).")


# public entry

def profile_run(tag: str, sampled: bool = False):
    """Return a profiling handle for the region tagged `tag`.

    Disabled (TVQ_PROFILE unset/false) → a zero-cost `_NoopRun`.
    Enabled → a `_ProfileRun`. `sampled=True` uses a step schedule (training
    loops, call `.step()` per iteration); `sampled=False` profiles the whole
    region (single-pass scripts).

    Never raises: if the profiler can't be constructed, prints a warning and
    falls back to the no-op handle so a profiling attempt can't break a run.
    """
    if not profiling_enabled():
        return _NoopRun()
    try:
        return _ProfileRun(tag, sampled)
    except Exception as exc:
        print(f"[profile] WARNING: profiler init failed ({exc!r}); continuing without profiling.")
        return _NoopRun()
