"""Give the current process a human-readable title.

Visible in ``ps`` / ``htop`` and — because NVML reads ``/proc/<pid>/cmdline`` —
also in ``nvidia-smi`` (instead of a wall of identical "python" rows).

Standalone + dependency-light ON PURPOSE: this is imported from the TimeVQVAE
pipeline scripts AND the TF1/py3.7 baseline wrappers (InterFusion / OmniAnomaly /
CATCH), whose envs lack torch/sklearn/etc. Only the stdlib is used here, and
``setproctitle`` is imported lazily — if the package isn't installed the call is
a silent no-op, so nothing breaks in an env that doesn't have it.

    pip install setproctitle      # per env where you want the names to show

The title is composed from, in order (blank parts dropped):

  * ``$TVQ_PROC_TAG``                 — a name YOU choose at launch
                                        (run.py --name / comparisons/run.py --name,
                                        or just ``export TVQ_PROC_TAG=...``)
  * ``$DATASET_NAME[/$DATASET_ENTITY]`` — set per-job by the launchers
  * the entry-script stem             — stage1, stage2, run_omnianomaly, ...
  * an optional caller-supplied suffix

e.g.  ``exp42 msl/omi-1 stage2``
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def build_title(suffix: str = "") -> str:
    """Compose the process title from the environment (see module docstring)."""
    tag = os.environ.get("TVQ_PROC_TAG", "").strip()
    ds = os.environ.get("DATASET_NAME", "").strip()
    ent = os.environ.get("DATASET_ENTITY", "").strip()
    dataset = f"{ds}/{ent}" if (ds and ent) else ds
    script = Path(sys.argv[0]).stem if (sys.argv and sys.argv[0]) else ""
    parts = [tag, dataset, script, suffix.strip()]
    return " ".join(p for p in parts if p)


def set_process_title(suffix: str = "") -> str:
    """Set this process's title from the env. Returns the title (or the would-be
    title if ``setproctitle`` is unavailable). NEVER raises — a missing
    ``setproctitle`` simply means the title isn't applied."""
    title = build_title(suffix)
    try:
        from setproctitle import setproctitle
    except Exception:
        return title
    if title:
        setproctitle(title)
    return title
