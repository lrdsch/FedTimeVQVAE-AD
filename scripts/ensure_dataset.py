"""Ensure a dataset is present and intact, provisioning it automatically if not.

This is the single "check → switch (download / generate)" gate the launchers
call before they run anything on a dataset. For the datasets the baselines and
the main pipeline run on, it answers one question per dataset:

    is `data/raw/<name>/` present and structurally intact?
      yes  -> log OK, do nothing.
      no   -> log what's missing, run the right provisioner (download or
              generate), re-check, and log the outcome.

Managed datasets (everything else is left untouched — reported "unmanaged"):

  * toy_fed_uni[_full|_t<N>]    -> GENERATE via scripts/build_toy_fed_uni.py.

(The multivariate corpora -- smap/msl, asd -- and the toy_*_channel_anomalies
families were removed on 2026-07-23 with the closed -M study.)

"Intact" is a fast, self-contained structural check (no manifest dependency —
the repo manifest doesn't track the generated toys): the per-entity
layout must have matching train/test/test_label entity sets, every .npy must be
non-empty and have a sane shape (train/test 2-D, label 1-D with len == test len).
.npy shapes are read from the header only (no full load, no numpy dependency),
so this stays importable from the dependency-light launcher processes.

Provisioning subprocesses inherit stdout/stderr, so the download/generation
progress is visible live on the terminal.

CLI::

    python scripts/ensure_dataset.py toy_fed_uni
    python scripts/ensure_dataset.py toy_fed_uni --force        # re-provision even if present
    python scripts/ensure_dataset.py toy_fed_uni --check-only   # report, never provision
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
RAW = ROOT / "data" / "raw"

# The federated benchmark: toy_fed_uni / toy_fed_uni_full / toy_fed_uni_t<N>  (C=1, clustered).
# `_t<N>` encodes the per-client train length (the scarcity knob); `_full` is the
# larger client tier. Everything else falls back to the tier default.
_FED_RE = re.compile(r"^toy_fed_uni(_full|_t(?P<train_len>\d+))?$")


# ─────────────────────────────────────────────────────────────────────────────
# .npy header reader (stdlib only — no numpy import needed to validate shapes)
# ─────────────────────────────────────────────────────────────────────────────

def npy_shape(path: Path) -> Optional[tuple[int, ...]]:
    """Return the array shape stored in a .npy file's header, or None if the
    file is not a readable .npy (truncated / corrupt / wrong magic)."""
    try:
        with path.open("rb") as f:
            if f.read(6) != b"\x93NUMPY":
                return None
            major = f.read(1)
            f.read(1)  # minor
            if major == b"\x01":
                raw = f.read(2)
                if len(raw) != 2:
                    return None
                (hlen,) = struct.unpack("<H", raw)
            else:
                raw = f.read(4)
                if len(raw) != 4:
                    return None
                (hlen,) = struct.unpack("<I", raw)
            header = f.read(hlen)
            if len(header) != hlen:
                return None
            meta = ast.literal_eval(header.decode("latin1"))
            shape = meta.get("shape")
            return tuple(int(x) for x in shape) if shape is not None else None
    except (OSError, ValueError, SyntaxError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnsureReport:
    name: str
    status: str   # "ok" | "provisioned" | "failed" | "unmanaged" | "missing_no_provision"
    detail: str

    @property
    def ok(self) -> bool:
        # "unmanaged" is OK (the caller's own missing-handling takes over).
        return self.status in ("ok", "provisioned", "unmanaged")


# ─────────────────────────────────────────────────────────────────────────────
# Integrity checks (return (ok, detail))
# ─────────────────────────────────────────────────────────────────────────────

def _check_per_entity(root: Path) -> tuple[bool, str]:
    """Validate the per-entity .npy layout used by asd + the channel-anomaly toys:
    train/ test/ test_label/ with a matching entity set, each .npy non-empty and
    sanely shaped (train/test 2-D, label 1-D with len == test len)."""
    if not root.is_dir():
        return False, "folder absent"
    train_dir, test_dir, label_dir = root / "train", root / "test", root / "test_label"
    if not train_dir.is_dir():
        return False, "no train/ dir"
    train_ids = sorted(p.stem for p in train_dir.glob("*.npy"))
    if not train_ids:
        return False, "train/ has no .npy files"
    test_ids = sorted(p.stem for p in test_dir.glob("*.npy")) if test_dir.is_dir() else []
    label_ids = sorted(p.stem for p in label_dir.glob("*.npy")) if label_dir.is_dir() else []

    missing_test = set(train_ids) - set(test_ids)
    if missing_test:
        return False, f"{len(missing_test)} entit(y/ies) missing in test/ (e.g. {sorted(missing_test)[:3]})"
    missing_label = set(train_ids) - set(label_ids)
    if missing_label:
        return False, f"{len(missing_label)} entit(y/ies) missing in test_label/ (e.g. {sorted(missing_label)[:3]})"

    for eid in train_ids:
        tr, te, la = train_dir / f"{eid}.npy", test_dir / f"{eid}.npy", label_dir / f"{eid}.npy"
        for p in (tr, te, la):
            if p.stat().st_size == 0:
                return False, f"{p.name} is empty (0 bytes)"
        s_tr, s_te, s_la = npy_shape(tr), npy_shape(te), npy_shape(la)
        if s_tr is None or len(s_tr) != 2:
            return False, f"{tr.name}: unreadable or not 2-D (shape={s_tr})"
        if s_te is None or len(s_te) != 2:
            return False, f"{te.name}: unreadable or not 2-D (shape={s_te})"
        if s_la is None or len(s_la) != 1:
            return False, f"{la.name}: unreadable or not 1-D (shape={s_la})"
        if s_la[0] != s_te[0]:
            return False, f"{eid}: label len {s_la[0]} != test len {s_te[0]}"
    return True, f"{len(train_ids)} entit(y/ies), train/test/test_label aligned"


# ─────────────────────────────────────────────────────────────────────────────
# Provisioner resolution: (check_fn, build the provisioning command)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Provisioner:
    root: Path                                  # data/raw folder for this dataset
    check: Callable[[], tuple[bool, str]]       # () -> (ok, detail)
    base_cmd: list[str]                         # provisioning command (sans interpreter / overwrite flag)
    overwrite_flag: str                         # flag that forces the provisioner to rewrite ("--force" | "--overwrite")

    def command(self, *, overwrite: bool) -> list[str]:
        """Full provisioning command. The overwrite flag MUST be passed when the
        target already exists but is broken, otherwise the downloaders skip
        ("already present") and the toy builders abort (FileExistsError)."""
        return [*self.base_cmd, self.overwrite_flag] if overwrite else list(self.base_cmd)


def _resolve(name: str) -> Optional[Provisioner]:
    """Map a dataset name to its (check, provisioning command), or None if the
    name is not one this module manages."""
    low = name.lower()

    fed = _FED_RE.match(name)
    if fed:
        builder = SCRIPTS / "build_toy_fed_uni.py"
        if not builder.exists():
            return None
        tier = "full" if name.endswith("_full") else "dev"
        cmd = [str(builder), "--tier", tier,
               "--output-dir", str(RAW / name), "--dataset-name", name]
        if fed.group("train_len"):
            cmd += ["--train-length", fed.group("train_len")]
        return Provisioner(RAW / name, lambda: _check_per_entity(RAW / name),
                           cmd, "--overwrite")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dataset(
    name: str,
    *,
    force: bool = False,
    allow_provision: bool = True,
    python: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> EnsureReport:
    """Ensure dataset `name` is present and intact; provision it if not.

    Returns an EnsureReport. Never raises for an expected condition (missing
    data, failed provisioning) — those are reported via the status so a sweep
    keeps going. `python` selects the interpreter used to run the provisioning
    script (default: the current interpreter)."""
    prov = _resolve(name)
    if prov is None:
        log(f"[ensure] {name}: unmanaged (no auto-provision rule) - leaving as-is.")
        return EnsureReport(name, "unmanaged", "no provisioner")

    py = python or sys.executable
    ok, detail = prov.check()

    if ok and not force:
        log(f"[ensure] {name}: OK - {detail}.")
        return EnsureReport(name, "ok", detail)

    if not allow_provision:
        log(f"[ensure] {name}: NOT INTACT - {detail} (provisioning disabled).")
        return EnsureReport(name, "missing_no_provision", detail)

    # Re-provisioning a folder that already exists requires the overwrite flag,
    # else the downloaders skip ("already present") and the toy builders abort.
    overwrite = force or (prov.root.exists() and any(prov.root.iterdir()))
    cmd = prov.command(overwrite=overwrite)
    reason = "forced refresh" if (ok and force) else f"not intact - {detail}"
    log(f"[ensure] {name}: {reason}; provisioning...")
    log(f"[ensure] {name}: $ {py} {' '.join(cmd)}")
    rc = subprocess.run([py, *cmd], cwd=str(ROOT), check=False).returncode
    if rc != 0:
        log(f"[ensure] {name}: provisioner exited rc={rc}.")
        return EnsureReport(name, "failed", f"provisioner rc={rc}")

    ok2, detail2 = prov.check()
    if ok2:
        log(f"[ensure] {name}: provisioned OK - {detail2}.")
        return EnsureReport(name, "provisioned", detail2)
    log(f"[ensure] {name}: still NOT intact after provisioning - {detail2}.")
    return EnsureReport(name, "failed", f"still broken: {detail2}")


def ensure_datasets(
    names: list[str],
    *,
    force: bool = False,
    allow_provision: bool = True,
    python: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> list[EnsureReport]:
    """Ensure each unique dataset in `names` (order preserved). Logs a header so
    the check is visible on the terminal even when everything is already fine."""
    seen: set[str] = set()
    uniq = [n for n in names if not (n in seen or seen.add(n))]
    managed = [n for n in uniq if _resolve(n) is not None]
    if managed:
        log(f"[ensure] checking {len(managed)} dataset(s): {', '.join(managed)}")
    return [ensure_dataset(n, force=force, allow_provision=allow_provision,
                           python=python, log=log) for n in uniq]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+",
                   help="Dataset names to ensure (e.g. toy_fed_uni).")
    p.add_argument("--force", action="store_true",
                   help="Re-provision even if the dataset is already present and intact.")
    p.add_argument("--check-only", action="store_true",
                   help="Report status only; never download or generate.")
    p.add_argument("--python", default=None,
                   help="Interpreter for provisioning scripts (default: this interpreter).")
    args = p.parse_args(argv)

    reports = ensure_datasets(
        args.datasets, force=args.force,
        allow_provision=not args.check_only, python=args.python,
    )
    bad = [r for r in reports if not r.ok]
    if bad:
        print(f"[ensure] {len(bad)} dataset(s) not ready: "
              f"{', '.join(f'{r.name} ({r.status})' for r in bad)}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
