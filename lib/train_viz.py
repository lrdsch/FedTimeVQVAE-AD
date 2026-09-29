"""Opt-in training-time visualization CAPTURE for the federated cb_only path.

FULLY OPT-IN, DEFAULT-OFF. Nothing here runs unless ``TVQ_VIZ=1`` is set in the
environment. When it is unset, ``pipeline.federated`` never imports this module
(the import is lazy, gated on the env var), so any in-flight experiment is
byte-identical and untouched. All output is written under ``TVQ_VIZ_DIR`` — a
root kept deliberately separate from ``artifacts/fed_eval`` so real runs are
never touched.

Design: CAPTURE-in-loop, RENDER-offline. This module only DUMPS cheap numpy
arrays (.npz) during training — no matplotlib, no PCA in the hot path. Figures
are produced afterwards by ``scripts/plot_train_viz.py`` from those dumps.

What it captures, per ``federated_stage1`` run:
  * per client, per (captured) round: input windows, their reconstruction, the
    pre-quantizer encoder latent z folded to (token, d), and each token's
    assigned stage-0 code id.  -> features (1) recon before/after, (2) latent 2D.
  * per (captured) round: the merged GLOBAL codebook (per RVQ stage), usage
    counts, the scalar cb_drift, and — when the server-side codebook EMA is on —
    the raw/smoothed EMA accumulators (C, S, w).  -> feature (3) codebook drift.

Env knobs (all optional except TVQ_VIZ_DIR, required when enabled):
  TVQ_VIZ=1                 enable capture
  TVQ_VIZ_DIR=<path>        output ROOT (a per-run <tag>/ subdir is created)
  TVQ_VIZ_TAG=<str>         subdir name for this run (default: fed_<pid>_<time>)
  TVQ_VIZ_CLIENTS=a,b|all   which client entities to capture (default: all)
  TVQ_VIZ_EVERY=<int>       capture every Nth round, round 0 always (default: 1)
  TVQ_VIZ_NPROBE=<int>      probe windows per client (default: 8)
  TVQ_VIZ_SERIES=1          ALSO capture the full-series reconstruction BAND per round
                            (opt-in, heavier): reconstructs the whole (capped) train
                            series with the stride-1 overlapping windows and stores the
                            per-timestep spread across all concurrent windows — the
                            "shade of all overlapping windows" view, per epoch/round.
  TVQ_VIZ_SERIES_MAX=<int>  cap on windows reconstructed for the band (default: 2000)

Safety contract: every public method is wrapped so a viz failure can NEVER crash
training — it prints a one-line warning and disables itself. The capture forward
runs under eval()+no_grad() (no BN/EMA/stat pollution, no dropout, no RNG draws),
so a viz-on run is bit-identical to a viz-off run.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

_TRUTHY = {"1", "true", "yes", "on"}


def viz_enabled() -> bool:
    """True iff TVQ_VIZ requests capture. The ONLY gate; default path is off."""
    return os.environ.get("TVQ_VIZ", "").strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@torch.no_grad()
def _fold_latent(model, latent: torch.Tensor):
    """Fold encoder latent (B, C*d, F, W) to per-token z (B, N, d) in the SAME
    (channel, spatial) order the quantizer / encode_tokens use:
    position p = c*(F*W) + f*W + w. Mirrors scripts/fa_pca.py:_encode_latents."""
    B, Cd, F_, W = latent.shape
    groups = getattr(model.quantizer, "groups", None)
    C = int(groups) if groups else 1
    if Cd % C != 0:                       # defensive: fall back to a single group
        C = 1
    d = Cd // C
    z = latent.reshape(B, C, d, F_, W).permute(0, 1, 3, 4, 2).reshape(B, C * F_ * W, d)
    return z.float(), C, F_, W, d


class FedStage1Viz:
    """Recorder for a single ``federated_stage1`` run. Construct via ``from_env``.

    Instances are cheap and hold: a per-client fixed probe (captured once, reused
    every round so windows are comparable), and an in-memory list of per-round
    codebook snapshots flushed to disk in :meth:`finalize`.
    """

    def __init__(self, entities, base_cfg, *, root: Path, clients_filter,
                 every: int, n_probe: int, series: bool = False, series_max: int = 2000):
        self.entities = list(entities)
        self.every = max(1, int(every))
        self.n_probe = max(1, int(n_probe))
        self.series = bool(series)
        self.series_max = max(1, int(series_max))
        self.enabled = True
        if clients_filter in (None, "", "all"):
            self.selected = set(self.entities)
        else:
            want = {c.strip() for c in str(clients_filter).split(",") if c.strip()}
            self.selected = {e for e in self.entities if e in want}
        self.root = Path(root)
        self.clients_dir = self.root / "clients"
        self._probes: dict[str, torch.Tensor] = {}     # entity -> (n, C, W) cpu
        self._cb_hist: list[dict] = []                 # per captured round
        self.clients_dir.mkdir(parents=True, exist_ok=True)
        ds = getattr(getattr(base_cfg, "dataset", None), "name", "?")
        meta = {
            "dataset": ds,
            "entities": self.entities,
            "selected": sorted(self.selected),
            "every": self.every,
            "n_probe": self.n_probe,
            "codebook_size": int(getattr(getattr(base_cfg, "quantizer", None),
                                         "codebook_size", -1)),
            "token_embedding_dim": int(getattr(getattr(base_cfg, "quantizer", None),
                                               "token_embedding_dim", -1)),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[fed:viz] capture ON -> {self.root}  "
              f"(clients={len(self.selected)}/{len(self.entities)} "
              f"every={self.every} n_probe={self.n_probe})")

    # ── construction ──────────────────────────────────────────────────────────
    @classmethod
    def from_env(cls, entities, base_cfg) -> "FedStage1Viz | None":
        root_env = os.environ.get("TVQ_VIZ_DIR", "").strip()
        if not root_env:
            print("[fed:viz] TVQ_VIZ=1 but TVQ_VIZ_DIR unset -> capture DISABLED")
            return None
        tag = os.environ.get("TVQ_VIZ_TAG", "").strip() or f"fed_{os.getpid()}_{int(time.time())}"
        root = Path(root_env) / tag / "stage1"
        return cls(entities, base_cfg, root=root,
                   clients_filter=os.environ.get("TVQ_VIZ_CLIENTS", "all"),
                   every=_env_int("TVQ_VIZ_EVERY", 1),
                   n_probe=_env_int("TVQ_VIZ_NPROBE", 8),
                   series=os.environ.get("TVQ_VIZ_SERIES", "").strip().lower() in _TRUTHY,
                   series_max=_env_int("TVQ_VIZ_SERIES_MAX", 2000))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _should_capture_round(self, r: int) -> bool:
        return self.enabled and (r == 0 or r % self.every == 0)

    def _probe_for(self, client, device) -> torch.Tensor:
        """First (up to n_probe) windows from the client's val loader (train
        fallback), captured ONCE and reused every round on `device`."""
        ent = client.entity_id
        cached = self._probes.get(ent)
        if cached is None:
            loader = getattr(client.data, "val_loader", None) or client.data.train_loader
            batch = next(iter(loader))
            cached = batch["inputs"][: self.n_probe].detach().cpu().clone()
            self._probes[ent] = cached
        return cached.to(device)

    def _disable(self, where: str, exc: Exception) -> None:
        self.enabled = False
        print(f"[fed:viz] DISABLED after error in {where}: {type(exc).__name__}: {exc}")

    # ── full-series reconstruction band (all overlapping windows per timestep) ─
    @torch.no_grad()
    def _series_band(self, model, client, device):
        """Reconstruct the whole (capped) train series with the stride-1 overlapping
        windows and return per-timestep spread stats across all concurrent windows.
        Same scatter-along-the-diagonal logic as scripts/recon_full_series.py:recon_band;
        valid because window_normalization='none' keeps every window in one space."""
        ds = getattr(client.data, "train_dataset", None)
        if ds is None or len(ds) < 1:
            return None
        N = min(len(ds), self.series_max)
        xs = torch.stack([ds[i]["inputs"] for i in range(N)])            # (N, C, W) ordered stride-1
        recs = []
        for i in range(0, N, 512):
            recs.append(model(xs[i:i + 512].to(device))["reconstructed"].detach().cpu())
        rec = torch.cat(recs).numpy()                                    # (N, C, W)
        xin = xs.numpy()
        W = xin.shape[-1]; L = N + W - 1
        orig = np.empty(L, np.float32)                                   # stitch the z-scored series
        orig[:W - 1] = xin[0, 0, :W - 1]
        orig[W - 1:] = xin[:, 0, -1]
        band = np.full((L, W), np.nan, np.float32)                       # scatter each window on its diagonal
        ar = np.arange(N)
        for j in range(W):
            band[ar + j, j] = rec[:, 0, j]
        mean = np.nanmean(band, 1)
        lo, hi = np.nanpercentile(band, [10, 90], axis=1)
        nmse = float(np.nansum((mean - orig) ** 2) / max(float(np.sum(orig ** 2)), 1e-12))
        return dict(orig=orig, mean=mean.astype(np.float32),
                    bmin=np.nanmin(band, 1).astype(np.float32),
                    bmax=np.nanmax(band, 1).astype(np.float32),
                    p10=lo.astype(np.float32), p90=hi.astype(np.float32),
                    W=np.int64(W), N=np.int64(N), nmse=np.float32(nmse), round=np.int64(-1))

    # Features (1)+(2): reconstruction and latent statistics per client and round
    @torch.no_grad()
    def capture_client(self, client, r: int, device) -> None:
        if not self._should_capture_round(r) or client.entity_id not in self.selected:
            return
        try:
            model = client.model
            was_training = model.training
            model.eval()
            try:
                x = self._probe_for(client, device)
                out = model(x)
                recon = out["reconstructed"].detach().cpu().numpy()
                z, C, F_, W, d = _fold_latent(model, out["latent"])          # (B, N, d)
                cb0 = client.vqs[0].codebook.weight.detach().to(z.device)     # (K, d)
                zf = z.reshape(-1, d)                                         # (B*N, d)
                idx = torch.cdist(zf, cb0).argmin(dim=-1)                     # (B*N,)
                win = torch.arange(z.shape[0]).repeat_interleave(z.shape[1])  # window id per token
                band = self._series_band(model, client, device) if self.series else None
            finally:
                if was_training:
                    model.train()
            out_dir = self.clients_dir / client.entity_id
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / f"round_{r:04d}.npz",
                inputs=x.detach().cpu().numpy().astype(np.float32),
                recon=recon.astype(np.float32),
                z=zf.detach().cpu().numpy().astype(np.float32),
                idx=idx.detach().cpu().numpy().astype(np.int64),
                win=win.numpy().astype(np.int64),
                round=np.int64(r), C=np.int64(C), F=np.int64(F_), W=np.int64(W), d=np.int64(d),
            )
            if band is not None:
                band["round"] = np.int64(r)
                np.savez_compressed(out_dir / f"series_round_{r:04d}.npz", **band)
        except Exception as e:                       # viz must never crash training
            self._disable("capture_client", e)

    # Feature (3): global codebook and EMA drift per round
    @torch.no_grad()
    def capture_codebook(self, r: int, global_cbs, merged, server_ema,
                         cb_drift: float, N) -> None:
        # Captured EVERY round (cheap: a (K,D) table + counts) so the drift curve /
        # trails are full-resolution, independent of the per-client figure cadence.
        if not self.enabled:
            return
        try:
            rec = {
                "round": int(r),
                "codebooks": np.stack([g.detach().cpu().numpy().astype(np.float32)
                                       for g in global_cbs]),          # (S, K, D)
                "counts": np.stack([m[1].detach().cpu().numpy().astype(np.float32)
                                    for m in merged]),                 # (S, K)  broadcast N_s
                "n_dead": np.array([int(m[3]) for m in merged], dtype=np.int64),
                "cb_drift": float(cb_drift),
            }
            if server_ema is not None:
                rec["ema_C"] = np.stack([s["C"].cpu().numpy().astype(np.float32)
                                         for s in server_ema])          # (S, K)
                rec["ema_S"] = np.stack([s["S"].cpu().numpy().astype(np.float32)
                                         for s in server_ema])          # (S, K, D)
                rec["ema_w"] = np.array([float(s["w"]) for s in server_ema], dtype=np.float32)
            self._cb_hist.append(rec)
        except Exception as e:
            self._disable("capture_codebook", e)

    # ── flush ─────────────────────────────────────────────────────────────────
    def finalize(self) -> None:
        if not self.enabled or not self._cb_hist:
            return
        try:
            rounds = np.array([h["round"] for h in self._cb_hist], dtype=np.int64)
            payload = {
                "rounds": rounds,
                "codebooks": np.stack([h["codebooks"] for h in self._cb_hist]),   # (R, S, K, D)
                "counts": np.stack([h["counts"] for h in self._cb_hist]),         # (R, S, K)
                "n_dead": np.stack([h["n_dead"] for h in self._cb_hist]),         # (R, S)
                "cb_drift": np.array([h["cb_drift"] for h in self._cb_hist], dtype=np.float32),
            }
            if all("ema_C" in h for h in self._cb_hist):
                payload["ema_C"] = np.stack([h["ema_C"] for h in self._cb_hist])
                payload["ema_S"] = np.stack([h["ema_S"] for h in self._cb_hist])
                payload["ema_w"] = np.stack([h["ema_w"] for h in self._cb_hist])
            np.savez_compressed(self.root / "codebook_history.npz", **payload)
            print(f"[fed:viz] wrote codebook_history.npz ({len(rounds)} rounds) "
                  f"and per-client dumps under {self.clients_dir}")
        except Exception as e:
            self._disable("finalize", e)
