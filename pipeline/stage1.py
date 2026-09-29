# %%
"""
=============================================================================
  Stage 1 — train the VQ-VAE. Plain PyTorch, monolithic main().
=============================================================================

Pipeline: input (B, C, T)
  → transform         (STFT / Identity)
  → encoder           (per_channel / ChIndep Conv2d / ChIndep Conv3d)
  → quantizer         (shared_vq / per_channel_vq)
  → decoder           (Conv2d TF / ChIndep Conv2d / ChIndep Conv3d)
  → inverse transform
  → refinement head   (learnable residual in time-domain)
  = reconstruction

Loss: MSE(input, reconstruction) + commitment_weight * commitment_loss

Outputs written to artifacts/runs/stage1/<run_name>/:
  * checkpoints/best.ckpt, last.ckpt
  * logs/metrics.csv
  * losses/loss.png, loss_logy.png
  * reconstruction_snapshots/<i>_<tag>/epoch_NNNN.png
  * token_cache.pt       (used by stage 2)

Debug-friendly:
  * `Stage1VQVAE`  — plain nn.Module wiring the 5 pieces + refinement head.
  * `main()`       — inline training loop. Set a breakpoint anywhere and
                     step through: no Lightning, no callbacks, no hooks.
"""
from __future__ import annotations

# repo root on sys.path so this pipeline/ script can import the shared
# libs (config / data / utils / metrics_core) that live at the project root.
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from lib.proctitle import set_process_title  # noqa: E402

import csv
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, format_config, load_config
from data import encode_and_cache_tokens, make_dataloaders
from model.decoder import RefinementHead, TemporalReconstructor, build_decoder
from model.encoder import build_encoder
from model.transforms import build_transform
from model.vector_quantizer import QuantizerOutput, build_quantizer
from lib.profiling import profile_run
from utils import (
    configure_logging, log_gpu_peak, run_dir_for, save_loss_plots,
    seed_everything, token_cache_path,
    apply_determinism, build_fingerprint, build_payload, save_resumable,
    load_resumable, snapshot_rng, restore_rng, _assert_fingerprint, resolve_device,
)


# ─── The VQ-VAE: 5 pieces wired together ────────────────────────────────────

class Stage1VQVAE(nn.Module):
    """Transform → Encoder → Quantizer → Decoder → inverse transform → Refinement.

    Lazy: encoder and decoder build themselves on the first forward, once
    they've seen the real tensor shapes (STFT output depends on window length
    and n_fft). Always run one forward pass before building the optimizer or
    loading a checkpoint.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # ── Lock: only the grouped target path is accepted ────────────────
        if cfg.encoder.name != "channel_independent_conv2d":
            raise ValueError(
                f"Target path requires encoder.name = 'channel_independent_conv2d', "
                f"got {cfg.encoder.name!r}"
            )
        if cfg.quantizer.name not in ("shared_codebook_per_channel_vq",
                                      "residual_shared_codebook_per_channel_vq"):
            raise ValueError(
                f"Target path requires quantizer.name in {{'shared_codebook_per_channel_vq', "
                f"'residual_shared_codebook_per_channel_vq'}}, got {cfg.quantizer.name!r}"
            )
        if cfg.decoder.name != "channel_independent_conv2d":
            raise ValueError(
                f"Target path requires decoder.name = 'channel_independent_conv2d', "
                f"got {cfg.decoder.name!r}"
            )

        if cfg.transform.name == "stft":
            self.transform = build_transform("stft", n_fft=cfg.transform.n_fft,
                                             normalized=cfg.transform.normalized)
        else:
            self.transform = build_transform("identity")

        # Single source of truth: cfg.quantizer.token_embedding_dim wires
        # encoder, quantizer, and decoder together (no projection between them).
        token_embedding_dim = cfg.quantizer.token_embedding_dim

        self.encoder = build_encoder(
            "channel_independent_conv2d",
            token_embedding_dim=token_embedding_dim,
            downsampled_width=cfg.encoder.downsampled_width,
            n_resnet_blocks=cfg.encoder.n_resnet_blocks,
            dropout=cfg.encoder.dropout,
            width_base=getattr(cfg.encoder, "width_base", 4),
        )

        q_kwargs = dict(
            token_embedding_dim=token_embedding_dim,
            codebook_size=cfg.quantizer.codebook_size,
            commitment_weight=cfg.quantizer.commitment_weight,
            ema_decay=cfg.quantizer.ema_decay,
            eps=cfg.quantizer.eps,
            threshold_ema_dead_code=cfg.quantizer.threshold_ema_dead_code,
            kmeans_init=getattr(cfg.quantizer, "kmeans_init", True),
        )
        if cfg.quantizer.name == "residual_shared_codebook_per_channel_vq":
            q_kwargs["n_stages"] = int(cfg.quantizer.n_residual_stages)
        self.quantizer = build_quantizer(cfg.quantizer.name, **q_kwargs)

        self.decoder_2d = build_decoder(
            cfg.decoder.name,
            token_embedding_dim=token_embedding_dim,
            n_resnet_blocks=cfg.decoder.n_resnet_blocks,
            dropout=cfg.decoder.dropout,
            width_base=getattr(cfg.encoder, "width_base", 4),
        )
        self.decoder_1d = TemporalReconstructor()        # only used if quantized is 3D
        self.refinement = RefinementHead(mode=getattr(cfg.decoder, "refine_mode", "linear"))
        # Learned residual on the time-domain waveform after inverse STFT.
        # Adds a Linear(T, T) head that the decoder uses to clean up STFT
        # round-trip artefacts.
        self.use_refinement = True

    def reconstruct_representation(self, quantized: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        if quantized.ndim == 4:
            return self.decoder_2d(quantized, target_shape)
        return self.decoder_1d(quantized, target_shape)

    def forward(self, x: torch.Tensor) -> dict:
        """Returns a dict with 'reconstructed', losses, and the quantizer output."""
        tf = self.transform(x)                           # TransformOutput
        # One-shot: tell the shared-codebook per-channel VQ how many groups (= C) to use.
        if hasattr(self.quantizer, "set_groups") and getattr(self.quantizer, "groups", None) is None:
            self.quantizer.set_groups(tf.spec.original_channels)
        latent = self.encoder(tf)                        # (B, C*d, F, W')
        quant_out: QuantizerOutput = self.quantizer(latent)
        reconstructed_repr = self.reconstruct_representation(quant_out.quantized, tf.tensor.shape)
        reconstructed = self.transform.inverse(reconstructed_repr, tf.spec)
        if self.use_refinement:
            reconstructed = self.refinement(reconstructed)

        loss_time = F.mse_loss(reconstructed, x)
        loss_spec = F.mse_loss(reconstructed_repr, tf.tensor)
        loss_vq = quant_out.loss
        # Backprop loss matches the original TimeVQVAE-AD recipe: time-domain
        # MSE + VQ commitment. `loss_spec` is logged for diagnostics but not
        # backpropped by default — STFT-coefficient outliers on near-constant
        # channels otherwise dominate the gradient on SMAP. cfg.decoder.spec_weight
        # (default 0.0) opts it back in as a high-frequency-aware objective.
        spec_w = float(getattr(self.cfg.decoder, "spec_weight", 0.0))
        total = loss_time + loss_vq + (spec_w * loss_spec if spec_w else 0.0)
        # Per-channel diagnostics — logged only, NOT in `total`.
        C = tf.spec.original_channels
        Z = tf.spec.components_per_channel
        time_per_ch = (reconstructed - x).pow(2).mean(dim=(0, 2))                      # (C,)
        spec_err = (reconstructed_repr - tf.tensor).pow(2)
        spec_per_ch = spec_err.reshape(spec_err.shape[0], C, Z, -1).mean(dim=(0, 2, 3))  # (C,)
        losses = {
            "loss": total,
            "loss_time": loss_time,
            "loss_spec": loss_spec,
            "loss_vq": loss_vq,
            "loss_time_worst_ch": time_per_ch.max(),
            # Backward-compat aliases (CSV columns already in existing runs):
            "reconstruction_loss": loss_time,
            "quantizer_loss": loss_vq,
        }
        for c in range(C):
            losses[f"loss_time_ch{c}"] = time_per_ch[c]
            losses[f"loss_spec_ch{c}"] = spec_per_ch[c]
        for name, value in quant_out.stats.items():
            if "loss" in name.lower():
                losses[name] = value

        return {
            "reconstructed": reconstructed,
            "reconstructed_repr": reconstructed_repr,
            "latent": latent,
            "quantizer_output": quant_out,
            "losses": losses,
            "transform_output": tf,
        }

    @torch.no_grad()
    def encode_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        """(quantized, indices, latent_spatial_shape) — used by stage 2 and score caching."""
        tf = self.transform(x)
        if hasattr(self.quantizer, "set_groups") and getattr(self.quantizer, "groups", None) is None:
            self.quantizer.set_groups(tf.spec.original_channels)
        latent = self.encoder(tf)
        quant_out = self.quantizer(latent)
        latent_spatial = tuple(int(s) for s in latent.shape[2:])
        return quant_out.quantized, quant_out.indices, latent_spatial


# ─── Checkpoint helpers ──────────────────────────────────────────────────────

def save_stage1_checkpoint(path: Path, model: Stage1VQVAE, cfg: Config, step: int, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "cfg_dict": asdict(cfg),
        "step": step,
        "epoch": epoch,
    }, path)


def load_stage1(
    ckpt_path: str | Path, cfg: Config,
    example_inputs: torch.Tensor,
    device: str | torch.device = "cpu",
) -> Stage1VQVAE:
    """Rebuild a Stage1VQVAE and load its weights. Returns frozen model in eval mode."""
    state = torch.load(str(ckpt_path), map_location="cpu")
    model = Stage1VQVAE(cfg)
    # eval() before the materialisation forward keeps the VQ from running its
    # kmeans-init / EMA update / dead-code expiry on a random tensor — those paths
    # mutate the codebook state in-place and would be overwritten by the strict
    # load below anyway, so skipping them is faster and avoids spurious work.
    model.eval()
    with torch.no_grad():
        model(example_inputs.cpu())                      # materialise lazy modules
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ─── Snapshot helpers (periodic reconstruction dumps during training) ───────

def _select_diverse_windows(dataset, n_select: int = 5):
    """Pick `n_select` windows at evenly spaced pattern-richness quantiles.

    Richness = mean std across channels + 3 × mean derivative-sign-flip rate
    (peaks/valleys). Rewards windows oscillating across all channels, not just
    one. Fixes "highest = flat window" when channel 0 happens to be a step.

    Works for any SlidingWindowDataset (train or val) — only reads
    `dataset[i]["inputs"]` and `dataset.records`.
    """
    n = len(dataset)
    if n == 0:
        return torch.zeros((0,)), []
    richness = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = dataset[i]["inputs"].numpy()                         # (C, T)
        d = np.diff(x, axis=-1)
        zc = np.mean(np.sign(d[..., 1:]) != np.sign(d[..., :-1]), axis=-1)
        richness[i] = float(x.std(axis=-1).mean() + 3.0 * zc.mean())
    order = np.argsort(richness, kind="stable")
    n_sel = min(n_select, n)
    quantile_positions = np.linspace(0, n - 1, n_sel).astype(int)
    pick_order = list(dict.fromkeys(order[quantile_positions].tolist()))
    tags = ["flattest", "low", "medium", "high", "highest"][:n_sel]

    windows, meta = [], []
    for tag, idx in zip(tags, pick_order):
        item = dataset[idx]
        windows.append(item["inputs"])
        md = item["metadata"]
        record = dataset.records[int(md["record_index"])]
        feats = [] if record.metadata is None else list(record.metadata.feature_names)
        meta.append({
            "tag": tag,
            "entity_id": str(md["entity_id"]),
            "dataset_name": str(md["dataset"]),
            "window_start": int(md["window_start"]),
            "window_stop": int(md["window_stop"]),
            "feature_names": feats,
            "richness": float(richness[idx]),
        })
    return torch.stack(windows, dim=0), meta


def _save_snapshot_figure(
    path: Path, original: np.ndarray, reconstructed: np.ndarray,
    meta: dict, epoch: int, global_step: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    n = original.shape[0]
    names = meta["feature_names"] or [f"channel_{i}" for i in range(n)]
    t = np.arange(original.shape[-1])

    fig, axes = plt.subplots(n, 1, figsize=(14, max(3.5, 1.4 * n)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lo = float(min(original.min(), reconstructed.min()))
    hi = float(max(original.max(), reconstructed.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    for c, ax in enumerate(axes):
        ax.plot(t, original[c], color="steelblue", linewidth=0.9, label="Original")
        ax.plot(t, reconstructed[c], color="darkorange", linewidth=0.9, label="Reconstructed")
        ax.set_ylabel(names[c], fontsize=7, rotation=0, labelpad=48, va="center")
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(alpha=0.2, linewidth=0.4)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Time step")
    fig.suptitle(
        f"epoch={epoch} | step={global_step} | {meta['tag']} "
        f"({meta['entity_id']} [{meta['window_start']}:{meta['window_stop']}], richness={meta['richness']:.3f})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── Main — plain-PyTorch training loop, everything inline ──────────────────

def main(cfg: Config | None = None) -> Path:
    """Train stage 1 end-to-end. Returns path to the best checkpoint."""
    configure_logging()
    cfg = cfg or load_config()
    print(format_config(cfg))
    seed_everything(cfg.seed)
    # Resume / determinism flags (design: resume couples Tier S)
    strict_bitexact = os.environ.get("STRICT_BITEXACT") == "1"
    resume = bool(getattr(cfg.training, "resume", False)) or os.environ.get("RESUME") == "1"
    deterministic = (bool(getattr(cfg.training, "deterministic", False))
                     or os.environ.get("DETERMINISTIC") == "1"
                     or strict_bitexact or resume)
    if os.environ.get("ALLOW_NONDETERMINISTIC_RESUME") == "1":
        deterministic = (bool(getattr(cfg.training, "deterministic", False))
                         or os.environ.get("DETERMINISTIC") == "1" or strict_bitexact)
    apply_determinism(deterministic, strict_bitexact)   # sets matmul/cudnn per tier

    device = resolve_device()
    run_dir = run_dir_for(cfg, "stage1")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stage1] run dir: {run_dir.resolve()}")
    print(f"[stage1] device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────
    data = make_dataloaders(cfg, stage="stage1")
    train_loader = data.train_loader
    val_loader = data.val_loader
    batches_per_epoch = len(train_loader)
    example_inputs = next(iter(train_loader))["inputs"][:1].cpu()

    # ── Model (materialise lazy modules with one CPU forward pass) ─────────
    model = Stage1VQVAE(cfg)
    # Materialise in eval mode so the VQ doesn't trigger kmeans-init / EMA /
    # dead-code expiry on this throw-away forward. The training loop calls
    # `model.train()` at the start of every epoch, so we don't need to flip back.
    model.eval()
    with torch.no_grad():
        model(example_inputs)
    model.to(device)

    # ── Architecture summary (per-component param counts + layer trees) ────
    _sep = "─" * 80
    print(f"[stage1] {_sep}")
    for _name, _mod in [
        ("transform",  model.transform),
        ("encoder",    model.encoder),
        ("quantizer",  model.quantizer),
        ("decoder_2d", model.decoder_2d),
        ("decoder_1d", model.decoder_1d),
        ("refinement", model.refinement),
    ]:
        _n = sum(p.numel() for p in _mod.parameters())
        print(f"[stage1] {_name}: {_n:,} params")
        print(_mod)
        print(f"[stage1] {_sep}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage1] TOTAL: {n_params:,} params")
    print(f"[stage1] {_sep}")

    # ── Snapshot windows (5 diverse windows from train AND val for periodic plots) ──
    train_snap_windows, train_snap_meta = _select_diverse_windows(
        data.train_dataset, n_select=5,
    )
    val_snap_windows, val_snap_meta = _select_diverse_windows(
        data.val_dataset, n_select=5,
    )
    snapshot_dir = run_dir / "reconstruction_snapshots"
    if train_snap_windows.numel() > 0:
        tags = ", ".join(f"{m['tag']}(r={m['richness']:.3f})" for m in train_snap_meta)
        print(f"[stage1] train snapshot windows: {tags}")
    if val_snap_windows.numel() > 0:
        tags = ", ".join(f"{m['tag']}(r={m['richness']:.3f})" for m in val_snap_meta)
        print(f"[stage1] val snapshot windows:   {tags}")

    # Optimiser + warmup-cosine schedule
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                            fused=(torch.device(device).type == "cuda"))
    max_steps = cfg.training.stage1_max_steps
    min_epochs = getattr(cfg.training, "stage1_min_epochs", 0)
    if min_epochs > 0:
        floor = min_epochs * batches_per_epoch
        if floor > max_steps:
            print(f"[stage1] step budget raised {max_steps} -> {floor} "
                  f"({min_epochs} epochs x {batches_per_epoch} batches/epoch).")
            max_steps = floor
    warmup_steps = max(1, int(max_steps * cfg.training.warmup_rate))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ── Mixed precision ────────────────────────────────────────────────────
    # fp16 below Ampere (this node's RTX 8000 is sm_75: fp16 tensor cores, NO bf16),
    # bf16 from Ampere on. `AMP=0` → GradScaler(enabled=False), a pure pass-through,
    # and autocast(enabled=False) → bit-identical fp32.
    use_amp = bool(getattr(cfg.training, "amp", False)) and torch.cuda.is_available()
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    if use_amp:
        print(f"[stage1] AMP {str(amp_dtype).replace('torch.', '')} autocast ON "
              f"(AMP=0 to disable)")

    # Re-seed: the materialisation forward + diverse-window selection consume
    # python/numpy/torch random state. Re-seeding here makes the actual training
    # loop deterministic regardless of what example_inputs/snapshot picks did.
    seed_everything(cfg.seed)

    # CSV log setup
    # Buffer all rows in memory; rewrite the whole CSV every epoch so its
    # header always matches the data (no "header narrower than rows" problem
    # when a column like `val/loss` first appears mid-training). The file is
    # always up-to-date for live `tail -f`, and a mid-training crash still
    # leaves the CSV consistent through the last completed epoch.
    csv_path = run_dir / "logs" / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, float | int]] = []

    def _write_metrics_csv() -> None:
        if not csv_rows:
            return
        all_fields: list[str] = []
        seen: set[str] = set()
        for r in csv_rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    all_fields.append(k)
        import io as _io
        from utils import atomic_write_text
        buf = _io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
        atomic_write_text(csv_path, buf.getvalue())   # crash-safe: temp+fsync+replace

    # Early-stopping bookkeeping (step-based patience)
    cb_size = cfg.quantizer.codebook_size
    best_val = float("inf")
    best_step = 0
    best_ckpt_path = ckpt_dir / "best.ckpt"
    last_ckpt_path = ckpt_dir / "last.ckpt"
    patience_steps = cfg.training.stage1_patience_steps

    # ── Resume (design: full-state restore at epoch boundary) ──────────────
    # fingerprint = pure fn of cfg+run-shape; asserted (not restored) so any
    # config/data/env drift hard-fails instead of silently diverging.
    fingerprint = build_fingerprint(
        cfg, batches_per_epoch=batches_per_epoch, max_steps=max_steps,
        warmup_steps=warmup_steps, batch_size=cfg.dataset.batch_size_stage1,
        min_epochs=min_epochs, patience_steps=patience_steps,
        device=device, deterministic=deterministic,
    )
    start_step, start_epoch = 0, 0
    if resume:
        ck = load_resumable(ckpt_dir)
        if ck is None:
            print("[resume] stage1: no valid checkpoint; starting fresh.")
        else:
            _assert_fingerprint(ck["loader_fingerprint"], fingerprint,
                                deterministic=deterministic)
            model.load_state_dict(ck["state_dict"])          # weights + VQ EMA buffers
            opt.load_state_dict(ck["optimizer"])             # AdamW moments
            sched.load_state_dict(ck["scheduler"])           # cosine position
            best_val = float(ck["best_val"]); best_step = int(ck["best_step"])
            start_step, start_epoch = int(ck["step"]), int(ck["epoch"])
            csv_rows[:] = list(ck["csv_rows"])               # verbatim, no reparse
            restore_rng(ck["rng"], device)                   # LAST: overrides pre-loop seed
            print(f"[resume] stage1 resumed at epoch={start_epoch} step={start_step} "
                  f"best_val={best_val:.6f} best_step={best_step}")

    # ── Training loop ──────────────────────────────────────────────────────
    # Stops when ANY of these fires: epoch >= stage1_max_epochs (primary control),
    # step >= stage1_max_steps (safety cap), or early-stopping patience exhausted.
    step = start_step
    epoch = start_epoch
    stop = False
    # Opt-in profiler (TVQ_PROFILE=1). No-op + zero overhead when disabled.
    # Sampled mode: records a small window of training steps (see lib/profiling.py).
    prof = profile_run("stage1", sampled=True)
    prof.start()
    while not stop and epoch < cfg.training.stage1_max_epochs:
        # ─ Train epoch ─
        model.train()
        train_losses: dict[str, list[float]] = {}
        train_ppl: list[float] = []
        train_usage: torch.Tensor | None = None      # (num_codebooks, cb_size) bool

        for batch in train_loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(inputs)
                loss = out["losses"]["loss"]

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            _prev_scale = scaler.get_scale()
            scaler.step(opt)                    # skipped iff grads overflowed under fp16
            scaler.update()
            # A skipped step means no weights moved: the LR schedule must not advance
            # either, or it silently desyncs from the optimiser under fp16.
            if scaler.get_scale() >= _prev_scale:
                sched.step()
            prof.step()                         # advance profiler one step (no-op when off)

            for name, value in out["losses"].items():
                train_losses.setdefault(name, []).append(value.item())
            ppl = out["quantizer_output"].perplexity
            if ppl is not None:
                train_ppl.append(ppl.item())

            # Track codebook usage across the epoch (OR into a bool mask).
            idx = out["quantizer_output"].indices.detach().cpu().long()
            if idx.ndim == 1:      view = idx.reshape(1, 1, -1)
            elif idx.ndim == 2:    view = idx.unsqueeze(1)
            else:                  view = idx.reshape(idx.shape[0], idx.shape[1], -1)
            if train_usage is None:
                train_usage = torch.zeros((view.shape[1], cb_size), dtype=torch.bool)
            for ci in range(view.shape[1]):
                codes = torch.unique(view[:, ci, :])
                codes = codes[(codes >= 0) & (codes < cb_size)]
                if codes.numel() > 0:
                    train_usage[ci, codes] = True

            step += 1
            if step >= max_steps:
                stop = True
                break

        # ─ Val epoch (optional, every check_val_every_n_epoch) ─
        val_losses: dict[str, list[float]] = {}
        val_ppl: list[float] = []
        val_usage: torch.Tensor | None = None
        ran_val = False
        if val_loader is not None and (epoch % cfg.training.check_val_every_n_epoch == 0):
            model.eval()
            ran_val = True
            # Validation stays fp32 on purpose: the val loss drives early stopping and
            # checkpoint selection, so it must not depend on the training precision.
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["inputs"].to(device, non_blocking=True)
                    out = model(inputs)
                    for name, value in out["losses"].items():
                        val_losses.setdefault(name, []).append(value.item())
                    ppl = out["quantizer_output"].perplexity
                    if ppl is not None:
                        val_ppl.append(ppl.item())
                    idx = out["quantizer_output"].indices.detach().cpu().long()
                    if idx.ndim == 1:      view = idx.reshape(1, 1, -1)
                    elif idx.ndim == 2:    view = idx.unsqueeze(1)
                    else:                  view = idx.reshape(idx.shape[0], idx.shape[1], -1)
                    if val_usage is None:
                        val_usage = torch.zeros((view.shape[1], cb_size), dtype=torch.bool)
                    for ci in range(view.shape[1]):
                        codes = torch.unique(view[:, ci, :])
                        codes = codes[(codes >= 0) & (codes < cb_size)]
                        if codes.numel() > 0:
                            val_usage[ci, codes] = True

        # ─ Build CSV row ─
        row: dict[str, float | int] = {"epoch": epoch, "step": step,
                                       "lr": float(opt.param_groups[0]["lr"])}
        for name, vals in train_losses.items():
            row[f"train/{name}"] = float(np.mean(vals)) if vals else float("nan")
        if train_ppl:
            row["train/perplexity"] = float(np.mean(train_ppl))
        if train_usage is not None:
            total = int(train_usage.numel())
            used = int(train_usage.sum().item())
            row["train/channel_code_occupancy_percent"] = 100.0 * used / total
            row["train/inactive_channel_code_pairs"] = float(total - used)
            global_used = int(train_usage.any(dim=0).sum().item())
            row["train/global_codes_used"] = float(global_used)
            row["train/global_dead_codes"] = float(cb_size - global_used)
            row["train/global_codebook_usage_percent"] = 100.0 * global_used / cb_size
        for name, vals in val_losses.items():
            row[f"val/{name}"] = float(np.mean(vals)) if vals else float("nan")
        if val_ppl:
            row["val/perplexity"] = float(np.mean(val_ppl))
        if val_usage is not None:
            total = int(val_usage.numel())
            used = int(val_usage.sum().item())
            row["val/channel_code_occupancy_percent"] = 100.0 * used / total
            row["val/inactive_channel_code_pairs"] = float(total - used)
            global_used = int(val_usage.any(dim=0).sum().item())
            row["val/global_codes_used"] = float(global_used)
            row["val/global_dead_codes"] = float(cb_size - global_used)
            row["val/global_codebook_usage_percent"] = 100.0 * global_used / cb_size

        csv_rows.append(row)
        _write_metrics_csv()

        # ─ Console line ─
        tl = row.get("train/loss", float("nan"))
        vl = row.get("val/loss", float("nan"))
        up = row.get("train/global_codebook_usage_percent", float("nan"))
        print(f"[stage1] epoch={epoch:4d} step={step:6d} "
              f"train/loss={tl:.4f} val/loss={vl:.4f} train/global_usage={up:.1f}% "
              f"lr={row['lr']:.2e}")

        # ─ Checkpoint (last always; best if val improved) ─
        save_stage1_checkpoint(last_ckpt_path, model, cfg, step, epoch)
        if ran_val:
            current_val = row.get("val/loss", float("inf"))
            if current_val < best_val - cfg.training.early_stopping_min_delta:
                best_val = float(current_val)
                best_step = step
                save_stage1_checkpoint(best_ckpt_path, model, cfg, step, epoch)
            elif (cfg.training.early_stopping and step >= warmup_steps
                  and step - best_step >= patience_steps):
                # Gate early stopping on WARMUP, not on `min_epochs`. `min_epochs` raises
                # `max_steps` to span the LR schedule, so gating on it made early stopping
                # fire only at the very last epoch — i.e. it was effectively disabled.
                print(f"[stage1] early stopping at epoch={epoch} step={step} "
                      f"(no val improvement for {step - best_step} steps "
                      f"since best at step {best_step}; past warmup={warmup_steps})")
                stop = True

        # ─ Resumable checkpoint: written LAST, AFTER best_val/best_step update
        #   (design §4.3 BLOCKER fix), with epoch+1 = next epoch to run. RNG is
        #   snapshotted here = last training-RNG event of the epoch (the val and
        #   snapshot passes below run under eval/no_grad and draw no training RNG).
        if resume:                       # write resume checkpoints only when opted-in (OFF = unchanged)
            save_resumable(ckpt_dir, build_payload(
                stage="stage1", model=model, cfg=cfg, opt=opt, sched=sched,
                step=step, epoch=epoch + 1, best_val=best_val, best_step=best_step,
                rng=snapshot_rng(device), fingerprint=fingerprint, csv_rows=csv_rows,
            ))

        # ─ Reconstruction snapshots (every epoch < 10, then every 100 epochs) ─
        # Save train and val snapshots in separate subfolders so val drift vs.
        # train fit is visible at a glance — gap = overfitting.
        if epoch < 10 or epoch % 100 == 0:
            model.eval()
            for split_name, snap_windows, snap_meta in [
                ("train", train_snap_windows, train_snap_meta),
                ("val",   val_snap_windows,   val_snap_meta),
            ]:
                if snap_windows.numel() == 0:
                    continue
                with torch.no_grad():
                    recon = model(snap_windows.to(device))["reconstructed"].detach().cpu().numpy()
                orig = snap_windows.cpu().numpy()
                for i, (o, r, m) in enumerate(zip(orig, recon, snap_meta)):
                    _save_snapshot_figure(
                        snapshot_dir / split_name / f"{i}_{m['tag']}" / f"epoch_{epoch:04d}.png",
                        o, r, m, epoch, step,
                    )

        epoch += 1

    prof.finish()                                     # stop profiler + write trace/table
    _write_metrics_csv()                              # final flush

    # Peak-VRAM telemetry -> stdout + logs/vram.csv for batch-size tuning.
    log_gpu_peak("stage1", cfg.dataset.batch_size_stage1, cfg, reset=True)

    # Post-training: loss plots + token cache
    save_loss_plots(csv_path, run_dir / "losses", stage_label="Stage 1")

    best_ckpt = best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path
    print(f"[stage1] best checkpoint: {best_ckpt.resolve()}")

    cache_path = token_cache_path(cfg)
    try:
        cached_model = load_stage1(best_ckpt, cfg, example_inputs, device=device)
        encode_and_cache_tokens(
            cached_model, data.train_loader, data.val_loader,
            cache_path, cfg, device=device,
        )
    except Exception as e:
        # Token cache is an optimisation — stage 2 falls back to on-the-fly
        # encoding if the file is missing. Print traceback so real bugs surface.
        import traceback
        print(f"[stage1] WARNING: failed to cache tokens ({type(e).__name__}: {e}). "
              f"Stage 2 will encode on-the-fly.")
        traceback.print_exc()

    return best_ckpt

# %%
if __name__ == "__main__":
    set_process_title()
    main()
