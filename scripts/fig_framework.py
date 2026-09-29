#!/usr/bin/env python3
"""A2 Arm Framework (federated_enc_fedavg + suffstat + prior partial + BN shared),
in the same style of the MLISE figure — see STYLE.md.

Each number in the figure is VERIFICATE, undeclared:
  - 163 encoder weight loaders + 44 buffer BN = 207 shared (run button)
  - 6 tensors of the identical quantzer among clients
  - 209 decoders + 2 of refinement DIVERGONO -> remain local
  - 55 shared prior body keys, local channel_embedding/output_bias heads
Measuring 5 stage1.ckpt artifacts/runs/c50_a2/.../ucr_187/seed0/.

    python3 scripts/fig_framework.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, FancyArrowPatch, Rectangle
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

# palettes and conventions: identical to STYLE.md
STAGE_A, STAGE_B = "#FF8029", "#FFFF00"
PANEL, PANEL_B = "#71CEDC", "#71E2D5"
ENC, INPUT, VIEW = "#FF9A9A", "#AEC6FF", "#90EF90"
INNER = "#F9F9FA"
A_STAGE, A_PANEL, A_BOX = 0.19, 0.59, 0.49
ACC = dict(blue="#4567A9", orange="#D17920", green="#2B7B4F",
           purple="#7B3A99", red="#B0302F")

LW = 1.0
FS = 1.30                      # the squares are wide: the writings can grow
FONT = "DejaVu Serif"          # the original uses Bookman Demi Italic
TXT = dict(family=FONT, style="italic", weight="bold", ha="center", va="center")

W, H = 100.0, 132.0
# Single-column figure, just under half of an IEEE text page in height.
fig = plt.figure(figsize=(3.5, 3.5 * H / W))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc, alpha, fs=4.4*FS, lines=(), rr=None, ec="black"):
    r = rr if rr is not None else min(0.35 * h, 0.45 * w)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, alpha=alpha, ec=ec, lw=LW, zorder=3))
    for i, t in enumerate(lines):
        ax.text(x, y + (len(lines) - 1 - 2 * i) * fs * 0.248, t, fontsize=fs, zorder=4, **TXT)


def trapez(x, y, w, h, fc, lines, fs=4.4*FS, flip=False, taper=0.30):
    t = taper * w
    top, bot = (w / 2, w / 2 - t) if not flip else (w / 2 - t, w / 2)
    ax.add_patch(Polygon([(x - top, y + h / 2), (x + top, y + h / 2),
                          (x + bot, y - h / 2), (x - bot, y - h / 2)],
                         closed=True, fc=fc, alpha=A_BOX, ec="black", lw=LW,
                         joinstyle="round", zorder=3))
    for i, s in enumerate(lines):
        ax.text(x, y + (len(lines) - 1 - 2 * i) * fs * 0.248, s, fontsize=fs, zorder=4, **TXT)


def stage(x, y, w, h, fc, tag):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=2.4",
                                fc=fc, alpha=A_STAGE, ec="none", zorder=0))
    # black label at the TOP-LEFT of the region, above every other patch
    ax.text(x - w / 2 + 2.2, y + h / 2 - 1.5, tag, fontsize=5.4*FS, zorder=7,
            color="black", family=FONT, style="italic", weight="bold",
            ha="left", va="top")


def arrow(p0, p1, rad=0.0, lw=LW, color="black", head=7.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head,
                                 lw=lw, color=color, linestyle=ls, zorder=5,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


def loss(x, y, sub, fs=6.6*FS):
    ax.text(x, y, r"$\mathcal{L}$", fontsize=fs, zorder=6,
            family=FONT, style="italic", weight="bold", ha="right", va="center")
    ax.text(x + 0.3, y - fs * 0.13, sub, fontsize=fs * 0.46, zorder=6,
            family=FONT, style="italic", weight="bold", ha="left", va="center")


def snowflake(x, y, s=1.3, color=None):
    for a in np.arange(0, np.pi, np.pi / 3):
        ax.plot([x - s * np.cos(a), x + s * np.cos(a)],
                [y - s * np.sin(a), y + s * np.sin(a)],
                color=color or ACC["blue"], lw=LW * 0.9, zorder=6, solid_capstyle="round")


def frozen(x_left, y_top, s=1.05, dx=1.7, dy=1.3):
    """Frozen marker OUTSIDE the box, off its top-left corner (x_left, y_top)."""
    snowflake(x_left - dx, y_top + dy, s)


def series(x, y, w, h, seed, spike=None, lw=0.45):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc="white", alpha=0.95,
                           ec="black", lw=LW * 0.8, zorder=3))
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 240)
    v = np.sin(2 * np.pi * 6 * t) * 0.5 + rng.normal(0, 0.15, t.size)
    if spike is not None:
        m = (t > spike) & (t < spike + 0.10)
        v[m] += np.sin(2 * np.pi * 40 * t[m]) * 0.8
    v = v / (np.abs(v).max() * 2.35)
    ax.plot(x - w / 2 + t * w, y + v * h, color="black", lw=LW * lw, zorder=4)
    if spike is not None:
        ax.plot(x - w / 2 + t[m] * w, y + v[m] * h, color=ACC["red"], lw=LW * 0.6, zorder=5)


def tokens(x, y, n, m, cell=1.3, seed=0, anomalous=None, masked=()):
    rng = np.random.default_rng(seed)
    pal = [INPUT, VIEW, ENC, PANEL, "#D9C7F0"]
    for i in range(n):
        for j in range(m):
            c = pal[rng.integers(0, len(pal))]
            if anomalous and anomalous[0] <= i < anomalous[1]:
                c = ACC["red"]
            if i in masked:
                c = "#9A9A9A"
            ax.add_patch(Rectangle((x + i * cell, y + j * cell), cell * 0.92, cell * 0.92,
                                   fc=c, alpha=0.85, ec="black", lw=LW * 0.35, zorder=4))


# ═════════════ STAGE 1 ═════════════
# The two training stages share one grammar: client pipeline on the left, server on the
# right, communication confined to the central gutter, and the stage label at the top-left.
# The frozen marker (snowflake) sits OUTSIDE its box, off the top-left corner.
stage(50, 108.9, 94, 46.2, STAGE_A, "Stage 1  ·  tokenizer")          # y 85.8 .. 132

# ---- client pipeline ----
series(25, 125.2, 25, 2.6, seed=30, lw=0.4)
box(25, 120.0, 27, 4.6, INPUT, A_BOX, lines=["STFT"], fs=4.8*FS)
trapez(25, 113.6, 34, 5.2, ENC, [r"Encoder  $E$"], fs=4.6*FS, taper=0.22)
box(25, 106.0, 36, 6.2, PANEL, A_PANEL,
    lines=["Vector Quantizer", "codebook FROZEN in-round"], fs=3.9*FS)
frozen(25 - 18, 106.0 + 3.1)
trapez(25, 98.4, 34, 5.4, ENC, [r"Decoder  $D_k$   local"], fs=4.1*FS,
       flip=True, taper=0.22)
box(25, 90.6, 39, 6.2, INNER, 0.99,
    lines=[r"$\mathcal{L}_\mathrm{reconstruction} + \mathcal{L}_\mathrm{commit}$"], fs=4.8*FS)

arrow((25, 123.7), (25, 122.4))
arrow((25, 117.6), (25, 116.6))
arrow((25, 110.8), (25, 109.2))
arrow((25, 102.7), (25, 101.1))
arrow((25, 95.5), (25, 93.9))

# ---- server and ownership note ----
box(73, 113.3, 46, 18.2, PANEL_B, A_PANEL, lines=[], rr=3.2)
ax.text(73, 119.7, "Server (no date)", fontsize=4.9*FS, zorder=6, **TXT)
ax.text(73, 115.7, "codebook:  k-means M-step", fontsize=4.0*FS, zorder=6, **TXT)
ax.text(73, 111.8, "encoder:  FedAvg", fontsize=4.0*FS, zorder=6, **TXT)
ax.text(73, 107.9, "BN running stats:  pooled", fontsize=4.0*FS, zorder=6, **TXT)
arrow((42.5, 109.2), (50.0, 115.6), rad=-0.17)
arrow((50.0, 108.6), (42.5, 104.2), rad=-0.17)

box(71, 94.1, 48, 6.4, INNER, 0.99,
    lines=[r"$D_k$ + refinement never leave client $k$"], fs=4.0*FS)
ax.plot([39.0, 49.5], [98.6, 95.4], color="black", lw=LW * 0.7,
        ls=(0, (2.2, 1.8)), zorder=2)

# ═════════════ STAGE 2 ═════════════
stage(50, 68.95, 94, 29.7, STAGE_B, "Stage 2  ·  prior")               # y 54.1 .. 83.8

tokens(18.8, 74.3, 8, 3, cell=1.35, seed=11, masked=(2, 3, 6))
box(25, 67.8, 40, 6.6, PANEL, A_PANEL,
    lines=["MaskGIT prior", "body shared  /  head local"], fs=4.0*FS)
box(25, 58.7, 25, 5.6, INNER, 0.99,
    lines=[r"$\mathcal{L}_\mathrm{stage\,2}$"], fs=5.0*FS)
arrow((25, 73.8), (25, 71.3))
arrow((25, 64.4), (25, 61.7))

box(73, 67.8, 46, 14.6, PANEL_B, A_PANEL, lines=[], rr=3.0)
ax.text(73, 72.9, "Server", fontsize=4.7*FS, zorder=6, **TXT)
ax.text(73, 68.9, "FedAvg on the body keys", fontsize=4.0*FS, zorder=6, **TXT)
ax.text(73, 64.7, "channel embedding + bias  stay local",
        fontsize=3.5*FS, zorder=6, **TXT)
arrow((43.0, 70.3), (50.0, 72.3), rad=-0.14)
arrow((50.0, 64.0), (43.0, 65.4), rad=-0.14)

# ═════════════ INFERENCE ═════════════
ax.plot([4, 96], [51.1, 51.1], color="black", lw=LW * 0.7,
        ls=(0, (3.2, 2.4)), zorder=2)
ax.text(5.5, 48.9, r"Inference   (entirely on client $k$)", fontsize=5.2*FS, zorder=6,
        family=FONT, style="italic", weight="bold", ha="left", va="center")

# Shared trunk: one row followed by a centred prior.
series(14, 41.9, 21, 3.0, seed=21, spike=0.62)
ax.text(14, 38.0, "test window", fontsize=4.0*FS, zorder=6, **TXT)
box(44, 41.9, 30, 5.8, INPUT, A_BOX,
    lines=["encoder + VQ  shared"], fs=4.0*FS)
frozen(44 - 15, 41.9 + 2.9)
arrow((24.7, 41.9), (29.7, 41.9))

tokens(64.5, 39.9, 7, 3, cell=1.35, seed=13, anomalous=(4, 6))
arrow((58.2, 41.9), (64.2, 41.9))
box(50, 31.3, 44, 6.8, PANEL, A_PANEL,
    lines=["prior surprise", "shared body + local head"], fs=4.0*FS)
arrow((70.0, 39.4), (61.0, 35.3), rad=0.16)

# The only branch in inference: detection and explanation.
arrow((43.0, 27.8), (26.0, 23.6), rad=0.10)
arrow((57.0, 27.8), (74.0, 23.6), rad=-0.10)
box(25, 19.9, 44, 6.0, VIEW, A_BOX,
    lines=["smoothing over channel and time"], fs=3.8*FS)
box(75, 19.9, 44, 6.0, PANEL_B, A_PANEL,
    lines=["explainable sampling"], fs=4.1*FS)

box(25, 7.6, 44, 7.4, PANEL_B, A_PANEL,
    lines=["detection", "anomaly score  vs  threshold"], fs=3.8*FS)
trapez(75, 7.6, 44, 7.4, ENC,
       ["counterfactual", r"Decoder  $D_k$   local"], fs=4.0*FS,
       flip=True, taper=0.20)
arrow((25, 16.8), (25, 11.5))
arrow((75, 16.8), (75, 11.5))
# flipped trapezoid: the top edge is the short one, its left corner is at x - (w/2 - taper*w)
frozen(75 - (22 - 0.20 * 44), 7.6 + 3.7, dx=1.2)


fig.savefig("a2_framework.pdf")
fig.savefig("a2_framework.svg")
fig.savefig("a2_framework_check.png", dpi=250)
print("writings a2_framework.pdf + a2_framework.svg + _check.png")
