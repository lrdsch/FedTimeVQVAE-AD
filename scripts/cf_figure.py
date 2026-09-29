#!/usr/bin/env python3.10
"""Figure of the counterfactual section: the proposed by the federate.

He reads the dumps of `scripts/cf_quality.py` (`--dump`) and draws, for a series and
some windows, the original signal and the counterfactuals of the 5 client federated
overlapping, with the region rewritten in evidence.

The rewritten region is the same in each panel and in each arm for construction:
the mask comes from the labels, not from the model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402


def masked_spans(colmask: np.ndarray, t_len: int) -> list[tuple[int, int]]:
    """Latent columns rewritten -> timestep intervals, united if adjacent."""
    w_lat = colmask.shape[0]
    edges = np.linspace(0, t_len, w_lat + 1).round().astype(int)
    spans: list[tuple[int, int]] = []
    for c in np.flatnonzero(colmask):
        a, b = int(edges[c]), int(edges[c + 1])
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="es. evidence/cf_quality/arrays/A2_ucr_043.npz")
    ap.add_argument("--windows", default="0,1", help="window indexes to be drawn")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", type=int, default=0)
    a = ap.parse_args()

    z = np.load(a.npz)
    x, d, cols = z["x"], z["d"], z["cols"]          # (B,C,T) · (n_cli,B,C,T) · (B,W_lat)
    starts = z["starts"]
    wins = [int(w) for w in a.windows.split(",")]
    n_cli, _, _, T = d.shape
    ch = a.channel

    # IEEEtran with two columns: figures* is wide ~7,16in.
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    fig, axes = plt.subplots(1, len(wins), figsize=(7.0, 2.1), sharey=True)
    axes = np.atleast_1d(axes)
    palette = plt.cm.viridis(np.linspace(0.15, 0.8, n_cli))

    for k, (ax, w) in enumerate(zip(axes, wins)):
        t = np.arange(T)
        spans = masked_spans(cols[w], T)
        for a0, b0 in spans:
            ax.axvspan(a0, b0, color="0.90", lw=0, zorder=0)
        for c in range(n_cli):
            ax.plot(t, (x[w, ch] + d[c, w, ch]), lw=0.8, color=palette[c],
                    alpha=0.9, zorder=2, label=f"client {c}" if k == 0 else None)
        ax.plot(t, x[w, ch], lw=1.3, color="k", zorder=3,
                label="observed" if k == 0 else None)
        if spans:
            ax.annotate("rewritten", xy=((spans[0][0] + spans[-1][1]) / 2, 1.0),
                        xycoords=("data", "axes fraction"), ha="center", va="top",
                        fontsize=7, color="0.35")
        ax.set_title(f"window at $t={int(starts[w])}$", fontsize=8)
        ax.set_xlabel("time step", fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("normalised\namplitude", fontsize=8)
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left",
                   handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)
    fig.tight_layout(pad=0.4)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
