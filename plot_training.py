"""Plot training curves from one or more saved runs.

Run:  python plot_training.py results/baseline_qlearning_seed0.npz [more.npz ...]

Panels: episode return (with moving average), episode length, and the
fraction of recent episodes ending in success / unsafe / time limit, plus
the fraction that reached the upright region at least once.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def label_for(path: Path) -> str:
    return path.stem.replace("_seed", " seed ")


def plot(paths, window: int = 100, save: str | None = None, show: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    (ax_ret, ax_len), (ax_out, ax_up) = axes
    for path in map(Path, paths):
        d = np.load(path)
        ep = np.arange(1, len(d["return"]) + 1)
        name = label_for(path)
        line, = ax_ret.plot(ep, moving_average(d["return"], window), label=name)
        ax_ret.plot(ep, d["return"], color=line.get_color(), alpha=0.15, lw=0.5)
        ax_len.plot(ep, moving_average(d["steps"] * 0.01, window),
                    color=line.get_color(), label=name)
        out = d["outcome"]
        ax_out.plot(ep, moving_average(out == 0, window), color=line.get_color(),
                    label=f"{name}: success")
        ax_out.plot(ep, moving_average(out == 1, window), color=line.get_color(),
                    ls="--", label=f"{name}: unsafe")
        ax_up.plot(ep, moving_average(d["first_upright"] >= 0, window),
                   color=line.get_color(), label=name)

    ax_ret.set_title(f"Episode return (light) and {window}-episode moving average")
    ax_ret.set_ylabel("return")
    ax_len.set_title("Episode duration (moving average)")
    ax_len.set_ylabel("seconds")
    ax_out.set_title("How episodes ended (moving average)")
    ax_out.set_ylabel("fraction of episodes")
    ax_out.set_ylim(-0.02, 1.02)
    ax_up.set_title("Reached the upright region at least once")
    ax_up.set_ylabel("fraction of episodes")
    ax_up.set_ylim(-0.02, 1.02)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("training episode")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
        print(f"Saved {save}")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    plot(sys.argv[1:], save="results/training_curves.png")
