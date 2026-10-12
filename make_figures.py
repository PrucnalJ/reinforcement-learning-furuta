"""Produce every figure and table used in the report from saved runs.

Run (after the training commands in README.md):
    python make_figures.py

Reads results/final/*.npz and writes figures/ and figures/summary.json:
  fig_training.png     return (moving average), duration, outcomes,
                       reached-upright fraction during training
  fig_eval_traces.png  wrapped pendulum angle vs time, evaluation episodes
  fig_eval_bars.png    evaluation success / reached-upright fractions
  fig_comparison.png   controlled comparison: fine vs coarse arm bins
  fig_visits.png       where in the Q table the agent spent its time
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate import evaluate
from plot_training import moving_average

RES = Path("results/final")
OUT = Path("figures")
SEEDS = [0, 1, 2]
WINDOW = 500
N_EVAL = 20
COLORS = {"qlearning": "tab:blue", "sarsa": "tab:orange"}
NAMES = {"qlearning": "Q-learning", "sarsa": "SARSA"}


def runs(tag: str, method: str):
    return [np.load(RES / f"{tag}_{method}_seed{s}.npz") for s in SEEDS
            if (RES / f"{tag}_{method}_seed{s}.npz").exists()]


def band(ax, data, key, color, label, transform=lambda x: x):
    curves = np.array([moving_average(transform(d[key]), WINDOW) for d in data])
    ep = np.arange(1, curves.shape[1] + 1)
    m = curves.mean(0)
    ax.plot(ep, m, color=color, label=label)
    if len(curves) > 1:
        ax.fill_between(ep, curves.min(0), curves.max(0), color=color, alpha=0.2)


def fig_training():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for m in ("qlearning", "sarsa"):
        data = runs("baseline", m)
        c, n = COLORS[m], NAMES[m]
        band(axes[0, 0], data, "return", c, n)
        band(axes[0, 1], data, "steps", c, n, lambda x: x * 0.01)
        band(axes[1, 0], data, "outcome", c, f"{n}: success", lambda x: (x == 0).astype(float))
        band(axes[1, 0], data, "outcome", c, f"{n}: unsafe", lambda x: (x == 1).astype(float))
        axes[1, 0].lines[-1].set_linestyle("--")
        band(axes[1, 1], data, "first_upright", c, n, lambda x: (x >= 0).astype(float))
    titles = [f"Episode return ({WINDOW}-episode moving average)",
              "Episode duration (s)", "Success (solid) and unsafe (dashed) rate",
              "Fraction of episodes reaching |theta| <= 20 deg"]
    for ax, t in zip(axes.flat, titles):
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 1].set_ylim(-0.02, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("training episode")
    fig.suptitle(f"Training, baseline configuration (mean of {len(SEEDS)} seeds, "
                 f"band = min/max)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_training.png", dpi=150)


def fig_eval(results):
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.8 * len(results)), squeeze=False)
    for ax, (s, rows) in zip(axes[:, 0], results):
        for r in rows[:5]:
            t = np.arange(1, len(r["theta_trace"]) + 1) * 0.05
            ax.plot(t, np.rad2deg(r["theta_trace"]), lw=1, label=r["outcome"])
        ax.axhspan(-20, 20, color="tab:green", alpha=0.15)
        ax.set_ylim(-185, 185)
        ax.set_yticks([-180, -90, -20, 0, 20, 90, 180])
        ax.set_ylabel("wrapped theta (deg)")
        ax.set_title(f"{s['method']} (seed 0): first 5 evaluation episodes, "
                     f"green = upright region", fontsize=10)
        ax.legend(fontsize=7, ncol=5, loc="lower right")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_eval_traces.png", dpi=150)


def fig_bars(table):
    labels = list(table)
    succ = [np.mean([s["success_rate"] for s in table[k]]) for k in labels]
    up = [np.mean([1 - s["never_reached_upright"] / s["episodes"] for s in table[k]]) for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.2, succ, 0.4, label="5 s balance (success)", color="tab:green")
    ax.bar(x + 0.2, up, 0.4, label="reached upright at least once", color="tab:blue")
    for i in range(len(labels)):
        ax.text(x[i] - 0.2, succ[i] + 0.02, f"{succ[i]:.0%}", ha="center", fontsize=8)
        ax.text(x[i] + 0.2, up[i] + 0.02, f"{up[i]:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction of evaluation episodes")
    ax.set_title(f"Greedy evaluation: {N_EVAL} episodes x {len(SEEDS)} trained seeds, "
                 f"identical initial conditions", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_eval_bars.png", dpi=150)


def fig_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, m in zip(axes, ("qlearning", "sarsa")):
        band(ax, runs("baseline", m), "first_upright", "tab:blue",
             "fine arm bins (baseline, 10 bins)", lambda x: (x >= 0).astype(float))
        band(ax, runs("coarsephi", m), "first_upright", "tab:red",
             "coarse arm bins (6 bins)", lambda x: (x >= 0).astype(float))
        ax.set_title(f"{NAMES[m]}: fraction of episodes reaching upright", fontsize=10)
        ax.set_xlabel("training episode")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("fraction (moving average)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_comparison.png", dpi=150)


def fig_visits():
    d = np.load(RES / "baseline_qlearning_seed0.npz")
    cfg = json.loads(str(d["config"]))
    edges = [np.array(cfg["bins"][k]) for k in ("phi", "phi_dot", "theta", "theta_dot")]
    shape = tuple(len(e) + 1 for e in edges)
    v = d["visits"].sum(1).reshape(shape).astype(float)
    names = ["arm angle phi", "arm velocity phi_dot", "pendulum angle theta (wrapped)",
             "pendulum velocity theta_dot"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for k, ax in enumerate(axes):
        other = tuple(i for i in range(4) if i != k)
        frac = v.sum(axis=other) / v.sum()
        e = edges[k]
        labels = ([f"<{e[0]:.2g}"] + [f"{e[i]:.2g}..{e[i + 1]:.2g}" for i in range(len(e) - 1)]
                  + [f">{e[-1]:.2g}"])
        if k == 2:
            labels = ([f"<{np.rad2deg(e[0]):.0f}"]
                      + [f"{np.rad2deg(e[i]):.0f}..{np.rad2deg(e[i + 1]):.0f}" for i in range(len(e) - 1)]
                      + [f">{np.rad2deg(e[-1]):.0f}"])
        ax.bar(range(len(frac)), frac, color="tab:purple")
        ax.set_xticks(range(len(frac)), labels, rotation=70, fontsize=6)
        ax.set_title(names[k], fontsize=9)
    axes[0].set_ylabel("share of all updates")
    fig.tight_layout()
    fig.savefig(OUT / "fig_visits.png", dpi=150)
    counts = d["visits"].ravel()
    return {
        "entries": int(counts.size),
        "entries_visited": int(np.count_nonzero(counts)),
        "entries_visited_10plus": int(np.sum(counts >= 10)),
        "states_visited": int(np.count_nonzero(d["visits"].sum(1))),
        "share_of_updates_top_1pct_entries": float(np.sort(counts)[::-1][: counts.size // 100].sum() / counts.sum()),
        "share_upright_region": float(v[:, :, 4:10, :].sum() / v.sum()),  # theta bins within +-20 deg
    }


def main():
    OUT.mkdir(exist_ok=True)
    fig_training()
    fig_comparison()

    table, seed0 = {}, []
    for tag, label in (("baseline", "baseline"), ("coarsephi", "coarse arm bins")):
        for m in ("qlearning", "sarsa"):
            summaries = []
            for s in SEEDS:
                f = RES / f"{tag}_{m}_seed{s}.npz"
                if not f.exists():
                    continue
                summ, rows = evaluate(str(f), N_EVAL)
                summaries.append(summ)
                if tag == "baseline" and s == 0:
                    seed0.append((summ, rows))
            table[f"{NAMES[m]}\n{label}"] = summaries
    fig_eval(seed0)
    fig_bars(table)
    visits = fig_visits()

    def agg(summaries, key):
        vals = [s[key] for s in summaries if s[key] is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    summary = {"per_config": {}, "visits": visits, "training": {}}
    for k, summaries in table.items():
        summary["per_config"][k.replace("\n", " / ")] = {
            key: agg(summaries, key) for key in
            ("success_rate", "unsafe_rate", "time_limit_rate", "return_mean",
             "return_std", "never_reached_upright", "time_to_upright_mean_s",
             "duration_mean_s")} | {"per_seed": summaries}
    for tag in ("baseline", "coarsephi"):
        for m in ("qlearning", "sarsa"):
            data = runs(tag, m)
            last = [d["first_upright"][-1000:] >= 0 for d in data]
            summary["training"][f"{tag}/{m}"] = {
                "reached_upright_last1000_mean": float(np.mean([x.mean() for x in last])),
                "success_any_episode": int(sum(int(np.sum(d["outcome"] == 0)) for d in data)),
                "train_seconds": [json.loads(str(d["config"]))["train_seconds"] for d in data],
            }
    long = RES / "long50k_qlearning_seed0.npz"
    if long.exists():
        d = np.load(long)
        summary["training"]["long50k/qlearning"] = {
            "reached_upright_last1000": float(np.mean(d["first_upright"][-1000:] >= 0)),
            "success_any_episode": int(np.sum(d["outcome"] == 0)),
        }
        s, _ = evaluate(str(long), N_EVAL)
        summary["long50k_eval"] = s
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_config"}, indent=2))
    for k, v in summary["per_config"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "per_seed"})


if __name__ == "__main__":
    main()


def fig_eval_success():
    """Required plot: fraction of evaluation episodes with a 5 s hold, every policy.

    Reads figures/summary.json (20k runs) and figures/long_eval.json (50k runs,
    written by the evaluation of the long runs), so it can be redrawn alone.
    """
    summ = json.loads((OUT / "summary.json").read_text())["per_config"]
    long = json.loads((OUT / "long_eval.json").read_text())
    def pooled(key):
        per = summ[key]["per_seed"]
        n = sum(s["episodes"] for s in per)
        succ = sum(round(s["success_rate"] * s["episodes"]) for s in per)
        up = sum(s["episodes"] - s["never_reached_upright"] for s in per)
        return succ, up, n
    def one(name):
        s = long[name]
        return round(s["success_rate"] * s["episodes"]), s["episodes"] - s["never_reached_upright"], s["episodes"]
    rows = [
        ("Q-learning\n20k, 3 seeds", pooled("Q-learning / baseline")),
        ("SARSA\n20k, 3 seeds", pooled("SARSA / baseline")),
        ("Q-learning\n20k coarse, 3 seeds", pooled("Q-learning / coarse arm bins")),
        ("SARSA\n20k coarse, 3 seeds", pooled("SARSA / coarse arm bins")),
        ("Q-learning\n50k", one("long50k_qlearning_seed0")),
        ("SARSA\n50k", one("long50k_sarsa_seed0")),
        ("Q-learning\n50k coarse", one("long50kcoarse_qlearning_seed0")),
        ("SARSA\n50k coarse, 2 seeds", tuple(a + b for a, b in zip(one("long50kcoarse_sarsa_seed0"),
                                                                one("long50kcoarse_sarsa_seed1")))),
    ]
    x = np.arange(len(rows))
    succ = [s / n for _, (s, _, n) in rows]
    up = [u / n for _, (_, u, n) in rows]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, up, 0.4, color="tab:blue", alpha=0.35, label="reached upright at least once")
    ax.bar(x + 0.2, succ, 0.4, color="tab:green", label="held 5 s (success)")
    for i, (_, (s, u, n)) in enumerate(rows):
        ax.text(x[i] - 0.2, up[i] + 0.02, f"{u}/{n}", ha="center", fontsize=8)
        ax.text(x[i] + 0.2, succ[i] + 0.02, f"{s}/{n}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of evaluation episodes")
    ax.set_title("Greedy evaluation (20 episodes per policy, seeds 100000-100019): "
                 "five-second balancing was reached only after 50,000 episodes", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_eval_success.png", dpi=150)
