"""Evaluate saved Q tables with a frozen, greedy policy (required item 4/5).

Run:  python evaluate.py results/baseline_sarsa_seed0.npz results/baseline_qlearning_seed0.npz

For every saved table this runs the same N evaluation episodes (identical
initial conditions, from seeds EVAL_SEED_BASE + i), with no exploration and
no learning, and reports:
  success rate, mean and std of return, time to first enter the upright
  region (episodes that never reach upright are counted and reported
  explicitly), and episode duration.
It also saves a plot of the wrapped pendulum angle vs time for
representative episodes and a bar chart of the success fraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agents import QLearningAgent, SarsaAgent, run_greedy_episode
from discretize import BinConfig, Discretizer
from swingup_env import SwingUpConfig, SwingUpEnv

EVAL_SEED_BASE = 100_000


def load(path: str):
    d = np.load(path)
    cfg = json.loads(str(d["config"]))
    bins = BinConfig(**{k: np.array(v) for k, v in cfg["bins"].items()})
    env_cfg = SwingUpConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                               for k, v in cfg["env"].items()})
    method = cfg["args"]["method"]
    cls = QLearningAgent if method == "qlearning" else SarsaAgent
    disc = Discretizer(bins)
    agent = cls(disc.n_states, len(env_cfg.torque_values))
    agent.Q = d["Q"].copy()
    agent.Q.setflags(write=False)       # frozen: any write would raise
    return agent, disc, env_cfg, cfg


def evaluate(path: str, n: int):
    agent, disc, env_cfg, cfg = load(path)
    rows = []
    for i in range(n):
        seed = EVAL_SEED_BASE + i
        env = SwingUpEnv(env_cfg, seed=seed)
        out = run_greedy_episode(agent, env, disc,
                                 np.random.default_rng(seed), record=True)
        rows.append(out)
    returns = np.array([r["return"] for r in rows])
    durations = np.array([r["sim_steps"] for r in rows]) * env_cfg.dt
    outcomes = [r["outcome"] for r in rows]
    first = [r["first_upright_step"] for r in rows]
    reached = [f for f in first if f is not None]
    summary = {
        "file": path,
        "method": agent.name,
        "episodes": n,
        "eval_seeds": f"{EVAL_SEED_BASE}..{EVAL_SEED_BASE + n - 1}",
        "success_rate": outcomes.count("success") / n,
        "unsafe_rate": outcomes.count("unsafe") / n,
        "time_limit_rate": outcomes.count("time_limit") / n,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)),
        "never_reached_upright": n - len(reached),
        "time_to_upright_mean_s": (float(np.mean(reached)) * env_cfg.dt if reached else None),
        "time_to_upright_std_s": (float(np.std(reached, ddof=1)) * env_cfg.dt
                                  if len(reached) > 1 else None),
        "duration_mean_s": float(durations.mean()),
        "duration_std_s": float(durations.std(ddof=1)),
    }
    return summary, rows


def print_summary(s: dict) -> None:
    print(f"\n{s['method']}  ({s['file']})")
    print(f"  {s['episodes']} greedy episodes, seeds {s['eval_seeds']}")
    print(f"  success rate        {s['success_rate']:6.1%}   "
          f"(unsafe {s['unsafe_rate']:.0%}, time limit {s['time_limit_rate']:.0%})")
    print(f"  return              {s['return_mean']:9.1f} +- {s['return_std']:.1f}")
    if s["time_to_upright_mean_s"] is not None:
        sd = s["time_to_upright_std_s"]
        print(f"  time to upright     {s['time_to_upright_mean_s']:6.2f} s"
              + (f" +- {sd:.2f}" if sd is not None else ""))
    print(f"  never reached upright: {s['never_reached_upright']} of {s['episodes']}")
    print(f"  episode duration    {s['duration_mean_s']:6.2f} +- {s['duration_std_s']:.2f} s")


def plot(results, out_dir: Path, n_traces: int = 5) -> None:
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 3.2 * len(results)),
                             squeeze=False)
    for ax, (s, rows) in zip(axes[:, 0], results):
        for r in rows[:n_traces]:
            n = len(r["theta_trace"])          # one point per agent decision
            t = np.arange(1, n + 1) * (r["sim_steps"] * 0.01 / n)
            ax.plot(t, np.rad2deg(r["theta_trace"]), lw=1,
                    label=f"{r['outcome']}")
        ax.axhspan(-20, 20, color="tab:green", alpha=0.15)
        ax.set_ylim(-185, 185)
        ax.set_yticks([-180, -90, -20, 0, 20, 90, 180])
        ax.set_ylabel("wrapped pendulum angle (deg)")
        ax.set_title(f"{s['method']}: first {n_traces} evaluation episodes "
                     f"(0 = upright, green = +-20 deg region)", fontsize=10)
        ax.legend(fontsize=7, loc="lower right", ncol=n_traces)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "eval_theta_traces.png", dpi=130)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    names = [s["method"] for s, _ in results]
    vals = [s["success_rate"] for s, _ in results]
    ax.bar(names, vals, color=["tab:blue", "tab:orange", "tab:green", "tab:red"][:len(vals)])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction of evaluation episodes")
    ax.set_title(f"Five seconds of continuous balancing\n"
                 f"({results[0][0]['episodes']} episodes, same seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "eval_success.png", dpi=130)
    print(f"\nSaved {out_dir / 'eval_theta_traces.png'} and {out_dir / 'eval_success.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("-n", "--episodes", type=int, default=20)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    results = [evaluate(f, args.episodes) for f in args.files]
    for s, _ in results:
        print_summary(s)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "eval_summary.json").write_text(json.dumps([s for s, _ in results], indent=2))
    plot(results, out)
