"""Exploration study: compare three epsilon schedules at the baseline settings.

Run after training the epsfixed01 and epsfast runs (see README):
    python exploration_study.py

Schedules (20,000 episodes, fine arm bins, seeds 0-2):
  baseline    epsilon 1.0 -> 0.05 over 15,000 episodes
  epsfast     epsilon 1.0 -> 0.05 over 5,000 episodes
  epsfixed01  epsilon fixed at 0.1
Prints training and frozen-greedy evaluation results and writes
figures/exploration.json.
"""

import json

import numpy as np

from evaluate import evaluate

SCHEDULES = {"baseline": "1.0 to 0.05 over 15,000 episodes",
             "epsfast": "1.0 to 0.05 over 5,000 episodes",
             "epsfixed01": "fixed 0.1"}
SEEDS = (0, 1, 2)
N_EVAL = 20


def main():
    out = {}
    for tag, label in SCHEDULES.items():
        for method in ("qlearning", "sarsa"):
            reached_train, reached_eval, succ_eval, holds, dur = [], [], [], 0, []
            for seed in SEEDS:
                path = f"results/final/{tag}_{method}_seed{seed}.npz"
                log = np.load(path, allow_pickle=True)
                first = log["first_upright"][-1000:]
                reached_train.append(float(np.mean(first >= 0)))
                holds += int(np.sum(log["outcome"] == 0))   # 0 = success
                s, _ = evaluate(path, N_EVAL)
                reached_eval.append(N_EVAL - s["never_reached_upright"])
                succ_eval.append(round(s["success_rate"] * N_EVAL))
                dur.append(s["duration_mean_s"])
            row = {"schedule": label, "method": method,
                   "train_reached_last1000": float(np.mean(reached_train)),
                   "train_holds": holds,
                   "eval_reached": int(sum(reached_eval)), "eval_reached_per_seed": reached_eval,
                   "eval_success": int(sum(succ_eval)), "eval_duration_s": float(np.mean(dur))}
            out[f"{tag}_{method}"] = row
            print(f"{tag:11s} {method:9s} train upright {row['train_reached_last1000']:5.1%}  "
                  f"holds {holds:3d}  eval upright {row['eval_reached']:2d}/60 {reached_eval}  "
                  f"success {row['eval_success']}/60  duration {row['eval_duration_s']:.1f} s")
    with open("figures/exploration.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
