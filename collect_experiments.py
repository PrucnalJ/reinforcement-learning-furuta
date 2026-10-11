"""Collect every saved training run into one table: experiments.csv.

Run:  python collect_experiments.py

For each results/**/*.npz it reads the saved configuration and per-episode
logs and reports, over the last 1000 training episodes: how often episodes
reached the upright region, how often they crashed, average episode length,
and the total number of successful five-second holds.
"""

import csv
import glob
import json
from pathlib import Path

import numpy as np

ROWS = []
for path in sorted(glob.glob("results/**/*.npz", recursive=True)):
    d = np.load(path)
    cfg = json.loads(str(d["config"]))
    a, env, bins = cfg["args"], cfg["env"], cfg["bins"]
    n = len(d["return"])
    last = slice(max(0, n - 1000), n)
    ROWS.append({
        "file": Path(path).as_posix(),
        "tag": a["tag"],
        "method": a["method"],
        "seed": a["seed"],
        "episodes": n,
        "gamma": a["gamma"],
        "alpha": a["alpha"],
        "crash_penalty": env.get("unsafe_penalty"),
        "action_repeat": env.get("action_repeat", 1),
        "arm_bins": len(bins["phi"]) + 1,
        "reached_upright_last1000": round(float(np.mean(d["first_upright"][last] >= 0)), 4),
        "crashed_last1000": round(float(np.mean(d["outcome"][last] == 1)), 4),
        "avg_seconds_last1000": round(float(np.mean(d["steps"][last])) * env["dt"], 2),
        "successes_total": int(np.sum(d["outcome"] == 0)),
    })

with open("experiments.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ROWS[0]))
    w.writeheader()
    w.writerows(ROWS)
for r in ROWS:
    print(f"{r['tag']:16s} {r['method']:9s} s{r['seed']} ep{r['episodes']:6d} "
          f"g{r['gamma']} rep{r['action_repeat']} pen{r['crash_penalty']} arm{r['arm_bins']:2d}  "
          f"up {r['reached_upright_last1000']:5.1%} crash {r['crashed_last1000']:5.1%} "
          f"len {r['avg_seconds_last1000']:5.2f}s succ {r['successes_total']}")
print(f"\nWrote experiments.csv ({len(ROWS)} runs)")
