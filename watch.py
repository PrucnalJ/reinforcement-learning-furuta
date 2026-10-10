"""Watch a trained agent play the swing-up task.

Run:  python watch.py results/baseline_qlearning_seed0.npz

The agent acts greedily from its saved Q table (no exploration, no learning),
exactly as in evaluation.  Episodes restart automatically; r resets, q quits.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

from discretize import BinConfig, Discretizer
from play import Game


def load_policy(path: str):
    data = np.load(path)
    Q = data["Q"]
    bins = json.loads(str(data["config"]))["bins"]
    disc = Discretizer(BinConfig(**{k: np.array(v) for k, v in bins.items()}))
    rng = np.random.default_rng(0)

    def policy(state) -> int:
        row = Q[disc(state)]
        best = np.flatnonzero(row == row.max())
        return int(rng.choice(best))

    return policy


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    game = Game(policy=load_policy(sys.argv[1]),
                title=f"Agent: {sys.argv[1]}", auto_restart=True)
    plt.show()
