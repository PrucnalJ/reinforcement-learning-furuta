"""Watch a trained agent play the swing-up task.

Run:  python watch.py results/final/long50kcoarse_qlearning_seed0.npz
      python watch.py results/final/crash100_qlearning_seed0.npz --eval-episode 1 3 5 6 4
      python watch.py results/final/long50kcoarse_qlearning_seed0.npz --eval-episode 10

The agent acts greedily from its saved Q table (no exploration, no learning),
exactly as in evaluation, in the environment settings it was trained with.
Episodes restart automatically; r resets, q quits.

--eval-episode K [K ...] replays those evaluation episodes (seed 100000 + K, the same
initial condition and tie-breaking as evaluate.py) once, then stops; r replays.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from discretize import BinConfig, Discretizer
from evaluate import EVAL_SEED_BASE
from play import Game
from swingup_env import SwingUpConfig


def load_env_config(cfg: dict) -> SwingUpConfig:
    env = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["env"].items()}
    # Runs saved before the 0.05 s control period decided every simulation step.
    env.setdefault("action_repeat", 1)
    return SwingUpConfig(**env)


class GreedyPolicy:
    """Greedy action with random tie-breaking, as TabularAgent.greedy."""

    def __init__(self, Q, disc, seed):
        self.Q, self.disc, self.seed = Q, disc, seed
        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(self.seed if seed is None else seed)

    def __call__(self, state) -> int:
        row = self.Q[self.disc(state)]
        best = np.flatnonzero(row == row.max())
        return int(best[0]) if best.size == 1 else int(self.rng.choice(best))


def load_policy(path: str, seed: int = 0):
    data = np.load(path)
    cfg = json.loads(str(data["config"]))
    disc = Discretizer(BinConfig(**{k: np.array(v) for k, v in cfg["bins"].items()}))
    return GreedyPolicy(data["Q"], disc, seed), load_env_config(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", help="saved .npz from train.py")
    ap.add_argument("--eval-episode", type=int, nargs="+", default=None,
                    help="replay these evaluation episodes (0-19) in order, then stop")
    args = ap.parse_args()
    if args.eval_episode is None:
        policy, env_cfg = load_policy(args.path)
        game = Game(policy=policy, title=f"Agent: {args.path}",
                    auto_restart=True, env_cfg=env_cfg)
    else:
        seeds = [EVAL_SEED_BASE + k for k in args.eval_episode]
        policy, env_cfg = load_policy(args.path, seeds[0])
        listed = " ".join(map(str, args.eval_episode))
        game = Game(policy=policy, auto_restart=True, env_cfg=env_cfg,
                    replay_seeds=seeds,
                    title=f"Agent: {args.path}  (evaluation episodes {listed})")
    plt.show()
