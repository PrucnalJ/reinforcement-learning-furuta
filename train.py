"""Train a SARSA or Q-learning agent on the swing-up task.

Examples
--------
python train.py --method sarsa --episodes 3000 --seed 0
python train.py --method qlearning --episodes 3000 --seed 0
python train.py --method qlearning --episodes 3000 --live   # watch it learn

Each run saves results/<tag>_<method>_seed<seed>.npz containing the learned
Q table, visit counts, per-episode training logs, and the full configuration,
so evaluation and plots can be reproduced from the saved file.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from agents import AgentConfig, QLearningAgent, SarsaAgent, run_greedy_episode
from discretize import BinConfig, Discretizer
from swingup_env import SwingUpConfig, SwingUpEnv

AGENTS = {"sarsa": SarsaAgent, "qlearning": QLearningAgent}
OUTCOMES = ["success", "unsafe", "time_limit"]


class LivePendulum:
    """--live: after each printed terminal line, replay one greedy test
    episode of the current agent in the pendulum window, then keep training."""

    def __init__(self, title: str, speed: float) -> None:
        import matplotlib.pyplot as plt
        from play import PendulumView
        self.plt = plt
        plt.ion()
        self.view = PendulumView(title)
        self.steps_per_frame = max(1, int(round(3 * speed)))
        plt.show(block=False)

    def replay(self, header: str, test: dict) -> None:
        from play import STATUS, status_line
        self.view.header.set_text(header)
        states = test["state_trace"]
        n = len(states)
        frames = list(range(0, n, self.steps_per_frame)) + [n - 1]
        for i in frames:
            if not self.plt.fignum_exists(self.view.fig.number):
                return  # window closed: keep training without the view
            phi, _, theta, _ = states[i]
            done = i == n - 1
            line = status_line(i + 1, test["hold_trace"][i], phi, test["return_trace"][i])
            status = STATUS[test["outcome"]] if done else ""
            self.view.draw(phi, theta, f"{line}\n{status}")
            self.plt.pause(0.001)
        self.plt.pause(0.6)  # hold the final frame briefly

    def finish(self) -> None:
        self.plt.ioff()
        if self.plt.fignum_exists(self.view.fig.number):
            self.plt.show()


def epsilon_at(episode: int, args) -> float:
    """Linear decay from eps_start to eps_end over the first decay episodes."""
    if args.eps_decay_episodes <= 0:
        return args.eps_start
    frac = min(1.0, episode / args.eps_decay_episodes)
    return args.eps_start + frac * (args.eps_end - args.eps_start)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=AGENTS, required=True)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--q-init", type=float, default=0.0)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=SwingUpConfig.max_steps)
    p.add_argument("--tag", default="baseline")
    p.add_argument("--out", default="results")
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--live", action="store_true",
                   help="replay the current agent in the pendulum window at each printed line")
    p.add_argument("--live-speed", type=float, default=4.0,
                   help="replay speed relative to real time (default 4x)")
    return p.parse_args(argv)


def train(args) -> Path:
    env_cfg = SwingUpConfig(max_steps=args.max_steps)
    bins = BinConfig()
    agent_cfg = AgentConfig(alpha=args.alpha, gamma=args.gamma,
                            epsilon=args.eps_start, q_init=args.q_init)

    env = SwingUpEnv(env_cfg, seed=args.seed)
    disc = Discretizer(bins)
    agent = AGENTS[args.method](disc.n_states, env.number_of_actions,
                                agent_cfg, seed=args.seed)

    n = args.episodes
    log = {
        "return": np.zeros(n),
        "steps": np.zeros(n, dtype=int),
        "outcome": np.zeros(n, dtype=int),       # index into OUTCOMES
        "first_upright": np.full(n, -1, dtype=int),
        "epsilon": np.zeros(n),
    }

    print(f"Training {agent.name}: {n} episodes, seed {args.seed}, "
          f"alpha {args.alpha}, gamma {args.gamma}, "
          f"epsilon {args.eps_start} -> {args.eps_end} over {args.eps_decay_episodes}")
    print(f"Q table: {disc.n_states} states x {env.number_of_actions} actions")
    live = (LivePendulum(f"Training {agent.name}, seed {args.seed}", args.live_speed)
            if args.live else None)
    # Separate env and RNG for the live greedy test episodes, so watching
    # does not change the training run (results are identical with/without).
    test_env = SwingUpEnv(env_cfg, seed=10_000 + args.seed)
    test_rng = np.random.default_rng(10_000 + args.seed)
    t0 = time.perf_counter()
    for ep in range(n):
        agent.cfg.epsilon = epsilon_at(ep, args)
        out = agent.run_episode(env, disc)
        log["return"][ep] = out["return"]
        log["steps"][ep] = out["steps"]
        log["outcome"][ep] = OUTCOMES.index(out["outcome"])
        log["first_upright"][ep] = out["first_upright_step"] or -1
        log["epsilon"][ep] = agent.cfg.epsilon

        if (ep + 1) % args.print_every == 0 or ep == n - 1:
            w = slice(max(0, ep + 1 - args.print_every), ep + 1)
            succ = np.mean(log["outcome"][w] == 0)
            unsafe = np.mean(log["outcome"][w] == 1)
            reached = np.mean(log["first_upright"][w] >= 0)
            print(f"  ep {ep + 1:5d} | eps {agent.cfg.epsilon:.3f} | "
                  f"avg return {log['return'][w].mean():9.1f} | "
                  f"avg steps {log['steps'][w].mean():6.0f} | "
                  f"reached upright {reached:4.0%} | success {succ:4.0%} | "
                  f"unsafe {unsafe:4.0%} | {time.perf_counter() - t0:6.0f} s")
            if live:
                test = run_greedy_episode(agent, test_env, disc, test_rng, record=True)
                live.replay(f"training episode {ep + 1}/{n}   eps {agent.cfg.epsilon:.3f}   "
                            f"success {succ:.0%}   unsafe {unsafe:.0%}\n"
                            f"test run of the current agent (no exploration), "
                            f"{args.live_speed:g}x speed", test)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.tag}_{args.method}_seed{args.seed}.npz"
    config = {
        "args": vars(args),
        "env": {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(env_cfg).items()},
        "bins": {k: np.asarray(v).tolist() for k, v in asdict(bins).items()},
        "agent": asdict(agent_cfg) | {"epsilon": "schedule"},
        "train_seconds": time.perf_counter() - t0,
    }
    np.savez_compressed(path, Q=agent.Q, visits=agent.visits,
                        config=json.dumps(config), **log)
    print(f"Saved {path}")
    if live:
        print("Close the pendulum window to exit.")
        live.finish()
    return path


if __name__ == "__main__":
    train(parse_args())
