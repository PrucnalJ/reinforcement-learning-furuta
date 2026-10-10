"""Tabular SARSA and Q-learning (required work item 3).

The episode loops follow the Lecture 5 pseudocode (slides 31 and 36).
The two methods share everything except the update target, which is
written out in sarsa_target() and q_learning_target() below.

Terminal transitions (success or unsafe) use continuation value zero:
target = r.  A time-limit truncation is not a real terminal state, so the
target still bootstraps from the next state there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass
class AgentConfig:
    alpha: float = 0.1          # learning rate
    gamma: float = 0.99         # discount factor
    epsilon: float = 0.1        # exploration rate (may be changed by a schedule)
    q_init: float = 0.0         # explicit Q-table initialization


class TabularAgent:
    """Q table, epsilon-greedy behavior, and greedy evaluation policy."""

    name = "tabular"

    def __init__(self, n_states: int, n_actions: int,
                 config: AgentConfig | None = None, seed: int | None = None) -> None:
        self.cfg = config or AgentConfig()
        self.n_actions = n_actions
        self.Q = np.full((n_states, n_actions), self.cfg.q_init, dtype=float)
        self.visits = np.zeros((n_states, n_actions), dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    # ---- action selection (Lecture 5, slides 26-28) -----------------------
    def greedy(self, s: int, rng: np.random.Generator | None = None) -> int:
        """argmax_a Q(s, a); ties are broken uniformly at random."""
        rng = rng or self.rng
        row = self.Q[s]
        best = np.flatnonzero(row == row.max())
        return int(best[0]) if best.size == 1 else int(rng.choice(best))

    def epsilon_greedy(self, s: int) -> int:
        """With probability epsilon a uniformly random action, else greedy.

        This gives each of m tied greedy actions (1 - eps)/m + eps/n and
        every other action eps/n, as on Lecture 5 slide 28.
        """
        if self.rng.random() < self.cfg.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(s)

    # ---- shared update rule (Lecture 5, slide 7) --------------------------
    def update(self, s: int, a: int, target: float) -> None:
        """Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)]"""
        self.Q[s, a] += self.cfg.alpha * (target - self.Q[s, a])
        self.visits[s, a] += 1

    def run_episode(self, env, discretize: Callable[[np.ndarray], int]) -> Dict:
        raise NotImplementedError


class SarsaAgent(TabularAgent):
    """On-policy: the target uses the next action the agent will actually take."""

    name = "SARSA"

    def sarsa_target(self, r: float, s_next: int, a_next: int) -> float:
        # R_{t+1} + gamma * Q(S_{t+1}, A_{t+1})
        return r + self.cfg.gamma * self.Q[s_next, a_next]

    def run_episode(self, env, discretize):
        state, _ = env.reset()
        s = discretize(state)
        a = self.epsilon_greedy(s)
        total, steps = 0.0, 0
        while True:
            state, r, terminated, truncated, info = env.step(a)
            s_next = discretize(state)
            total += r
            steps += 1
            if terminated:
                target = r
                a_next = None
            else:
                a_next = self.epsilon_greedy(s_next)
                target = self.sarsa_target(r, s_next, a_next)
            self.update(s, a, target)
            if terminated or truncated:
                return {"return": total, "steps": steps, **info}
            s, a = s_next, a_next


class QLearningAgent(TabularAgent):
    """Off-policy: the target uses the greedy (max) next action value."""

    name = "Q-learning"

    def q_learning_target(self, r: float, s_next: int) -> float:
        # R_{t+1} + gamma * max_a Q(S_{t+1}, a)
        return r + self.cfg.gamma * self.Q[s_next].max()

    def run_episode(self, env, discretize):
        state, _ = env.reset()
        s = discretize(state)
        total, steps = 0.0, 0
        while True:
            a = self.epsilon_greedy(s)
            state, r, terminated, truncated, info = env.step(a)
            s_next = discretize(state)
            total += r
            steps += 1
            if terminated:
                target = r
            else:
                target = self.q_learning_target(r, s_next)
            self.update(s, a, target)
            if terminated or truncated:
                return {"return": total, "steps": steps, **info}
            s = s_next


def run_greedy_episode(agent: TabularAgent, env, discretize,
                       rng: np.random.Generator, record: bool = False) -> Dict:
    """Evaluation (required work item 4): Q frozen, no exploration, no updates."""
    state, _ = env.reset()
    total, steps = 0.0, 0
    thetas = [] if record else None
    while True:
        a = agent.greedy(discretize(state), rng)   # reads Q, never writes it
        state, r, terminated, truncated, info = env.step(a)
        total += r
        steps += 1
        if record:
            thetas.append(info["theta_wrapped"])
        if terminated or truncated:
            out = {"return": total, "steps": steps, **info}
            if record:
                out["theta_trace"] = np.array(thetas)
            return out
