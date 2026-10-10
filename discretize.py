"""Finite state representation for the tabular agents (required work item 2).

Each continuous state variable is split into intervals by a list of interior
bin edges.  np.digitize maps any real value to an interval index:
values below the first edge go to bin 0 and values above the last edge go to
the final bin, so out-of-range values always map to a valid table index.

The pendulum angle is wrapped to [-pi, pi) first, so the full angle range is
covered.  Its edges are finer near upright (balancing needs precision) and
coarser near the bottom (swing-up only needs the rough phase of the swing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from furuta_env import discretize_state
from swingup_env import wrap_to_pi


def _symmetric(values) -> np.ndarray:
    """Edges at -v ... -v1, 0, v1 ... v (sorted, symmetric about zero)."""
    v = np.asarray(values, dtype=float)
    return np.concatenate([-v[::-1], [0.0], v])


@dataclass
class BinConfig:
    """Interior bin edges for [phi, phi_dot, theta_wrapped, theta_dot]."""

    phi: np.ndarray = field(default_factory=lambda: _symmetric([0.5, 1.5]))
    phi_dot: np.ndarray = field(default_factory=lambda: _symmetric([1.0, 4.0]))
    theta: np.ndarray = field(default_factory=lambda: _symmetric(
        np.deg2rad([5, 12, 20, 45, 90, 135])))
    theta_dot: np.ndarray = field(default_factory=lambda: _symmetric(
        [0.5, 1.5, 3.0, 6.0]))

    @property
    def edges(self) -> Tuple[np.ndarray, ...]:
        return (self.phi, self.phi_dot, self.theta, self.theta_dot)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Number of intervals per variable (edges + 1)."""
        return tuple(len(e) + 1 for e in self.edges)

    @property
    def n_states(self) -> int:
        return int(np.prod(self.shape))


class Discretizer:
    """Maps a continuous state to (a) a 4-tuple of bin indices, (b) one row."""

    def __init__(self, bins: BinConfig | None = None) -> None:
        self.bins = bins or BinConfig()

    def indices(self, state: np.ndarray) -> Tuple[int, int, int, int]:
        phi, phi_dot, theta, theta_dot = np.asarray(state, dtype=float)
        wrapped = np.array([phi, phi_dot, wrap_to_pi(theta), theta_dot])
        return discretize_state(wrapped, self.bins.edges)  # supplied helper

    def __call__(self, state: np.ndarray) -> int:
        """Single integer row index into a Q table of n_states rows."""
        return int(np.ravel_multi_index(self.indices(state), self.bins.shape))

    @property
    def n_states(self) -> int:
        return self.bins.n_states


def describe(bins: BinConfig, n_actions: int = 3) -> str:
    names = ["phi", "phi_dot", "theta (wrapped)", "theta_dot"]
    lines = []
    for name, e, n in zip(names, bins.edges, bins.shape):
        lines.append(f"  {name:16s} {n:2d} bins, edges {np.round(e, 3).tolist()}")
    lines.append(f"  states = {' x '.join(map(str, bins.shape))} = {bins.n_states}")
    lines.append(f"  state-action entries = {bins.n_states} x {n_actions} "
                 f"= {bins.n_states * n_actions}")
    return "\n".join(lines)


def run_tests() -> None:
    """Test values below, inside, and above every range (brief, item 2)."""
    d = Discretizer()
    shape = d.bins.shape
    big = 1e6
    names = ["phi", "phi_dot", "theta", "theta_dot"]
    for k, (name, edges) in enumerate(zip(names, d.bins.edges)):
        inside = 0.5 * (edges[len(edges) // 2 - 1] + edges[len(edges) // 2])
        cases = {"below": -big, "inside": inside, "above": big,
                 "on edge": edges[0]}
        for label, value in cases.items():
            state = np.zeros(4)
            state[k] = value
            idx = d.indices(state)
            row = d(state)
            assert all(0 <= i < n for i, n in zip(idx, shape)), (name, label, idx)
            assert 0 <= row < d.n_states
            print(f"  {name:10s} {label:8s} {value:+12.4g} -> bin {idx[k]:2d} "
                  f"of {shape[k]}  (row {row})")
    # Angle wrapping: physically identical angles share a bin.
    for theta in (np.pi - 0.01, -np.pi + 0.01, 3 * np.pi - 0.01, 0.1, 0.1 + 4 * np.pi):
        print(f"  theta {theta:+8.3f} rad wraps to {wrap_to_pi(theta):+.3f} "
              f"-> bin {d.indices([0, 0, theta, 0])[2]}")
    assert d.indices([0, 0, 0.1, 0]) == d.indices([0, 0, 0.1 + 4 * np.pi, 0])
    # Every row is reachable and the mapping is one-to-one on bin tuples.
    rows = {int(np.ravel_multi_index(t, shape)) for t in np.ndindex(*shape)}
    assert rows == set(range(d.n_states))
    print("  all tests passed")


if __name__ == "__main__":
    print("Bin configuration:")
    print(describe(BinConfig()))
    print("\nMapping tests:")
    run_tests()
