"""Minimal Furuta-pendulum environment for tabular reinforcement learning.

This is a Python restructuring of the supplied MATLAB model.  It uses the
lumped parameters reported in Gaefvert (1998); it is not a calibrated model
of the Quanser QUBE-Servo 3.

State convention
----------------
x = [phi, phi_dot, theta, theta_dot]

The upright equilibrium is theta = 0.  The only control input is the torque
applied to the horizontal arm.  For a tabular agent, an integer action is
mapped to one of a small set of torque values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FurutaParameters:
    """Lumped parameters from the supplied MATLAB simulation."""

    alpha: float = 0.0033472
    beta: float = 0.0038852
    gamma: float = 0.0024879
    delta: float = 0.097625


class FurutaEnv:
    """Small Gym-like environment with ``reset`` and ``step`` methods.

    The environment intentionally has no dependency on Gymnasium.  This keeps
    the starter code transparent and allows students to see the complete
    interaction loop.
    """

    def __init__(
        self,
        dt: float = 0.01,
        max_steps: int = 2000,
        torque_values: Tuple[float, ...] = (-0.02, 0.0, 0.02),
        angle_limit: float = np.deg2rad(20.0),
        arm_angle_limit: float = np.pi,
        process_noise_std: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.p = FurutaParameters()
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.torque_values = np.asarray(torque_values, dtype=float)
        self.angle_limit = float(angle_limit)
        self.arm_angle_limit = float(arm_angle_limit)
        self.process_noise_std = float(process_noise_std)
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=float)
        self.steps = 0

    @property
    def number_of_actions(self) -> int:
        return int(self.torque_values.size)

    def reset(
        self,
        seed: Optional[int] = None,
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Begin a balancing episode near the upright equilibrium."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if initial_state is None:
            # Small random displacement makes repeated episodes non-identical.
            self.state = np.array(
                [
                    self.rng.uniform(-0.02, 0.02),
                    0.0,
                    self.rng.uniform(-0.02, 0.02),
                    0.0,
                ],
                dtype=float,
            )
        else:
            initial_state = np.asarray(initial_state, dtype=float)
            if initial_state.shape != (4,):
                raise ValueError("initial_state must have shape (4,)")
            self.state = initial_state.copy()

        self.steps = 0
        return self.state.copy(), {}

    def dynamics(self, state: np.ndarray, torque: float) -> np.ndarray:
        """Evaluate the continuous-time nonlinear equations of motion."""
        phi, phi_dot, theta, theta_dot = np.asarray(state, dtype=float)
        del phi  # The ideal model is rotationally symmetric in phi.

        alpha = self.p.alpha
        beta = self.p.beta
        gamma = self.p.gamma
        delta = self.p.delta

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        scale = 1.0 / (
            alpha * beta
            - gamma**2
            + (beta**2 + gamma**2) * sin_theta**2
        )

        phi_ddot = scale * (
            beta * gamma * (sin_theta**2 - 1.0) * sin_theta * phi_dot**2
            - 2.0 * beta**2 * cos_theta * sin_theta * phi_dot * theta_dot
            + beta * gamma * sin_theta * theta_dot**2
            - gamma * delta * cos_theta * sin_theta
            + beta * torque
        )

        theta_ddot = scale * (
            beta
            * (alpha + beta * sin_theta**2)
            * cos_theta
            * sin_theta
            * phi_dot**2
            - 2.0
            * beta
            * gamma
            * (1.0 - sin_theta**2)
            * sin_theta
            * phi_dot
            * theta_dot
            - gamma**2 * cos_theta * sin_theta * theta_dot**2
            + delta * (alpha + beta * sin_theta**2) * sin_theta
            - gamma * cos_theta * torque
        )

        return np.array([phi_dot, phi_ddot, theta_dot, theta_ddot])

    def _rk4(self, state: np.ndarray, torque: float) -> np.ndarray:
        """Advance the continuous model by one sample using RK4."""
        h = self.dt
        k1 = self.dynamics(state, torque)
        k2 = self.dynamics(state + 0.5 * h * k1, torque)
        k3 = self.dynamics(state + 0.5 * h * k2, torque)
        k4 = self.dynamics(state + h * k3, torque)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        """Apply one discrete action and return a Gym-like transition tuple."""
        if not 0 <= int(action) < self.number_of_actions:
            raise ValueError(f"action must be in [0, {self.number_of_actions - 1}]")

        torque = float(self.torque_values[int(action)])
        next_state = self._rk4(self.state, torque)

        if self.process_noise_std > 0.0:
            next_state += self.rng.normal(0.0, self.process_noise_std, size=4)

        self.state = next_state
        self.steps += 1

        theta = float(next_state[2])
        phi = float(next_state[0])
        terminated = abs(theta) > self.angle_limit or abs(phi) > self.arm_angle_limit
        truncated = self.steps >= self.max_steps

        # A dense balancing reward.  Students may be asked to justify and test
        # an alternative reward as a project design decision.
        reward = 1.0 - (theta / self.angle_limit) ** 2 - 0.001 * torque**2
        if terminated:
            reward = -10.0

        info = {"torque": torque, "theta": theta, "phi": phi}
        return self.state.copy(), float(reward), terminated, truncated, info


def discretize_state(
    state: np.ndarray,
    bins: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[int, int, int, int]:
    """Convert a continuous state into a tuple suitable for a Q table.

    Each array in ``bins`` contains internal bin boundaries.  For example,
    four boundaries produce five discrete intervals.
    """
    if len(bins) != 4:
        raise ValueError("bins must contain one boundary array per state variable")
    return tuple(
        int(np.digitize(value, boundaries))
        for value, boundaries in zip(np.asarray(state, dtype=float), bins)
    )

