"""Swing-up task built on top of the supplied FurutaEnv.

The supplied environment is configured for balancing near upright.  The
project brief asks for swing-up from the downward position, so this wrapper
changes the *task* (initial condition, reward, episode outcomes) while
reusing the supplied plant dynamics and RK4 integration unchanged.

Episode outcomes
----------------
success     the pendulum stays within 20 deg of upright for 500 consecutive
            steps (5 s at dt = 0.01 s)
unsafe      |phi| > arm_angle_limit or a velocity exceeds its safety limit
time limit  max_steps reached before success (truncation, not failure)

Reward (baseline, from the brief)
---------------------------------
R_{t+1} = cos(theta_wrapped) - lambda_phi * phi^2 - lambda_u * u^2
          + success_bonus  on success
          + unsafe_penalty on unsafe termination
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from furuta_env import FurutaEnv


def wrap_to_pi(angle: float) -> float:
    """Map an angle to [-pi, pi), so upright is 0 and downward is +-pi."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class SwingUpConfig:
    """Every task parameter in one place so experiments can change them."""

    dt: float = 0.01
    max_steps: int = 3000                 # 30 s
    torque_values: Tuple[float, ...] = (-0.02, 0.0, 0.02)
    upright_limit: float = np.deg2rad(20.0)
    hold_steps: int = 500                 # 5 s / 0.01 s
    arm_angle_limit: float = np.pi
    arm_velocity_limit: float = 30.0      # rad/s
    pendulum_velocity_limit: float = 30.0 # rad/s
    initial_angle_noise: float = np.deg2rad(5.0)
    initial_velocity_noise: float = 0.05  # rad/s
    lambda_phi: float = 0.05
    lambda_u: float = 1.0
    success_bonus: float = 100.0
    unsafe_penalty: float = -100.0
    process_noise_std: float = 0.0


class SwingUpEnv:
    """Gym-like wrapper: reset() and step(action) with the swing-up task."""

    def __init__(self, config: Optional[SwingUpConfig] = None,
                 seed: Optional[int] = None) -> None:
        self.cfg = config or SwingUpConfig()
        # The supplied environment's own termination is disabled here
        # (angle_limit = inf, arm limit = inf, very long horizon); this
        # wrapper decides when an episode ends.
        self.env = FurutaEnv(
            dt=self.cfg.dt,
            max_steps=10**9,
            torque_values=self.cfg.torque_values,
            angle_limit=np.inf,
            arm_angle_limit=np.inf,
            process_noise_std=self.cfg.process_noise_std,
            seed=seed,
        )
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.hold_counter = 0
        self.first_upright_step: Optional[int] = None

    @property
    def number_of_actions(self) -> int:
        return self.env.number_of_actions

    @property
    def state(self) -> np.ndarray:
        return self.env.state.copy()

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Start near the downward equilibrium: theta = pi + U(-5, 5) deg."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        initial = np.array([
            0.0,
            self.rng.uniform(-c.initial_velocity_noise, c.initial_velocity_noise),
            np.pi + self.rng.uniform(-c.initial_angle_noise, c.initial_angle_noise),
            self.rng.uniform(-c.initial_velocity_noise, c.initial_velocity_noise),
        ])
        self.env.reset(initial_state=initial)
        self.steps = 0
        self.hold_counter = 0
        self.first_upright_step = None
        return self.state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        c = self.cfg
        state, _, _, _, info = self.env.step(action)  # supplied reward unused
        self.steps += 1
        phi, phi_dot, theta, theta_dot = state
        theta_w = wrap_to_pi(theta)
        torque = info["torque"]

        upright = abs(theta_w) <= c.upright_limit
        self.hold_counter = self.hold_counter + 1 if upright else 0
        if upright and self.first_upright_step is None:
            self.first_upright_step = self.steps

        success = self.hold_counter >= c.hold_steps
        unsafe = (abs(phi) > c.arm_angle_limit
                  or abs(phi_dot) > c.arm_velocity_limit
                  or abs(theta_dot) > c.pendulum_velocity_limit)
        terminated = success or unsafe
        truncated = (not terminated) and self.steps >= c.max_steps

        reward = np.cos(theta_w) - c.lambda_phi * phi**2 - c.lambda_u * torque**2
        if success:
            reward += c.success_bonus
        if unsafe:
            reward += c.unsafe_penalty

        outcome = ("success" if success else "unsafe" if unsafe
                   else "time_limit" if truncated else "running")
        info = {
            "torque": torque,
            "theta_wrapped": theta_w,
            "phi": phi,
            "hold_counter": self.hold_counter,
            "first_upright_step": self.first_upright_step,
            "outcome": outcome,
        }
        return state.copy(), float(reward), terminated, truncated, info
