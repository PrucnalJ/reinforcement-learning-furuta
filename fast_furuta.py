"""Speed-only re-implementation of the supplied FurutaEnv step.

FurutaEnv evaluates its equations with small numpy arrays, which is slow
when called one step at a time.  FastFurutaEnv evaluates the *same*
equations (copied term by term from furuta_env.FurutaEnv.dynamics) with
scalar math and the same fourth-order Runge-Kutta step, so it produces the
same trajectories to floating-point precision but several times faster.

Nothing about the plant model, parameters, integrator, or time step changes.
check_equivalence() compares it against the supplied environment; run
`python fast_furuta.py` to see the comparison.
"""

from __future__ import annotations

import math

import numpy as np

from furuta_env import FurutaEnv


class FastFurutaEnv(FurutaEnv):
    def _accel(self, phi_dot, theta, theta_dot, torque):
        """phi_ddot and theta_ddot exactly as in FurutaEnv.dynamics."""
        alpha, beta = self.p.alpha, self.p.beta
        gamma, delta = self.p.gamma, self.p.delta
        s = math.sin(theta)
        c = math.cos(theta)
        s2 = s * s
        scale = 1.0 / (alpha * beta - gamma**2 + (beta**2 + gamma**2) * s2)
        phi_ddot = scale * (
            beta * gamma * (s2 - 1.0) * s * phi_dot**2
            - 2.0 * beta**2 * c * s * phi_dot * theta_dot
            + beta * gamma * s * theta_dot**2
            - gamma * delta * c * s
            + beta * torque
        )
        theta_ddot = scale * (
            beta * (alpha + beta * s2) * c * s * phi_dot**2
            - 2.0 * beta * gamma * (1.0 - s2) * s * phi_dot * theta_dot
            - gamma**2 * c * s * theta_dot**2
            + delta * (alpha + beta * s2) * s
            - gamma * c * torque
        )
        return phi_ddot, theta_ddot

    def _rk4(self, state, torque):
        """Same RK4 step as FurutaEnv._rk4, written with scalars."""
        h = self.dt
        p, pd, t, td = (float(v) for v in state)
        a1, b1 = self._accel(pd, t, td, torque)
        p2, pd2, t2, td2 = p + 0.5*h*pd, pd + 0.5*h*a1, t + 0.5*h*td, td + 0.5*h*b1
        a2, b2 = self._accel(pd2, t2, td2, torque)
        p3, pd3, t3, td3 = p + 0.5*h*pd2, pd + 0.5*h*a2, t + 0.5*h*td2, td + 0.5*h*b2
        a3, b3 = self._accel(pd3, t3, td3, torque)
        p4, pd4, t4, td4 = p + h*pd3, pd + h*a3, t + h*td3, td + h*b3
        a4, b4 = self._accel(pd4, t4, td4, torque)
        return np.array([
            p + (h / 6.0) * (pd + 2*pd2 + 2*pd3 + pd4),
            pd + (h / 6.0) * (a1 + 2*a2 + 2*a3 + a4),
            t + (h / 6.0) * (td + 2*td2 + 2*td3 + td4),
            td + (h / 6.0) * (b1 + 2*b2 + 2*b3 + b4),
        ])


def check_equivalence(n_steps: int = 5000, seed: int = 0) -> float:
    """Max absolute state difference between supplied and fast env."""
    kwargs = dict(angle_limit=np.inf, arm_angle_limit=np.inf, max_steps=10**9)
    slow, fast = FurutaEnv(**kwargs), FastFurutaEnv(**kwargs)
    x0 = np.array([0.0, 0.03, np.pi + 0.05, -0.02])
    slow.reset(initial_state=x0)
    fast.reset(initial_state=x0)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_steps):
        a = int(rng.integers(3))
        s1 = slow.step(a)[0]
        s2 = fast.step(a)[0]
        worst = max(worst, float(np.max(np.abs(s1 - s2))))
    return worst


if __name__ == "__main__":
    import time
    print(f"max |difference| over 5000 random steps: {check_equivalence():.2e}")
    for cls in (FurutaEnv, FastFurutaEnv):
        env = cls(angle_limit=np.inf, arm_angle_limit=np.inf, max_steps=10**9)
        env.reset(initial_state=np.array([0.0, 0.0, np.pi, 0.0]))
        t0 = time.perf_counter()
        for i in range(20000):
            env.step(i % 3)
        print(f"{cls.__name__:14s} {20000 / (time.perf_counter() - t0):8.0f} steps/s")
