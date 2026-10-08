"""Required work item 1: verify the supplied environment interface.

Run:  python verify_env.py
"""

import numpy as np

from furuta_env import FurutaEnv

STATE_NAMES = ["phi (arm angle, rad)", "phi_dot (arm velocity, rad/s)",
               "theta (pendulum angle, rad)", "theta_dot (pendulum velocity, rad/s)"]


def main() -> None:
    env = FurutaEnv(seed=0)

    # 1. The four state components returned by reset().
    print("=== 1. State returned by reset() ===")
    state, _ = env.reset(seed=0)
    for name, value in zip(STATE_NAMES, state):
        print(f"  {name:38s} {value:+.5f}")
    print(f"  dt = {env.dt} s, max_steps = {env.max_steps} "
          f"({env.dt * env.max_steps:.0f} s), "
          f"angle_limit = {np.rad2deg(env.angle_limit):.0f} deg, "
          f"arm_angle_limit = {env.arm_angle_limit:.4f} rad")

    # 2. Apply every legal action from the same state and check the torque.
    print("\n=== 2. Every legal action and the torque reported in info ===")
    for action in range(env.number_of_actions):
        env.reset(initial_state=state)
        next_state, reward, terminated, truncated, info = env.step(action)
        expected = env.torque_values[action]
        ok = "OK" if info["torque"] == expected else "MISMATCH"
        print(f"  action {action}: torque {info['torque']:+.3f} N m "
              f"(expected {expected:+.3f}) {ok}; "
              f"theta_ddot effect -> theta_dot = {next_state[3]:+.6f}")
    try:
        env.step(3)
    except ValueError as err:
        print(f"  illegal action 3 rejected: {err}")

    # 3. One transition that continues the episode.
    print("\n=== 3. One continuing transition (S_t, A_t, R_t+1, S_t+1) ===")
    s, _ = env.reset(initial_state=state)
    s_next, r, terminated, truncated, info = env.step(1)
    print(f"  S_t     = {np.round(s, 5)}")
    print(f"  A_t     = 1 (torque {info['torque']:+.2f} N m)")
    print(f"  R_t+1   = {r:.5f}")
    print(f"  S_t+1   = {np.round(s_next, 5)}")
    print(f"  terminated = {terminated}, truncated = {truncated}")

    # 4a. Unsafe termination: pendulum leaves the allowed angle.
    print("\n=== 4a. Termination: pendulum angle leaves the limit ===")
    env.reset(initial_state=np.array([0.0, 0.0, 0.05, 0.0]))
    for _ in range(env.max_steps):
        s, r, terminated, truncated, info = env.step(1)
        if terminated or truncated:
            break
    print(f"  stopped after {env.steps} steps, theta = "
          f"{np.rad2deg(s[2]):.2f} deg, reward = {r}, "
          f"terminated = {terminated}, truncated = {truncated}")

    # 4b. Unsafe termination: arm angle leaves the limit.
    print("\n=== 4b. Termination: arm angle leaves the limit ===")
    env.reset(initial_state=np.array([np.pi - 0.001, 1.0, 0.0, 0.0]))
    s, r, terminated, truncated, info = env.step(1)
    print(f"  phi = {s[0]:.4f} rad (> pi), reward = {r}, "
          f"terminated = {terminated}, truncated = {truncated}")

    # 4c. Time-limit truncation: use a short episode so it is reached.
    print("\n=== 4c. Truncation: maximum episode duration reached ===")
    short_env = FurutaEnv(max_steps=5, seed=0)
    short_env.reset(initial_state=np.zeros(4))  # exact equilibrium: stays put
    for _ in range(10):
        s, r, terminated, truncated, info = short_env.step(1)
        if terminated or truncated:
            break
    print(f"  stopped after {short_env.steps} steps, "
          f"terminated = {terminated}, truncated = {truncated}")

    # 5. Who chooses what.
    print("\n=== 5. Agent vs environment ===")
    print("  Chosen by the agent:      the action A_t (an integer 0, 1 or 2)")
    print("  Produced by environment:  next state S_t+1, reward R_t+1, "
          "terminated/truncated flags, info (torque, theta, phi)")


if __name__ == "__main__":
    main()
