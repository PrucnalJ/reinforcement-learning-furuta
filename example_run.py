"""Smoke test and minimal interaction loop for ``FurutaEnv``."""

from furuta_env import FurutaEnv


def main() -> None:
    env = FurutaEnv(seed=7)
    state, _ = env.reset()
    total_return = 0.0

    # This is deliberately not a controller.  It only demonstrates the API.
    for _ in range(200):
        action = 1  # zero torque for torque_values=(-0.02, 0.0, 0.02)
        state, reward, terminated, truncated, info = env.step(action)
        total_return += reward
        if terminated or truncated:
            break

    print("final state:", state)
    print("last applied torque:", info["torque"])
    print("episode return:", total_return)
    print("steps:", env.steps)


if __name__ == "__main__":
    main()

