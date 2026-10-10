"""Play the swing-up task yourself with the keyboard.

Run:  python play.py

Controls
--------
left arrow   apply -0.02 N m  (action 0)
right arrow  apply +0.02 N m  (action 2)
no key       zero torque      (action 1)
r            reset
q            quit

Goal: swing the pendulum up and keep it within 20 deg of upright for 5 s,
without the arm passing +-180 deg.  This is exactly the task the RL agent
must learn, using the same three actions.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from swingup_env import SwingUpEnv

STEPS_PER_FRAME = 3          # 3 x 10 ms per 30 ms frame = roughly real time
FRAME_INTERVAL_MS = 30


class Game:
    """policy=None: you press the keys.  policy=f(state)->action: f plays."""

    def __init__(self, policy=None, title="Furuta swing-up", auto_restart=False,
                 seed=None) -> None:
        self.env = SwingUpEnv(seed=seed)
        self.policy = policy
        self.auto_restart = auto_restart
        self.keys = set()
        self.episode = 0
        self.pause_frames = 0
        self.reset()

        self.fig, (self.ax_side, self.ax_top) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.canvas.manager.set_window_title(title)
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.fig.canvas.mpl_connect("key_release_event", self.on_release)

        # Side view: the pendulum angle. Up is theta = 0.
        ax = self.ax_side
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.set_title("Pendulum (side view)")
        ax.axis("off")
        for sign in (-1, 1):  # the +-20 deg upright region
            a = sign * np.deg2rad(20)
            ax.plot([0, np.sin(a)], [0, np.cos(a)], color="tab:green", lw=1, ls="--")
        (self.pend_line,) = ax.plot([], [], lw=4, color="tab:red")
        (self.pend_bob,) = ax.plot([], [], "o", ms=14, color="tab:red")
        ax.plot([0], [0], "ko")

        # Top view: the arm angle with the +-180 deg limit behind it.
        ax = self.ax_top
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.set_title("Arm (top view)")
        ax.axis("off")
        circle = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(circle), np.sin(circle), color="0.85")
        ax.plot([0, 0], [-1.15, -1.0], color="tab:orange", lw=3)  # +-180 limit
        (self.arm_line,) = ax.plot([], [], lw=4, color="tab:blue")
        ax.plot([0], [0], "ko")

        self.text = self.fig.text(0.5, 0.04, "", ha="center", family="monospace")
        controls = ("<- / ->  push arm     r  reset     q  quit" if policy is None
                    else "agent is playing     r  reset     q  quit")
        self.fig.text(0.5, 0.95, controls, ha="center")

        self.anim = FuncAnimation(self.fig, self.update,
                                  interval=FRAME_INTERVAL_MS,
                                  cache_frame_data=False)

    def reset(self) -> None:
        self.state, _ = self.env.reset()
        self.total_reward = 0.0
        self.outcome = "running"
        self.episode += 1

    def on_press(self, event) -> None:
        if event.key == "r":
            self.reset()
        elif event.key == "q":
            plt.close(self.fig)
        else:
            self.keys.add(event.key)

    def on_release(self, event) -> None:
        self.keys.discard(event.key)

    def action(self) -> int:
        if self.policy is not None:
            return self.policy(self.state)
        if "left" in self.keys and "right" not in self.keys:
            return 0
        if "right" in self.keys and "left" not in self.keys:
            return 2
        return 1

    def update(self, _frame):
        info = {"hold_counter": self.env.hold_counter}
        if self.outcome == "running":
            for _ in range(STEPS_PER_FRAME):
                self.state, r, term, trunc, info = self.env.step(self.action())
                self.total_reward += r
                if term or trunc:
                    self.outcome = info["outcome"]
                    self.pause_frames = 40   # hold the final frame ~1 s
                    break
        elif self.auto_restart:
            self.pause_frames -= 1
            if self.pause_frames <= 0:
                self.reset()

        phi, _, theta, _ = self.state
        # Side view: theta = 0 points up, theta = pi points down.
        x, y = np.sin(theta), np.cos(theta)
        self.pend_line.set_data([0, x], [0, y])
        self.pend_bob.set_data([x], [y])
        # Top view: phi = 0 points up the screen.
        self.arm_line.set_data([0, np.sin(phi)], [0, np.cos(phi)])

        status = {
            "running": "",
            "success": "SUCCESS! 5 s upright.  Press r to play again.",
            "unsafe": "FAILED: arm or speed limit hit.  Press r.",
            "time_limit": "Time up.  Press r.",
        }[self.outcome]
        self.text.set_text(
            f"episode {self.episode}   "
            f"time {self.env.steps * self.env.cfg.dt:5.2f} s   "
            f"upright hold {self.env.hold_counter * self.env.cfg.dt:4.2f}/5.00 s   "
            f"arm {np.rad2deg(phi):+6.0f} deg   return {self.total_reward:8.1f}\n"
            f"{status}"
        )
        return self.pend_line, self.pend_bob, self.arm_line, self.text


if __name__ == "__main__":
    game = Game()
    plt.show()
