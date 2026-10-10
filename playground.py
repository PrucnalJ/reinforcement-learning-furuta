"""Interactive playground: tweak the main RL values and watch the effect.

Run:  python playground.py

Drag the sliders at any time; changes take effect immediately, even in the
middle of training.  Every few training episodes the window replays one test
episode of the current agent (greedy, no exploration) so you can see how it
plays right now.

Buttons
-------
Start / Pause   run or pause training
Reset           throw away the Q table and start learning from scratch
                (needed after switching between SARSA and Q-learning)

This is for building intuition only.  Report results must come from
train.py, where every setting is fixed and recorded for reproducibility.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from agents import AgentConfig, QLearningAgent, SarsaAgent, run_greedy_episode
from discretize import Discretizer
from play import STATUS, PendulumView, status_line
from swingup_env import SwingUpConfig, SwingUpEnv

EPISODES_PER_TICK = 3      # training episodes between screen refreshes
REPLAY_EVERY = 50          # training episodes between replays
REPLAY_STEPS_PER_FRAME = 12  # 4x real time
WINDOW = 50                # episodes in the rolling statistics


class Playground:
    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(12, 8.5))
        self.fig.canvas.manager.set_window_title("RL playground: Furuta swing-up")
        ax_side = self.fig.add_axes([0.03, 0.47, 0.45, 0.42])
        ax_top = self.fig.add_axes([0.52, 0.47, 0.45, 0.42])
        self.view = PendulumView(axes=(ax_side, ax_top), text_y=0.43)

        self.stats_text = self.fig.text(0.5, 0.985, "", ha="center", va="top",
                                        family="monospace", fontsize=10)

        # --- sliders ---------------------------------------------------------
        def slider(y, label, lo, hi, init, fmt="%.3f"):
            ax = self.fig.add_axes([0.25, y, 0.5, 0.025])
            return Slider(ax, label, lo, hi, valinit=init, valfmt=fmt)

        self.s_alpha = slider(0.33, "alpha  (learning rate)", 0.01, 1.0, 0.1)
        self.s_gamma = slider(0.29, "gamma  (discount)", 0.80, 0.999, 0.99)
        self.s_eps = slider(0.25, "epsilon  (exploration)", 0.0, 1.0, 1.0)
        self.s_crash = slider(0.21, "crash penalty", -3000, 0, -100, "%.0f")
        self.s_bonus = slider(0.17, "success bonus", 0, 3000, 100, "%.0f")

        ax_decay = self.fig.add_axes([0.80, 0.24, 0.17, 0.05], frameon=False)
        self.c_decay = CheckButtons(ax_decay, ["auto-decay epsilon"], [True])

        ax_method = self.fig.add_axes([0.03, 0.04, 0.17, 0.11])
        ax_method.set_title("method", fontsize=9)
        self.r_method = RadioButtons(ax_method, ["Q-learning", "SARSA"])

        ax_start = self.fig.add_axes([0.30, 0.06, 0.15, 0.06])
        ax_reset = self.fig.add_axes([0.50, 0.06, 0.15, 0.06])
        self.b_start = Button(ax_start, "Start")
        self.b_reset = Button(ax_reset, "Reset")
        self.b_start.on_clicked(self.toggle)
        self.b_reset.on_clicked(lambda _e: self.reset())

        self.fig.text(0.70, 0.07,
                      "Sliders apply immediately.\n"
                      "Reset = fresh Q table.\n"
                      "Switch method -> press Reset.",
                      fontsize=8, va="bottom")

        self.disc = Discretizer()
        self.running = False
        self.reset()

        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.add_callback(self.tick)
        self.timer.start()

    # ---- setup --------------------------------------------------------------
    def reset(self) -> None:
        self.cfg = SwingUpConfig()
        self.env = SwingUpEnv(self.cfg, seed=0)
        self.test_env = SwingUpEnv(self.cfg, seed=10_000)  # shares the config
        self.test_rng = np.random.default_rng(10_000)
        agent_cls = QLearningAgent if self.r_method.value_selected == "Q-learning" else SarsaAgent
        self.agent = agent_cls(self.disc.n_states, self.env.number_of_actions,
                               AgentConfig(), seed=0)
        self.episode = 0
        self.history = []          # (outcome, reached_upright) per episode
        self.replay = None
        self.s_eps.set_val(1.0)
        self.view.draw(0.0, np.pi, "press Start")
        self.update_stats()

    def toggle(self, _event) -> None:
        self.running = not self.running
        self.b_start.label.set_text("Pause" if self.running else "Start")

    def apply_sliders(self) -> None:
        self.agent.cfg.alpha = self.s_alpha.val
        self.agent.cfg.gamma = self.s_gamma.val
        self.agent.cfg.epsilon = self.s_eps.val
        self.cfg.unsafe_penalty = self.s_crash.val
        self.cfg.success_bonus = self.s_bonus.val

    # ---- main loop ----------------------------------------------------------
    def tick(self) -> None:
        if self.replay is not None:
            self.step_replay()
        elif self.running:
            self.train_some()
        self.fig.canvas.draw_idle()

    def train_some(self) -> None:
        for _ in range(EPISODES_PER_TICK):
            self.apply_sliders()
            out = self.agent.run_episode(self.env, self.disc)
            self.episode += 1
            self.history.append((out["outcome"], out["first_upright_step"] is not None))
            if self.c_decay.get_status()[0]:
                # linear decay to 0.05 over about 2000 episodes
                self.s_eps.set_val(max(0.05, self.s_eps.val - 0.95 / 2000))
            if self.episode % REPLAY_EVERY == 0:
                self.apply_sliders()
                test = run_greedy_episode(self.agent, self.test_env, self.disc,
                                          self.test_rng, record=True)
                self.replay = {"test": test, "i": 0, "hold": 0}
                break
        self.update_stats()

    def step_replay(self) -> None:
        r = self.replay
        test = r["test"]
        n = len(test["state_trace"])
        if r["i"] >= n:
            r["hold"] += 1           # keep the final frame on screen briefly
            if r["hold"] > 40:
                self.replay = None
            return
        i = min(r["i"], n - 1)
        phi, _, theta, _ = test["state_trace"][i]
        done = r["i"] + REPLAY_STEPS_PER_FRAME >= n
        status = STATUS[test["outcome"]] if done else ""
        self.view.draw(phi, theta,
                       f"test run after {self.episode} episodes:  "
                       + status_line(i + 1, test["hold_trace"][i], phi,
                                     test["return_trace"][i]) + f"\n{status}")
        r["i"] += REPLAY_STEPS_PER_FRAME

    def update_stats(self) -> None:
        recent = self.history[-WINDOW:]
        if recent:
            succ = np.mean([o == "success" for o, _ in recent])
            unsafe = np.mean([o == "unsafe" for o, _ in recent])
            up = np.mean([u for _, u in recent])
        else:
            succ = unsafe = up = 0.0
        self.stats_text.set_text(
            f"{self.agent.name}   episode {self.episode}   "
            f"last {WINDOW} episodes:  reached upright {up:4.0%}   "
            f"success {succ:4.0%}   crashed {unsafe:4.0%}")


if __name__ == "__main__":
    pg = Playground()
    plt.show()
