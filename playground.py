"""Interactive playground: tweak the main RL values and watch the effect.

Run:  python playground.py

Drag the sliders at any time; changes take effect immediately, even in the
middle of training.  The progress chart on the right shows how often recent
episodes reached upright, succeeded, or crashed; every slider change is
marked on it, so you can see whether things improved after the change.

Every 50 training episodes the window replays one test episode of the
current agent (greedy, no exploration).  Tick "turbo" to skip the replays
and train much faster.

Buttons
-------
Start / Pause   run or pause training
Reset           throw away the Q table and start learning from scratch
                (needed after switching between SARSA and Q-learning)

This is for building intuition only.  Report results must come from
train.py, where every setting is fixed and recorded for reproducibility.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from agents import AgentConfig, QLearningAgent, SarsaAgent, run_greedy_episode
from discretize import Discretizer
from play import STATUS, PendulumView, status_line
from swingup_env import SwingUpConfig, SwingUpEnv

EPISODES_PER_TICK = 3        # training episodes between screen refreshes
TURBO_EPISODES_PER_TICK = 15
REPLAY_EVERY = 50            # training episodes between replays
REPLAY_STEPS_PER_FRAME = 12  # 4x real time
WINDOW = 50                  # episodes in the rolling statistics


class Playground:
    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(14, 8.5))
        self.fig.canvas.manager.set_window_title("RL playground: Furuta swing-up")

        # --- pendulum (left) and progress chart (right) ----------------------
        ax_side = self.fig.add_axes([0.01, 0.50, 0.26, 0.40])
        ax_top = self.fig.add_axes([0.27, 0.50, 0.25, 0.40])
        self.view = PendulumView(axes=(ax_side, ax_top), text_y=0.455, text_x=0.27)

        self.ax_chart = self.fig.add_axes([0.60, 0.52, 0.38, 0.38])
        ax = self.ax_chart
        (self.l_up,) = ax.plot([], [], color="tab:blue", label="reached upright")
        (self.l_succ,) = ax.plot([], [], color="tab:green", lw=2, label="success (5 s hold)")
        (self.l_crash,) = ax.plot([], [], color="tab:red", ls="--", label="crashed")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 100)
        ax.set_xlabel("training episode", fontsize=8)
        ax.set_ylabel(f"fraction of last {WINDOW} episodes", fontsize=8)
        ax.set_title("Progress  (grey lines = you changed a setting)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="center left")
        self.markers = []

        self.stats_text = self.fig.text(0.5, 0.985, "", ha="center", va="top",
                                        family="monospace", fontsize=10)

        # --- sliders ---------------------------------------------------------
        def slider(y, label, lo, hi, init, fmt="%.3f"):
            ax = self.fig.add_axes([0.20, y, 0.40, 0.025])
            s = Slider(ax, label, lo, hi, valinit=init, valfmt=fmt)
            s.on_changed(lambda _v, name=label.split()[0]: self.note_change(name))
            return s

        self.s_alpha = slider(0.36, "alpha  (learning rate)", 0.01, 1.0, 0.1)
        self.s_gamma = slider(0.32, "gamma  (discount)", 0.80, 0.999, 0.99)
        self.s_eps = slider(0.28, "epsilon  (exploration)", 0.0, 1.0, 1.0)
        self.s_crash = slider(0.24, "crash penalty", -3000, 0, -100, "%.0f")
        self.s_bonus = slider(0.20, "success bonus", 0, 3000, 100, "%.0f")

        ax_checks = self.fig.add_axes([0.68, 0.22, 0.25, 0.14], frameon=False)
        self.c_opts = CheckButtons(ax_checks, ["auto-decay epsilon", "turbo (no replays)"],
                                   [True, False])
        self.c_opts.on_clicked(lambda label: label.startswith("auto")
                               and self.note_change("auto-decay"))

        ax_method = self.fig.add_axes([0.03, 0.04, 0.14, 0.11])
        ax_method.set_title("method", fontsize=9)
        self.r_method = RadioButtons(ax_method, ["Q-learning", "SARSA"])

        ax_start = self.fig.add_axes([0.22, 0.06, 0.13, 0.06])
        ax_reset = self.fig.add_axes([0.38, 0.06, 0.13, 0.06])
        self.b_start = Button(ax_start, "Start")
        self.b_reset = Button(ax_reset, "Reset")
        self.b_start.on_clicked(self.toggle)
        self.b_reset.on_clicked(lambda _e: self.reset())

        self.fig.text(0.56, 0.05,
                      "How to test an idea:  Reset, set your values, Start, and give it\n"
                      "a few hundred episodes.  Then Reset and try a different value.\n"
                      "Compare the progress charts.  One change at a time tells you\n"
                      "what actually helped.   Switch method -> press Reset.",
                      fontsize=8, va="bottom")

        self.disc = Discretizer()
        self.running = False
        self._auto = False      # True while the program itself moves a slider
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
        self.pending = set()
        self.rate = 0.0
        for m in self.markers:
            m.remove()
        self.markers = []
        self._auto = True
        self.s_eps.set_val(1.0)
        self._auto = False
        self.view.draw(0.0, np.pi, "press Start")
        self.update_stats()
        self.update_chart()

    def toggle(self, _event) -> None:
        self.running = not self.running
        self.b_start.label.set_text("Pause" if self.running else "Start")

    def note_change(self, name: str) -> None:
        """Remember that the user changed something; marked on the chart."""
        if not self._auto and self.episode > 0:
            self.pending.add(name)

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
        if self.pending:
            self.add_marker(", ".join(sorted(self.pending)))
            self.pending = set()
        auto_decay, turbo = self.c_opts.get_status()
        n = TURBO_EPISODES_PER_TICK if turbo else EPISODES_PER_TICK
        t0 = time.perf_counter()
        for _ in range(n):
            self.apply_sliders()
            out = self.agent.run_episode(self.env, self.disc)
            self.episode += 1
            self.history.append((out["outcome"], out["first_upright_step"] is not None))
            if auto_decay:
                # linear decay to 0.05 over about 2000 episodes
                self._auto = True
                self.s_eps.set_val(max(0.05, self.s_eps.val - 0.95 / 2000))
                self._auto = False
            if not turbo and self.episode % REPLAY_EVERY == 0:
                self.apply_sliders()
                test = run_greedy_episode(self.agent, self.test_env, self.disc,
                                          self.test_rng, record=True)
                self.replay = {"test": test, "i": 0, "hold": 0}
                break
        dt = time.perf_counter() - t0
        self.rate = 0.8 * self.rate + 0.2 * (n / dt if dt > 0 else 0)
        self.update_stats()
        self.update_chart()

    def add_marker(self, label: str) -> None:
        ax = self.ax_chart
        line = ax.axvline(self.episode, color="0.5", lw=1)
        text = ax.text(self.episode, 1.0, f" {label}", rotation=90, va="top",
                       fontsize=7, color="0.35")
        self.markers += [line, text]

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
                       f"test run after {self.episode} episodes\n"
                       + status_line(i + 1, test["hold_trace"][i], phi,
                                     test["return_trace"][i]) + f"\n{status}")
        r["i"] += REPLAY_STEPS_PER_FRAME

    def rolling(self):
        if not self.history:
            return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        outcomes = np.array([o for o, _ in self.history])
        up = np.array([u for _, u in self.history], dtype=float)
        k = np.ones(WINDOW)
        count = np.convolve(np.ones(len(up)), k)[: len(up)]
        roll = lambda x: np.convolve(x, k)[: len(x)] / count
        x = np.arange(1, len(up) + 1)
        return (x, roll(up), roll((outcomes == "success").astype(float)),
                roll((outcomes == "unsafe").astype(float)))

    def update_chart(self) -> None:
        x, up, succ, crash = self.rolling()
        self.l_up.set_data(x, up)
        self.l_succ.set_data(x, succ)
        self.l_crash.set_data(x, crash)
        self.ax_chart.set_xlim(0, max(100, self.episode))

    def update_stats(self) -> None:
        recent = self.history[-WINDOW:]
        if recent:
            succ = np.mean([o == "success" for o, _ in recent])
            unsafe = np.mean([o == "unsafe" for o, _ in recent])
            up = np.mean([u for _, u in recent])
        else:
            succ = unsafe = up = 0.0
        self.stats_text.set_text(
            f"{self.agent.name}   episode {self.episode}   ({self.rate:.0f} episodes/s)   "
            f"last {WINDOW}:  reached upright {up:4.0%}   "
            f"success {succ:4.0%}   crashed {unsafe:4.0%}")


if __name__ == "__main__":
    pg = Playground()
    plt.show()
