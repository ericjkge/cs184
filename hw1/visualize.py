import argparse

import matplotlib.pyplot as plt
import numpy as np
from IH import InfiniteHorizon
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from MDP import build_infinite_horizon_maze_MDP

ACTION_TITLES = [
    "Up-action (a=0)",
    "Down-action (a=1)",
    "Left-action (a=2)",
    "Right-action (a=3)",
]
POLICY_CMAP = ListedColormap(["w"])
VALUE_CMAP = colormaps["seismic"]


def _span(data):
    """
    Return the half-width of a symmetric colour scale for `seismic`.

    Note:
        Centring on 0 puts it at the colormap's white midpoint, so hue reads as the
        sign of the value. Sharing one span across panels that show the same quantity
        keeps them comparable, and the floor stops an all-zero Q from rendering as a
        saturated block.
    """
    return max(float(np.abs(data).max()), 1.0)


def _label_colour(value, span):
    """
    Return a label colour that stays legible on the cell behind it.

    Note:
        Sampling the colormap beats thresholding on magnitude: `seismic` darkens at
        both ends, and faster on the red side, so a value well short of the maximum
        can still land on a near-black cell.
    """
    r, g, b, _ = VALUE_CMAP((value + span) / (2 * span))
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "black"


class Visualize:
    def __init__(self, DP):
        self.DP = DP

        self.arrows = ["↑", "↓", "←", "→"]
        self.iteration = 0

    def visualize_dp(self, policy, value_function):
        self.visualize_V_and_Pi(value_function, policy)

    def _q_panels(self):
        return [self.Q[:-1, a].reshape(4, 4) for a in range(4)]

    def visualize_value_iteration(self, initialQ):
        self.Q = initialQ
        self.iteration = 0
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(
            f"Q-function values for iteration {self.iteration} of value iteration"
        )

        axes = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]
        span = _span(self.Q[:-1])
        a_img = {}
        text_dictionary = {0: {}, 1: {}, 2: {}, 3: {}}
        for a, (data, axis) in enumerate(zip(self._q_panels(), axes)):
            a_img[a] = axis.matshow(data, cmap=VALUE_CMAP, vmin=-span, vmax=span)
            axis.set_title(ACTION_TITLES[a])
            for i, j in np.ndindex(data.shape):
                text_dictionary[a][(i, j)] = axis.text(
                    j,
                    i,
                    f"{data[i][j]:0.1f}",
                    ha="center",
                    va="center",
                    color=_label_colour(data[i][j], span),
                )

        fig.canvas.mpl_connect(
            "key_press_event",
            lambda event: self.VI_on_keyboard(event, fig, a_img, text_dictionary),
        )

        plt.show()

    def VI_on_keyboard(self, event, fig, a_grid, text_dic):
        if event.key == "right":
            self.iteration += 1
            self.Q = self.DP.valueIterationStep(self.Q)
            span = _span(self.Q[:-1])
            for a, data in enumerate(self._q_panels()):
                a_grid[a].set_array(data)
                a_grid[a].set_clim(vmin=-span, vmax=span)
                for i, j in np.ndindex(data.shape):
                    text_dic[a][(i, j)].set_text(f"{data[i][j]:0.1f}")
                    text_dic[a][(i, j)].set_color(_label_colour(data[i][j], span))

        elif event.key == "enter":
            pi = self.DP.extractPifromQ(self.Q)
            V_pi = self.DP.computeVfromQ(self.Q, pi)
            self.visualize_V_and_Pi(V_pi, pi)
        fig.suptitle(
            f"Q-function values for iteration {self.iteration} of value iteration"
        )

        fig.canvas.flush_events()
        fig.canvas.draw()

    def _draw_V_and_Pi(self, ax, V_data, policy_data):
        """Fill a value panel and an arrow panel, returning their images and labels."""
        span = _span(V_data)
        a_img = ax[0].matshow(V_data, cmap=VALUE_CMAP, vmin=-span, vmax=span)
        b_img = ax[1].matshow(np.zeros((4, 4)), cmap=POLICY_CMAP)

        a_text_dictionary = {}
        b_text_dictionary = {}
        for (i, j), z in np.ndenumerate(V_data):
            a_text_dictionary[(i, j)] = ax[0].text(
                j,
                i,
                f"{z:0.1f}",
                ha="center",
                va="center",
                color=_label_colour(z, span),
            )
        for (i, j), z in np.ndenumerate(policy_data):
            b_text_dictionary[(i, j)] = ax[1].text(
                j, i, self.arrows[z], ha="center", va="center"
            )
        return a_img, b_img, a_text_dictionary, b_text_dictionary

    def _update_V_and_Pi(self, a_grid, a_dic, b_dic, V_data, policy_data):
        span = _span(V_data)
        a_grid.set_array(V_data)
        a_grid.set_clim(vmin=-span, vmax=span)
        for (i, j), z in np.ndenumerate(V_data):
            a_dic[(i, j)].set_text(f"{z:0.1f}")
            a_dic[(i, j)].set_color(_label_colour(z, span))
        for (i, j), z in np.ndenumerate(policy_data):
            b_dic[(i, j)].set_text(self.arrows[z])

    def visualize_policy_iteration(self, initial_pi, exact):
        self.pi = initial_pi
        self.iteration = 0
        fig, ax = plt.subplots(2)
        self.V = (
            self.DP.exactPolicyEvaluation(initial_pi)
            if exact
            else self.DP.iterPolicyEvaluation(initial_pi)[0]
        )
        a_img, _, a_text_dictionary, b_text_dictionary = self._draw_V_and_Pi(
            ax, np.reshape(self.V[:-1], (4, 4)), np.reshape(self.pi[:-1], (4, 4))
        )
        fig.canvas.mpl_connect(
            "key_press_event",
            lambda event: self.PI_on_keyboard(
                event, exact, fig, a_img, a_text_dictionary, b_text_dictionary
            ),
        )

        fig.suptitle(
            f"Iteration {self.iteration} of {'exact' if exact else 'iterative'} policy iteration"
        )

        plt.show()

    def PI_on_keyboard(self, event, exact, fig, a_grid, a_dic, b_dic):
        if event.key == "right":
            self.pi = self.DP.policyIterationStep(self.pi, exact)
            self.V = (
                self.DP.exactPolicyEvaluation(self.pi)
                if exact
                else self.DP.iterPolicyEvaluation(self.pi)[0]
            )
            self._update_V_and_Pi(
                a_grid,
                a_dic,
                b_dic,
                np.reshape(self.V[:-1], (4, 4)),
                np.reshape(self.pi[:-1], (4, 4)),
            )
            self.iteration += 1
            fig.suptitle(
                f"Iteration {self.iteration} of {'exact' if exact else 'iterative'} policy iteration"
            )
            fig.canvas.flush_events()
            fig.canvas.draw()

    def visualize_V_and_Pi(self, V, pi):
        """A static snapshot of one value function and its policy."""
        fig, ax = plt.subplots(2)
        self._draw_V_and_Pi(
            ax,
            np.reshape(V[:-1], (4, 4)),
            np.reshape(pi[:-1].astype(int), (4, 4)),
        )
        fig.suptitle(f"Value Function and Policy on t={self.iteration}")
        plt.show()


if __name__ == "__main__":
    # read system arguments argparse
    parser = argparse.ArgumentParser(
        prog="GridWorldVisualizer",
        description="Visualizer for gridworld MDPs",
        epilog="Example: python3 visualize.py VI",
    )
    parser.add_argument(
        "alg",
        choices=["VI", "PI_exact", "PI_iter"],
        help="Algorithm to visualize",
    )
    parser.add_argument(
        "--gamma", "-G", type=float, default=0.9, help="Discount factor: 0 <= gamma < 1"
    )

    args = parser.parse_args()
    assert 0 <= args.gamma < 1, "Invalid gamma"

    dp = InfiniteHorizon(build_infinite_horizon_maze_MDP(args.gamma))
    v = Visualize(dp)
    if args.alg == "VI":
        initial_Q = np.zeros((dp.nStates, dp.nActions))
        v.visualize_value_iteration(initial_Q)
    else:
        initial_pi = np.zeros(dp.nStates, dtype="int")
        v.visualize_policy_iteration(initial_pi, args.alg == "PI_exact")
