"""Experiments and figures for HW3 Q4(d), Q5(b) and Q6(a).

Put all of your experiment and plotting code in this file, and submit it with triad.py.
It is not autograded, but `python experiments.py` should reproduce every figure in your
PDF. run_cube() and plot_cube() (the Q5 figure) are provided; q4d() and q6a() are yours
to write.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from triad import mu, MLP, Table, train, N_STATES, STATES, true_values, GAMMA

SEEDS = (0, 1, 2)
OFF_KAPPA = 3.0  # "off-policy" in the handout


# ----------------------------------------------------------------------------------
# Provided
# ----------------------------------------------------------------------------------


def run_cube(steps=1000):
    """Train all eight corners of the cube, with 3 seeds each (24 runs, under a minute).

    Returns:
        A dict mapping each corner, e.g. ("MLP", "TD(0)", "off-policy"), to a list with
        one (steps, norms) pair per seed: steps are the logged training steps (0, 100,
        ..., 1000, fewer if the run became non-finite) and norms the matching values
        of ||V_theta||_inf.
    """
    results = {}
    for model_cls in (Table, MLP):  # function approximation: absent, present
        for n, target in ((None, "MC"), (1, "TD(0)")):  # bootstrapping
            for kappa, data in ((0.0, "on-policy"), (OFF_KAPPA, "off-policy")):  # off-policy data
                runs = []
                for seed in SEEDS:
                    torch.manual_seed(seed)  # seeds the initialization
                    model = model_cls()
                    # train's seed argument seeds the data
                    history = train(model, kappa=kappa, n=n, steps=steps, seed=seed)
                    norms = [np.abs(v).max() for v in history["values"]]
                    runs.append((history["step"], norms))
                results[(model_cls.__name__, target, data)] = runs
    return results


def plot_cube(results, path="cube.png"):
    """Q5(b): ||V_theta||_inf over training for every corner, one subplot each."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, ((model, target, data), runs) in zip(axes.flat, results.items()):
        for seed, (steps, norms) in zip(SEEDS, runs):
            ax.plot(steps, norms, label=f"seed {seed}")
        ax.axhline(1, color="black", lw=0.8, ls=":")  # ||V^pi||_inf = 1
        ax.set_yscale("log")
        ax.set_title(f"{model}, {target}, {data}", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("training step")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\|V_\theta\|_\infty$")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(r"$\|V_\theta\|_\infty$ over training (dotted: $\|V^\pi\|_\infty = 1$)")
    fig.tight_layout()
    fig.savefig(path)
    print(f"saved {path}")


# ----------------------------------------------------------------------------------
# Your figures
# ----------------------------------------------------------------------------------


def q4d():
    """Q4(d): the MLP with Monte Carlo targets on on-policy data, against V^pi."""
    # TODO: train MLP() with n=None, kappa=0 for 3 seeds; plot V_theta(s) against s for
    # each seed with true_values() overlaid; print ||V_theta - V^pi||_inf for each seed.
    s = STATES.numpy()
    v_true = true_values().numpy()

    _, ax = plt.subplots(figsize=(6, 4))
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = MLP()
        history = train(model, kappa=0.0, n=None, seed=seed)
        v = history["values"][-1]  # V_theta after the last step
        ax.plot(s, v, marker="o", label=f"$V_\\theta$, seed {seed}")
        print(f"seed {seed}: ||V_theta - V^pi||_inf = {np.abs(v - v_true).max():.4f}")
    ax.plot(s, v_true, "k--", lw=2, label=r"$V^\pi$")
    ax.set_xlabel("state $s$")
    ax.set_ylabel("value")
    ax.set_title("MLP, Monte Carlo, on-policy")
    ax.legend()


def q6a():
    """Q6(a): how often each state is updated vs used as a target."""
    # TODO: plot mu_kappa(s) / (GAMMA * mu_kappa(s - 1)) for s = 1, ..., 19 and
    # kappa in {0, 1, 2, 3}. triad.mu(kappa) gives mu_kappa as an array.
    _, ax = plt.subplots(figsize=(6, 4))
    s = np.arange(1, N_STATES)
    for kappa in (0, 1, 2, 3):
        m = mu(kappa)
        ax.plot(s, m[1:] / (GAMMA * m[:-1]), marker="o", label=f"$\\kappa={kappa}$")
    ax.axhline(1, color="black", ls=":")
    ax.set_xlabel("state $s$")
    ax.set_ylabel(r"$\mu_\kappa(s) / (\gamma \mu_\kappa(s-1))$")
    ax.legend()



if __name__ == "__main__":
    # q4d()
    # plot_cube(run_cube())
    q6a()
    plt.show()
