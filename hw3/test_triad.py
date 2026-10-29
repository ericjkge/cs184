"""Sanity checks for triad.py. Run with `python -m pytest test_triad.py` (a few seconds).

Passing these means your pieces are wired up correctly, not that your figures or
explanations are right -- those are graded from your PDF.
"""

import numpy as np
import torch

import triad
from triad import GAMMA, MLP, N_STATES, STATES, Table


def test_mlp_shapes():
    model = MLP()
    assert model(torch.arange(7)).shape == (7,)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == (1 * 64 + 64) + (64 * 64 + 64) + (64 * 1 + 1)


def test_monte_carlo_target_is_true_value():
    y = triad.n_step_target(lambda s: torch.full(s.shape, 123.0), STATES, None)
    assert torch.allclose(y, triad.true_values(), atol=1e-5)


def test_n_step_target_bootstraps():
    # Far from the exit, the n-step target is just GAMMA^n times the bootstrap value.
    s = torch.tensor([0, 2, 6])
    for n in (1, 3):
        y = triad.n_step_target(lambda t: 10.0 * t, s, n)
        assert torch.allclose(y, GAMMA**n * 10.0 * (s + n))


def test_n_step_target_stops_at_exit():
    # From the last state, one step exits with reward 1: no bootstrap, for any n.
    last = torch.tensor([N_STATES - 1])
    for n in (1, 5, None):
        y = triad.n_step_target(lambda t: torch.full(t.shape, 1e6), last, n)
        assert torch.allclose(y, torch.ones(1))
    # Two steps from the exit with n = 4: reward arrives on the 2nd step.
    y = triad.n_step_target(lambda t: torch.full(t.shape, 1e6), torch.tensor([N_STATES - 2]), 4)
    assert torch.allclose(y, torch.tensor([GAMMA]))


def test_mu_is_a_distribution():
    for kappa in (0, 1.5, 3):
        assert np.isclose(triad.mu(kappa).sum(), 1)
    assert np.allclose(triad.mu(0), 1 / N_STATES)


def test_train_tabular_on_policy_converges():
    h = triad.train(Table(), kappa=0.0, n=1, steps=3000)
    assert np.abs(h["values"][-1] - triad.true_values().numpy()).max() < 0.05
