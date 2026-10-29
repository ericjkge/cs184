"""The deadly triad, in one corridor.

Everything in this file is about ONE tiny problem, policy evaluation on a corridor:

    s =  0      1      2    ...   18     19  | exit
        [ ] -> [ ] -> [ ] -> ... -> [ ] -> [ ] -> +1

The states are the integers 0, 1, ..., 19. The policy we evaluate always steps right.
The step out of state 19 ends the episode with reward +1; every other step has reward
0. So the true value is V(s) = GAMMA ** (19 - s), which climbs from about 0.83 at the
left end to exactly 1 at the right end.

What we vary is HOW we learn V, along the three legs of the deadly triad:

    function approximation   Table (one number per state)  vs  MLP (a neural network)
    bootstrapping            n-step target with n = 1 (TD)  vs  n = None (Monte Carlo)
    off-policy data          kappa = 0 (on-policy)          vs  kappa > 0 (skewed to the left)

Data. We do not simulate whole episodes. Each training step draws a batch of states
s ~ mu_kappa and asks the corridor where they go. mu_0 is uniform, which is exactly how
often an episode that starts at s = 0 visits each state: this is the ON-POLICY
distribution. For kappa > 0, mu_kappa(s) is proportional to (N_STATES - s) ** kappa,
so states near the exit are rarely sampled -- as if the data came from some other
policy that usually wanders back to the left before reaching the exit.
"""

import numpy as np
import torch
from torch import nn

GAMMA = 0.99
N_STATES = 20

# Every state, in order. Every plot and every diagnostic is evaluated on these.
STATES = torch.arange(N_STATES)


# ----------------------------------------------------------------------------------
# The corridor (provided)
# ----------------------------------------------------------------------------------


def step(s):
    """One move of the policy we are evaluating, applied to a batch of states.

    Args:
        s: integer tensor of shape (B,), states in {0, ..., N_STATES - 1}.

    Returns:
        (s_next, reward, done), each of shape (B,). Where done is 1, the episode has
        ended, s_next == N_STATES is not a real state, and it must not be bootstrapped
        from.
    """
    s_next = s + 1
    done = (s_next == N_STATES).float()
    reward = done.clone()
    return s_next, reward, done


def true_values(s=STATES):
    """V(s) for the corridor, in closed form."""
    return GAMMA ** (N_STATES - 1 - s).float()


def mu(kappa):
    """The sampling distribution mu_kappa, as a numpy array of shape (N_STATES,).

    mu_kappa(s) is proportional to (N_STATES - s) ** kappa. kappa = 0 is uniform, which
    is the on-policy distribution of the corridor.
    """
    weights = (N_STATES - np.arange(N_STATES)) ** float(kappa)
    return weights / weights.sum()


def sample_states(batch_size, kappa, rng):
    """Draw a batch of states from mu_kappa."""
    return torch.tensor(rng.choice(N_STATES, size=batch_size, p=mu(kappa)))


# ----------------------------------------------------------------------------------
# Function approximators
# ----------------------------------------------------------------------------------


class Table(nn.Module):
    """One learnable number per state: the tabular case (provided)."""

    def __init__(self):
        super().__init__()
        self.values = nn.Parameter(torch.zeros(N_STATES))

    def forward(self, s):
        return self.values[s]


class MLP(nn.Module):
    """A small fully connected network that maps a state s to a value V(s).

    Architecture: 1 -> hidden -> hidden -> 1 with a ReLU after each hidden layer.
    The input is the state rescaled to [0, 1), i.e. s / N_STATES.
    """

    def __init__(self, hidden=64):
        super().__init__()
        # TODO: build self.net, a torch.nn.Sequential with two hidden layers of width
        # `hidden`.
        # HINT: nn.Linear(in_features, out_features), nn.ReLU().
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s):
        """Maps integer states of shape (B,) to values of shape (B,)."""
        # TODO: rescale s to a float in [0, 1). self.net expects input of shape (B, 1)
        # and returns shape (B, 1).
        x = s / N_STATES # Rescale to [0, 1)
        return self.net(x.unsqueeze(1)).squeeze(1) # Convert (B,) to (B, 1) to (B,)


def make_optimizer(model):
    """Plain SGD for the table, so that it is exactly tabular TD; Adam for networks."""
    if isinstance(model, Table):
        return torch.optim.SGD(model.parameters(), lr=1.0)
    return torch.optim.Adam(model.parameters(), lr=1e-3)


# ----------------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------------


@torch.no_grad()
def n_step_target(value_fn, s, n):
    """The n-step return from states s, bootstrapping from value_fn.

        y = r_0 + GAMMA r_1 + ... + GAMMA^(n-1) r_(n-1) + GAMMA^n value_fn(s_n),

    where the bootstrap term is dropped if the episode ends within n steps.
    n = None means never bootstrap: roll out to the end of the episode (Monte Carlo).

    Args:
        value_fn: a callable mapping integer states (B,) to values (B,).
        s: integer tensor of shape (B,).
        n: a positive int, or None.

    Returns:
        A float tensor of shape (B,).
    """
    # TODO: roll every state in the batch forward with step(), accumulating discounted
    # reward, and add the discounted bootstrap value for the states whose episode has
    # not ended. Use only step() and value_fn -- not true_values().
    # HINT: keep a (B,) mask of which episodes are still running. Once an episode ends,
    # stop adding its rewards, and do not call value_fn on s == N_STATES, which is not
    # a state (clamp it, or mask the result out).

    B = s.shape[0]
    y = torch.zeros(B)
    mask = torch.ones(B) # (B,) mask of running episodes (1 = running, 0 = ended)
    discount = 1.0 # GAMMA^t
    t = 0

    if n is None: # Keep running until every episode ends
        while mask.any():
            s_next, reward, done = step(s)
            y += mask * discount * reward # Non-running episodes get 0 reward
            mask = mask * (1 - done) # done = 1 for ended episodes
            s = torch.where(mask.bool(), s_next, s) # Take s_next if running, else freeze on s
            discount *= GAMMA
            t += 1
    else: # Step n times first
        for t in range(n): 
            s_next, reward, done = step(s)
            y += mask * discount * reward
            mask = mask * (1 - done)
            s = torch.where(mask.bool(), s_next, s)
            discount *= GAMMA
        y += mask * discount * value_fn(s) # Bootstrap remaining steps from V(s_n)

    return y

# ----------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------


def train(
    model,
    kappa=0.0,
    n=1,
    steps=1000,
    batch_size=128,
    log_every=100,
    seed=0,
):
    """Semi-gradient value learning on the corridor.

    Each step: sample a batch s ~ mu_kappa, build the n-step target y (held fixed, i.e.
    no gradient flows through it), and take one optimizer step on the loss
    mean(0.5 * (V(s) - y)^2).

    Args:
        model: a Table or an MLP.
        kappa: off-policy skew of the sampling distribution (0 = on-policy).
        n: bootstrap horizon for n_step_target (None = Monte Carlo).

    Returns:
        A dict of lists, one entry per logged step:
            "step":   the step number,
            "values": model(STATES) as a numpy array of shape (N_STATES,).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = make_optimizer(model)
    history = {"step": [], "values": []}

    for t in range(steps + 1):
        if t % log_every == 0:
            with torch.no_grad():
                history["step"].append(t)
                history["values"].append(model(STATES).numpy().copy())
            if not np.isfinite(history["values"][-1]).all():
                break
        if t == steps:
            break

        # Calculate loss
        s = sample_states(batch_size, kappa, rng)
        y = n_step_target(model, s, n)
        loss = (0.5 * (model(s) - y)**2).mean()

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return history

if __name__ == "__main__":
    model = MLP()
    print(model.forward(STATES))