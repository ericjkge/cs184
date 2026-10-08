import numpy as np


class MDP:
    def __init__(self, P, R, discount=None, horizon=None):
        """
        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object
        :param P: Transition function: |S| x |A| x |S'| array
        :param R: Reward function: |S| x |A| array
        :param discount: discount factor: scalar in [0,1), for the infinite-horizon case
        :param horizon: number of steps, for the finite-horizon case
        """

        assert not (discount is None and horizon is None), (
            "Exactly one of discount and horizon must be given"
        )
        if discount is not None:
            assert 0 <= discount < 1, "Discount factor must be in [0,1)"
        self.discount = discount
        self.horizon = horizon
        assert P.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nStates = P.shape[0]
        self.nActions = P.shape[1]
        assert P.shape == (self.nStates, self.nActions, self.nStates), (
            "Invalid transition function: it has dimensionality "
            + repr(P.shape)
            + ", but it should be (nStates,nActions,nStates)"
        )
        assert (abs(P.sum(2) - 1) < 1e-5).all(), (
            "Invalid transition function: some transition probability does not equal 1"
        )
        self.P = P
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions"
        assert R.shape == (self.nStates, self.nActions), (
            "Invalid reward function: it has dimensionality "
            + repr(R.shape)
            + ", but it should be (nStates,nActions)"
        )
        self.R = R

    def isTerminal(self, state):
        return state == self.nStates - 1


def _build_maze_dynamics():
    """
    adapted from https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring18/assignments/asst1/TestRLmaze.py

    Construct a simple maze MDP

    Grid world layout:

    ---------------------
    |  0 |  1 |  2 |  3 |
    ---------------------
    |  4 |  5 |  6 |  7 |
    ---------------------
    |  8 |  9 | 10 | 11 |
    ---------------------
    | 12 | 13 | 14 | 15 |
    ---------------------

    Goal state: 15
    Bad state: 7,8,9
    End state: 16

    The end state is an absorbing state that the agent transitions
    to after visiting the goal state.

    There are 17 states in total (including the end state)
    and 4 actions (up, down, left, right).
    :return: (P, R)
    """
    # Transition function: |S| x |A| x |S'| array
    P = np.zeros([17, 4, 17])
    a = 0.7  # intended move
    b = 0.15  # lateral move

    # up (a = 0)

    P[0, 0, 0] = a + b
    P[0, 0, 1] = b

    P[1, 0, 0] = b
    P[1, 0, 1] = a
    P[1, 0, 2] = b

    P[2, 0, 1] = b
    P[2, 0, 2] = a
    P[2, 0, 3] = b

    P[3, 0, 2] = b
    P[3, 0, 3] = a + b

    P[4, 0, 4] = b
    P[4, 0, 0] = a
    P[4, 0, 5] = b

    P[5, 0, 4] = b
    P[5, 0, 1] = a
    P[5, 0, 6] = b

    P[6, 0, 5] = b
    P[6, 0, 2] = a
    P[6, 0, 7] = b

    P[7, 0, 6] = b
    P[7, 0, 3] = a
    P[7, 0, 7] = b

    P[8, 0, 8] = b
    P[8, 0, 4] = a
    P[8, 0, 9] = b

    P[9, 0, 8] = b
    P[9, 0, 5] = a
    P[9, 0, 10] = b

    P[10, 0, 9] = b
    P[10, 0, 6] = a
    P[10, 0, 11] = b

    P[11, 0, 10] = b
    P[11, 0, 7] = a
    P[11, 0, 11] = b

    P[12, 0, 12] = b
    P[12, 0, 8] = a
    P[12, 0, 13] = b

    P[13, 0, 12] = b
    P[13, 0, 9] = a
    P[13, 0, 14] = b

    P[14, 0, 13] = b
    P[14, 0, 10] = a
    P[14, 0, 15] = b

    P[15, 0, 16] = 1
    P[16, 0, 16] = 1

    # down (a = 1)

    P[0, 1, 0] = b
    P[0, 1, 4] = a
    P[0, 1, 1] = b

    P[1, 1, 0] = b
    P[1, 1, 5] = a
    P[1, 1, 2] = b

    P[2, 1, 1] = b
    P[2, 1, 6] = a
    P[2, 1, 3] = b

    P[3, 1, 2] = b
    P[3, 1, 7] = a
    P[3, 1, 3] = b

    P[4, 1, 4] = b
    P[4, 1, 8] = a
    P[4, 1, 5] = b

    P[5, 1, 4] = b
    P[5, 1, 9] = a
    P[5, 1, 6] = b

    P[6, 1, 5] = b
    P[6, 1, 10] = a
    P[6, 1, 7] = b

    P[7, 1, 6] = b
    P[7, 1, 11] = a
    P[7, 1, 7] = b

    P[8, 1, 8] = b
    P[8, 1, 12] = a
    P[8, 1, 9] = b

    P[9, 1, 8] = b
    P[9, 1, 13] = a
    P[9, 1, 10] = b

    P[10, 1, 9] = b
    P[10, 1, 14] = a
    P[10, 1, 11] = b

    P[11, 1, 10] = b
    P[11, 1, 15] = a
    P[11, 1, 11] = b

    P[12, 1, 12] = a + b
    P[12, 1, 13] = b

    P[13, 1, 12] = b
    P[13, 1, 13] = a
    P[13, 1, 14] = b

    P[14, 1, 13] = b
    P[14, 1, 14] = a
    P[14, 1, 15] = b

    P[15, 1, 16] = 1
    P[16, 1, 16] = 1

    # left (a = 2)

    P[0, 2, 0] = a + b
    P[0, 2, 4] = b

    P[1, 2, 1] = b
    P[1, 2, 0] = a
    P[1, 2, 5] = b

    P[2, 2, 2] = b
    P[2, 2, 1] = a
    P[2, 2, 6] = b

    P[3, 2, 3] = b
    P[3, 2, 2] = a
    P[3, 2, 7] = b

    P[4, 2, 0] = b
    P[4, 2, 4] = a
    P[4, 2, 8] = b

    P[5, 2, 1] = b
    P[5, 2, 4] = a
    P[5, 2, 9] = b

    P[6, 2, 2] = b
    P[6, 2, 5] = a
    P[6, 2, 10] = b

    P[7, 2, 3] = b
    P[7, 2, 6] = a
    P[7, 2, 11] = b

    P[8, 2, 4] = b
    P[8, 2, 8] = a
    P[8, 2, 12] = b

    P[9, 2, 5] = b
    P[9, 2, 8] = a
    P[9, 2, 13] = b

    P[10, 2, 6] = b
    P[10, 2, 9] = a
    P[10, 2, 14] = b

    P[11, 2, 7] = b
    P[11, 2, 10] = a
    P[11, 2, 15] = b

    P[12, 2, 8] = b
    P[12, 2, 12] = a + b

    P[13, 2, 9] = b
    P[13, 2, 12] = a
    P[13, 2, 13] = b

    P[14, 2, 10] = b
    P[14, 2, 13] = a
    P[14, 2, 14] = b

    P[15, 2, 16] = 1
    P[16, 2, 16] = 1

    # right (a = 3)

    P[0, 3, 0] = b
    P[0, 3, 1] = a
    P[0, 3, 4] = b

    P[1, 3, 1] = b
    P[1, 3, 2] = a
    P[1, 3, 5] = b

    P[2, 3, 2] = b
    P[2, 3, 3] = a
    P[2, 3, 6] = b

    P[3, 3, 3] = a + b
    P[3, 3, 7] = b

    P[4, 3, 0] = b
    P[4, 3, 5] = a
    P[4, 3, 8] = b

    P[5, 3, 1] = b
    P[5, 3, 6] = a
    P[5, 3, 9] = b

    P[6, 3, 2] = b
    P[6, 3, 7] = a
    P[6, 3, 10] = b

    P[7, 3, 3] = b
    P[7, 3, 7] = a
    P[7, 3, 11] = b

    P[8, 3, 4] = b
    P[8, 3, 9] = a
    P[8, 3, 12] = b

    P[9, 3, 5] = b
    P[9, 3, 10] = a
    P[9, 3, 13] = b

    P[10, 3, 6] = b
    P[10, 3, 11] = a
    P[10, 3, 14] = b

    P[11, 3, 7] = b
    P[11, 3, 11] = a
    P[11, 3, 15] = b

    P[12, 3, 8] = b
    P[12, 3, 13] = a
    P[12, 3, 12] = b

    P[13, 3, 9] = b
    P[13, 3, 14] = a
    P[13, 3, 13] = b

    P[14, 3, 10] = b
    P[14, 3, 15] = a
    P[14, 3, 14] = b

    P[15, 3, 16] = 1
    P[16, 3, 16] = 1

    # Reward function: |S| x |A| array
    R = -1 * np.ones([17, 4])

    # set rewards
    R[15, :] = 150  # goal state
    R[7, :] = -60  # bad state
    R[8, :] = -60  # bad state
    R[9, :] = -60  # bad state
    R[16, :] = 0  # end state

    return P, R


def build_infinite_horizon_maze_MDP(discount_factor=0.9):
    """Construct the maze MDP under the infinite-horizon discounted formulation."""
    P, R = _build_maze_dynamics()
    return MDP(P, R, discount=discount_factor)


def build_finite_horizon_maze_MDP(time_horizon=10):
    """Construct the maze MDP under the finite-horizon formulation."""
    P, R = _build_maze_dynamics()
    return MDP(P, R, horizon=time_horizon)
