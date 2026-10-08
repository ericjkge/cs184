import numpy as np


class InfiniteHorizon:
    def __init__(self, MDP):
        self.R = MDP.R  # |S|x|A|
        self.P = MDP.P  # |S|x|A|x|S|
        self.discount = MDP.discount
        self.nStates = MDP.nStates
        self.nActions = MDP.nActions

    def extractRfrompi(self, pi: np.ndarray) -> np.ndarray:
        """
        Return R(s, pi(s)) for all states.

        Note:
            This should be used in policy evaluation and policy iteration.

            Given an m x n matrix A, the expression

                A[row_indices, col_indices] (len(row_indices) == len(col_indices))

            returns a matrix of size len(row_indices) that contains the elements

                A[row_indices[i], col_indices[i]] in a row for all indices i.

        Parameters:
            `pi`: A deterministic policy. An array of |S| integers, each of which
                specifies an action (column) for a given state s.

        Returns:
            An array of |S| entries.
        """
        return self.R[np.arange(len(self.R)), pi]

    def extractPfrompi(self, pi: np.ndarray) -> np.ndarray:
        """
        Returns P^pi.

        Parameters:
            `pi`: A deterministic policy. An array of |S| integers.

        Returns:
            An |S|x|S| matrix where the (i,j) entry corresponds to P(j|i, pi(i)).
        """
        return self.P[np.arange(len(self.P)), pi]

    def computeVfromQ(self, Q: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """
        Return the V function for a given Q function corresponding to a deterministic
        policy pi.

        Note:
            Remember that

                V^pi(s) = Q^pi(s, pi(s))

        Parameters:
            `Q`: Q function. An array of |S|x|A| numbers.
            `pi`: Policy. An array of |S| integers.

        Returns:
            An array of |S| numbers.
        """
        # TODO 1
        return Q[np.arange(len(pi)), pi] # Fancy indexing in np

    def computeQfromV(self, V: np.ndarray) -> np.ndarray:
        """
        Return the Q function given a V function corresponding to a policy pi.

        Note:
            Use the bellman equation for Q-function to compute Q from V.

        Parameters:
            `V`: Value function. An array of |S| numbers.

        Returns:
            An |S|x|A| array.
        """
        # TODO 2
        return self.R + self.discount * (self.P @ V)

    def extractPifromQ(self, Q: np.ndarray) -> np.ndarray:
        """
        Return the policy pi corresponding to the Q-function.

        Note:
            The policy is determined by

                pi(s) = argmax_a Q(s,a)

        Parameters:
            `Q`: Q function. An array of |S|x|A| numbers.

        Returns:
            An array of |S| integers.
        """
        # TODO 3
        return np.argmax(Q, axis=1) # Optimal (greedy) policy


    def extractPifromV(self, V: np.ndarray) -> np.ndarray:
        """
        Return the policy corresponding to the V-function.

        Note:
            Compute the Q-function from the given V-function and then extract the
            policy via:
                pi(s) = argmax_a Q(s,a)

        Parameters:
            `V`: V function. An array of |S| numbers.

        Returns:
            An array of |S| integers.
        """
        # TODO 4
        Q = self.computeQfromV(V)
        return self.extractPifromQ(Q)

    def valueIterationStep(self, Q: np.ndarray) -> np.ndarray:
        """
        Return the Q function after one step of value iteration.

        Note:
            The input Q can be thought of as the Q-value at iteration t. Return
            Q^{t+1}.

        Parameters:
            `Q`: Q function. An array of |S|x|A| numbers.

        Returns:
            An array of |S|x|A| numbers.
        """
        # TODO 5
        V = Q.max(axis=1) # V_t(s') = max_{a'} Q_t(s',a')
        return self.computeQfromV(V)

    def valueIteration(
        self, initialQ: np.ndarray, tolerance: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """
        Run value iteration on the input initial Q-function until a certain tolerance
        is met.

        Note:
            Specifically, value iteration should continue to run until

                ||Q^t-Q^{t+1}||_inf <= tolerance

            Recall that for a vector v, ||v||_inf is the maximum absolute element of v.

        Parameters:
            `initialQ`: Initial Q-function. An array of |S|x|A| entries.
            `tolerance`: Threshold on ||Q^t-Q^{t+1}||_inf. Float >= 0 (default: 0.01).

        Returns:
            The policy, value function, number of iterations required for convergence,
            and the end epsilon where the epsilon is ||Q^t-Q^{t+1}||_inf.
        """
        # TODO 6
        Q = initialQ
        iterations = 0
        while True:
            Q_next = self.valueIterationStep(Q)
            epsilon = np.max(np.abs(Q - Q_next))
            Q = Q_next
            iterations += 1
            if epsilon <= tolerance:
                break
        pi = self.extractPifromQ(Q)
        V = self.computeVfromQ(Q, pi)
        return pi, V, iterations, epsilon


    ### EXACT POLICY EVALUATION  ###
    def exactPolicyEvaluation(self, pi: np.ndarray) -> np.ndarray:
        """
        Evaluate a policy by solving a system of linear equations.

        Note:
            V^pi = R^pi + gamma P^pi V^pi

        Parameters:
            `pi`: Deterministic policy. An array of |S| integers.

        Returns:
            The value function. An array of |S| numbers.
        """
        # TODO 7
        R = self.extractRfrompi(pi)
        P = self.extractPfrompi(pi)
        return np.linalg.solve(np.eye(P.shape[0]) - self.discount * P, R) # Solve (I - gamma * P^pi) V^pi = R^pi for V^pi

    ### ITERATIVE POLICY EVALUATION ###
    def iterPolicyEvaluation(
        self,
        pi: np.ndarray,
        initialV: np.ndarray | None = None,
        tolerance: float = 0.01,
    ) -> tuple[np.ndarray, int, float]:
        """
        Evaluate a policy using iterative policy evaluation.

        Note:
            Like value iteration, iterative policy evaluation should continue until

                ||V_n - V_{n+1}||_inf <= tolerance

        Parameters:
            `pi`: Deterministic policy. An array of |S| integers.
            `initialV`: Initial value function. An array of |S| numbers
                (default: all zeros).
            `tolerance`: Threshold on ||V^n-V^n+1||_inf. Float >= 0 (default: 0.01).

        Returns:
            The value function, number of iterations required to get to exactness
            criterion, and final epsilon value.
        """
        # TODO 8
        R = self.extractRfrompi(pi)
        P = self.extractPfrompi(pi)
        V = np.zeros(self.nStates) if initialV is None else initialV
        iterations = 0
        while True:
            V_next = R + self.discount * (P @ V)
            epsilon = np.max(np.abs(V - V_next))
            V = V_next
            iterations += 1
            if epsilon <= tolerance:
                break
        return V, iterations, epsilon

    def policyIterationStep(self, pi: np.ndarray, exact: bool) -> np.ndarray:
        """
        Run one step of policy evaluation, followed by one step of policy improvement.

        Note:
            Return pi^{t+1} as a new numpy array. Do not modify pi^t.

        Parameters:
            `pi`: Current policy pi^t. An array of |S| integers.
            `exact`: Indicate whether to use exact policy evaluation.

        Returns:
            An array of |S| integers.
        """
        # TODO 9
        if exact:
            V = self.exactPolicyEvaluation(pi)
        else:
            V, _, _ = self.iterPolicyEvaluation(pi)
        return self.extractPifromV(V) # Calls self.extractPifromQ, which returns the optimal (greedy) policy based on Q calculated from V


    def policyIteration(
        self, initial_pi: np.ndarray, exact: bool
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Alternate between policy evaluation and policy improvement.

        Note:
            Policy evaluation solves V^pi = R^pi + gamma T^pi V^pi; policy improvement
            takes pi <-- argmax_a Q^pi(s,a).

            This function should run policyIteration until convergence where
            convergence is defined as pi^{t+1}(s) == pi^t(s) for all states s. As in
            valueIteration, count only the steps that changed the policy; the final
            step that detects convergence does not count.

        Parameters:
            `initial_pi`: Initial policy. An array of |S| entries.
            `exact`: Indicate whether to use exact policy evaluation.

        Returns:
            The final policy, its corresponding value function, and the number
            of iterations that were required until convergence.
        """
        # TODO 10
        pi = initial_pi
        iterations = 0
        while True:
            pi_next = self.policyIterationStep(pi, exact)
            if np.array_equal(pi, pi_next):
                break
            pi = pi_next
            iterations += 1
        if exact:
            V = self.exactPolicyEvaluation(pi)
        else:
            V, _, _ = self.iterPolicyEvaluation(pi)
        return pi, V, iterations
