from env_MAB import *

def random_argmax(a):
    """
    Select the index corresponding to the maximum in the input list.
    Ties are randomly broken.
    """
    return np.random.choice(np.where(a == a.max())[0])


class Explore:
    def __init__(self, MAB: MAB):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        pulls = self.MAB.get_record().sum(axis=1)
        arm = random_argmax(-pulls) # Choose arm with least pulls
        self.MAB.pull(arm)

class Greedy:
    """
    Pull each arm once. After that, at every step pull the arm with the highest
    current empirical mean (its running average, updated after every pull).
    """

    def __init__(self, MAB: MAB):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        record = self.MAB.get_record()
        pulls = record.sum(axis=1)

        # Pull each arm once
        for i, pull in enumerate(pulls):
            if pull == 0:
                self.MAB.pull(i)
                return

        # Pull arm with highest empirical mean
        averages = record[:, 1] / record.sum(axis=1)
        self.MAB.pull(random_argmax(averages))

class ETC:
    """
    Explore each arm N_e times, then commit to the arm with the highest mean
    from the exploration phase forever (no more updating).
    """

    def __init__(self, MAB: MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta
        self.committed = None

    def reset(self):
        self.MAB.reset()
        self.committed = None

    def play_one_step(self):
        T, K = self.MAB.get_T(), self.MAB.get_K()
        N_e = int(np.floor((T * np.sqrt(np.log(2 * K / self.delta) / 2) / K) ** (2/3)))

        record = self.MAB.get_record()
        pulls = record.sum(axis=1)

        # Pull each arm N_e times
        for i, pull in enumerate(pulls):
            if pull < N_e:
                self.MAB.pull(i)
                return

        # Commit to the arm with highest empirical mean
        if self.committed is None:
            averages = record[:, 1] / pulls
            self.committed = random_argmax(averages)
        self.MAB.pull(self.committed)

class Epgreedy:
    """
    Pull each arm once. After that, with probability
    epsilon_t = min(1, (K ln(t) / t)^(1/3)), where t is the total number of
    pulls so far, pull an arm chosen uniformly from all K arms; otherwise pull
    the arm with the highest running average.
    """

    def __init__(self, MAB: MAB):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        K = self.MAB.get_K()
        record = self.MAB.get_record()
        pulls = record.sum(axis=1)

        # Pull each arm once
        for i, pull in enumerate(pulls):
            if pull == 0:
                self.MAB.pull(i)
                return

        t = pulls.sum()
        epsilon_t = min(1, (K * np.log(t) / t) ** (1/3))

        if np.random.rand() < epsilon_t: # Explore
            self.MAB.pull(np.random.randint(K)) # Random pull
        else: # Exploit
            averages = record[:, 1] / pulls
            self.MAB.pull(random_argmax(averages))

class UCB:
    def __init__(self, MAB: MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta

    def reset(self):
        """
        Reset the instance and eliminate history.
        """
        self.MAB.reset()

    def play_one_step(self):
        """
        Implement one step of the UCB algorithm.
        """
        T, K = self.MAB.get_T(), self.MAB.get_K()
        record = self.MAB.get_record()
        pulls = record.sum(axis=1)

        # Pull each arm once
        for i, pull in enumerate(pulls):
            if pull == 0:
                self.MAB.pull(i)
                return

        averages = record[:, 1] / pulls
        bonus = np.sqrt(np.log(2 * K * T / self.delta) / (2 * pulls))
        self.MAB.pull(random_argmax(averages + bonus))