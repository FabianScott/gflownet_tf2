import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CubeEnv:
    def __init__(self, dim=2, size=10, r0=0.01, r1=0.5, r2=1):
        self.dim = dim
        self.size = size
        self.r = np.array([r0, r1, r2])
        self.fill_board()

    def fill_board(self):
        self.reward_space = np.ones([self.size] * self.dim) * self.r[0]
        for coord, i in np.ndenumerate(self.reward_space):
            self.reward_space[coord] = self.calculate_reward(coord)
        self.prob_space = self.reward_space / np.sum(self.reward_space)

    def calculate_reward(self, coord):
        reward = np.ones(2)
        for c in coord:
            reward[0] *= int(0.25 < np.abs(c / (self.size - 1) - 0.5))
            reward[1] *= int(0.3 < np.abs(c / (self.size - 1) - 0.5) <= 0.4)
        return self.r[0] + sum(reward * self.r[1:])

    def get_reward(self, coord):
        return self.reward_space[tuple(coord)]

    def plot_reward_2d(self):
        """Matplotlib output of first two dimensions of reward environment.
        :return: (None) Matplotlib figure object
        """
        top_slice = tuple([slice(0, self.size), slice(0, self.size)] + [0] * (self.dim - 2))
        # plt.imshow(self.reward_space[top_slice], origin='lower')
        # plt.show()
        sns.heatmap(self.reward_space[top_slice])
        plt.imshow(self.reward_space[top_slice], origin='lower')
        plt.show()


if __name__ == '__main__':
    env = CubeEnv()
    env.plot_reward_2d()
