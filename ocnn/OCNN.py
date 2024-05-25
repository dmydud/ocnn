from enum import Enum

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class ActivationFunction(Enum):
    LOGISTIC_MAP = 'logistic map'

    @property
    def to_func(self):
        if self is ActivationFunction.LOGISTIC_MAP:
            return lambda x: 1 - 2 * x ** 2
        raise ValueError

    def __call__(self, x):
        if self is ActivationFunction.LOGISTIC_MAP:
            return self.to_func(x)
        raise ValueError


class InteractionFunction(Enum):
    DIPOLE = 'dipole'
    GAUSSIAN = 'gaussian'

    def __call__(self, dist, a=1, with_zero_on_diag=True):
        if self is InteractionFunction.DIPOLE:
            interaction = a ** 3 / (a ** 3 + dist ** 3)
        elif self is InteractionFunction.GAUSSIAN:
            interaction = np.exp(-dist ** 2/(2*a**2))
        else:
            raise ValueError

        if with_zero_on_diag:
            np.fill_diagonal(interaction, 0)
        return interaction


class OCNN:
    def __init__(self, input_data, k=5,
                 interaction_function=InteractionFunction.DIPOLE,
                 activation_function=ActivationFunction.LOGISTIC_MAP):

        if not callable(activation_function):
            raise TypeError("Invalid activation function specified")

        if not callable(interaction_function):
            raise TypeError("Invalid interaction function specified")

        input_data = input_data.copy()
        self._net_size = input_data.shape[0]

        self._k = k
        self._interaction_function = interaction_function
        self._activation_function = activation_function

        interaction = self._get_interaction(
            dist=euclidean_distances(input_data, input_data)
        )
        self._weight = interaction / interaction.sum(axis=1)[:, np.newaxis]

    @property
    def size(self):
        return self._net_size

    def _get_interaction(self, dist):
        knn_indices = np.argpartition(dist, kth=self._k + 1, axis=1)[:, 1:self._k + 1]
        indices = np.arange(dist.shape[0])[:, np.newaxis]
        a = dist[indices, knn_indices].mean()

        return self._interaction_function(dist, a=a)

    def _get_init_state(self, seed):
        np.random.seed(seed)
        return np.random.uniform(low=-1, high=1, size=self._net_size)

    def __call__(self, observation_count=100, transfer_count=10, seed=None):
        curr_state = self._get_init_state(seed=seed)
        for t in range(0, transfer_count):
            curr_state = self._weight @ self._activation_function(curr_state)

        net_states = np.empty((self._net_size, observation_count), dtype=np.float32)
        net_states[:, 0] = curr_state
        for t in range(1, observation_count):
            net_states[:, t] = self._weight @ self._activation_function(net_states[:, t - 1])

        return net_states
