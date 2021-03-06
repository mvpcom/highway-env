from __future__ import division, print_function
from abc import ABCMeta, abstractmethod


class AbstractAgent(object):
    """
        An abstract class specifying the interface of a generic agent.
        The agent interacts with an environment that implements the highway_env.mdp.abstract.MDP interface.
    """
    metaclass__ = ABCMeta

    @abstractmethod
    def plan(self, state):
        """
            Plan an optimal trajectory given an initial MDP state.

        :param state: the initial state
        :return: the optimal sequence of actions [a0, a1, a2...]
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """
            Reset the agent to its initial internal state
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """
            Seed the agent's random number generator
        :param seed: the seed to be used
        :return: the used seed
        """
        raise NotImplementedError()
