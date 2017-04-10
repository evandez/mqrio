"""Interfaces for Deep Q-Network."""
import tensorflow as tf

class DeepQLearner(object):
    """Provides wrapper around TensorFlow for Deep Q-Network."""
    def __init__(self, actions):
        self.actions = actions

    def step(self, frame, reward):
        """Steps the training algorithm given the current frame and previous reward.
        Assumes that the reward is a consequence of the previous action.

        Args:
            frame: Current game frame.
            reward: Reward value from previous action.

        Returns:
            The next action to perform.
        """
        return [self.actions[0]]
