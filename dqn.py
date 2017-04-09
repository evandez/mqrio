"""Interfaces for Deep Q-Network."""

class DeepQLearner(object):
    """Provides wrapper around TensorFlow for Deep Q-Network."""
    def __init__(self):
        print 'Initializing Deep Q-Network...'

    def step(self, frame, reward):
        """Steps the training algorithm given the current frame and previous reward.
        Assumes that the reward is a consequence of the previous action.

        Args:
            frame: Current game frame.
            reward: Reward value from previous action.

        Returns:
            The next action to perform.
        """
        print 'Stepping with frame', frame, 'and reward', reward
    