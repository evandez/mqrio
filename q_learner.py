"""Logic for training a DeepQ Learner."""

class QLearner(object):
    """State object for training the Deep Q-Network. Keeps track of
    the current training iteration, the current Q-approximator, the previously
    observed states, etc.
    """
    def __init__(self):
        print 'Initialized Q learner!'

    def step(self, r, s):
        """ 
        """
        print 'Stepping!'
