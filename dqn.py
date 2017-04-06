"""Wrapper around tensorflow for the nonlinear Q approximator."""

class DQN(object):
    """Defines an interface for interacting with the nonlinear Q approximator.
    Uses TensorFlow for all CNN logic.
    """
    def __init__(self):
        print 'Deep Q Network created successfully!'
