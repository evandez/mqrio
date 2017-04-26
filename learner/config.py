"""Defines hyperparameters and runtime settings for the Deep Q-Network."""

# The network learning rate.
LEARNING_RATE = 0.00025

# Initial probability of the learner taking a random action.
# This probability decays over time as the learner experiences the world.
EXPLORATION_START_RATE = 1

# Final probability of the learner taking a random action.
# The exploration rate should decay linearly to this value.
EXPLORATION_END_RATE = 0.1

# The number of frames that pass before the learner's exploration rate
# decays to its final value. Determines the rate at which the exploration
# rate decays.
FINAL_EXPLORATION_FRAME = 1000000

# When predicting the "value" of a state and action, we discount the value
# of potential future rewards by multiplying by this constant.
DISCOUNT = 0.99

# Weight of momentum term for gradient descent optimization.
MOMENTUM = 0.95

# When the network's parameters are updated, we sample this many previous
# state-action-reward triples to use as a training set for the network.
BATCH_SIZE = 32

# The number of iterations for which a chosen action is repeated.
# Necessary for actions to have "real" consequences in the game world.
ACTION_REPEAT = 4

# Interval (in number of iterations) at which the target network's parameters
# are updated. Recall that at each iteration we take SGD steps on a TEMPORARY network,
# and only update the network that is making the decisions after some number of iterations.
UPDATE_FREQUENCY = 4

# Number of iterations for which we take random actions, to build a foundation
# for our state-action-reward memory.
REPLAY_START_SIZE = 500 # 50000

# Maximum size of the replay memory.
REPLAY_MEMORY_SIZE = 100000
