"""Defines hyperparameters and runtime settings for the Deep Q-Network."""

# The network learning rate.
LEARNING_RATE = 2.5e-4

# Initial probability of the learner taking a random action.
# This probability decays over time as the learner experiences the world.
EXPLORATION_START_RATE = 1

# Final probability of the learner taking a random action.
# The exploration rate should decay linearly to this value.
EXPLORATION_END_RATE = 0.1

# The number of frames that pass before the learner's exploration rate
# decays to its final value. Determines the rate at which the exploration
# rate decays.
FINAL_EXPLORATION_FRAME = 1e6

# When predicting the "value" of a state and action, we discount the value
# of potential future rewards by multiplying by this constant.
DISCOUNT = 0.99

# When the network's parameters are updated, we sample this many previous
# state-action-reward triples to use as a training set for the network.
BATCH_SIZE = 32

# The number of iterations for which a chosen action is repeated.
# Necessary for actions to have "real" consequences in the game world.
ACTION_REPEAT = 4

# The number of frames used in a state object.
STATE_FRAMES = 4

# Number of iterations for which we take random actions, to build a foundation
# for our state-action-reward memory.
REPLAY_START_SIZE = 5e4

# Maximum size of the replay memory.
REPLAY_MEMORY_SIZE = int(1e6)

# How often the Q-learner should log its state.
LOGGING_FREQUENCY = 1e2

# How often the Q-learner should save its parameters.
SAVING_FREQUENCY = 1e3

SCORING_FUNCTION = 'HITS'
