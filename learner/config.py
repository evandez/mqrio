"""Defines hyperparameters and runtime settings for the Deep Q-Network."""

# The network learning rate.
LEARNING_RATE = 1e-6

# Initial probability of the learner taking a random action.
# This probability decays over time as the learner experiences the world.
EXPLORATION_START_RATE = 1

# Final probability of the learner taking a random action.
# The exploration rate should decay linearly to this value.
EXPLORATION_END_RATE = 0.05

# The number of time steps that pass before the learner's exploration rate
# decays to its final value. Determines the rate at which the exploration
# rate decays.
FINAL_EXPLORATION_TIME = 1000000

# When predicting the "value" of a state and action, we discount the value
# of potential future rewards by multiplying by this constant.
DISCOUNT = 0.9

# When the network's parameters are updated, we sample this many previous
# state-action-reward triples to use as a training set for the network.
BATCH_SIZE = 32

# The number of iterations for which a chosen action is repeated.
# Necessary for actions to have "real" consequences in the game world.
ACTION_REPEAT = 4

# Number of iterations for which we take random actions, to build a foundation
# for our state-action-reward memory.
REPLAY_START_SIZE = 50000

# Maximum size of the replay memory.
REPLAY_MEMORY_SIZE = 100000

# How many actions are to be performed between network updates.
UPDATE_FREQUENCY = 4

# How often the Q-learner should log its state.
LOG_FREQUENCY = 10000

# How often to write to log file.
WRITE_FREQUENCY = 10000

# How many times should the state be logged in a row.
LOG_IN_A_ROW = 1

# How often the Q-learner should save its parameters.
SAVE_FREQUENCY = 500000

# Shape of the frames to use in the network.
FRAME_HEIGHT, FRAME_WIDTH = (84, 84)

# The number of frames used in a state object.
STATE_FRAMES = 4

# Whether to use duel architecture or not
DUELLING_ARCHITECTURE = False

POOLING_ARCHITECTURE = False

# Where to log score ratio.
LOG_PATH = 'score_ratio_log.txt'

# Where to save and load network weights to/from.
CHK_PATH = './deep_q_model/'
