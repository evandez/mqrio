"""Interfaces for Deep Q-Network."""
from collections import deque
import random
import os
from learner.config import *
from learner.qnet import QNet
import numpy as np
from scipy.misc import imresize

class DeepQLearner(object):
    """Provides wrapper around TensorFlow for Deep Q-Network."""
    def __init__(self, actions, chk_path='deep_q_model/', save=True, restore=False):
        """Intializes the TensorFlow graph.
        Args:
            actions: List of viable actions learner can make. (Must be PyGame constants.)
        """
        # Initialize state variables.
        self.actions = actions
        self.net = QNet(len(actions))
        self.exploration_rate = EXPLORATION_START_RATE
        self.exploration_reduction = (EXPLORATION_START_RATE - EXPLORATION_END_RATE) \
            / float(REPLAY_MEMORY_SIZE - REPLAY_START_SIZE)
        self.iteration = -1
        self.previous_frames = deque(maxlen=STATE_FRAMES-1)
        self.loss = 0

        # Handle network save/restore.
        # TODO: Need to restore other settings from file too.
        self.chk_path = chk_path
        self.save = save
        if restore:
            if not os.path.exists(chk_path):
                raise Exception('No such checkpoint path %s!' % chk_path)
            self.net.restore(chk_path)

        # Store all previous transitions in a deque to allow for efficient
        # popping from the front and to allow for size management.
        #
        # Transitions are dictionaries of the form below.
        #     {
        #         'input': The Q-network input at this point in time.
        #         'action': The action index (indices) taken at this frame.
        #         'reward': The reward from the previous action.
        #         'terminal': True if the action led to a terminal state.
        #     }
        self.transitions = deque(maxlen=REPLAY_MEMORY_SIZE)

    def normalize_frame(self, frame):
        """Normalizes the screen array to be 84x84x1, with floating point values in
        the range [0, 1].
        Args:
            frame: The pixel values from the screen.
        Returns:
            An 84x84x1 floating point numpy array.
        """
        return np.reshape(
            np.mean(imresize(frame, (FRAME_HEIGHT, FRAME_WIDTH)), axis=2),
            (FRAME_HEIGHT, FRAME_WIDTH, 1))

    def preprocess(self, frame):
        """Resize image, pool across color channels, and normalize pixels.
        Args:
            frame: The frame to process.
        Returns:
            The preprocessed frame.
        """
        proc_frame = frame
        if not len(self.transitions) or len(self.previous_frames) < STATE_FRAMES - 1:
            return np.repeat(frame, STATE_FRAMES, axis=2)
        else:
            for recent_frame in self.previous_frames:
                proc_frame = np.append(proc_frame, recent_frame, axis=2)
            return proc_frame

    def remember_transition(self, pre_frame, action, terminal):
        """Returns the transition dictionary for the given data. Defer recording the
        reward and resulting state until they are observed.
        Args:
            pre_frame: The frame at the current time.
            action: The index of the action(s) taken at current time.
            terminal: True if the action at current time led to episode termination.
        """
        self.transitions.append({
            'state_in': pre_frame,
            'action': self.actions.index(action),
            'terminal': terminal
        })

    def observe_result(self, resulting_state, reward):
        """Records the resulting state and reward from the previous action.
        Clips reward as necessary.
        Args:
            resulting_state: The (preprocessed) state resulting from the previous action.
            reward: The reward from the previous transition.
        """
        if not len(self.transitions):
            return
        self.transitions[-1]['reward'] = np.clip(reward, -1, 1)
        self.transitions[-1]['state_out'] = resulting_state

    def is_burning_in(self):
        """Returns true if the network is still burning in (observing transitions)."""
        return len(self.transitions) < REPLAY_START_SIZE

    def do_explore(self):
        """Returns true if a random action should be taken, false otherwise.
        Decays the exploration rate if the final exploration frame has not been reached.
        """
        if not self.is_burning_in() and self.exploration_rate > EXPLORATION_END_RATE:
            self.exploration_rate -= self.exploration_reduction
        return random.random() < self.exploration_rate or self.is_burning_in()

    def best_action(self, frame):
        """Returns the best action to perform.
        Args:
            frame: The current (preprocessed) frame.
        """
        return self.actions[np.argmax(self.net.compute_q(frame))]

    def random_action(self):
        """Returns a random action to perform."""
        return self.actions[int(random.random() * len(self.actions))]

    def compute_target_reward(self, trans):
        """Computes the target reward for the given transition.
        Args:
            trans: The transition for which to compute the target reward.
        Returns:
            The target reward.
        """
        target_reward = trans['reward']
        if not trans['terminal']:
            target_reward += DISCOUNT * np.amax(self.net.compute_q(trans['state_out']))
        return target_reward

    def step(self, frame, reward, terminal):
        """Steps the training algorithm given the current frame and previous reward.
        Assumes that the reward is a consequence of the previous action.
        Args:
            frame: Current game frame.
            reward: Reward value from previous action.
            terminal: True if the previous action was termnial.
        Returns:
            The next action to perform.
        """
        self.iteration += 1

        # Log if necessary.
        if self.iteration % LOGGING_FREQUENCY < LOG_IN_A_ROW * STATE_FRAMES and self.iteration % STATE_FRAMES == 0:
            self.log_status()

        # Repeat previous action for some number of iterations.
        # If we ARE repeating an action, we pretend that we did not see
        # this frame and just keep doing what we're doing.
        if self.iteration % ACTION_REPEAT != 0:
            return [self.transitions[-1]['action']]

        frame = self.normalize_frame(frame)

        # Store this frame as a previous frame.
        self.previous_frames.appendleft(frame) # Left frame should be most recent.

        # Observe the previous reward.
        proc_frame = self.preprocess(frame)
        self.observe_result(proc_frame, reward)

        # Save network if necessary before updating.
        if self.save and self.iteration % SAVING_FREQUENCY == 0:
            self.net.save(self.chk_path)

        # update the network after every 4 actions are selected
        if not self.is_burning_in() and self.iteration % (ACTION_REPEAT * STATE_FRAMES) == 0:
            # Update network from the previous action.
            minibatch = random.sample(self.transitions, BATCH_SIZE)
            batch_frames = [trans['state_in'] for trans in minibatch]
            batch_actions = [trans['action'] for trans in minibatch]
            batch_targets = [self.compute_target_reward(trans) for trans in minibatch]
            self.loss = self.net.update(batch_frames, batch_actions, batch_targets)

        # Select the next action.
        action = self.random_action() if self.do_explore() else self.best_action(proc_frame)

        # Remember the action and the input frames, reward to be observed later.
        self.remember_transition(proc_frame, action, terminal)

        return [action]

    def log_status(self):
        """Print the current status of the Q-learner."""
        fmt = """
        \t\t-----------------\t\t
        Iteration: %d
        Replay capacity: %d (burn in %s)
        Exploration rate: %.9f (%s annealing)"""
        print(fmt % (
            self.iteration,
            len(self.transitions),
            'not done' if self.is_burning_in() else 'done',
            self.exploration_rate,
            'not' if self.is_burning_in() else (
                'still' if self.exploration_rate > EXPLORATION_END_RATE else 'done')
        ))

        # If we're using the network, print a sample of the output.
        if not self.is_burning_in():
            print('        Sample Q output:', self.net.compute_q(self.transitions[-1]['state_in']))
            print('        Clipped loss:', self.loss)