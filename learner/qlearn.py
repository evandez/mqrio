"""Interfaces for Deep Q-Network."""
from collections import deque
import random
import os
from learner import config as cg
from learner.qnet import QNet
import numpy as np
from scipy.misc import imresize

class DeepQLearner(object):
    """Provides wrapper around TensorFlow for Deep Q-Network."""
    def __init__(self, actions, chk_path='deep_q_model', save=False, restore=False):
        """Intializes the TensorFlow graph.

        Args:
            actions: List of viable actions learner can make. (Must be PyGame constants.)
        """
        # Initialize state variables.
        self.actions = actions
        self.net = QNet(len(actions))
        self.exploration_rate = cg.EXPLORATION_START_RATE
        self.iteration = -1

        # Handle network save/restore.
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
        self.transitions = deque(maxlen=cg.REPLAY_MEMORY_SIZE)

    def preprocess(self, frame):
        """Resize image, pool across color channels, and normalize pixels.

        Args:
            frame: The frame to process.

        Returns:
            The preprocessed frame.
        """
        proc_frame = np.reshape(
            [px / 255.0 for px in np.amax(imresize(frame, (84, 84)), axis=2)],
            (84, 84, 1))
        if not len(self.transitions):
            return np.repeat(proc_frame, 4, axis=2)
        else:
            return np.append(proc_frame, self.transitions[-1]['input'][:, :, -3:], axis=2)

    def remember_transition(self, pre_frame, action, terminal):
        """Returns the transition dictionary for the given data. Defer recording the
        reward until it is observed.

        Args:
            pre_frame: The frame at the current time.
            action: The index of the action(s) taken at current time.
            terminal: True if the action at current time led to episode termination.
        """
        self.transitions.append({
            'time': len(self.transitions),
            'input': pre_frame,
            'action': self.actions.index(action),
            'terminal': terminal
        })

    def observe_reward(self, reward):
        """Records the reward from the previous action. Clips as necessary.

        Args:
            reward: The reward from the previous transition.
        """
        if not len(self.transitions):
            return
        self.transitions[-1]['reward'] = np.clip(reward, -1, 1)

    def do_explore(self):
        """Returns true if a random action should be taken, false otherwise.
        Decays the exploration rate if the final exploration frame has not been reached.
        """
        if len(self.transitions) <= cg.FINAL_EXPLORATION_FRAME:
            self.exploration_rate -= (
                float(cg.EXPLORATION_START_RATE - cg.EXPLORATION_END_RATE)
                / (cg.FINAL_EXPLORATION_FRAME - cg.REPLAY_START_SIZE))
        return random.random() < self.exploration_rate

    def best_action(self, frame):
        """Returns the best action to perform.

        Args:
            frame: The current frame.
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
        if not trans['terminal'] and trans['time'] < len(self.transitions) - 1:
            next_input = self.transitions[trans['time']+1]['input']
            target_reward += cg.DISCOUNT * np.amax(self.net.compute_q(next_input))
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
        if self.iteration % cg.LOGGING_FREQUENCY == 0:
            self.log_status()

        # Handle burn in period.
        if self.iteration <= cg.REPLAY_START_SIZE:
            self.observe_reward(reward)
            proc_frame = self.preprocess(frame)
            action = self.random_action()
            self.remember_transition(proc_frame, action, terminal)
            return [action]

        # Repeat previous action for some number of iterations.
        # If we ARE repeating an action, we pretend that we did not see
        # this frame and just keep doing what we're doing.
        if self.iteration % cg.ACTION_REPEAT == 0:
            return [self.transitions[-1]['action']]

        # Observe the previous reward.
        self.observe_reward(reward)

        # Save network if necessary before updating.
        if self.save and self.iteration % cg.SAVING_FREQUENCY == 0:
            self.net.save(self.chk_path)

        # Update network from the previous action.
        minibatch = random.sample(self.transitions, cg.BATCH_SIZE)
        batch_frames = [trans['input'] for trans in minibatch]
        batch_actions = [trans['action'] for trans in minibatch]
        batch_targets = [self.compute_target_reward(trans) for trans in minibatch]
        self.net.update(batch_frames, batch_actions, batch_targets)

        # Select the next action.
        proc_frame = self.preprocess(frame)
        action = self.random_action() if self.do_explore() else self.best_action(proc_frame)
        self.remember_transition(proc_frame, action, terminal)
        return [action]

    def log_status(self):
        """Print the current status of the Q-learner."""
        fmt = """
        \t\t-----------------\t\t
        Iteration: %d
        Replay capacity: %d (burn in %s)
        Exploration rate: %.4f (%s annealing)"""
        print(fmt % (
            self.iteration,
            len(self.transitions),
            'not done' if self.iteration <= cg.REPLAY_START_SIZE else 'done',
            self.exploration_rate,
            'still' if len(self.transitions) <= cg.FINAL_EXPLORATION_FRAME else 'done'
        ))

        # If we're using the network, print a sample of the output.
        if self.iteration >= cg.REPLAY_START_SIZE:
            print('Sample Q output:', self.net.compute_q(self.transitions[-1]['input']))
