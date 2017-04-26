"""Interfaces for Deep Q-Network."""
from collections import deque
import random
import config as cg
import numpy as np
from scipy.misc import imresize
from qnet import QNet

class DeepQLearner(object):
    """Provides wrapper around TensorFlow for Deep Q-Network."""
    def __init__(self, actions):
        """Intializes the TensorFlow graph.

        Args:
            actions: List of viable actions learner can make. (Must be PyGame constants.)
        """
        self.actions = actions
        self.target_net = QNet(len(actions))
        self.training_net = self.target_net.deep_copy()
        self.exploration_rate = cg.EXPLORATION_START_RATE
        self.iteration = -1

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

    def remember_transition(self, time, pre_frame, action, reward, terminal):
        """Returns the transition dictionary for the given data.

        Args:
            time: The current time step.
            pre_frame: The frame at the current time.
            action: The action(s) taken at current time.
            reward: The reward received at current time (prior to taken the action above).
            terminal: True if the action at current time led to episode termination.
        """
        self.transitions.append({
            'time': time,
            'input': pre_frame,
            'action': action,
            'reward': reward,
            'terminal': terminal
        })

    def do_explore(self):
        """Returns true if a random action should be taken, false otherwise.
        Decays the exploration rate if the final exploration frame has not been reached.
        """
        if self.iteration <= cg.FINAL_EXPLORATION_FRAME:
            self.exploration_rate -= (
                float(cg.EXPLORATION_START_RATE - cg.EXPLORATION_END_RATE)
                / cg.FINAL_EXPLORATION_FRAME)
        return random.random() < self.exploration_rate

    def best_action(self, frame):
        """Returns the best action to perform.

        Args:
            frame: The current frame.
        """
        return self.actions[np.argmax(self.target_net.compute_q(frame))]

    def random_action(self):
        """Returns a random action to perform."""
        return self.actions[int(random.random() * len(self.actions))]

    def compute_target_reward(self, trans):
        """Computes the target reward for the given transition using the target Q-network.

        Args:
            trans: The transition for which to compute the target reward.

        Returns:
            The target reward.
        """
        target_reward = trans['reward']
        if not trans['terminal'] and trans['time'] != self.iteration - 1:
            next_input = self.transitions[trans['time']]['input']
            target_reward += cg.DISCOUNT * np.amax(self.target_net.compute_q(next_input))
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

        # Clip reward to limit gradient scale later on.
        reward = np.clip(reward, -1, 1)

        # Handle burn in period.
        if self.iteration <= cg.REPLAY_START_SIZE:
            proc_frame = self.preprocess(frame)
            action = self.random_action()
            self.remember_transition(self.iteration, proc_frame, action, reward, terminal)
            return [action]

        # Repeat previous action for some number of iterations.
        # If we ARE repeating an action, we pretend that we did not see
        # this frame and just keep doing what we're doing.
        if self.iteration % cg.ACTION_REPEAT == 0:
            return [self.transitions[-1]['action']]

        # Update network from the previous action.
        minibatch = random.sample(self.transitions, cg.BATCH_SIZE)
        batch_input = [trans['input'] for trans in minibatch]
        batch_target = [self.compute_target_reward(trans) for trans in minibatch]
        self.training_net.update(batch_input, batch_target)

        # If it's time to update the target network, do so.
        if self.iteration % cg.UPDATE_FREQUENCY == 0:
            self.target_net = self.training_net.deep_copy()

        # Select the next action.
        proc_frame = self.preprocess(frame)
        action = self.random_action() if self.do_explore() else self.best_action(proc_frame)
        self.remember_transition(self.iteration, proc_frame, action, reward, terminal)
        return [action]
