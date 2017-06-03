"""PyGamePlayer logic for Tetris. Taken and modified from PyGamePlayer example."""
import pygame.constants as pgc
from PyGamePlayer.pygame_player import PyGamePlayer, function_intercept
import games.tetris
from learner.qlearn import DeepQLearner
from learner.config import *


ACTIONS = [pgc.K_UNKNOWN, pgc.K_RIGHT, pgc.K_LEFT, pgc.K_DOWN, pgc.K_UP]


class TetrisPlayer(PyGamePlayer):
    """Implementation of PyGamePlayer for Tetris."""

    def __init__(self, force_game_fps=10, run_real_time=False):
        """Initializes the deep Q-network"""
        super(TetrisPlayer, self).__init__(force_game_fps=10, run_real_time=False)
        self.new_reward = 0.0
        self.terminal = False
        self.lines_removed = False
        self.dql = DeepQLearner(ACTIONS, save=True)

    def add_removed_lines_to_reward(self, lines_removed):
        """Title says all."""
        self.new_reward += lines_removed
        return lines_removed

    def check_for_game_over(self, ret, text):
        """Updates player state to determine if the game is over."""
        if text == 'Game Over':
            self.terminal = True

        # To get the reward we will intercept the removeCompleteLines method
        # and store what it returns
        games.tetris.removeCompleteLines = function_intercept(games.tetris.removeCompleteLines,
                                                              add_removed_lines_to_reward)
        # Find out if we have had a game over.
        games.tetris.showTextScreen = function_intercept(games.tetris.showTextScreen,
                                                         check_for_game_over)

    def get_keys_pressed(self, screen_array, feedback, terminal):
        """Returns the keys to press at the given timestep. See parent class function."""
        if self.terminal:
        	self.terminal = False
        	return [pgc.K_SPACE]
        return  self.dql.step(screen_array, feedback, terminal)

    def get_feedback(self):
        """Returns the feedback for the current state of the game. See parent class function."""
        if self.terminal:
            from games.tetris import blankSpaces
            terminal = self.terminal

            # Found the following reward/penalty strategy in a paper.
            # Coeff is taken from the paper.
            # Should play around with it a little
            return float(.35*blankSpaces), terminal
        temp = self.new_reward
        self.new_reward = 0.0
        self.lines_removed = False
        terminal = self.terminal
        return temp*temp, terminal

    def start(self):
        """Starts the player."""
        super(TetrisPlayer, self).start()
        games.tetris.main()
