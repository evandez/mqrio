"""PyGamePlayer logic for playing Flappy Bird."""
import pygame.constants as pgc
from PyGamePlayer.pygame_player import PyGamePlayer
import games.flappy_bird.flappy as game
from learner.qlearn import DeepQLearner
from learner.config import *


# Possible actions. Last one is equivalent to "do nothing."
ACTIONS = [pgc.K_SPACE, pgc.K_UNKNOWN]


class FlappyBirdPlayer(PyGamePlayer):
		"""Implementation of PyGamePlayer for Flappy Bird."""

		def __init__(self, force_game_fps=10, run_real_time=True):
				"""Initializes the deep Q-network."""
				super(FlappyBirdPlayer, self).__init__(
						force_game_fps=force_game_fps,
						run_real_time=run_real_time)
				self.dql = DeepQLearner(ACTIONS, save=True)

		def get_keys_pressed(self, screen_array, feedback, terminal):
				"""Returns the keys to press at the given timestep. See parent class function."""
				return self.dql.step(screen_array, feedback, terminal) if game.collision else [pgc.K_SPACE] 

		def get_feedback(self):
				"""Returns the feedback for the current state of the game. See parent class function."""
				# Rewarded only on dead or alive basis.
				reward = -500.0 if game.collision else 1.0
				return reward, game.collision

		def start(self):
				"""Starts the player."""
				super(FlappyBirdPlayer, self).start()
				game.main()
