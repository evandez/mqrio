"""Simple Pong Player for testing DeepQLearnerep Q logic."""
import pygame.constants as pgc
from PyGamePlayer.pygame_player import PyGamePlayer
import games.Flappy_Bird.flappy as game
from learner.qlearn import DeepQLearner
from learner.config import *


# Possible actions. Last one is equivalent to "do nothing."
ACTIONS = [pgc.K_SPACE, pgc.K_UNKNOWN]

class FlpBrdPlayer(PyGamePlayer):
	def __init__(self, force_game_fps=10, run_real_time=True):
		super(FlpBrdPlayer, self).__init__(force_game_fps=force_game_fps, run_real_time=run_real_time)

		self.dql = DeepQLearner(ACTIONS,save=True)
		
	def get_keys_pressed(self, screen_array, feedback, terminal):
		if game.collision :
			return [pgc.K_SPACE]

		return self.dql.step(screen_array, feedback, terminal)
		
	def get_feedback(self):
		# Rewarded only on dead or alive basis
		if game.collision :
			#print (game.collision)
			return (-500.0, game.collision)
		else :
			return (1.0, game.collision)

	def start(self):
		super(FlpBrdPlayer, self).start()
		game.main()
		

if __name__ == '__main__':
    FlpBrdPlayer().start()