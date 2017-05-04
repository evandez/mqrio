'Runs but still needs some work/enhancement'
import pygame.constants as pgc
from PyGamePlayer.pygame_player import PyGamePlayer, function_intercept
import games.tetris
# from learner.qlearn import DeepQLearner
# from learner.config import *


ACTIONS = [pgc.K_UNKNOWN, pgc.K_RIGHT, pgc.K_LEFT, pgc.K_DOWN, pgc.K_UP]

class TetrisPlayer(PyGamePlayer):
    def __init__(self, force_game_fps=10, run_real_time=False):
        super(TetrisPlayer, self).__init__(force_game_fps=10, run_real_time=False)
        self._new_reward = 0.0
        self._terminal = False
        # self._line_removed = False
        # self.dql = DeepQLearner(ACTIONS,save=True)

        def add_removed_lines_to_reward(lines_removed, *args, **kwargs):
            self._new_reward += lines_removed
            # self._line_removed = True
            return lines_removed

        def check_for_game_over(ret, text):
            if text == 'Game Over':
                self._terminal = True

        # to get the reward we will intercept the removeCompleteLines method and store what it returns
        games.tetris.removeCompleteLines = function_intercept(games.tetris.removeCompleteLines,
                                                              add_removed_lines_to_reward)
        # find out if we have had a game over
        games.tetris.showTextScreen = function_intercept(games.tetris.showTextScreen,
                                                         check_for_game_over)

    def get_keys_pressed(self, screen_array, feedback, terminal):
        if self._terminal :
        	self._terminal = False
        	return [pgc.K_SPACE]

        # return  self.dql.step(screen_array, feedback, terminal)
        return [pgc.K_DOWN]
        
    def get_feedback(self):
    	if self._terminal :
    		terminal = self._terminal
    		# print (terminal)
    		return float(-25), terminal
    	# if self._line_removed:
    	temp = self._new_reward
    	self._new_reward = 0.0
    	self.lines_removed = False
    	terminal = self._terminal
    	return temp*temp, terminal
    	"""
    	else :
    		
    			TODO :: potentially add reward for moves that don't remove lines
    			− 0.51 × Height + 0.76 × Lines − 0.36 × Holes − 0.18 × Bumpiness 
		"""


    def start(self):
        super(TetrisPlayer, self).start()
        games.tetris.main()

if __name__ == '__main__':
    player = TetrisPlayer()
    player.start()
