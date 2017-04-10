"""Controller for Super Mario Bros."""
from threading import Thread
import pygame as pg, pygame.constants as pgc
from games.super_mario_bros.gamelib.game import Game
from PyGamePlayer.pygame_player import PyGamePlayer
from dqn import DeepQLearner

SMB_ACTIONS = [pgc.K_RIGHT, pgc.K_LEFT, pgc.K_z]

class SMBPlayer(PyGamePlayer):
    """Simple implementation of PyGamePlayer for Super Mario Bros."""
    def __init__(self, smb_game):
        """Initializes the Deep Q-Learner.

        Args:
            smb_game: The Super Mario Bros instance.
        """
        super(SMBPlayer, self).__init__() # For now, use default parameters.

        # Uninitialized game state info.
        self.game = smb_game
        self.last_score = 0

        # Q approximator.
        self.dqn = DeepQLearner(SMB_ACTIONS)

    def get_keys_pressed(self, screen_array, feedback, terminal):
        """Override of get_keys_pressed from PyGamePlayer."""
        return self.dqn.step(screen_array, feedback)

    def get_feedback(self):
        """Override of get_feedback from PyGamePlayer.

        For now, just returns the game score and whether or not game is over.
        """
        print pg.key.get_pressed()
        reward = self.game.score - self.last_score
        self.last_score = self.game.score
        return (reward, self.game.player.alive())

if __name__ == '__main__':
    pg.init()
    game = Game(pg.display.set_mode((640, 480)))
    player = SMBPlayer(game)

    # Start the game.
    Thread(target=game.main_loop).start()
    player.start()
