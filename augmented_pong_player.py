"""Simple Pong Player for testing Deep Q logic."""
import pygame.constants as pgc
from PyGamePlayer.pygame_player import PyGamePlayer
from learner.config import *
from learner.qlearn import DeepQLearner

# Possible actions for Pong. Last one is equivalent to "do nothing."
ACTIONS = [pgc.K_DOWN, pgc.K_UP, pgc.K_UNKNOWN]

# Either 'HITS' or 'SCORES'.
SCORING_FUNCTION = 'SCORES'

class AugmentedPongPlayer(PyGamePlayer):
    """Implementation of PyGamePlayer for Pong."""
    def __init__(self, force_game_fps=10, run_real_time=False):
        """
        Example class for playing Pong
        """
        super(AugmentedPongPlayer, self).__init__(
            force_game_fps=force_game_fps,
            run_real_time=run_real_time)
        self.last_bar1_score = 0.0
        self.last_bar2_score = 0.0

        self.last_bar1_hit_count = 0.
        self.last_bar2_hit_count = 0.

        self.dql = DeepQLearner(ACTIONS, save=True)

    def get_keys_pressed(self, screen_array, feedback, terminal):
        return self.dql.step(screen_array, feedback, terminal)

    def get_feedback(self):
        # import must be done here because otherwise importing would cause the game to start playing
        if SCORING_FUNCTION == 'SCORES':
            from games.pong import bar1_score, bar2_score

            # get the difference in score between this and the last run
            score_change = (bar1_score - self.last_bar1_score) - (bar2_score - self.last_bar2_score)
            self.last_bar1_score = bar1_score
            self.last_bar2_score = bar2_score

        elif SCORING_FUNCTION == 'HITS':
            from games.augmented_pong import bar1_hit_count, bar2_hit_count

            score_change = ((bar1_hit_count - self.last_bar1_hit_count)
                            - (bar2_hit_count - self.last_bar2_hit_count))

            self.last_bar1_hit_count = bar1_hit_count
            self.last_bar2_hit_count = bar2_hit_count

        return float(score_change), score_change != 0

    def start(self):
        super(AugmentedPongPlayer, self).start()
        import games.augmented_pong

if __name__ == '__main__':
    AugmentedPongPlayer().start()
