"""Implementation of PyGamePlayer for Pong."""
from pygame.constants import K_DOWN, K_UP, K_UNKNOWN
from PyGamePlayer.pygame_player import PyGamePlayer
from learner.qlearn import DeepQLearner

# The valid actions for pong.
ACTIONS = [K_DOWN, K_UNKNOWN, K_UP]


class PongPlayer(PyGamePlayer):
    """Simple implementation of PyGamePlayer for Pong."""
    def __init__(self, force_game_fps=10, run_real_time=False):
        """Store necessary state information. See init function for superclass."""
        super(PongPlayer, self).__init__(
            force_game_fps=force_game_fps,
            run_real_time=run_real_time)
        self.last_bar1_score = 0.0
        self.last_bar2_score = 0.0

        self.dql = DeepQLearner(ACTIONS)

    def get_keys_pressed(self, screen_array, feedback, terminal):
        """Returns the keys to press at the given timestep. See parent class function."""
        return self.dql.step(screen_array, feedback, terminal)

    def get_feedback(self):
        """Returns the feedback for the current state of the game. In this case, just returns
        the difference in the learner's score minus the difference in the other player's score.
        See parent class function.
        """
        # import must be done here because otherwise importing would cause the game to start playing
        from games.pong import bar1_score, bar2_score

        # get the difference in score between this and the last run
        score_change = (bar1_score - self.last_bar1_score) - (bar2_score - self.last_bar2_score)
        self.last_bar1_score = bar1_score
        self.last_bar2_score = bar2_score

        return float(score_change), score_change != 0

    def start(self):
        """Starts the learner and game."""
        super(PongPlayer, self).start()
        import games.pong

if __name__ == '__main__':
    PongPlayer().start()
