"""Implementation of PyGamePlayer for Half Pong."""
from pygame.constants import K_DOWN, K_UP, K_UNKNOWN
from PyGamePlayer.pygame_player import PyGamePlayer
from learner.qlearn import DeepQLearner

ACTIONS = [K_DOWN, K_UNKNOWN, K_UP]

class HalfPongPlayer(PyGamePlayer):
    """Simple implementation of PyGamePlayer for Half Pong."""
    def __init__(self, force_game_fps=8, run_real_time=False):
        super(HalfPongPlayer, self).__init__(
            force_game_fps=force_game_fps,
            run_real_time=run_real_time)
        self.last_score = 0

        self.dql = DeepQLearner(ACTIONS)

    def get_keys_pressed(self, screen_array, reward, terminal):
        """Returns the keys to press at the given timestep. See parent class function."""
        return self.dql.step(screen_array, reward, terminal)

    def get_feedback(self):
        """Returns the feedback for the current state of the game. In this case, just returns
        the difference in the learner's score minus the difference in the other player's score.
        See parent class function.
        """
        # import must be done here because otherwise importing would cause the game to start playing
        from games.half_pong import score

        # get the difference in score between this and the last run
        score_change = (score - self.last_score)
        self.last_score = score

        return float(score_change), score_change == -1

    def start(self):
        super(HalfPongPlayer, self).start()
        import games.half_pong


if __name__ == '__main__':
    HalfPongPlayer().start()
    