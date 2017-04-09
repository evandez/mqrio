"""Controller for Super Mario Bros."""
from games.super_mario_bros.gamelib import main as smb_main
from PyGamePlayer.pygame_player import PyGamePlayer
from dqn import DeepQLearner

class SMBPlayer(PyGamePlayer):
    """Simple implementation of PyGamePlayer for Super Mario Bros."""
    def __init__(self):
        super(SMBPlayer, self).__init__() # For now, use default parameters.
        self.dqn = DeepQLearner()

    def get_keys_pressed(self, screen_array, feedback, terminal):
        """Override of get_keys_pressed from PyGamePlayer."""
        return self.dqn.step(screen_array, feedback)

    def get_feedback(self):
        """Override of get_feedback from PyGamePlayer."""
        # TODO: What is the feedback for Mario? Probably can use fitness score from MarI/O.
        return 0

if __name__ == '__main__':
    smb_main.main()
    SMBPlayer().start()
