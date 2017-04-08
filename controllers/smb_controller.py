"""Driver for Deep Q Learner."""
from games.super_mario_bros.gamelib import main as smb_main
from PyGamePlayer.pygame_player import PyGamePlayer

class SMBController(PyGamePlayer):
    """Thin wrapper around PyGame to provide utility for getting DQN data."""
    def __init__(self):
        print 'Initialized controller!'

    def get_keys_pressed(self, screen_array, feedback, terminal):
        print 'Getting feedback...'

    def get_feedback(self):
        print 'Getting feedback...'

    def start(self):
        """Starts the game."""
        smb_main.main()
