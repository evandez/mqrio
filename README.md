# Overview
Mqrio is a deep reinforcement learner that plays games. Deep reinforcement learning was popularized by Mnih et al. in this [seminal paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). It modifies the traditional Q-learning by approximating the Q function with a convolutional neural network. Thus, the learner learns to play a game by looking only at the game frames and current score.

This implementation has successfully played Half Pong and Pong at human level. Support for Tetris and Flappy Bird has been provided, but the learner has not been thoroughly tested on these games.

# Setup
First, clone this repository using the command `git clone --recursive https://github.com/EvanFredHernandez/mqrio.git`.

Then ensure you have installed the following Python libraries (e.g., using pip):
- numpy
- scipy
- pygame
- tensorflow (GPU version is ideal)

# Running the Learner
From the command line, run `python3 run_me.py` and navigate through the command line interface to start the DQN on a game of your choice.

# Contributing
If you are interested in running our learner on a different game, and in particular if you experience good results with our learner on a different game, feel free to submit a pull request.

# Acknowledgements
We would like to thank Daniel Slater for PyGamePlayer, a central underlying component of our code that abstracts away the gross internals of PyGame.