"""Interfaces for Deep Q-Network."""
import numpy as np
import tensorflow as tf

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[stride, stride, 1, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class DeepQLearner(object):
    """Provides wrapper around TensorFlow for Deep Q-Network."""
    def __init__(self, actions):
        self.actions = actions

        # Set up tf graph.
        self.graph_in = tf.placeholder(tf.float32, shape=[640, 480, 3])
        pre_layer = tf.image.resize_images(self.graph_in, [84, 84])

        w_conv1 = weight_variable([8, 8, 3, 32])
        b_conv1 = bias_variable([32])
        conv_layer1 = tf.nn.relu(conv2d(pre_layer, w_conv1, 4) + b_conv1)

        w_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])
        conv_layer2 = tf.nn.relu(conv2d(conv_layer1, w_conv2, 2) + b_conv2)

        w_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])
        conv_layer3 = tf.nn.relu(conv2d(conv_layer2, w_conv3, 1) + b_conv3)

        conv_layer3_flat = tf.reshape(conv_layer3, [-1, 4]) # TODO: What shape?
        w_fc1 = weight_variable() # TODO: What shape?
        b_fc1 = bias_variable([512])
        fc_layer1 = tf.nn.relu(tf.matmul(w_fc1, conv_layer3_flat) + b_fc1)

        w_fc2 = weight_variable([512, len(self.actions)])
        b_fc2 = bias_variable([len(self.actions)])
        self.graph_out = tf.nn.relu(tf.matmul(w_fc2, fc_layer1) + b_fc2)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def step(self, frame, reward):
        """Steps the training algorithm given the current frame and previous reward.
        Assumes that the reward is a consequence of the previous action.

        Args:
            frame: Current game frame.
            reward: Reward value from previous action.

        Returns:
            The next action to perform.
        """
        # TODO: Train the network.
        output = self.sess.run(self.graph_out, feed_dict={self.graph_in:frame})
        return self.actions[np.argmax(output, 1)]
