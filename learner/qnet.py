"""Thin wrapper around TensorFlow logic."""
import learner.config as cg
import tensorflow as tf

class QNet(object):
    """A deep network Q-approximator implemented with TensorFlow.

    The network is structure is fixed, aside from the output width, which depends
    on the number of actions necessary to play a given game.
    """

    def __init__(self, output_width):
        """Initializes the TensorFlow graph.

        Args:
            output_width: The number of output units.
        """
        self.output_width = output_width # Hold onto this value for copying.

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.graph_in = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])

            w_conv1 = QNet._weight_variable([8, 8, 4, 32])
            b_conv1 = QNet._bias_variable([32])
            conv_layer1 = tf.nn.relu(QNet._conv2d(self.graph_in, w_conv1, 4) + b_conv1)

            w_conv2 = QNet._weight_variable([4, 4, 32, 64])
            b_conv2 = QNet._bias_variable([64])
            conv_layer2 = tf.nn.relu(QNet._conv2d(conv_layer1, w_conv2, 2) + b_conv2)

            w_conv3 = QNet._weight_variable([3, 3, 64, 64])
            b_conv3 = QNet._bias_variable([64])
            conv_layer3 = tf.nn.relu(QNet._conv2d(conv_layer2, w_conv3, 1) + b_conv3)

            conv_layer3_flat = tf.reshape(conv_layer3, [-1, 7744])
            w_fc1 = QNet._weight_variable([7744, 512])
            b_fc1 = QNet._bias_variable([512])
            fc_layer1 = tf.nn.relu(tf.matmul(conv_layer3_flat, w_fc1) + b_fc1)

            w_fc2 = QNet._weight_variable([512, output_width])
            b_fc2 = QNet._bias_variable([output_width])
            self.graph_out = tf.nn.relu(tf.matmul(fc_layer1, w_fc2) + b_fc2)

            self.target_reward = tf.placeholder(tf.float32)
            loss = tf.nn.l2_loss(self.target_reward - tf.reduce_max(self.graph_out))
            clipped_loss = tf.clip_by_value(loss, -1, 1)
            self.optimizer = tf.train.RMSPropOptimizer(
                cg.LEARNING_RATE, momentum=cg.MOMENTUM).minimize(clipped_loss)

            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())

    def __del__(self):
        """Closes the TensorFlow session, freeing resources."""
        self.sess.close()

    @staticmethod
    def _conv2d(data, weights, stride):
        """Returns a TensforFlow 2D convolutional layer.

        Args:
            data: The input tensor to the convolutional layer.
            weights: The convolutional weights for this layer.
            stride: The x and y stride for the convolution.

        Returns:
            The TensorFlow convolutional layer.
        """
        return tf.nn.conv2d(data, weights, strides=[1, stride, stride, 1], padding='SAME')

    @staticmethod
    def _weight_variable(shape):
        """Returns a TensforFlow weight variable.

        Args:
            shape: The shape of the weight variable.

        Returns:
            A TensorFlow weight variable of the given size.
        """
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    @staticmethod
    def _bias_variable(shape):
        """Returns a TensforFlow 2D bias variable.

        Args:
            shape: The shape of the bias variable.

        Returns:
            A TensorFlow bias variable of the specified shape.
        """
        return tf.Variable(tf.constant(0.1, shape=shape))

    def save(self, filename):
        """Saves this network's current graph variables.

        Args:
            filename: The name of the file to save the variables to.
        """
        with self.graph.as_default():
            tf.train.Saver().save(self.sess, filename)

    def load(self, filename):
        """Loads the network variables from a checkpoint file.

        Args:
            filename: The name of the checkpoint file.
        """
        with self.graph.as_default():
            tf.train.Saver().restore(self.sess, filename)

    def deep_copy(self):
        """Writes the current network to a checkpoint, then loads a new network
        of the same structure from that checkpoint. Effectively creates a deep
        copy of this network.

        Returns:
            A Q-net with the same structure and parameters as this network, but under
            a different TensorFlow session.
        """
        filename = '/tmp/transfer.chk'
        copy = QNet(self.output_width)
        self.save(filename)
        copy.load(filename)
        return copy

    def compute_q(self, net_in):
        """Forward-propagates the given input and returns the array of outputs.

        Args:
            net_in: Image to forward-prop through the network. Must be 1x84x84x4.

        Returns:
            The array of network outputs.
        """
        return self.sess.run(self.graph_out, feed_dict={self.graph_in:[net_in]})[0]

    def update(self, batch_input, batch_target):
        """Updates the network with the given batch input/target values using RMSProp.

        Args:
            batch_input: Set of Nx84x84x4 network inputs, where N is the batch size.
            batch_target: Corresponding target Q values for each input.
        """
        self.sess.run(
            self.optimizer,
            feed_dict={self.graph_in:batch_input, self.target_reward:batch_target})
