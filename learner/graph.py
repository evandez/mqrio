"""Since we use the same graph for all Q-learners, centralize its construction."""
from learner.config import *
import tensorflow as tf

G_IN = 'frame_input'              # The graph input (84x84x4 frame sets).
G_CONV1_W = 'w_conv_1'            # First convolutional layer weights.
G_CONV1_B = 'b_conv_1'            # First convolutional layer bias
G_CONV2_W = 'w_conv_2'            # etc...
G_CONV2_B = 'b_conv_2'
G_CONV3_W = 'w_conv_3'
G_CONV3_B = 'b_conv_3'
G_FC1_W = 'w_fc_1'                # First fully connected layer weights.
G_FC1_B = 'b_fc_1'                # First fully connected layer bias.
G_FC2_W = 'w_fc_2'                # etc...
G_FC2_B = 'b_fc_2'
G_FC3_W = 'w_fc_3'                # etc...
G_FC3_B = 'b_fc_3'
G_FC4_W = 'w_fc_4'
G_FC4_B = 'b_fc_4'
G_OUT = 'q_value'                 # Graph output.


def construct_graph(output_width):
    """Creates a new TensorFlow graph with predetermined structure.

    Args:
        output_width: The number of output units for the graph.

    Returns:
        The graph input and output tensors (in that order).
    """
    graph_in = tf.placeholder(
        tf.float32,
        shape=[None, FRAME_HEIGHT, FRAME_WIDTH, STATE_FRAMES],
        name=G_IN)

    w_conv1 = _weight_variable([8, 8, STATE_FRAMES, 32], G_CONV1_W)
    b_conv1 = _bias_variable([32], G_CONV1_B)
    conv_layer1 = tf.nn.relu(_conv2d(graph_in, w_conv1, 4) + b_conv1)

    w_conv2 = _weight_variable([4, 4, 32, 64], G_CONV2_W)
    b_conv2 = _bias_variable([64], G_CONV2_B)
    conv_layer2 = tf.nn.relu(_conv2d(conv_layer1, w_conv2, 2) + b_conv2)

    w_conv3 = _weight_variable([3, 3, 64, 64], G_CONV3_W)
    b_conv3 = _bias_variable([64], G_CONV3_B)
    conv_layer3 = tf.nn.relu(_conv2d(conv_layer2, w_conv3, 1) + b_conv3)

    # TODO: Reshape this so we don't have to hardcode the the number of inputs
    # and can freely change the frame height/width.
    conv_layer3_flat = tf.reshape(conv_layer3, [-1, 7744])

    # Fully connected layer 1
    w_fc1 = _weight_variable([7744, 512], G_FC1_W)
    b_fc1 = _bias_variable([512], G_FC1_B)
    fc_layer1 = tf.nn.relu(tf.matmul(conv_layer3_flat, w_fc1) + b_fc1)

    # Fully connected layer 2
    w_fc2 = _weight_variable([7744, 512], G_FC2_W)
    b_fc2 = _bias_variable([512], G_FC2_B)
    fc_layer2 = tf.nn.relu(tf.matmul(conv_layer3_flat, w_fc2) + b_fc2)

    # Bias and weights for fully connected layer 1
    b_fc3 = _bias_variable([output_width], G_FC3_B)
    w_fc3 = _weight_variable([512, output_width], G_FC3_W)

    if DUEL_ARCHITECTURE:
        # Bias and weights for fully connected layer 2
        b_fc4 = _bias_variable([output_width], G_FC4_B)
        w_fc4 = _weight_variable([512, output_width], G_FC4_W)
        # Outputs of both fully connected layer
        output_fc1 = tf.add(tf.matmul(fc_layer1, w_fc3), b_fc3)
        output_fc2 = tf.add(tf.matmul(fc_layer2, w_fc4), b_fc4)
        graph_out = tf.add(output_fc1, output_fc2, name=G_OUT)
    else:
        graph_out = tf.add(tf.matmul(fc_layer1, w_fc3), b_fc3, name=G_OUT)

    return graph_in, graph_out


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


def _weight_variable(shape, name):
    """Returns a TensforFlow weight variable.

    Args:
        shape: The shape of the weight variable.
        name: The name for the variable.

    Returns:
        A TensorFlow weight variable of the given size.
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)


def _bias_variable(shape, name):
    """Returns a TensforFlow 2D bias variable.

    Args:
        shape: The shape of the bias variable.
        name: The name for the variable.

    Returns:
        A TensorFlow bias variable of the specified shape.
    """
    return tf.Variable(tf.constant(0.01, shape=shape), name=name)
