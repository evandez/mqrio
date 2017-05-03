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
    pool_layer1 = _pool(conv_layer1)

    w_conv2 = _weight_variable([4, 4, 32, 64], G_CONV2_W)
    b_conv2 = _bias_variable([64], G_CONV2_B)
    conv_layer2 = tf.nn.relu(_conv2d(pool_layer1, w_conv2, 2) + b_conv2)
    pool_layer2 = _pool(conv_layer2)

    w_conv3 = _weight_variable([3, 3, 64, 64], G_CONV3_W)
    b_conv3 = _bias_variable([64], G_CONV3_B)
    conv_layer3 = tf.nn.relu(_conv2d(pool_layer2, w_conv3, 1) + b_conv3)
    pool_layer3 = _pool(conv_layer3)

    conv_layer3_flat = tf.reshape(pool_layer3, [-1, 256])
    w_fc1 = _weight_variable([256, 256], G_FC1_W)
    b_fc1 = _bias_variable([256], G_FC1_B)
    fc_layer1 = tf.nn.relu(tf.matmul(conv_layer3_flat, w_fc1) + b_fc1)

    w_fc2 = _weight_variable([256, output_width], G_FC2_W)
    b_fc2 = _bias_variable([output_width], G_FC2_B)
    graph_out = tf.add(tf.matmul(fc_layer1, w_fc2), b_fc2, name=G_OUT)
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

def _pool(data):
    """Returns a TensforFlow pooling layer.

    Args:
        data: The input tensor to the pooling layer.

    Returns:
        The TensorFlow pooling layer.
    """
    return tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

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

