#cython: language_level=3, boundscheck=False

import tensorflow as tf
import numpy as np
import os
import multiprocessing
from tensorflow.python.framework.ops import reset_default_graph

def call_policy_net(board_representation):#Deprecated: instantiate session directly and call inside of NeuralNet Class
    reset_default_graph()#TODO: keep policy net open for entire game instead of opening it on every move
    sess, y_pred, X = instantiate_session()
    if isinstance(board_representation, list):  # batch of positions to evaluate
        predicted_moves = sess.run(y_pred, feed_dict={X: board_representation})
    else: #just one position
        predicted_moves = sess.run(y_pred, feed_dict={X: [board_representation]})
    return predicted_moves


def instantiate_session_both():#todo: return the graph as well in case we want to run multiple NNs in the same session?
    reset_default_graph()

    with tf.variable_scope('black_net', reuse=False):
        y_pred_black, X_black = build_policy_net()
    with tf.variable_scope('white_net', reuse=False):
        y_pred_white, X_white = build_policy_net()
    NUM_CORES = multiprocessing.cpu_count()
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                   intra_op_parallelism_threads=NUM_CORES))

    saver = tf.train.Saver()
    # path = os.path.join(r'..',r'policy_net_model',  r'combined_policy_nets', r'4')
    path = os.path.join(r'..', r'policy_net_model', r'DualWinningNets065Accuracy', r'DualWinningNets065Accuracy')
    # path = os.path.join(r'C:\Users\damon\PycharmProjects\BreakthroughANN\policy_net_model',  r'combined_policy_nets', r'4')
    # path = os.path.join(r'C:\Users\damon\PycharmProjects\BreakthroughANN\policy_net_model\DualWinningNets065Accuracy', r'DualWinningNets065Accuracy')

#C:\Users\damon\PycharmProjects\BreakthroughANN\policy_net_model\DualWinningNets065Accuracy
    saver.restore(sess, path)

    return sess, y_pred_white, X_white, y_pred_black, X_black
def instantiate_session_both_128():#todo: return the graph as well in case we want to run multiple NNs in the same session?
    reset_default_graph()

    with tf.variable_scope('black_net', reuse=False):
        y_pred_black, X_black = build_policy_net_128()
    with tf.variable_scope('white_net', reuse=False):
        y_pred_white, X_white = build_policy_net_128()
    NUM_CORES = multiprocessing.cpu_count()
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                   intra_op_parallelism_threads=NUM_CORES))

    saver = tf.train.Saver()
    # path = os.path.join(r'..',r'policy_net_model',  r'combined_policy_nets', r'4')
    path = os.path.join(r'..', r'policy_net_model', r'685AccWhite', r'0507WinningWhiteNet065Accuracy_192_4_')
    # path = os.path.join(r'C:\Users\damon\PycharmProjects\BreakthroughANN\policy_net_model',  r'combined_policy_nets', r'4')
    # path = os.path.join(r'C:\Users\damon\PycharmProjects\BreakthroughANN\policy_net_model\DualWinningNets065Accuracy', r'DualWinningNets065Accuracy')

#C:\Users\damon\PycharmProjects\BreakthroughANN\policy_net_model\DualWinningNets065Accuracy
    saver.restore(sess, path)

    return sess, y_pred_white, X_white, y_pred_black, X_black



def instantiate_session():  # todo: return the graph as well in case we want to run multiple NNs in the same session?
    reset_default_graph()

    y_pred, X = build_policy_net()
    saver = tf.train.Saver()
    path = os.path.join(r'..', r'policy_net_model', r'model')
    NUM_CORES = multiprocessing.cpu_count()
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                                            intra_op_parallelism_threads=NUM_CORES))
    saver.restore(sess, path)
    # tf.train.write_graph(sess.graph_def, 'cppModels/', 'policyNet.pb', as_text=False)
    # tf.python.framework.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), y_pred.name)
    return sess, y_pred, X

def instantiate_session_white():  # todo: return the graph as well in case we want to run multiple NNs in the same session?
    reset_default_graph()
    with tf.variable_scope('white_net', reuse=False):
       y_pred, X = build_policy_net()
    saver = tf.train.Saver()
    path = os.path.join(r'..', r'policy_net_model', r'white_policy_net', r'4')
    NUM_CORES = multiprocessing.cpu_count()
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                                            intra_op_parallelism_threads=NUM_CORES))
    saver.restore(sess, path)
    # tf.train.write_graph(sess.graph_def, 'cppModels/', 'policyNet.pb', as_text=False)
    # tf.python.framework.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), y_pred.name)
    return sess, y_pred, X

def instantiate_session_black():  # todo: return the graph as well in case we want to run multiple NNs in the same session?
    reset_default_graph()
    with tf.variable_scope('black_net', reuse=False):
        y_pred, X = build_policy_net()
    saver = tf.train.Saver()
    path = os.path.join(r'..', r'policy_net_model', r'black_policy_net', r'BLACK_Policy_Net')
    NUM_CORES = multiprocessing.cpu_count()
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                                            intra_op_parallelism_threads=NUM_CORES))
    saver.restore(sess, path)
    # tf.train.write_graph(sess.graph_def, 'cppModels/', 'policyNet.pb', as_text=False)
    # tf.python.framework.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), y_pred.name)
    return sess, y_pred, X

def build_policy_net():
    input = tf.placeholder(tf.float32, [None, 8, 8, 4])
    # y = tf.placeholder(tf.float32, [None, 155])
    filter_size = 3
    n_filters = 192
    num_hidden = 4
    n_filters_out = [n_filters] * num_hidden + [1]
    n_layers = len(n_filters_out)

    # input layer to first hidden layer
    h_layers = [hidden_layer_init(input, input.get_shape()[-1],  # n_filters in == n_feature_planes
                                  n_filters_out[0], filter_size, name='hidden_layer/1', reuse=None)]
    # hidden layers
    for i in range(0, n_layers - 1):
        h_layers.append(hidden_layer_init(h_layers[i], n_filters_out[i], n_filters_out[i + 1], filter_size,
                                          name='hidden_layer/{num}'.format(num=i + 2), reuse=None))

    output, _ = output_layer_init(h_layers[-1], reuse=None)
    return output, input


def build_policy_net_128():
    input = tf.placeholder(tf.float32, [None, 8, 8, 4])
    # y = tf.placeholder(tf.float32, [None, 155])
    filter_size = 3
    n_filters = 192
    num_hidden = 4
    n_filters_out = [n_filters] * num_hidden + [1]
    n_layers = len(n_filters_out)
    filter_sizes = [filter_size]*num_hidden + [1]

    # input layer to first hidden layer
    h_layers = [hidden_layer_init_128(input, input.get_shape()[-1],  # n_filters in == n_feature_planes
                                  n_filters_out[0], filter_sizes[0], name='hidden_layer/1', reuse=None)]
    # hidden layers
    for i in range(0, n_layers - 1):
        h_layers.append(hidden_layer_init_128(h_layers[i], n_filters_out[i], n_filters_out[i + 1], filter_sizes[i+1],
                                          name='hidden_layer/{num}'.format(num=i + 2), reuse=None))

    output, _ = output_layer_init(h_layers[-1], reuse=None)
    return output, input


def hidden_layer_init(prev_layer, n_filters_in, n_filters_out, filter_size, name=None, activation=tf.nn.relu,
                      reuse=None):

    std_dev_He = np.sqrt(2 / np.prod(prev_layer.get_shape().as_list()[1:])) #He et. al
    #initialize layer in the given namespace
    with tf.variable_scope(name or 'hidden_layer', reuse=reuse):
        # convolutional filters
        kernel = tf.get_variable(
            name='weights',
            shape=[filter_size, filter_size,  # h x w
            n_filters_in,
            n_filters_out],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=std_dev_He))

        bias = tf.get_variable(
            name='bias',
            shape=[n_filters_out],
            initializer=tf.constant_initializer(
            0.01))  # karpathy: for relu, 0.01 initializer ensures all relus fire in the beginning

        hidden_layer = activation(
            tf.nn.bias_add(
                tf.nn.conv2d(
                    input=prev_layer,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME'),
                bias))

        return hidden_layer

def hidden_layer_init_128(prev_layer, n_filters_in, n_filters_out, filter_size, name=None, activation=tf.nn.relu,
                      reuse=None):

    std_dev_He = np.sqrt(2 / np.prod(prev_layer.get_shape().as_list()[1:])) #He et. al
    #initialize layer in the given namespace
    with tf.variable_scope(name or 'hidden_layer', reuse=reuse):
        if name == "hidden_layer/1":
            # paddings = [[0, 0], [4, 4], [4, 4], [0, 0]]
            # tf.pad(prev_layer, paddings, "CONSTANT")
            filter_size = 5
        # convolutional filters
        kernel = tf.get_variable(
            name='weights',
            shape=[filter_size, filter_size,  # h x w
            n_filters_in,
            n_filters_out],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=std_dev_He))

        bias = tf.get_variable(
            name='bias',
            shape=[n_filters_out],
            initializer=tf.constant_initializer(
            0.01))  # karpathy: for relu, 0.01 initializer ensures all relus fire in the beginning

        hidden_layer = activation(
            tf.nn.bias_add(
                tf.nn.conv2d(
                    input=prev_layer,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME'),
                bias))

        return hidden_layer

def output_layer_init(layer_in, name='output_layer', reuse=None):
    layer_in = tf.reshape(layer_in, [-1, 8 * 8])
    activation = tf.nn.softmax
    n_features = layer_in.get_shape().as_list()[1]

    #initialize layer in the given namespace
    with tf.variable_scope(name or 'output_layer', reuse=reuse):
        #convolutional filters
        kernel = tf.get_variable(
            name='weights',
            shape=[n_features, 155],  # 1 x 64 filter in, 1 class out
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        bias = tf.get_variable(
            name='bias',
            shape=[155],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        predictions = activation(tf.nn.bias_add(
            name='output',
            value=tf.matmul(layer_in, kernel),
            bias=bias))
        return predictions, kernel