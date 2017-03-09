import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework.ops import reset_default_graph

def call_policy_net(board_representation):
    reset_default_graph()#TODO: keep policy net open for entire game instead of opening it on every move
    #
    #
    #
    # saver = tf.train.Saver()
    # path = os.path.join(r'..', r'policy_net_model', r'model')
    # with tf.Session() as sess:
    #     saver.restore(sess, path)
    #     if isinstance(board_representation, list): #batch of positions to evaluate
    #         predicted_moves = sess.run(y_pred, feed_dict={X: board_representation})
    #     else: #just one position
    #         predicted_moves = sess.run(y_pred, feed_dict={X: [board_representation]})
    #     sess.close()
    #
    # return predicted_moves
    sess, y_pred, X = instantiate_session()
    if isinstance(board_representation, list):  # batch of positions to evaluate
        predicted_moves = sess.run(y_pred, feed_dict={X: board_representation})
    else: #just one position
        predicted_moves = sess.run(y_pred, feed_dict={X: [board_representation]})
    return predicted_moves

def instantiate_session():
    reset_default_graph()
    y_pred, X = build_policy_net()
    saver = tf.train.Saver()
    path = os.path.join(r'..', r'policy_net_model', r'model')
    sess = tf.Session()
    saver.restore(sess, path)
    return sess, y_pred, X

def build_policy_net():
    X = tf.placeholder(tf.float32, [None, 8, 8, 4])
    y = tf.placeholder(tf.float32, [None, 155])
    filter_size = 3  # AlphaGo used 5x5 followed by 3x3, but Go is 19x19 whereas breakthrough is 8x8 => 3x3 filters seems reasonable
    n_filters = 192
    num_hidden = 4
    n_filters_out = [n_filters] * num_hidden + [
        1]  # " # of filters in each layer ranged from 64-192; layer prior to softmax was # filters = # num_softmaxes
    n_layers = len(n_filters_out)

    # input layer
    h_layers = [hidden_layer_init(X, X.get_shape()[-1],  # n_filters in == n_feature_planes
                                  n_filters_out[0], filter_size, name='hidden_layer/1', reuse=None)]
    # hidden layers
    for i in range(0, n_layers - 1):
        h_layers.append(hidden_layer_init(h_layers[i], n_filters_out[i], n_filters_out[i + 1], filter_size,
                                          name='hidden_layer/{num}'.format(num=i + 2), reuse=None))

    outer_layer, _ = output_layer_init(h_layers[-1], reuse=None)
    return tf.nn.softmax(outer_layer), X

def hidden_layer_init(prev_layer, n_filters_in, n_filters_out, filter_size, name=None, activation=tf.nn.relu,
                      reuse=None):
    # of filters in each layer ranged from 64-192
    std_dev_He = np.sqrt(2 / np.prod(prev_layer.get_shape().as_list()[1:]))
    with tf.variable_scope(name or 'hidden_layer', reuse=reuse):
        kernel = tf.get_variable(name='weights',
                                 shape=[filter_size, filter_size,  # h x w
                                        n_filters_in,
                                        n_filters_out],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=std_dev_He)  # mean, std?
                                 )
        bias = tf.get_variable(name='bias',
                               shape=[n_filters_out],
                               initializer=tf.constant_initializer(
                                   0.01))  # karpathy: for relu, 0.01 ensures all relus fire in the beginning
        hidden_layer = activation(
            tf.nn.bias_add(
                tf.nn.conv2d(input=prev_layer,
                             filter=kernel,
                             strides=[1, 1, 1, 1],
                             padding='SAME'),
                bias
            )
        )
        return hidden_layer


def output_layer_init(layer_in, name='output_layer', reuse=None):
    layer_in = tf.reshape(layer_in, [-1, 8 * 8])
    activation = tf.nn.softmax
    n_features = layer_in.get_shape().as_list()[1]
    with tf.variable_scope(name or 'output_layer', reuse=reuse):
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