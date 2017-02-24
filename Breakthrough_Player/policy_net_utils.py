import tensorflow as tf
import numpy as np
def call_policy_net(board_representation):
    sess = tf.Session()
    saver = tf.train.Saver()

    #TODO: restore policy net from graphdef checkpoint
    X = tf.placeholder(tf.float32, [None, 8, 8, 4])
    y = tf.placeholder(tf.float32, [None, 155])
    filter_size = 3  # AlphaGo used 5x5 followed by 3x3, but Go is 19x19 whereas breakthrough is 8x8 => 3x3 filters seems reasonable
    n_filters = 64
    num_hidden = 11
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

    # output layer = softmax. in paper, also convolutional, but 19x19 softmax for player move.
    inference, _ = output_layer_init(h_layers[-1], reuse=None)

    path = r'G:\TruncatedLogs\PythonDatasets\Datastructures\NumpyArrays\PolicyNet\4DArraysHDF5(RxCxF)POEPolicyNet3rdThird'
    saver.restore(sess, path)
    predicted_moves = sess.run(inference, feed_dict={X: [board_representation]})
    sess.close()
    return predicted_moves



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