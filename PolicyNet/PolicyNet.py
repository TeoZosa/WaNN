import tensorflow as tf

class Policy_net():
    X = tf.placeholder(tf.float32, [None, 8, 8,
                                    4])  # in demo, this was reshaped afterwards; I don't need to since I know shape?
    # TODO: consider reshaping for C++ input; could also put it into 3d matrix on the fly, ex. if player == board[i][j], X[n][i][j] = [1, 0, 0]
    y = tf.placeholder(tf.float32, [None, 155])  # TODO: pass it the tf.one_hot or redo conversion
    # def _hidden_layer_init(prev_layer, n_filters_in, n_filters_out, filter_size, name=None, activation=tf.nn.relu, reuse=None):
    #     # of filters in each layer ranged from 64-192
    #     with tf.variable_scope(name or 'hidden_layer', reuse=reuse):
    #         weight = tf.get_variable(name='weight',
    #                                  shape=[filter_size, filter_size,  # h x w
    #                                         n_filters_in,
    #                                         n_filters_out],
    #                                  initializer=tf.random_normal_initializer()  # mean, std?
    #                                  )
    #         bias = tf.get_variable(name='bias',
    #                                shape=[n_filters_out],
    #                                initializer=tf.constant_initializer())
    #         hidden_layer = activation(
    #             tf.nn.bias_add(
    #                 tf.nn.conv2d(input=prev_layer,
    #                              filter=weight,
    #                              strides=[1, 1, 1, 1],
    #                              padding='SAME'),
    #                 bias
    #             )
    #         )
    #         return hidden_layer
    #
    # def _output_layer_init(layer_in, name='output_layer_init', reuse=None):
    #     layer_in = tf.reshape(layer_in, [-1, 8 * 8])
    #     activation = tf.nn.softmax
    #     n_features = layer_in.get_shape().as_list()[1]
    #     with tf.variable_scope(name or 'output_layer_init', reuse=reuse):
    #         weight = tf.get_variable(
    #             name='W',
    #             shape=[n_features, 155],  # 1 x 64 filter in, 155 classes out
    #             dtype=tf.float32,
    #             initializer=tf.contrib.layers.xavier_initializer())
    #
    #         bias = tf.get_variable(
    #             name='b',
    #             shape=[155],
    #             dtype=tf.float32,
    #             initializer=tf.constant_initializer(0.0))
    #
    #         predicted_output = activation(tf.nn.bias_add(
    #             name='h',
    #             value=tf.matmul(layer_in, weight),
    #             bias=bias))
    #         return predicted_output, weight

    def __init__(self, learning_rate=0.001, optimizer=tf.train.AdamOptimizer, activation=tf.nn.relu, epochs=4096, batch_size=128):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size

        filter_size = 3  # AlphaGo used 5x5 followed by 3x3, but Go is 19x19 whereas breakthrough is 8x8 => 3x3 filters seems reasonable
        # TODO: consider doing a grid search type experiment where n_filters = rand_val in [2**i for i in range(0,8)]
        n_filters_out = [128] * 11 + [1]  # " # of filters in each layer ranged from 64-192; layer prior to softmax was # filters = # num_softmaxes

        n_layers = 12
        with tf.variable_scope('hidden_layer/1', reuse=True):
            weight = tf.get_variable(name='weight',
                                     shape=[filter_size, filter_size,  # h x w
                                            self.X.get_shape().as_list()[-1],
                                            n_filters_out[0]],
                                     initializer=tf.random_normal_initializer()  # mean, std?
                                     )
            bias = tf.get_variable(name='bias',
                                   shape=[n_filters_out[0]],
                                   initializer=tf.constant_initializer())
            hidden_layer = activation(
                tf.nn.bias_add(
                    tf.nn.conv2d(input=self.X,
                                 filter=weight,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME'),
                    bias
                )
            )
        self.hidden_layers = [hidden_layer]
        for i in range(0, n_layers):
            with tf.variable_scope('hidden_layer/{num}'.format(num=i + 2), reuse=True):
                weight = tf.get_variable(name='weight',
                                         shape=[filter_size, filter_size,  # h x w
                                                n_filters_out[i],
                                                n_filters_out[i+1]],
                                         initializer=tf.random_normal_initializer()  # mean, std?
                                         )
                bias = tf.get_variable(name='bias',
                                       shape=[n_filters_out[i+1]],
                                       initializer=tf.constant_initializer())
                hidden_layer = activation(
                    tf.nn.bias_add(
                        tf.nn.conv2d(input=self.hidden_layers[i],
                                     filter=weight,
                                     strides=[1, 1, 1, 1],
                                     padding='SAME'),
                        bias
                    )
                )
            self.hidden_layers.extend(hidden_layer)
        with tf.variable_scope('output_layer', reuse=True):
            layer_in = tf.reshape(self.hidden_layers[-1], [-1, 8 * 8])
            n_features = layer_in.get_shape().as_list()[1]
            weight = tf.get_variable(
                name='weight',
                shape=[n_features, 155],  # 1 x 64 filter in, 155 classes out
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

            bias = tf.get_variable(
                name='bias',
                shape=[155],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))

            self.predicted_output = tf.nn.softmax(tf.nn.bias_add(
                name='h',
                value=tf.matmul(layer_in, weight),
                bias=bias))
        # output layer = softmax. in paper, also convolutional, but 19x19 softmax for player move.


    def fit(self, X, y, X_validation, y_validation, X_test, y_test, learning_rate=0.001, batch_size=128):
        y_pred = self.predicted_output
        cross_entropy = -tf.reduce_sum(tf.one_hot(y, depth=155, on_value=1.0, off_value=0.0) * tf.log(y_pred + 1e-12))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(self.epochs):
            for batch_xs, batch_ys in (X, y):
                sess.run(optimizer, feed_dict={
                    X: batch_xs,
                    y: batch_ys
                })

            print(sess.run(accuracy,#validation is data from training set randomly (at each epoch) left out to test prior to next epoch
                           feed_dict={
                               X: X_validation,
                               y: y_validation
                           }))

        # Print final test accuracy:
        print(sess.run(accuracy,  # test is data from training set left out prior to entire training, only used at end
                       feed_dict={
                           X: X_test,
                           y: y_test
                       }))
