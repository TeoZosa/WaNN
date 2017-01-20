import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from random import randint
from Tools import utils
import os
import numpy as np

class Policy_net(object):
    def __init__(self, learning_rate=0.001, optimizer=tf.train.AdamOptimizer, activation=tf.nn.relu,
                 kernel_dim=3, num_filters=128, num_hidden_layers=12, epochs=5, batch_size=128):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = 155 #154 moves + no move
        self.kernel_dim = kernel_dim
        self.num_filters = num_filters
        self.input_image = tf.placeholder(tf.float32, [None, 8, 8, 4])  
        # TODO: consider reshaping for C++ input; could also put it into 3d matrix on the fly, ex. if player == board[i][j], X[n][i][j] = [1, 0, 0]
        self.labels = tf.placeholder(tf.float32, [None, 155])  # TODO: pass it the tf.one_hot or redo conversion

          # AlphaGo used 5x5 followed by 3x3, but Go is 19x19 whereas breakthrough is 8x8 => 3x3 filters seems reasonable
        # TODO: consider doing a grid search type experiment where n_filters = rand_val in [2**i for i in range(0,8)]
        self.num_hidden_layers = num_hidden_layers
        self.num_filters_out = [self.num_filters] * self.num_hidden_layers + [1]  # " # of filters in each layer ranged from 64-192; layer prior to softmax was # filters = # num_softmaxes

        self.num_layers = len(self.num_filters_out)
        self.hidden_layers = []
        # Hidden layer 1
        with tf.variable_scope('hidden_layer/1', reuse=None):
            kernel = tf.get_variable(name='weight',
                                     shape=[kernel_dim, kernel_dim,  # h x w
                                            self.input_image.get_shape().as_list()[-1],
                                            self.num_filters_out[0]],
                                     initializer=tf.random_normal_initializer()  # mean, std?
                                     )
            bias = tf.get_variable(name='bias',
                                   shape=[self.num_filters_out[0]],
                                   initializer=tf.constant_initializer())
            hidden_layer = activation(
                tf.nn.bias_add(
                    tf.nn.conv2d(input=self.input_image,
                                 filter=kernel,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME'),
                    bias
                )
            )
        self.hidden_layers.append(hidden_layer)
        
        # Hidden layers 2-12
        for i in range(0, self.num_layers):
            with tf.variable_scope('hidden_layer/{num}'.format(num=i + 2), reuse=None):
                kernel = tf.get_variable(name='weight',
                                         shape=[kernel_dim, kernel_dim,  # h x w
                                                self.num_filters_out[i],
                                                self.num_filters_out[i+1]],
                                         initializer=tf.random_normal_initializer()  # mean, std?
                                         )
                bias = tf.get_variable(name='bias',
                                       shape=[self.num_filters_out[i+1]],
                                       initializer=tf.constant_initializer())
                hidden_layer = activation(
                    tf.nn.bias_add(
                        tf.nn.conv2d(input=self.hidden_layers[i],
                                     filter=kernel,
                                     strides=[1, 1, 1, 1],
                                     padding='SAME'),
                        bias
                    )
                )
            self.hidden_layers.append(hidden_layer)
            
        # Output layer
        with tf.variable_scope('output_layer', reuse=None):
            layer_in = tf.reshape(self.hidden_layers[-1], [-1, 8 * 8])
            n_features = layer_in.get_shape().as_list()[1]
            kernel = tf.get_variable(
                name='weight',
                shape=[n_features, self.num_classes],  # 1 x 64 filter in, 155 classes out
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

            bias = tf.get_variable(
                name='bias',
                shape=[self.num_classes],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))

            self.unscaled_output = tf.nn.bias_add(
                name='h',
                value=tf.matmul(layer_in, kernel),
                bias=bias)
            self.prediction = tf.nn.softmax(self.unscaled_output, self.labels)

        with tf.name_scope('Loss'):
            self.cost = tf.nn.softmax_cross_entropy_with_logits(self.unscaled_output, self.labels)
            self.loss = tf.reduce_mean(self.cost)
        
        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.input_image, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    def fit(self, X, y, X_test, y_test, optimizer=tf.train.AdamOptimizer, learning_rate=0.001, batch_size=None, log_dir=r'tmp\policy_net'):
        file = os.path.join(log_dir, 'text_logs')+'log'
        if batch_size is None:
            batch_size = self.batch_size
       
        X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size=512, random_state=randint(1, 1024))  # keep validation outside

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        
        for epoch_i in range(self.epochs):
            # Shuffle and batch split data
            X_train, y_train = shuffle(X_train, y_train, random_state=randint(1, 1024))
            training_batches_input, training_batches_label = utils.batch_split(X_train, y_train, batch_size)

            for i in range(0, len(training_batches_input)):
                sess.run(optimizer(learning_rate=learning_rate).minimize(self.cost), feed_dict={
                    X: training_batches_input[i],
                    y: training_batches_label[i]
                })
                # Show stats at every 1/10th interval of epoch
                if (i + 1) % (len(training_batches_input) // 10) == 0:
                    loss = sess.run(self.cost, feed_dict={
                        X: training_batches_input[i],
                        y: training_batches_label[i]
                    })
                    accuracy_score = sess.run(self.accuracy, feed_dict={
                        X: X_valid,
                        y: y_valid
                    })
                    print("Loss: {}".format(loss), end="\n", file=file)
                    print("Loss Reduced Mean: {}".format(sess.run(tf.reduce_mean(loss))), end="\n", file=file)
                    print("Loss Reduced Sum: {}".format(sess.run(tf.reduce_sum(loss))), end="\n", file=file)
                    print('Interval {interval} of 10 Accuracy: {accuracy}'.format(
                        interval=(i + 1) // (len(training_batches_input) // 10),
                        accuracy=accuracy_score), end="\n", file=file)

            # Show accuracy at end of epoch
            print('Epoch {epoch_num} Accuracy: {accuracy_score}'.format(
                epoch_num=epoch_i + 1,
                accuracy_score=sess.run(self.accuracy,
                                        feed_dict={
                                            X: X_valid,
                                            y: y_valid
                                        })), end="\n", file=file)
            # Show example of what network is predicting vs the move oracle
            label_predicted = sess.run(self.prediction, feed_dict={X: [X_valid[0]]})
            print("Sample Predicted Probabilities = "
                  "\n{y_pred}"
                  "\nPredicted vs. Actual Move = "
                  "\nIf white: {y_pred_white} vs. {y_act_white}"
                  "\nIf black: {y_pred_black} vs. {y_act_black}".format(
                y_pred=label_predicted,
                y_pred_white=utils.move_lookup(np.argmax(label_predicted), 'White'),
                y_pred_black=utils.move_lookup(np.argmax(label_predicted), 'Black'),
                y_act_white=utils.move_lookup(np.argmax(y_valid[0]), 'White'),
                y_act_black=utils.move_lookup(np.argmax(y_valid[0]), 'Black')),
                end="\n", file=file)

        # Print final test accuracy:
        print(sess.run(self.accuracy,  # test is data from training set left out prior to entire training, only used at end
                       feed_dict={
                           X: X_test,
                           y: y_test
                       }))
    
