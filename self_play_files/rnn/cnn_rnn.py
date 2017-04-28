
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
from sklearn.utils import shuffle
import time
import seaborn as sns
from tools import utils
import h5py
import os
from PIL import Image
import random
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.contrib import rnn
import sys
import multiprocessing
import pickle

# from Breakthrough_Player.policy_net_utils import instantiate_session_both_RNN

sns.set(color_codes=True)


def WriteToDisk(path, estimator):
    write_path = path + r'Models/'
    estimator.save(write_path)


def image_stuff():
    # Train/validation/test split or reducing number of training examples
    sampleInput = X_WHITE[30]
    who_dict = {
        1.0: 127,
        0.0: 0,
    }
    who_dict1 = {
        1.0: 255,
        0.0: 0,
    }
    mean_player = np.mean(X_WHITE, axis=0)
    std_ = np.std(X_WHITE, axis=0)
    var = np.mean(std_, axis=2)

    mode_args = ['CMYK', 'RGB', 'YCbCr']
    for mode_arg in mode_args:
        plot.figure()
        plot.imshow(Image.fromarray(mean_player[:, :, 0], mode=mode_arg))
        plot.figure()
        plot.imshow(Image.fromarray(std_[:, :, 0], mode=mode_arg))

        plot.figure()
        plot.imshow(Image.fromarray(mean_player[:, :, 1], mode=mode_arg))
        plot.figure()
        plot.imshow(Image.fromarray(std_[:, :, 1], mode=mode_arg))

        plot.figure()
        plot.imshow(Image.fromarray(mean_player[:, :, 2], mode=mode_arg))
        plot.figure()
        plot.imshow(Image.fromarray(std_[:, :, 2], mode=mode_arg))

        plot.figure()
        plot.imshow(Image.fromarray(var[:, :], mode=mode_arg))

    sampleInput = sampleInput.transpose()
    colorE = pd.DataFrame(sampleInput[2]).apply(lambda x: x.apply(lambda y: who_dict[y])).as_matrix()
    colorO = pd.DataFrame(sampleInput[1]).apply(lambda x: x.apply(lambda y: who_dict1[y])).as_matrix()
    sampleInput[2] = colorE
    sampleInput[1] = colorO
    sampleInput = sampleInput.transpose()

    normal = sampleInput.transpose()
    player = normal[0].transpose()
    opponent = normal[1].transpose()
    empty = normal[2].transpose()
    fr_p = (pd.DataFrame(player))
    fr_o = pd.DataFrame(opponent)
    fr_e = pd.DataFrame(empty)

    print(fr_p, '\n', fr_o, '\n', fr_e)
    mode_arg = 'CMYK'
    POEB_img = Image.fromarray(sampleInput, mode_arg)
    P_img = Image.fromarray(sampleInput[:, :, 0], mode_arg)
    O_img = Image.fromarray(sampleInput[:, :, 1], mode_arg)
    E_img = Image.fromarray(sampleInput[:, :, 2], mode_arg)
    OE_img = Image.fromarray(sampleInput[:, :, 1:3], mode_arg)

    # plot.figure()
    # plot.imshow(P_img)
    # plot.figure()
    # plot.imshow(O_img)
    # plot.figure()
    # plot.imshow(OE_img)
    # plot.figure()
    # plot.imshow(POE_img)
    # plot.figure()
    # plot.imshow(POEB_img)


def assign_device(path):
    if path == r'/Users/teofilozosa/PycharmProjects/BreakthroughANN/':
        device = 'MBP2011_'
    elif path == r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/':
        device = 'MBP2014'
    elif path == r'/Users/Home/PycharmProjects/BreakthroughANN/':
        device = 'MBP2011'
    else:
        device = 'AWS'  # todo: configure AWS path
    return device


def assign_path(deviceName='Workstation'):
    if deviceName == 'MBP2011_':
        path = r'/Users/teofilozosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2014':
        path = r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2011':
        path = r'/Users/Home/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'Workstation':
        path = r'G:\TruncatedLogs\PythonDatasets\Datastructures\NumpyArrays\{net_type}\{features}\4DArraysHDF5(RxCxF){features}{net_type}AllThirdWhite_CNN_RNN'.format(
            features='POE', net_type='PolicyNet')
    elif deviceName == 'Workstation_Black':
        path = r'G:\TruncatedLogs\PythonDatasets\Datastructures\NumpyArrays\{net_type}\{features}\4DArraysHDF5(RxCxF){features}{net_type}AllThirdBlack_CNN_RNN'.format(
            features='POE', net_type='PolicyNet')
    else:
        path = ''  # todo:error checking
    return path


def get_net_type(type):
    type_dict = {'1stThird': 'Start',
                 '2ndThird': 'Mid',
                 '3rdThird': 'End',
                 'AllThird': 'Full'}
    return type_dict[type]


def load_examples_and_labels(path):
    example_files = [file_name for file_name in utils.find_files(path, "*X_CNNRNNSeparatedGames.p")]
    label_files = [file_name for file_name in utils.find_files(path, "*yCNNRNNSeparatedGames.p")]
    examples, labels = ([] for i in range(2))
    for example_file in example_files:  # add training examples and corresponding labels from each file to associated arrays
        file = open(os.path.join(path, example_file), 'r+b')
        example = pickle.load(file)
        examples.extend(example)
        file.close()
    for label_file in label_files:  # add training examples and corresponding labels from each file to associated arrays
        file = open(os.path.join(path, label_file), 'r+b')
        label = pickle.load(file)
        labels.extend(label)
        file.close()
    return examples, labels
    # return np.array(examples, dtype=np.float32), np.array(labels, dtype=np.float32)  # b x r x c x f & label



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
        # variable_summaries(kernel)

        bias = tf.get_variable(name='bias',
                               shape=[n_filters_out],
                               initializer=tf.constant_initializer(
                                   0.01))  # karpathy: for relu, 0.01 ensures all relus fire in the beginning

        # variable_summaries(bias)

        hidden_layer = activation(
            tf.nn.bias_add(
                tf.nn.conv2d(input=prev_layer,
                             filter=kernel,
                             strides=[1, 1, 1, 1],
                             padding='SAME'),
                bias
            )
        )
        # tf.summary.histogram('activations', hidden_layer)
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

        # variable_summaries(kernel)

        bias = tf.get_variable(
            name='bias',
            shape=[155],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        # variable_summaries(bias)

        unscaled_output = (tf.nn.bias_add(
            name='output',
            value=tf.matmul(layer_in, kernel),
            bias=bias))
        # tf.summary.histogram('unscaled_output', unscaled_output)
        return unscaled_output, kernel

def RNN(layer_in, num_hidden_layers, num_hidden_units, num_inputs_in=155):
    layer_in = tf.reshape(layer_in, [-1, 8 * 8])

    n_features = layer_in.get_shape().as_list()[1]
    num_inputs_in = 155
    num_classes = 155
    # reshape to [1, n_input]
    X = tf.reshape(layer_in, [-1, n_features])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    X = tf.split(X, n_features, 1)

    # 1-layer LSTM with n_hidden units.

    # rnn_cell = rnn.BasicLSTMCell(num_hidden)

    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden_units)] * num_hidden_layers)


    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, X, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden_units, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
def loss_init(output_layer, labels):
    with tf.variable_scope("cross_entropy"):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=labels)
        # loss_init = tf.reduce_mean(losses)
        # tf.summary.histogram('cross_entropy', losses)
    return losses


def compute_accuracy(examples, labels, X, y, accuracy_function, merged, test_writer, batch):
    if merged is not None:
        summary, accuracy_score = sess.run([merged, accuracy_function], feed_dict={X: examples, y: labels})
        test_writer.add_summary(summary, batch)
    else:
        accuracy_score = sess.run([accuracy_function], feed_dict={X: examples, y: labels})
    return accuracy_score


def train_model(examples, labels, X, y, optimizer, merged, train_writer, batch):
    if merged is not None:
        summary, _ = sess.run([merged, optimizer], feed_dict={
            X: examples,
            y: labels
        })
        train_writer.add_summary(summary, batch)
    else:
        sess.run([optimizer], feed_dict={
            X: examples,
            y: labels
        })


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.variable_scope(var.op.name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def print_hyperparameters(learning_rate, batch_size, n_epochs, n_filters, num_hidden, file_to_write):
    print("\nAdam Optimizer"
          "\nNum Filters: {num_filters}"
          "\nNum Hidden Layers: {num_hidden}"
          "\nLearning Rate: {learning_rate}"
          "\nBatch Size: {batch_size}"
          "\n# of Epochs: {n_epochs}".format(
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        num_filters=n_filters,
        num_hidden=num_hidden), end="\n", file=file_to_write)


def print_partition_accuracy_statistics(examples, labels, X, y, partition, accuracy_function, file_to_write):
    with tf.name_scope(partition + r'-states'):
        print("Final Test Accuracy ({partition}-states): {accuracy}".format(
            partition=partition,
            accuracy=compute_accuracy(examples, labels, X, y, accuracy_function, merged, test_writer, 0)),
            end="\n",
            file=file_to_write)


def print_prediction_statistics(examples, labels, y_pred, X, file_to_write):
    num_top_moves = 10
    random_example = random.randrange(0, len(examples))
    labels_predictions = sess.run(y_pred, feed_dict={X: [examples[random_example]]})
    correct_move_index = np.argmax(labels[random_example])
    predicted_move_index = np.argmax(labels_predictions)
    top_n_indexes = sorted(range(len(labels_predictions[0])), key=lambda i: labels_predictions[0][i], reverse=True)[
                    :num_top_moves]
    if (correct_move_index in top_n_indexes):
        rank_in_prediction = top_n_indexes.index(correct_move_index) + 1  # offset 0-indexing
        in_top_n = True
    else:
        rank_in_prediction = in_top_n = False
    top_n_white_moves = list(map(lambda index: utils.move_lookup_by_index(index, 'White'), top_n_indexes))
    top_n_black_moves = list(map(lambda index: utils.move_lookup_by_index(index, 'Black'), top_n_indexes))
    print("Sample Predicted Probabilities = "
          "\n{y_pred}"
          "\nIn top {num_top_moves} = {in_top_n}"
          "\nTop move's rank in prediction = {move_rank}"
          "\nTop {num_top_moves} moves ranked first to last = "
          "\nif White: \n {top_white_moves}"
          "\nif Black: \n {top_black_moves}"
          "\nActual Move = "
          "\nif White: {y_act_white}"
          "\nif Black: {y_act_black}".format(
        y_pred=labels_predictions,
        in_top_n=in_top_n,
        num_top_moves=num_top_moves,
        move_rank=rank_in_prediction,
        top_white_moves=top_n_white_moves,
        top_black_moves=top_n_black_moves,
        y_pred_white=utils.move_lookup_by_index(predicted_move_index, 'White'),
        y_pred_black=utils.move_lookup_by_index(predicted_move_index, 'Black'),
        y_act_white=utils.move_lookup_by_index(correct_move_index, 'White'),
        y_act_black=utils.move_lookup_by_index(correct_move_index, 'Black')),
        end="\n", file=file_to_write)

    # correct_prediction_position = [0] * 155
    # not_in_top_10 = 0
    # labels_predictions = sess.run(y_pred, feed_dict={X: examples})
    # with tf.name_scope('partition_probability_difference'):
    #     for example_num in range(0, len(labels_predictions)):
    #         correct_move_index = np.argmax(labels[example_num])
    #         predicted_move_index = np.argmax(labels_predictions[example_num])
    #         top_n_indexes = sorted(range(len(labels_predictions[example_num])),
    #                                key=lambda i: labels_predictions[example_num][i], reverse=True)[
    #                         :
    #                         # num_top_moves
    #                         ]
    #         if (correct_move_index in top_n_indexes):
    #             rank_in_prediction = top_n_indexes.index(correct_move_index)  # 0 indexed
    #             correct_prediction_position[rank_in_prediction] += 1
    #             correct_move_predicted_prob = labels_predictions[example_num][correct_move_index] * 100
    #
    #             if correct_move_index != predicted_move_index:
    #                 print("Incorrect prediction. Correct move was ranked {}".format(rank_in_prediction + 1),
    #                       file=file_to_write)
    #                 top_move_predicted_prob = labels_predictions[example_num][predicted_move_index] * 100
    #                 difference = sess.run(tf.add(top_move_predicted_prob, - correct_move_predicted_prob))
    #                 print("Predicted move probability = %{pred_prob}. \n"
    #                       "Correct move probability = %{correct_prob}\n"
    #                       "Difference = %{prob_diff}\n".format(pred_prob=top_move_predicted_prob,
    #                                                            correct_prob=correct_move_predicted_prob,
    #                                                            prob_diff=difference),
    #                       file=file_to_write)
    #                 in_top_n = True
    #                 # tf.summary.scalar('incorrect prediction: probability difference between top move and correct',
    #                 #                   difference)
    #                 # difference = tf.add(100, - correct_move_predicted_prob)
    #                 # tf.summary.scalar('incorrect prediction: predicted probability difference of correct move',
    #                 #                   difference)
    #
    #
    #             else:
    #                 difference = tf.add(100, - correct_move_predicted_prob)
    #                 # tf.summary.scalar('correct prediction: probability difference', difference)
    #
    #
    #         else:
    #             # rank_in_prediction = in_top_n = False
    #             not_in_top_10 += 1
    #
    # total_predictions = sum(correct_prediction_position) + not_in_top_10
    # percent_in_top = 0
    # for i in range(0, len(correct_prediction_position)):
    #     rank_percent = (correct_prediction_position[i] * 100) / total_predictions
    #     print("Correct move was in predicted rank {num} slot = %{percent}".format(num=i + 1, percent=rank_percent),
    #           file=file_to_write)
    #     percent_in_top += rank_percent
    #     print("Percent in top {num} predictions = %{percent}\n".format(num=i + 1, percent=percent_in_top),
    #           file=file_to_write)
    # print("Percentage of time not in predictions = {}".format((not_in_top_10 * 100) / total_predictions),
    #       file=file_to_write)


def print_partition_statistics(examples, labels, X, y, partition, accuracy_function, file):
    print_partition_accuracy_statistics(examples, labels, X, y, partition, accuracy_function, file)
    # print_prediction_statistics(examples, labels, file)


# TODO: excise code to be run in main script.

# main script

# config = run_config.RunConfig(num_cores=-1)
input_path_WHITE = assign_path()
input_path_BLACK = assign_path("Workstation_Black")
device = assign_device(input_path_WHITE)
game_stage = input_path_WHITE[-21:-13]  # ex. 1stThird
net_type = get_net_type(game_stage)

# for experiment with states from entire games, test data are totally separate games
training_examples_WHITE, training_labels_WHITE = load_examples_and_labels(os.path.join(input_path_WHITE, r'TrainingData'))
validation_examples_WHITE, validation_labels_WHITE = load_examples_and_labels(os.path.join(input_path_WHITE, r'ValidationData'))

training_examples_BLACK, training_labels_BLACK = load_examples_and_labels(os.path.join(input_path_BLACK, r'TrainingData'))
validation_examples_BLACK, validation_labels_BLACK = load_examples_and_labels(os.path.join(input_path_BLACK, r'ValidationData'))
#
#
# if (net_type == 'Start'):
#     partition_i = 'Mid'
#     partition_j = 'End'
#     WHITE_testing_examples_partition_i, WHITE_testing_labels_partition_i = load_examples_and_labels(
#         os.path.join(input_path_WHITE, r'TestDataMid'))  # 210659 states
#     WHITE_testing_examples_partition_j, WHITE_testing_labels_partition_j = load_examples_and_labels(
#         os.path.join(input_path_WHITE, r'TestDataEnd'))
#
#     BLACK_testing_examples_partition_i, BLACK_testing_labels_partition_i = load_examples_and_labels(
#         os.path.join(input_path_BLACK, r'TestDataMid'))  # 210659 states
#     BLACK_testing_examples_partition_j, BLACK_testing_labels_partition_j = load_examples_and_labels(
#         os.path.join(input_path_BLACK, r'TestDataEnd'))
# elif (net_type == 'Mid'):
#     partition_i = 'Start'
#     partition_j = 'End'
#     WHITE_testing_examples_partition_i, WHITE_testing_labels_partition_i = load_examples_and_labels(
#         os.path.join(input_path_WHITE, r'TestDataStart'))  # 210659 states
#     WHITE_testing_examples_partition_j, WHITE_testing_labels_partition_j = load_examples_and_labels(
#         os.path.join(input_path_WHITE, r'TestDataEnd'))
#
#     BLACK_testing_examples_partition_i, BLACK_testing_labels_partition_i = load_examples_and_labels(
#         os.path.join(input_path_BLACK, r'TestDataStart'))  # 210659 states
#     BLACK_testing_examples_partition_j, BLACK_testing_labels_partition_j = load_examples_and_labels(
#         os.path.join(input_path_BLACK, r'TestDataEnd'))
# elif (net_type == 'End'):
#     partition_i = 'Start'
#     partition_j = 'Mid'
#     WHITE_testing_examples_partition_i, WHITE_testing_labels_partition_i = load_examples_and_labels(
#         os.path.join(input_path_WHITE, r'TestDataStart'))  # 210659 states
#     WHITE_testing_examples_partition_j, WHITE_testing_labels_partition_j = load_examples_and_labels(
#         os.path.join(input_path_WHITE, r'TestDataMid'))
#
#     BLACK_testing_examples_partition_i, BLACK_testing_labels_partition_i = load_examples_and_labels(
#         os.path.join(input_path_BLACK, r'TestDataStart'))  # 210659 states
#     BLACK_testing_examples_partition_j, BLACK_testing_labels_partition_j = load_examples_and_labels(
#         os.path.join(input_path_BLACK, r'TestDataMid'))
# else:
#     partition_i = 'Start'
#     partition_j = 'Mid'
#     partition_k = 'End'
#     WHITE_testing_examples_partition_i, WHITE_testing_labels_partition_i = load_examples_and_labels(
#         os.path.join(input_path_WHITE, r'TestDataStart'))  # 210659 states
#     WHITE_testing_examples_partition_j, WHITE_testing_labels_partition_j = load_examples_and_labels(
#         os.path.join(input_path_WHITE, r'TestDataMid'))
#     WHITE_testing_examples_partition_k, WHITE_testing_labels_partition_k = load_examples_and_labels(
#         os.path.join(input_path_WHITE, r'TestDataEnd'))
#
#     BLACK_testing_examples_partition_i, BLACK_testing_labels_partition_i = load_examples_and_labels(
#         os.path.join(input_path_BLACK, r'TestDataStart'))  # 210659 states
#     BLACK_testing_examples_partition_j, BLACK_testing_labels_partition_j = load_examples_and_labels(
#         os.path.join(input_path_BLACK, r'TestDataMid'))
#     BLACK_testing_examples_partition_k, BLACK_testing_labels_partition_k = load_examples_and_labels(
#         os.path.join(input_path_BLACK, r'TestDataEnd'))


for num_hidden in [i for i in [4, 9]
                   # range(1,10)
                   ]:
    file_WHITE = open(os.path.join(input_path_WHITE,
                                   r'ExperimentLogs',
                                   game_stage + '065RNN_0427WHITE192Filters{}LayersTF_CE__He_weightsPOE.txt'.format(num_hidden)),
                      'a')

    file_BLACK = open(os.path.join(input_path_WHITE,
                                   r'ExperimentLogs',
                                   game_stage + '065RNN_0427BLACK192Filters{}LayersTF_CE__He_weightsPOE.txt'.format(
                                       num_hidden)), 'a')

    print("# of Testing Examples: {}".format(len(validation_examples_BLACK)), end='\n', file=file_WHITE)
    for n_filters in [
        #  64,
        # 128,
        192,
        # 256,
        # 512
    ]:
        for learning_rate in [
            0.001,
            # 0.0011,
            # 0.0012, 0.0013,
            #                   0.0014, 0.0015
        ]:
            reset_default_graph()
            batch_size = 128

            filter_size = 3  # AlphaGo used 5x5 followed by 3x3, but Go is 19x19 whereas breakthrough is 8x8 => 3x3 filters seems reasonable

            # TODO: consider doing a grid search type experiment where n_filters = rand_val in [2**i for i in range(0,8)]
            # TODO: dropout? regularizers? different combinations of weight initialization, cost func, num_hidden, etc.
            # Yoshua Bengio: "Because of early stopping and possible regularizers, it is mostly important to choose n sub h large enough.
            # Larger than optimal values typically do not hurt generalization performance much, but of course they require proportionally more computation..."
            # "...same size for all layers worked generally better or the same as..."
            # num_hidden = 11
            n_filters_out = [n_filters] * num_hidden + [
                1]  # " # of filters in each layer ranged from 64-192; layer prior to softmax was # filters = # num_softmaxes
            n_layers = len(n_filters_out)


            with tf.variable_scope('white_net', reuse=False):

                # build graph
                X_WHITE = tf.placeholder(tf.float32, [None, 8, 8, 4])
                # TODO: consider reshaping for C++ input; could also put it into 3d matrix on the fly, ex. if player == board[i][j], X[n][i][j] = [1, 0, 0, 1]
                y_WHITE = tf.placeholder(tf.float32, [None, 155])

                # input layer
                h_layers_WHITE = [
                    hidden_layer_init(X_WHITE, X_WHITE.get_shape()[-1],  # n_filters in == n_feature_planes
                                      n_filters_out[0], filter_size, name='hidden_layer/1', reuse=None)]
                # hidden layers
                for i in range(0, n_layers - 1):
                    h_layers_WHITE.append(
                        hidden_layer_init(h_layers_WHITE[i], n_filters_out[i], n_filters_out[i + 1], filter_size,
                                          name='hidden_layer/{num}'.format(num=i + 2), reuse=None))

                # output layer = softmax. in paper, also convolutional, but 19x19 softmax for player move.
                outer_layer_WHITE, _ = output_layer_init(h_layers_WHITE[-1], reuse=None)
                # outer_layer_WHITE = RNN(h_layers_WHITE[-1], 1)

                # TODO: if making 2 filters, 1 for each player color softmax, have a check that dynamically makes y_pred correspond to the right filter
                y_pred_WHITE = tf.nn.softmax(outer_layer_WHITE)

                # tf's internal softmax; else, put softmax back in output layer
                # cost = tf.nn.softmax_cross_entropy_with_logits(logits=outer_layer, labels=y)
                cost_WHITE = loss_init(outer_layer_WHITE, y_WHITE)

                # # alternative implementation
                # cost = tf.reduce_mean(cost) #used in MNIST tensorflow

                # kadenze cross_entropy cost function
                # cost = -tf.reduce_sum(y * tf.log(y_pred + 1e-12))


                # way better performance
                optimizer_WHITE = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_WHITE)

                # SGD used in AlphaGO
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
                with tf.name_scope('accuracy'):
                    correct_prediction_WHITE = tf.equal(tf.argmax(y_pred_WHITE, 1), tf.argmax(y_WHITE, 1))
                    accuracy_function_WHITE = tf.reduce_mean(tf.cast(correct_prediction_WHITE, 'float'))
                    # tf.summary.scalar('accuracy', accuracy_function_WHITE)

            with tf.variable_scope('black_net', reuse=False):

                # build graph
                X_BLACK = tf.placeholder(tf.float32, [None, 8, 8, 4])
                # TODO: consider reshaping for C++ input; could also put it into 3d matrix on the fly, ex. if player == board[i][j], X[n][i][j] = [1, 0, 0, 1]
                y_BLACK = tf.placeholder(tf.float32, [None, 155])

                # input layer
                h_layers_BLACK = [
                    hidden_layer_init(X_BLACK, X_BLACK.get_shape()[-1],  # n_filters in == n_feature_planes
                                      n_filters_out[0], filter_size, name='hidden_layer/1', reuse=None)]
                # hidden layers
                for i in range(0, n_layers - 1):
                    h_layers_BLACK.append(
                        hidden_layer_init(h_layers_BLACK[i], n_filters_out[i], n_filters_out[i + 1], filter_size,
                                          name='hidden_layer/{num}'.format(num=i + 2), reuse=None))

                # output layer = softmax. in paper, also convolutional, but 19x19 softmax for player move.
                outer_layer_BLACK, _ = output_layer_init(h_layers_BLACK[-1], reuse=None)


                # TODO: if making 2 filters, 1 for each player color softmax, have a check that dynamically makes y_pred correspond to the right filter
                y_pred_BLACK = tf.nn.softmax(outer_layer_BLACK)

                # tf's internal softmax; else, put softmax back in output layer
                # cost = tf.nn.softmax_cross_entropy_with_logits(logits=outer_layer, labels=y)
                cost_BLACK = loss_init(outer_layer_BLACK, y_BLACK)

                # # alternative implementation
                # cost = tf.reduce_mean(cost) #used in MNIST tensorflow

                # kadenze cross_entropy cost function
                # cost = -tf.reduce_sum(y * tf.log(y_pred + 1e-12))


                # way better performance
                optimizer_BLACK = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_BLACK)

                # SGD used in AlphaGO
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
                with tf.name_scope('accuracy'):
                    correct_prediction_BLACK = tf.equal(tf.argmax(y_pred_BLACK, 1), tf.argmax(y_BLACK, 1))
                    accuracy_function_BLACK = tf.reduce_mean(tf.cast(correct_prediction_BLACK, 'float'))
                    # tf.summary.scalar('accuracy', accuracy_function_BLACK)

            # save the model
            saver = tf.train.Saver()

            NUM_CORES = multiprocessing.cpu_count()
            sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                                                    intra_op_parallelism_threads=NUM_CORES))

            path = os.path.join(r'..', r'..', r'policy_net_model', r'DualWinningNets065Accuracy',
                                r'DualWinningNets065Accuracy')
            #
            saver.restore(sess, path)

            num_layers_rnn = 3
            num_hidden_rnn = 512
            with tf.variable_scope('white_net_RNN', reuse=False):

                outer_layer_WHITE = RNN(h_layers_WHITE[-1], num_layers_rnn, num_hidden_rnn)

                # TODO: if making 2 filters, 1 for each player color softmax, have a check that dynamically makes y_pred correspond to the right filter
                y_pred_WHITE = tf.nn.softmax(outer_layer_WHITE)

                # tf's internal softmax; else, put softmax back in output layer
                # cost = tf.nn.softmax_cross_entropy_with_logits(logits=outer_layer, labels=y)
                cost_WHITE = loss_init(outer_layer_WHITE, y_WHITE)

                # # alternative implementation
                # cost = tf.reduce_mean(cost) #used in MNIST tensorflow

                # kadenze cross_entropy cost function
                # cost = -tf.reduce_sum(y * tf.log(y_pred + 1e-12))


                # way better performance
                optimizer_WHITE = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_WHITE)

                # SGD used in AlphaGO
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
                with tf.name_scope('accuracy'):
                    correct_prediction_WHITE = tf.equal(tf.argmax(y_pred_WHITE, 1), tf.argmax(y_WHITE, 1))
                    accuracy_function_WHITE = tf.reduce_mean(tf.cast(correct_prediction_WHITE, 'float'))
                    # tf.summary.scalar('accuracy', accuracy_function_WHITE)

            with tf.variable_scope('black_net_RNN', reuse=False):
                outer_layer_BLACK= RNN(h_layers_BLACK[-1], num_layers_rnn, num_hidden_rnn)

                # TODO: if making 2 filters, 1 for each player color softmax, have a check that dynamically makes y_pred correspond to the right filter
                y_pred_BLACK = tf.nn.softmax(outer_layer_BLACK)

                # tf's internal softmax; else, put softmax back in output layer
                # cost = tf.nn.softmax_cross_entropy_with_logits(logits=outer_layer, labels=y)
                cost_BLACK = loss_init(outer_layer_BLACK, y_BLACK)

                # # alternative implementation
                # cost = tf.reduce_mean(cost) #used in MNIST tensorflow

                # kadenze cross_entropy cost function
                # cost = -tf.reduce_sum(y * tf.log(y_pred + 1e-12))


                # way better performance
                optimizer_BLACK = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_BLACK)

                # SGD used in AlphaGO
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
                with tf.name_scope('accuracy'):
                    correct_prediction_BLACK = tf.equal(tf.argmax(y_pred_BLACK, 1), tf.argmax(y_BLACK, 1))
                    accuracy_function_BLACK = tf.reduce_mean(tf.cast(correct_prediction_BLACK, 'float'))
                    # tf.summary.scalar('accuracy', accuracy_function_BLACK)



            # tensorboard summaries
            # merged = tf.summary.merge_all()
            merged = None
            train_writer = None
            test_writer = None
            # train_writer = tf.summary.FileWriter(os.path.join(input_path_WHITE,
            #                                                   r'ExperimentLogs', 'trainingSummaries',
            #                                                   str(num_hidden) + '_' + str(n_filters)),
            #                                      sess.graph)
            # test_writer = tf.summary.FileWriter(os.path.join(input_path_WHITE,
            #                                                  r'ExperimentLogs', 'testingSummaries',
            #                                                  str(num_hidden) + '_' + str(n_filters)))

            sess.run(tf.global_variables_initializer())

            # # We first get the graph that we used to compute the network
            # g = tf.get_default_graph()
            #
            # # And can inspect everything inside of it
            # pprint.pprint([op.name for op in g.get_operations()])

            n_epochs = 500  # mess with this a bit

            print_hyperparameters(learning_rate, batch_size, n_epochs, n_filters, num_hidden, file_WHITE)

            # split into testing and validation sets
            # valid_examples_WHITE, test_examples_WHITE, valid_labels_WHITE, test_labels_WHITE = model_selection.train_test_split(
            #     validation_examples_WHITE, validation_labels_WHITE, test_size=0.5, random_state=random.randint(1, 1024))
            #
            # valid_examples_BLACK, test_examples_BLACK, valid_labels_BLACK, test_labels_BLACK = model_selection.train_test_split(
            #     validation_examples_BLACK, validation_labels_BLACK, test_size=0.5, random_state=random.randint(1, 1024))
            epoch_i = 0
            # while compute_accuracy(BLACK_testing_examples_partition_k, BLACK_testing_labels_partition_k,  X_BLACK, y_BLACK, accuracy_function_BLACK, None, None, 0)[0] < 0.65 and \
            #         compute_accuracy(WHITE_testing_examples_partition_k, WHITE_testing_labels_partition_k, X_WHITE, y_WHITE, accuracy_function_WHITE, None, None, 0)[0] < 0.65 :
            for epoch_i in range(n_epochs):
                # WHITE
                # # reshuffle training set at each epoch
                # training_examples_WHITE, training_labels_WHITE = shuffle(training_examples_WHITE, training_labels_WHITE,
                #                                                          random_state=random.randint(1, 1024))

                sequence = sess.run(h_layers_BLACK[-1], feed_dict={X_BLACK: training_examples_BLACK[i]})

                sequence_label = training_labels_BLACK[i]
                # split training examples into batches
                batch_size = 1
                training_example_batches_WHITE, training_label_batches_WHITE = utils.batch_split(
                    training_examples_WHITE, training_labels_WHITE,
                    batch_size)

                # # BLACK
                # # reshuffle training set at each epoch
                # training_examples_BLACK, training_labels_BLACK = shuffle(training_examples_BLACK, training_labels_BLACK,
                #                                                          random_state=random.randint(1, 1024))

                # split training examples into batches
                training_example_batches_BLACK, training_label_batches_BLACK = utils.batch_split(
                    training_examples_BLACK, training_labels_BLACK,
                    batch_size)

                # BLACK
                startTime = time.time()  # start timer
                game_sequences = []
                for i in range(0, len(training_example_batches_BLACK)):

                    sequence = sess.run(h_layers_BLACK[-1], feed_dict={X_BLACK: training_example_batches_BLACK[i]})

                    sequence_label = training_label_batches_BLACK[i]

                    train_model(training_example_batches_BLACK[i], training_label_batches_BLACK[i], X_BLACK, y_BLACK,
                                optimizer_BLACK, merged,
                                train_writer,
                                (epoch_i * len(training_example_batches_BLACK)) + i)

                    # show stats at every 1/10th interval of epoch
                    if (i + 1) % (len(training_example_batches_BLACK) // 10) == 0:
                        loss_init = sess.run(cost_BLACK, feed_dict={
                            X_BLACK: training_example_batches_BLACK[i],
                            y_BLACK: training_label_batches_BLACK[i]
                        })

                        accuracy_score = compute_accuracy(valid_examples_BLACK, valid_labels_BLACK, X_BLACK, y_BLACK,
                                                          accuracy_function_BLACK,
                                                          merged,
                                                          test_writer,
                                                          (epoch_i * len(training_example_batches_BLACK)) + i)

                        print("Loss: {}".format(loss_init), end="\n", file=file_BLACK)
                        print("Loss Reduced Mean: {}".format(sess.run(tf.reduce_mean(loss_init))), end="\n", file=file_BLACK)
                        print("Loss Reduced Sum: {}".format(sess.run(tf.reduce_sum(loss_init))), end="\n", file=file_BLACK)
                        print('Interval {interval} of 10 Accuracy: {accuracy_score}'.format(
                            interval=(i + 1) // (len(training_example_batches_BLACK) // 10),
                            accuracy_score=accuracy_score), end="\n", file=file_BLACK)


                        # show accuracy at end of epoch
                accuracy_score = compute_accuracy(valid_examples_BLACK, valid_labels_BLACK, X_BLACK, y_BLACK,
                                                  accuracy_function_BLACK, merged,
                                                  test_writer,
                                                  (epoch_i * len(training_example_batches_BLACK)) + len(
                                                      training_example_batches_BLACK) - 1)
                print('Epoch {epoch_num} Accuracy: {accuracy_score}'.format(
                    epoch_num=epoch_i + 1,
                    accuracy_score=accuracy_score), end="\n", file=file_BLACK)

                # show example of what network is predicting vs the move oracle
                # print_prediction_statistics(valid_examples_BLACK, valid_labels_BLACK, y_pred_BLACK, X_BLACK, file_BLACK)
                print("\nMinutes between epochs: {time}".format(time=(time.time() - startTime) / 60), end="\n",
                      file=file_BLACK)

                # WHITE

                startTime = time.time()  # start timer

                # train model
                for i in range(0, len(training_example_batches_WHITE)):
                    train_model(training_example_batches_WHITE[i], training_label_batches_WHITE[i], X_WHITE, y_WHITE,
                                optimizer_WHITE, merged, train_writer,
                                (epoch_i * len(training_example_batches_WHITE)) + i)

                    # show stats at every 1/10th interval of epoch
                    if (i + 1) % (len(training_example_batches_WHITE) // 10) == 0:
                        loss_init = sess.run(cost_WHITE, feed_dict={
                            X_WHITE: training_example_batches_WHITE[i],
                            y_WHITE: training_label_batches_WHITE[i]
                        })

                        accuracy_score = compute_accuracy(valid_examples_WHITE, valid_labels_WHITE, X_WHITE, y_WHITE,
                                                          accuracy_function_WHITE, merged,
                                                          test_writer,
                                                          (epoch_i * len(training_example_batches_WHITE)) + i)

                        print("Loss: {}".format(loss_init), end="\n", file=file_WHITE)
                        print("Loss Reduced Mean: {}".format(sess.run(tf.reduce_mean(loss_init))), end="\n", file=file_WHITE)
                        print("Loss Reduced Sum: {}".format(sess.run(tf.reduce_sum(loss_init))), end="\n", file=file_WHITE)
                        print('Interval {interval} of 10 Accuracy: {accuracy_score}'.format(
                            interval=(i + 1) // (len(training_example_batches_WHITE) // 10),
                            accuracy_score=accuracy_score), end="\n", file=file_WHITE)


                        # show accuracy at end of epoch
                accuracy_score = compute_accuracy(valid_examples_WHITE, valid_labels_WHITE, X_WHITE, y_WHITE,
                                                  accuracy_function_WHITE, merged, test_writer,
                                                  (epoch_i * len(training_example_batches_WHITE)) + len(
                                                      training_example_batches_WHITE) - 1)
                print('Epoch {epoch_num} Accuracy: {accuracy_score}'.format(
                    epoch_num=epoch_i + 1,
                    accuracy_score=accuracy_score), end="\n", file=file_WHITE)


                # show example of what network is predicting vs the move oracle
                # print_prediction_statistics(valid_examples_WHITE, valid_labels_WHITE,  y_pred_WHITE, X_WHITE, file_WHITE)
                print("\nMinutes between epochs: {time}".format(time=(time.time() - startTime) / 60), end="\n",
                      file=file_WHITE)


                # if compute_accuracy(BLACK_testing_examples_partition_k, BLACK_testing_labels_partition_k,
                #                     X_BLACK, y_BLACK, accuracy_function_BLACK, None, None, 0)[0] < 0.10 or \
                #         compute_accuracy(WHITE_testing_examples_partition_k, WHITE_testing_labels_partition_k,
                #                          X_WHITE, y_WHITE, accuracy_function_WHITE, None, None, 0)[0] < 0.10:
                #     exit(100)

                # epoch_i += 1

                # save_path = saver.save(sess, os.path.join(input_path_BLACK, r'065AccIteration ({})'.format(epoch_i), r'DualWinningNets065Accuracy'))
                # save_path = saver.save(sess, os.path.join(input_path_BLACK, r'065AccIteration ({})'.format(epoch_i),
                #                                           r'1x1DualWinningNets070Accuracy_{}'.format(num_hidden)))     #  r'1x1DualWinningNets070Accuracy_{}'.format(num_hidden)

                print_partition_statistics(test_examples_WHITE, test_labels_WHITE, X_WHITE, y_WHITE, net_type,
                                           accuracy_function_WHITE, file_WHITE)
                print_prediction_statistics(test_examples_WHITE, test_labels_WHITE, y_pred_WHITE, X_WHITE, file_WHITE)




                # Print final test accuracy:
                # WHITE
            # this partition
            print_partition_statistics(test_examples_WHITE, test_labels_WHITE, X_WHITE, y_WHITE, net_type,
                                       accuracy_function_WHITE, file_WHITE)
            print_prediction_statistics(test_examples_WHITE, test_labels_WHITE, y_pred_WHITE, X_WHITE, file_WHITE)


            # save the model now
            # save_path = saver.save(sess, os.path.join(input_path_BLACK, r'model', r'1x1DualWinningNets070Accuracy_{}'.format(num_hidden)))

            sess.close()
    file_WHITE.close()
    file_BLACK.close()





