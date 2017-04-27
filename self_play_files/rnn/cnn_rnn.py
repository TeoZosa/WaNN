from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import rnn
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
import numpy as np
import collections
from tools.utils import find_files, batch_split, move_lookup_by_index

from Breakthrough_Player.policy_net_utils import instantiate_session_both_RNN
import h5py
import os


from sklearn import model_selection
from sklearn.utils import shuffle
import time

import random
import sys
import multiprocessing
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
        path = r'G:\TruncatedLogs\PythonDatasets\Datastructures\NumpyArrays\{net_type}\{features}\4DArraysHDF5(RxCxF){features}{net_type}AllThirdWhite'.format(
            features='POE', net_type='PolicyNet')
    elif deviceName == 'Workstation_Black':
        path = r'G:\TruncatedLogs\PythonDatasets\Datastructures\NumpyArrays\{net_type}\{features}\4DArraysHDF5(RxCxF){features}{net_type}AllThirdBlack'.format(
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
    files = [file_name for file_name in find_files(path, "*.hdf5")]
    examples, labels = ([] for i in range (2))
    for file in files:  # add training examples and corresponding labels from each file to associated arrays
        training_data = h5py.File(file, 'r')
        examples.extend(training_data['X'][:])
        labels.extend(training_data['y'][:])
        training_data.close()
    return np.array(examples, dtype=np.float32), np.array(labels, dtype=np.float32)  # b x r x c x f & label


def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def RNN(layer_in, num_hidden_layers, num_hidden_units):
    n_features = layer_in.get_shape().as_list()[1]
    num_inputs_in = 155
    num_classes = 155
    # reshape to [1, n_input]
    layer_in = tf.reshape(layer_in, [-1, n_features])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    layer_in = tf.split(layer_in, n_features, 1)

    # 1-layer LSTM with n_hidden units.

    # rnn_cell = rnn.BasicLSTMCell(num_hidden)

    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden_units)] * num_hidden_layers)


    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, layer_in, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden_units, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
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
    top_n_white_moves = list(map(lambda index: move_lookup_by_index(index, 'White'), top_n_indexes))
    top_n_black_moves = list(map(lambda index: move_lookup_by_index(index, 'Black'), top_n_indexes))
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
        y_pred_white=move_lookup_by_index(predicted_move_index, 'White'),
        y_pred_black=move_lookup_by_index(predicted_move_index, 'Black'),
        y_act_white=move_lookup_by_index(correct_move_index, 'White'),
        y_act_black=move_lookup_by_index(correct_move_index, 'Black')),
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


input_path = assign_path()
device = 'Workstation_Black'
game_stage = input_path[-13:-5]  # ex. 1stThird

# for experiment with states from entire games, test data are totally separate games
training_examples, training_labels = load_examples_and_labels(os.path.join(input_path, r'TrainingData'))
validation_examples, validation_labels = load_examples_and_labels(
    os.path.join(input_path, r'ValidationData'))

input_path_WHITE = assign_path()
input_path_BLACK = assign_path("Workstation_Black")
device = assign_device(input_path_WHITE)
game_stage = input_path_WHITE[-13:-5]  # ex. 1stThird
net_type = get_net_type(game_stage)

# for experiment with states from entire games, test data are totally separate games
training_examples_WHITE, training_labels_WHITE = load_examples_and_labels(os.path.join(input_path_WHITE, r'TrainingData'))
validation_examples_WHITE, validation_labels_WHITE = load_examples_and_labels(os.path.join(input_path_WHITE, r'ValidationData'))

training_examples_BLACK, training_labels_BLACK = load_examples_and_labels(os.path.join(input_path_BLACK, r'TrainingData'))
validation_examples_BLACK, validation_labels_BLACK = load_examples_and_labels(os.path.join(input_path_BLACK, r'ValidationData'))


if (net_type == 'Start'):
    partition_i = 'Mid'
    partition_j = 'End'
    WHITE_testing_examples_partition_i, WHITE_testing_labels_partition_i = load_examples_and_labels(
        os.path.join(input_path_WHITE, r'TestDataMid'))  # 210659 states
    WHITE_testing_examples_partition_j, WHITE_testing_labels_partition_j = load_examples_and_labels(
        os.path.join(input_path_WHITE, r'TestDataEnd'))

    BLACK_testing_examples_partition_i, BLACK_testing_labels_partition_i = load_examples_and_labels(
        os.path.join(input_path_BLACK, r'TestDataMid'))  # 210659 states
    BLACK_testing_examples_partition_j, BLACK_testing_labels_partition_j = load_examples_and_labels(
        os.path.join(input_path_BLACK, r'TestDataEnd'))
elif (net_type == 'Mid'):
    partition_i = 'Start'
    partition_j = 'End'
    WHITE_testing_examples_partition_i, WHITE_testing_labels_partition_i = load_examples_and_labels(
        os.path.join(input_path_WHITE, r'TestDataStart'))  # 210659 states
    WHITE_testing_examples_partition_j, WHITE_testing_labels_partition_j = load_examples_and_labels(
        os.path.join(input_path_WHITE, r'TestDataEnd'))

    BLACK_testing_examples_partition_i, BLACK_testing_labels_partition_i = load_examples_and_labels(
        os.path.join(input_path_BLACK, r'TestDataStart'))  # 210659 states
    BLACK_testing_examples_partition_j, BLACK_testing_labels_partition_j = load_examples_and_labels(
        os.path.join(input_path_BLACK, r'TestDataEnd'))
elif (net_type == 'End'):
    partition_i = 'Start'
    partition_j = 'Mid'
    WHITE_testing_examples_partition_i, WHITE_testing_labels_partition_i = load_examples_and_labels(
        os.path.join(input_path_WHITE, r'TestDataStart'))  # 210659 states
    WHITE_testing_examples_partition_j, WHITE_testing_labels_partition_j = load_examples_and_labels(
        os.path.join(input_path_WHITE, r'TestDataMid'))

    BLACK_testing_examples_partition_i, BLACK_testing_labels_partition_i = load_examples_and_labels(
        os.path.join(input_path_BLACK, r'TestDataStart'))  # 210659 states
    BLACK_testing_examples_partition_j, BLACK_testing_labels_partition_j = load_examples_and_labels(
        os.path.join(input_path_BLACK, r'TestDataMid'))
else:
    partition_i = 'Start'
    partition_j = 'Mid'
    partition_k = 'End'
    WHITE_testing_examples_partition_i, WHITE_testing_labels_partition_i = load_examples_and_labels(
        os.path.join(input_path_WHITE, r'TestDataStart'))  # 210659 states
    WHITE_testing_examples_partition_j, WHITE_testing_labels_partition_j = load_examples_and_labels(
        os.path.join(input_path_WHITE, r'TestDataMid'))
    WHITE_testing_examples_partition_k, WHITE_testing_labels_partition_k = load_examples_and_labels(
        os.path.join(input_path_WHITE, r'TestDataEnd'))

    BLACK_testing_examples_partition_i, BLACK_testing_labels_partition_i = load_examples_and_labels(
        os.path.join(input_path_BLACK, r'TestDataStart'))  # 210659 states
    BLACK_testing_examples_partition_j, BLACK_testing_labels_partition_j = load_examples_and_labels(
        os.path.join(input_path_BLACK, r'TestDataMid'))
    BLACK_testing_examples_partition_k, BLACK_testing_labels_partition_k = load_examples_and_labels(
        os.path.join(input_path_BLACK, r'TestDataEnd'))

for num_layers in [i for i in range(1, 10)
                   ]:

    for num_hidden in [
        # 128,
        # 256,
         512,
        #  1024,
        # 2048,
        # 4096,

    ]:
        file = open(os.path.join(input_path,
                                 r'ExperimentLogs',
                                 game_stage + '{n_filters}HiddenNeurons{num_hidden}LayersTF_CE__He_weightsPOE.txt'.format(
                                     n_filters=num_hidden, num_hidden=num_layers)),
                    'a')
        print("# of Testing Examples: {}".format(len(validation_examples)), end='\n', file=file)
        for learning_rate in [
            0.001,
            # 0.0011,
            # 0.0012, 0.0013,
            #                   0.0014, 0.0015
        ]:
            reset_default_graph()
            batch_size = 128

            num_classes = 155
            sequence_length = 40
            # number of units in RNN cell


            # RNN output node weights and biases
            # debug_peek1 = tf.reshape(training_examples[0], [-1, n_input])
            # debug_peek = tf.split(debug_peek1, n_input, 1)

            # build graph
            # X = tf.placeholder(tf.float32, [None, sequence_length])
            y = tf.placeholder(tf.float32, [None, num_classes])


            sess, CNN_output_WHITE, input_WHITE, CNN_output_BLACK, input_BLACK = instantiate_session_both_RNN()
            
            output_layer_BLACK = RNN(CNN_output_BLACK, num_layers, num_hidden)
            y_pred_BLACK = tf.nn.softmax(output_layer_BLACK)
            
            output_layer_WHITE = RNN(CNN_output_WHITE, num_layers, num_hidden)
            y_pred_WHITE = tf.nn.softmax(output_layer_WHITE)

            #40 zero elements => 1 move. append. can't batch this?
            # Loss and optimizer
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))

            cost = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer_BLACK, labels=y)


            # # way better performance
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(y_pred_BLACK, 1), tf.argmax(y, 1))
                accuracy_function = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # tf.summary.scalar('accuracy', accuracy_function)

            # save the model
            saver = tf.train.Saver()

            # NUM_CORES = multiprocessing.cpu_count()
            # sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
            #                                         intra_op_parallelism_threads=NUM_CORES))

            # tensorboard summaries
            # merged = tf.summary.merge_all()
            # train_writer = tf.summary.FileWriter(os.path.join(input_path,
            #                                                   r'ExperimentLogs', 'trainingSummaries',
            #                                                   str(num_hidden) + '_' + str(n_filters)),
            #                                      sess.graph)
            # test_writer = tf.summary.FileWriter(os.path.join(input_path,
            #                                                  r'ExperimentLogs', 'testingSummaries',
            #                                                  str(num_hidden) + '_' + str(n_filters)))


            # sess.run(tf.global_variables_initializer())


            # # We first get the graph that we used to compute the network
            # g = tf.get_default_graph()
            #
            # # And can inspect everything inside of it
            # pprint.pprint([op.name for op in g.get_operations()])

            n_epochs = 100  # mess with this a bit
            done = False

            print_hyperparameters(learning_rate, batch_size, n_epochs, num_hidden, num_layers, file)

            # split into testing and validation sets
            valid_examples, test_examples, valid_labels, test_labels = model_selection.train_test_split(
                validation_examples, validation_labels, test_size=0.5, random_state=random.randint(1, 1024))
            epoch_acc = [-1]*5
            epoch_i = 0


            while epoch_i <n_epochs and not done:




                # reshuffle training set at each epoch
                training_examples, training_labels = shuffle(training_examples, training_labels,
                                                             random_state=random.randint(1, 1024))

                # split training examples into batches
                training_example_batches, training_label_batches = batch_split(training_examples, training_labels,
                                                                                     batch_size)

                startTime = time.time()  # start timer

                # train model
                for i in range(0, len(training_example_batches)):
                    _, acc, loss, onehot_pred = sess.run([optimizer, accuracy_function, cost, y_pred_BLACK],
                                                         feed_dict={input_BLACK: training_example_batches[i], y: training_label_batches[i]})

                    # show stats at every 1/10th interval of epoch
                    if (i + 1) % (len(training_example_batches) // 10) == 0:
                        loss = sess.run(cost, feed_dict={
                            input_BLACK: training_example_batches[i],
                            y: training_label_batches[i]
                        })

                        accuracy_score = sess.run(accuracy_function, feed_dict={input_BLACK: valid_examples, y:valid_labels})
                        # compute_accuracy(valid_examples, valid_labels, accuracy_function, merged,
                        #                                   test_writer, (epoch_i * len(training_example_batches)) + i)

                        print("Loss: {}".format(loss), end="\n", file=file)
                        print("Loss Reduced Mean: {}".format(sess.run(tf.reduce_mean(loss))), end="\n", file=file)
                        print("Loss Reduced Sum: {}".format(sess.run(tf.reduce_sum(loss))), end="\n", file=file)
                        print('Interval {interval} of 10 Accuracy: {accuracy_score}'.format(
                            interval=(i + 1) // (len(training_example_batches) // 10),
                            accuracy_score=accuracy_score), end="\n", file=file)
                        print("Loss: {}".format(loss))
                        print("Loss Reduced Mean: {}".format(sess.run(tf.reduce_mean(loss))))
                        print("Loss Reduced Sum: {}".format(sess.run(tf.reduce_sum(loss))))
                        print('Interval {interval} of 10 Accuracy: {accuracy_score}'.format(
                            interval=(i + 1) // (len(training_example_batches) // 10),
                            accuracy_score=accuracy_score))


                        # show accuracy at end of epoch
                accuracy_score = sess.run(accuracy_function, feed_dict={input_BLACK: valid_examples, y: valid_labels})
                epoch_acc[epoch_i%5] = accuracy_score
                if epoch_i>4 and epoch_i%5 ==4:
                    first_epoch_acc = epoch_acc[0]
                    last_epoch_acc = epoch_acc[4]
                    if last_epoch_acc - first_epoch_acc > .01:  # if accuracy went up by at least 1 percent in the last 5 epochs, keep training.
                        done = False
                    else:
                        done = True
                # done = False
                    # for epoch_counter in range(1, len(epoch_acc)):
                    #     if abs(last_epoch_acc-epoch_acc[epoch_counter]) >.01: #if accuracy changed by at least a percent, keep training.
                    #         done = False
                    #     last_epoch_acc = epoch_acc[epoch_counter]

                save_path = saver.save(sess, os.path.join(input_path, r'model', str(num_layers),
                                                          str(num_hidden),
                                                          'CNNRNN{num_hidden}x{num_layers}'.format(num_hidden=num_hidden,
                                                                                                num_layers=num_layers)))

                print('Epoch {epoch_num} Accuracy: {accuracy_score}'.format(
                    epoch_num=epoch_i + 1,
                    accuracy_score=accuracy_score), end="\n", file=file)

                epoch_i +=1

                # show example of what network is predicting vs the move oracle
               # print_prediction_statistics(valid_examples, valid_labels, file)
                print("\nMinutes between epochs: {time}".format(time=(time.time() - startTime) / 60), end="\n",
                      file=file)
            accuracy_score = sess.run(accuracy_function, feed_dict={input_BLACK: test_examples, y: test_labels})

            print('Final Accuracy: {accuracy_score}'.format( accuracy_score=accuracy_score), end="\n", file=file)
            save_path = saver.save(sess, os.path.join(input_path, r'model', str(num_layers),
                                                      str(num_hidden),'CNNRNN{num_hidden}x{num_layers}'.format(num_hidden=num_hidden, num_layers=num_layers)))
