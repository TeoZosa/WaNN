from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import rnn
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
import numpy as np
import collections
from tools.utils import find_files, batch_split
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
          "\nHe std init"
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

def assign_path(deviceName ='Workstation'):
    if  deviceName == 'MBP2011_':
       path =  r'/Users/teofilozosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2014':
       path = r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2011':
       path = r'/Users/Home/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'Workstation':
        path =r'G:\TruncatedLogs\PythonDatasets\Datastructures\NumpyArrays\{net_type}\{features}\4DArraysHDF5(RxCxF){features}{net_type}AllThirdBlackRNN'.format(features='POE', net_type='PolicyNet')
    else:
        path = ''#todo:error checking
    return path

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

def RNN(X, num_hidden_layers):

    # reshape to [1, n_input]
    std_dev_He = np.sqrt(2 / np.prod(X.get_shape().as_list()[1:]))
    X = tf.reshape(X, [-1, sequence_length* 8*8])


    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    X = tf.split(X, sequence_length, 1)

    # 1-layer LSTM with n_hidden units.

    # rnn_cell = rnn.BasicLSTMCell(n_hidden)
    with tf.variable_scope('RNN', tf.random_normal_initializer(mean=0.0, stddev=std_dev_He)): #tf.random_normal_initializer(mean=0.0, stddev=std_dev_He) #initializer=tf.contrib.layers.xavier_initializer()
        # weights = {
        #     'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
        # }
        # biases = {
        #     'out': tf.Variable(tf.random_normal([num_classes]))
        # }
        weights = tf.get_variable(
            name='weights',
            shape=[num_hidden, num_classes],  # 1 x 64 filter in, 1 class out
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(
            name='biases',
            shape=[num_classes],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))
        GRU_cell_layer = [rnn.GRUCell(num_hidden)]
        # LSTM_cell_layer = [rnn.BasicLSTMCell(num_hidden, forget_bias=1)]
        rnn_cell = rnn.MultiRNNCell(GRU_cell_layer * num_hidden_layers)
        # generate prediction
        outputs, states = rnn.static_rnn(rnn_cell, X, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    # return tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.matmul(outputs[-1], weights) + biases


input_path = assign_path()
device = 'Workstation'
game_stage = input_path[-13:-5]  # ex. 1stThird

# for experiment with states from entire games, test data are totally separate games
training_examples, training_labels = load_examples_and_labels(os.path.join(input_path, r'TrainingDataCNN'))
validation_examples, validation_labels = load_examples_and_labels(
    os.path.join(input_path, r'ValidationDataCNN'))


for num_layers in [i for i in range(3, 4)
                   ]:

    for num_hidden in [
        # 128,
        # 256,
         512,
         # 1024,
        # 2048,
        # 4096,

    ]:
        file = open(os.path.join(input_path,
                                 r'ExperimentLogs',
                                 game_stage + '_CNNGRU_Stock{n_filters}HiddenNeurons{num_hidden}LayersTF_CE__He_weightsPOE.txt'.format(
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

            batch_size = 2048 *(4-num_layers)

            num_classes = 155
            sequence_length = 40
            # number of units in RNN cell


            # RNN output node weights and biases
            # debug_peek1 = tf.reshape(training_examples[0], [-1, n_input])
            # debug_peek = tf.split(debug_peek1, n_input, 1)

            # build graph
            X = tf.placeholder(tf.float32, [None, sequence_length, 8, 8, 1])
            y = tf.placeholder(tf.float32, [None, num_classes])




            output_layer = RNN(X, num_layers)
            y_pred = tf.nn.softmax(output_layer)

            # Loss and optimizer
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

            cost = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y)


            # # way better performance
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
                accuracy_function = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # tf.summary.scalar('accuracy', accuracy_function)

            # save the model
            saver = tf.train.Saver()

            NUM_CORES = multiprocessing.cpu_count()
            sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                                                    intra_op_parallelism_threads=NUM_CORES))

            # tensorboard summaries
            # merged = tf.summary.merge_all()
            # train_writer = tf.summary.FileWriter(os.path.join(input_path,
            #                                                   r'ExperimentLogs', 'trainingSummaries',
            #                                                   str(num_hidden) + '_' + str(n_filters)),
            #                                      sess.graph)
            # test_writer = tf.summary.FileWriter(os.path.join(input_path,
            #                                                  r'ExperimentLogs', 'testingSummaries',
            #                                                  str(num_hidden) + '_' + str(n_filters)))


            sess.run(tf.global_variables_initializer())


            # # We first get the graph that we used to compute the network
            # g = tf.get_default_graph()
            #
            # # And can inspect everything inside of it
            # pprint.pprint([op.name for op in g.get_operations()])

            n_epochs = 10  # mess with this a bit
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
                    # original = training_example_batches[0][1]
                    # npdebug = np.reshape(training_example_batches[0][1], [sequence_length* 8*8])
                    # npdebug_ = np.reshape(training_example_batches[0],[-1, sequence_length, 8*8])
                    # npdebug1 = np.split(npdebug_, sequence_length, 1)
                    # #
                    # debug = tf.reshape(training_example_batches[0][0], [-1, sequence_length*8*8])
                    # debug1 = tf.split(debug, sequence_length, 1)
                    _, acc, loss, onehot_pred = sess.run([optimizer, accuracy_function, cost, y_pred],
                                                        feed_dict={X: training_example_batches[i], y: training_label_batches[i]})

                    # show stats at every 1/10th interval of epoch
                    if (i + 1) % (len(training_example_batches) // 10) == 0:
                        loss = sess.run(cost, feed_dict={
                            X: training_example_batches[i],
                            y: training_label_batches[i]
                        })

                        accuracy_score = sess.run(accuracy_function, feed_dict={X: valid_examples, y:valid_labels})
                        # compute_accuracy(valid_examples, valid_labels, accuracy_function, merged,
                        #                                   test_writer, (epoch_i * len(training_example_batches)) + i)

                        print("Loss: {}".format(loss), end="\n", file=file)
                        print("Loss Reduced Mean: {}".format(sess.run(tf.reduce_mean(loss))), end="\n", file=file)
                        print("Loss Reduced Sum: {}".format(sess.run(tf.reduce_sum(loss))), end="\n", file=file)
                        print('Interval {interval} of 10 Test Accuracy: {accuracy_score}'.format(
                            interval=(i + 1) // (len(training_example_batches) // 10),
                            accuracy_score=accuracy_score), end="\n", file=file)
                        print('Interval {interval} of 10 Train Accuracy: {accuracy_score}'.format(
                            interval=(i + 1) // (len(training_example_batches) // 10),
                            accuracy_score=acc), end="\n", file=file)

                        print("Loss: {}".format(loss))
                        print("Loss Reduced Mean: {}".format(sess.run(tf.reduce_mean(loss))))
                        print("Loss Reduced Sum: {}".format(sess.run(tf.reduce_sum(loss))))
                        print('Interval {interval} of 10 Test Accuracy: {accuracy_score}'.format(
                            interval=(i + 1) // (len(training_example_batches) // 10),
                            accuracy_score=accuracy_score))
                        print('Interval {interval} of 10 Train Accuracy: {accuracy_score}'.format(
                            interval=(i + 1) // (len(training_example_batches) // 10),
                            accuracy_score=acc))


                        # show accuracy at end of epoch
                accuracy_score = sess.run(accuracy_function, feed_dict={X: valid_examples, y: valid_labels})
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
                                                          'RNN{num_hidden}x{num_layers}'.format(num_hidden=num_hidden,
                                                                                                num_layers=num_layers)))

                print('Epoch {epoch_num} Accuracy: {accuracy_score}'.format(
                    epoch_num=epoch_i + 1,
                    accuracy_score=accuracy_score), end="\n", file=file)

                epoch_i +=1

                # show example of what network is predicting vs the move oracle
               # print_prediction_statistics(valid_examples, valid_labels, file)
                print("\nMinutes between epochs: {time}".format(time=(time.time() - startTime) / 60), end="\n",
                      file=file)
            accuracy_score = sess.run(accuracy_function, feed_dict={X: test_examples, y: test_labels})

            print('Final Accuracy: {accuracy_score}'.format( accuracy_score=accuracy_score), end="\n", file=file)
            save_path = saver.save(sess, os.path.join(input_path, r'model', str(num_layers),
                                                      str(num_hidden),'RNN{num_hidden}x{num_layers}'.format(num_hidden=num_hidden, num_layers=num_layers)))
