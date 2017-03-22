
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import  pprint
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.ops import nn
from tensorflow.python.framework import dtypes
from sklearn import model_selection, metrics, grid_search, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.learning_curve import learning_curve, validation_curve
from tensorflow.contrib.learn.python.learn.estimators import run_config
import time
from scipy import stats ,integrate
import seaborn as sns
from tools import utils
import h5py
import os
from PIL import Image
import random
from tensorflow.python.framework.ops import reset_default_graph
import sys
import pickle

sns.set(color_codes=True)



def WriteToDisk(path, estimator):
    write_path = path + r'Models/'
    estimator.save(write_path)

def image_stuff():
    #Train/validation/test split or reducing number of training examples
    sampleInput = X[30]
    who_dict = {
                1.0: 127,
                0.0: 0,
                }
    who_dict1 = {
                1.0: 255,
                0.0: 0,
                }
    mean_player = np.mean(X, axis=0)
    std_ = np.std(X, axis=0)
    var = np.mean(std_, axis=2)


    mode_args = ['CMYK', 'RGB', 'YCbCr']
    for mode_arg in mode_args:
        plot.figure()
        plot.imshow(Image.fromarray(mean_player[:,:,0], mode=mode_arg))
        plot.figure()
        plot.imshow(Image.fromarray(std_[:, :, 0], mode=mode_arg))

        plot.figure()
        plot.imshow(Image.fromarray(mean_player[:,:,1], mode=mode_arg))
        plot.figure()
        plot.imshow(Image.fromarray(std_[:, :, 1], mode=mode_arg))

        plot.figure()
        plot.imshow(Image.fromarray(mean_player[:,:,2], mode=mode_arg))
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


    print(fr_p, '\n', fr_o, '\n', fr_e )
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
        device = 'AWS'#todo: configure AWS path
    return device


def assign_path(deviceName ='Workstation'):
    if  deviceName == 'MBP2011_':
       path =  r'/Users/teofilozosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2014':
       path = r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2011':
       path = r'/Users/Home/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'Workstation':
        path =r'G:\TruncatedLogs\PythonDatasets\Datastructures\NumpyArrays\{net_type}\{features}\4DArraysHDF5(RxCxF){features}{net_type}AllThird'.format(features='POE', net_type='PolicyNet')
    else:
        path = ''#todo:error checking
    return path

def get_net_type(type):
    type_dict = {'1stThird': 'Start',
                 '2ndThird': 'Mid',
                 '3rdThird': 'End',
                 'AllThird': 'Full'}
    return type_dict[type]

def load_examples_and_labels(path):
    files = [file_name for file_name in utils.find_files(path, "*.hdf5")]
    examples, labels = ([] for i in range (2))
    for file in files:  # add training examples and corresponding labels from each file to associated arrays
        training_data = h5py.File(file, 'r')
        examples.extend(training_data['X'][:])
        labels.extend(training_data['y'][:])
        training_data.close()
    return np.array(examples, dtype=np.float32), np.array(labels, dtype=np.float32)  # b x r x c x f & label



def hidden_layer_init(prev_layer, n_filters_in, n_filters_out, filter_size, name=None, activation=tf.nn.relu, reuse=None):
     # of filters in each layer ranged from 64-192
    std_dev_He = np.sqrt(2 / np.prod(prev_layer.get_shape().as_list()[1:]))
    with tf.variable_scope(name or 'hidden_layer', reuse=reuse):
        kernel = tf.get_variable(name='weights',
                                   shape=[filter_size, filter_size,   # h x w
                                          n_filters_in,
                                          n_filters_out],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=std_dev_He)#mean, std?
                                   )
        variable_summaries(kernel)

        bias = tf.get_variable(name='bias',
                                 shape=[n_filters_out],
                                 initializer=tf.constant_initializer(0.01))#karpathy: for relu, 0.01 ensures all relus fire in the beginning

        variable_summaries(bias)

        hidden_layer = activation(
            tf.nn.bias_add(
                tf.nn.conv2d(input=prev_layer,
                             filter=kernel,
                             strides=[1, 1, 1, 1],
                             padding='SAME'),
                bias
            )
        )
        tf.summary.histogram('activations', hidden_layer)
        return hidden_layer

def output_layer_init(layer_in, name='output_layer', reuse=None):
    layer_in = tf.reshape(layer_in, [-1, 8*8])
    activation = tf.nn.softmax
    n_features = layer_in.get_shape().as_list()[1]
    with tf.variable_scope(name or 'output_layer', reuse=reuse):
        kernel = tf.get_variable(
            name='weights',
            shape=[n_features, 155], # 1 x 64 filter in, 1 class out
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        variable_summaries(kernel)

        bias = tf.get_variable(
            name='bias',
            shape=[155],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))
        variable_summaries(bias)

        unscaled_output = (tf.nn.bias_add(
            name='output',
            value=tf.matmul(layer_in, kernel),
            bias=bias))
        tf.summary.histogram('unscaled_output', unscaled_output)
        return unscaled_output, kernel

def loss(output_layer, labels):
    with tf.variable_scope("cross_entropy"):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=labels)
        # loss = tf.reduce_mean(losses)
        tf.summary.histogram('cross_entropy', losses)
    return losses

def compute_accuracy(examples, labels, accuracy_function, merged, test_writer, batch):
    summary, accuracy_score = sess.run([merged, accuracy_function],feed_dict={X: examples, y: labels})
    test_writer.add_summary(summary, batch)
    return  accuracy_score

def train_model(examples, labels, optimizer, merged, train_writer, batch):
    summary, _ = sess.run([merged, optimizer], feed_dict={
        X: examples,
        y: labels
    })
    train_writer.add_summary(summary, batch)

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

def print_partition_accuracy_statistics(examples, labels, partition, file_to_write):
    with tf.name_scope(partition+r'-states'):
        print("Final Test Accuracy ({partition}-states): {accuracy}".format(
            partition=partition,
            accuracy=compute_accuracy(examples, labels, accuracy_function, merged, test_writer, 0 )),
            end="\n",
            file=file_to_write)

def print_prediction_statistics(examples, labels, file_to_write):
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

    correct_prediction_position = [0] * 155
    not_in_top_10 = 0
    labels_predictions = sess.run(y_pred, feed_dict={X: examples})
    with tf.name_scope('partition_probability_difference'):
        for example_num in range(0, len(labels_predictions)):
            correct_move_index = np.argmax(labels[example_num])
            predicted_move_index = np.argmax(labels_predictions[example_num])
            top_n_indexes = sorted(range(len(labels_predictions[example_num])),
                                   key=lambda i: labels_predictions[example_num][i], reverse=True)[
                            :
                            # num_top_moves
                            ]
            if (correct_move_index in top_n_indexes):
                rank_in_prediction = top_n_indexes.index(correct_move_index)  # 0 indexed
                correct_prediction_position[rank_in_prediction] += 1
                correct_move_predicted_prob = labels_predictions[example_num][correct_move_index] * 100

                if correct_move_index != predicted_move_index:
                    print("Incorrect prediction. Correct move was ranked {}".format(rank_in_prediction + 1),
                          file=file_to_write)
                    top_move_predicted_prob = labels_predictions[example_num][predicted_move_index] * 100
                    difference = tf.add(top_move_predicted_prob, - correct_move_predicted_prob)
                    print("Predicted move probability = %{pred_prob}. \n"
                          "Correct move probability = %{correct_prob}\n"
                          "Difference = %{prob_diff}\n".format(pred_prob=top_move_predicted_prob,
                                                               correct_prob=correct_move_predicted_prob,
                                                               prob_diff=difference),
                          file=file_to_write)
                    in_top_n = True
                    tf.summary.scalar('incorrect prediction: probability difference between top move and correct',difference)
                    difference = tf.add(100, - correct_move_predicted_prob)
                    tf.summary.scalar('incorrect prediction: predicted probability difference of correct move',difference)


                else:
                    difference = tf.add(100, - correct_move_predicted_prob)
                    tf.summary.scalar('correct prediction: probability difference',difference)


            else:
                # rank_in_prediction = in_top_n = False
                not_in_top_10 += 1

    total_predictions = sum(correct_prediction_position) + not_in_top_10
    percent_in_top = 0
    for i in range(0, len(correct_prediction_position)):
        rank_percent = (correct_prediction_position[i] * 100) / total_predictions
        print("Correct move was in predicted rank {num} slot = %{percent}".format(num=i + 1, percent=rank_percent),
              file=file_to_write)
        percent_in_top += rank_percent
        print("Percent in top {num} predictions = %{percent}\n".format(num=i + 1, percent=percent_in_top),
              file=file_to_write)
    print("Percentage of time not in predictions = {}".format((not_in_top_10 * 100) / total_predictions),
          file=file_to_write)


def print_partition_statistics(examples, labels, partition, file):
    print_partition_accuracy_statistics(examples, labels, partition, file)
    # print_prediction_statistics(examples, labels, file)

# TODO: excise code to be run in main script.

        #main script

# config = run_config.RunConfig(num_cores=-1)
input_path = assign_path()
device = assign_device(input_path)
game_stage = input_path[-8:]#ex. 1stThird
net_type = get_net_type(game_stage)

#for experiment with states from entire games, test data are totally separate games
training_examples, training_labels = load_examples_and_labels(os.path.join(input_path, r'TrainingData'))
validation_examples, validation_labels = load_examples_and_labels(os.path.join(input_path, r'ValidationData'))
if (net_type == 'Start'):
    partition_i = 'Mid'
    partition_j = 'End'
    testing_examples_partition_i, testing_labels_partition_i = load_examples_and_labels(os.path.join(input_path, r'TestDataMid'))  # 210659 states
    testing_examples_partition_j, testing_labels_partition_j = load_examples_and_labels(os.path.join(input_path, r'TestDataEnd'))
elif(net_type == 'Mid'):
    partition_i = 'Start'
    partition_j = 'End'
    testing_examples_partition_i, testing_labels_partition_i = load_examples_and_labels(os.path.join(input_path, r'TestDataStart'))  # 210659 states
    testing_examples_partition_j, testing_labels_partition_j = load_examples_and_labels(os.path.join(input_path, r'TestDataEnd'))
elif(net_type == 'End'):
    partition_i = 'Start'
    partition_j = 'Mid'
    testing_examples_partition_i, testing_labels_partition_i = load_examples_and_labels(os.path.join(input_path, r'TestDataStart'))  # 210659 states
    testing_examples_partition_j, testing_labels_partition_j = load_examples_and_labels(os.path.join(input_path, r'TestDataMid'))
else:
    partition_i = 'Start'
    partition_j = 'Mid'
    partition_k = 'End'
    testing_examples_partition_i, testing_labels_partition_i = load_examples_and_labels(os.path.join(input_path, r'TestDataStart')) # 210659 states
    testing_examples_partition_j, testing_labels_partition_j = load_examples_and_labels(os.path.join(input_path, r'TestDataMid'))
    testing_examples_partition_k, testing_labels_partition_k = load_examples_and_labels(os.path.join(input_path, r'TestDataEnd'))


# file = open(os.path.join(input_path,
#                          r'ExperimentLogs',
#                          game_stage + '03222017192Filters4and7LayersTF_CE__He_weightsPOE.txt'), 'a')
# file = sys.stdout
# print ("# of Testing Examples: {}".format(len(testing_examples_partition_i)), end='\n', file=file)

for num_hidden in [i for i in [4, 7]
                   # range(1,10)
                   ]:
    file = open(os.path.join(input_path,
                             r'ExperimentLogs',
                             game_stage + '03222017192Filters{}LayersTF_CE__He_weightsPOE.txt'.format(num_hidden)), 'a')
    print("# of Testing Examples: {}".format(len(testing_examples_partition_i)), end='\n', file=file)
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

            #build graph
            X = tf.placeholder(tf.float32, [None, 8, 8, 4])
            # TODO: consider reshaping for C++ input; could also put it into 3d matrix on the fly, ex. if player == board[i][j], X[n][i][j] = [1, 0, 0, 1]
            y = tf.placeholder(tf.float32, [None,  155])
            filter_size = 3 #AlphaGo used 5x5 followed by 3x3, but Go is 19x19 whereas breakthrough is 8x8 => 3x3 filters seems reasonable

            #TODO: consider doing a grid search type experiment where n_filters = rand_val in [2**i for i in range(0,8)]
            #TODO: dropout? regularizers? different combinations of weight initialization, cost func, num_hidden, etc.
            #Yoshua Bengio: "Because of early stopping and possible regularizers, it is mostly important to choose n sub h large enough.
            # Larger than optimal values typically do not hurt generalization performance much, but of course they require proportionally more computation..."
            # "...same size for all layers worked generally better or the same as..."
            # num_hidden = 11
            n_filters_out = [n_filters]*num_hidden + [1] #  " # of filters in each layer ranged from 64-192; layer prior to softmax was # filters = # num_softmaxes
            n_layers = len(n_filters_out)

            #input layer
            h_layers = [hidden_layer_init(X, X.get_shape()[-1],  # n_filters in == n_feature_planes
                                          n_filters_out[0], filter_size, name='hidden_layer/1', reuse=None)]
            #hidden layers
            for i in range(0, n_layers-1):
                h_layers.append(hidden_layer_init(h_layers[i], n_filters_out[i], n_filters_out[i+1], filter_size, name='hidden_layer/{num}'.format(num=i+2), reuse=None))

            #  output layer = softmax. in paper, also convolutional, but 19x19 softmax for player move.
            outer_layer, _ = output_layer_init(h_layers[-1], reuse=None)
            #TODO: if making 2 filters, 1 for each player color softmax, have a check that dynamically makes y_pred correspond to the right filter
            y_pred = tf.nn.softmax(outer_layer)

            #tf's internal softmax; else, put softmax back in output layer
            # cost = tf.nn.softmax_cross_entropy_with_logits(logits=outer_layer, labels=y)
            cost = loss(outer_layer, y)

            # # alternative implementation
            # cost = tf.reduce_mean(cost) #used in MNIST tensorflow

            #kadenze cross_entropy cost function
            # cost = -tf.reduce_sum(y * tf.log(y_pred + 1e-12))


            #way better performance
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            #SGD used in AlphaGO
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
                accuracy_function = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            tf.summary.scalar('accuracy', accuracy_function)



            #save the model
            saver = tf.train.Saver()

            sess = tf.Session()

            #tensorboard summaries
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(input_path,
                                                              r'ExperimentLogs', 'trainingSummaries'),
                                                 sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(input_path,
                                                             r'ExperimentLogs', 'testingSummaries'))

            sess.run(tf.global_variables_initializer())


            # # We first get the graph that we used to compute the network
            # g = tf.get_default_graph()
            #
            # # And can inspect everything inside of it
            # pprint.pprint([op.name for op in g.get_operations()])

            n_epochs = 5  # mess with this a bit

            print_hyperparameters(learning_rate, batch_size, n_epochs, n_filters, num_hidden, file)

            #split into testing and validation sets
            valid_examples, test_examples, valid_labels, test_labels = model_selection.train_test_split(
                validation_examples, validation_labels, test_size=0.5, random_state=random.randint(1, 1024))

            for epoch_i in range(n_epochs):

                # reshuffle training set at each epoch
                training_examples, training_labels = shuffle(training_examples, training_labels,
                                                             random_state=random.randint(1, 1024))

                #split training examples into batches
                training_example_batches, training_label_batches = utils.batch_split(training_examples, training_labels, batch_size)
                

                startTime = time.time()  #start timer

                #train model
                for i in range(0, len(training_example_batches)):
                    train_model(training_example_batches[i], training_label_batches[i], optimizer, merged, train_writer, (epoch_i*len(training_example_batches))+i)

                    # show stats at every 1/10th interval of epoch
                    if (i+1)%(len(training_example_batches)//10)==0:
                        loss = sess.run(cost, feed_dict={
                            X: training_example_batches[i],
                            y: training_label_batches[i]
                        })

                        accuracy_score = compute_accuracy(valid_examples, valid_labels, accuracy_function, merged, test_writer, (epoch_i*len(training_example_batches))+i)

                        print("Loss: {}".format(loss), end="\n", file=file)
                        print("Loss Reduced Mean: {}".format(sess.run(tf.reduce_mean(loss))), end="\n", file=file)
                        print("Loss Reduced Sum: {}".format(sess.run(tf.reduce_sum(loss))), end="\n", file=file)
                        print('Interval {interval} of 10 Accuracy: {accuracy_score}'.format(
                            interval=(i+1)//(len(training_example_batches) // 10),
                            accuracy_score=accuracy_score), end="\n", file=file)


#show accuracy at end of epoch
                accuracy_score = compute_accuracy(valid_examples, valid_labels, accuracy_function, merged, test_writer, (epoch_i*len(training_example_batches))+ len(training_example_batches)-1)
                print ('Epoch {epoch_num} Accuracy: {accuracy_score}'.format(
                    epoch_num=epoch_i+1,
                    accuracy_score=accuracy_score), end="\n", file=file)

                #show example of what network is predicting vs the move oracle
                print_prediction_statistics(valid_examples, valid_labels, file)
                print("\nMinutes between epochs: {time}".format(time=(time.time() - startTime) / 60), end="\n", file=file)

            # Print final test accuracy:
            
            #this partition
            print_partition_statistics(test_examples, test_labels, net_type, file)
            print_prediction_statistics(test_examples, test_labels, file)

            #partition i
            print_partition_statistics(testing_examples_partition_i, testing_labels_partition_i, partition_i, file)
            print_prediction_statistics(testing_examples_partition_i, testing_labels_partition_i, file)

            
            #partition j
            print_partition_statistics(testing_examples_partition_j, testing_labels_partition_j, partition_j, file)
            print_prediction_statistics(testing_examples_partition_j, testing_labels_partition_j, file)


            #partition k (only for full policy net)
            if (net_type == 'Full'):
                print_partition_statistics(testing_examples_partition_k, testing_labels_partition_k, partition_k, file)
                print_prediction_statistics(testing_examples_partition_k, testing_labels_partition_k, file)


            #save the model now
            save_path = saver.save(sess, os.path.join(input_path, r'model', num_hidden))

            sess.close()
file.close()



