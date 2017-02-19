
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
    elif deviceName == 'Workstation':#TODO: testing Start-Game Value Net
        path =r'G:\TruncatedLogs\PythonDatasets\Datastructures\NumpyArrays\ValueNet\4DArraysHDF5(RxCxF)POEValueNet50%DataSet'
    else:
        path = ''#todo:error checking
    return path


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
        bias = tf.get_variable(name='bias',
                                 shape=[n_filters_out],
                                 initializer=tf.constant_initializer(0.01))#karpathy: for relu, 0.01 ensures all relus fire in the beginning
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
    layer_in = tf.reshape(layer_in, [-1, 8*8])
    activation = tf.nn.softmax
    n_features = layer_in.get_shape().as_list()[1]
    with tf.variable_scope(name or 'output_layer', reuse=reuse):
        kernel = tf.get_variable(
            name='weights',
            shape=[n_features, 2], # 1 x 64 filter in, 1 class out
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        bias = tf.get_variable(
            name='bias',
            shape=[2],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        unscaled_output = (tf.nn.bias_add(
            name='output',
            value=tf.matmul(layer_in, kernel),
            bias=bias))
        return unscaled_output, kernel

def loss(output_layer, labels):
    with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(outer_layer, labels)
        loss = tf.reduce_mean(losses)

# TODO: excise code to be run in main script.

        #main script

# config = run_config.RunConfig(num_cores=-1)
input_path = assign_path()
device = assign_device(input_path)

#for experiment with states from entire games, test data are totally separate games
training_examples, training_labels = load_examples_and_labels(os.path.join(input_path, r'TrainingData'))
testing_examples_start, testing_labels_start = load_examples_and_labels(os.path.join(input_path, r'TestDataStart')) # 210659 states
testing_examples_mid, testing_labels_mid = load_examples_and_labels(os.path.join(input_path, r'TestDataMid'))
validation_examples, validation_labels = load_examples_and_labels(os.path.join(input_path, r'ValidationData'))
# X, y = load_examples_and_labels(inputPath)

# X[i] is a 3D matrix corresponding to label at y[i]
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
#     test_size=128, random_state=42)#sametrain/test split every time


file = open(os.path.join(input_path,
                         r'ExperimentLogs',
                         input_path[-10:] + 'AdamNumFiltersNumLayersTFCrossEntropy_He_weightsPOE.txt'), 'a')
# file = sys.stdout
print ("# of Testing Examples: {}".format(len(testing_examples_start)), end='\n', file=file)
for num_hidden in [i for i in range(1,10)]:
    for n_filters in [
                        16, 32, 64,
                       128,
                      192]:
        for learning_rate in [
            0.001,
            0.0011, 0.0012, 0.0013,
                              0.0014, 0.0015
        ]:
            reset_default_graph()
            batch_size = 128

            #build graph
            X = tf.placeholder(tf.float32, [None, 8, 8, 4])
            # TODO: consider reshaping for C++ input; could also put it into 3d matrix on the fly, ex. if player == board[i][j], X[n][i][j] = [1, 0, 0, 1]
            y = tf.placeholder(tf.float32, [None,  2])
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
            cost = tf.nn.softmax_cross_entropy_with_logits(logits=outer_layer, labels=y)
            # # alternative implementation
            # cost = tf.reduce_mean(cost) #used in MNIST tensorflow

            #kadenze cross_entropy cost function
            # cost = -tf.reduce_sum(y * tf.log(y_pred + 1e-12))


            #way better performance
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            #SGD used in AlphaGO
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            # # We first get the graph that we used to compute the network
            # g = tf.get_default_graph()
            #
            # # And can inspect everything inside of it
            # pprint.pprint([op.name for op in g.get_operations()])

            n_epochs = 5
            print("\nAdam Optimizer"
                  "\nNum Filters: {num_filters}"
                  "\nNum Hidden Layers: {num_hidden}"
                  "\nLearning Rate: {learning_rate}"
                  "\nBatch Size: {batch_size}"
                  "\n# of Epochs: {n_epochs}".format(
                learning_rate=learning_rate,
                batch_size = batch_size,
                n_epochs=n_epochs,
                num_filters=n_filters,
                num_hidden=num_hidden), end="\n", file=file)
            # X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_train, y_train,
            #                                                                       test_size=128,
            #                                                                       random_state=random.randint(1, 1024))  # keep validation outside

            #for entire dataset experiments, split the testing games into a validation and test split; 105330 & 105329 states each; does randomness matter?
            # testing accuracy of full value net on begin and mid-game examples
            test_set_examples = testing_examples_start
            test_set_labels = testing_labels_start
            validation_set_examples = validation_examples
            validation_set_labels = validation_labels
            # test_set_examples, validation_set_examples, test_set_labels, validation_set_labels = model_selection.train_test_split(testing_examples, testing_labels, test_size=0.5,
            #                                                                                                                       random_state=random.randint(1, 1024))

            for epoch_i in range(n_epochs):

                training_examples, training_labels = shuffle(training_examples, training_labels,
                                                             random_state=random.randint(1, 1024))  # reshuffling of training set at each epoch

                training_example_batches, training_label_batches = utils.batch_split(training_examples, training_labels, batch_size)

                startTime = time.time()  #start timer

                #train model
                for i in range (0, len(training_example_batches)):
                    sess.run(optimizer, feed_dict={
                        X: training_example_batches[i],
                        y: training_label_batches[i]
                    })

                    # show stats at every 1/10th interval of epoch
                    if (i+1)%(len(training_example_batches)//10)==0:
                        loss = sess.run(cost, feed_dict={
                            X: training_example_batches[i],
                            y: training_label_batches[i]
                            })
                        accuracy_score = sess.run(accuracy, feed_dict={
                                       X: validation_set_examples,
                                       y: validation_set_labels
                                       })
                        print("Loss: {}".format(loss), end="\n", file=file)
                        print("Loss Reduced Mean: {}".format(sess.run(tf.reduce_mean(loss))), end="\n", file=file)
                        print("Loss Reduced Sum: {}".format(sess.run(tf.reduce_sum(loss))), end="\n", file=file)
                        print('Interval {interval} of 10 Accuracy: {accuracy}'.format(
                            interval=(i+1)//(len(training_example_batches) // 10),
                            accuracy=accuracy_score), end="\n", file=file)


                #show accuracy at end of epoch
                print ('Epoch {epoch_num} Accuracy: {accuracy_score}'.format(
                    epoch_num=epoch_i+1,
                    accuracy_score=sess.run(accuracy,
                                   feed_dict={
                                       X: validation_set_examples,
                                       y: validation_set_labels
                                   })), end="\n", file=file)

                #show example of what network is predicting vs the move oracle
                example = random.randrange(0, len(validation_set_examples))
                y_pred_vector =sess.run(y_pred, feed_dict={X:[validation_set_examples[example]]})
                print("Sample Predicted Probabilities = "
                      "\n{y_pred}"
                      "\nPredicted: {predicted_outcome}"
                      "\nActual:  {actual_outcome}".format(
                        y_pred=y_pred_vector,
                        predicted_outcome=utils.win_lookup(np.argmax(y_pred_vector)),
                        actual_outcome=utils.win_lookup(np.argmax(validation_set_labels[example]))),
                        end="\n", file=file)

                print("\nMinutes between epochs: {time}".format(time=(time.time() - startTime) / 60), end="\n", file=file)

            # Print final test accuracy:
            print("Final Test Accuracy (start-states): {}".format(sess.run(accuracy,
                           feed_dict={
                               X: test_set_examples,
                               y: test_set_labels
                           })), end="\n", file=file)
            print("Final Test Accuracy (middle-states): {}".format(sess.run(accuracy,
                                                                           feed_dict={
                                                                               X: testing_examples_mid,
                                                                               y: testing_labels_mid
                                                                           })), end="\n", file=file)
            sess.close()
file.close()



