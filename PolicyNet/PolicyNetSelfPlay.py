
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
from Tools import utils
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

def K_FoldValidation(estimator, XMatrix, yVector, numFolds):
    numTrainingExamples = len(XMatrix)
    K = numFolds
    if K < 2:
        print("Error, K must be greater than or equal to 2")
        exit(-10)
    elif K > numTrainingExamples:
        print("Error, K must be less than or equal to the number of training examples")
        exit(-11)
    K_folds = model_selection.KFold(numTrainingExamples, K)

    for k, (train_index, test_index) in enumerate(K_folds):
        X_train, X_test = XMatrix[train_index], XMatrix[test_index]
        y_train, y_test = yVector[train_index], yVector[test_index]
        # Fit
        estimator.fit(X_train, y_train, logdir='')

        # Predict and score
        score = metrics.mean_squared_error(estimator.predict(X_test), y_test)
        print('Iteration {0:f} MSE: {1:f}'.format(k+1, score))

def StratifiedK_FoldValidation(XMatrix, yVector, learningModel, numFolds):
    num_cores = -1#use all cores
    scores = model_selection.cross_val_score(estimator=learningModel,
                                              X=XMatrix, y=yVector, cv = numFolds, n_jobs=num_cores)
    print('CV accuracy scores: {score}'.format(score = scores))


def GenerateLearningCurve(estimator, X, y):
    # generate learning curve data
    #todo: edit parameters
    num_resources = -1 #engage all available CPUs and GPUs
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X, y=y, train_sizes= np.linspace(0.1, 1.0, 10), cv=10, n_jobs=num_resources, scoring='mean_squared_error')
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    PlotLearningCurve(train_sizes, train_mean, train_std, test_mean, test_std)

def PlotLearningCurve(train_sizes, train_mean, train_std, test_mean, test_std):
    #todo: automatically save plot to disk for exportation
    #plot data
    plot.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plot.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')

    plot.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plot.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')

    plot.grid()
    plot.xlabel('Number of Training Samples')
    plot.ylabel('Mean Squared Error')
    plot.legend(loc='lower right')
    plot.ylim([0, 1.0])
    plot.show()

def GenerateValidationCurve(estimator, X, y):
    #todo: edit parameters
    #generate validation curve data
    num_resources = -1 #engage all available CPUs and GPUs
    alpha_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]#alpha
    #titrating hidden layer amount and size
    num_hidden_layers = 2
    hidden_units_range1 = [32]*num_hidden_layers
    hidden_units_range2 = [64]*num_hidden_layers
    hidden_units_range3 = [128]*num_hidden_layers
    hidden_units_range4 = [256]*num_hidden_layers
    hidden_units_range5 = [512]*num_hidden_layers
    hidden_units_range6 = [1024]*num_hidden_layers
    hidden_units_range7 = [2048]*num_hidden_layers
    hidden_units_range8 = [4096]*num_hidden_layers
    hidden_units_range = [hidden_units_range1, hidden_units_range2, hidden_units_range3, hidden_units_range4,
                          hidden_units_range5, hidden_units_range6, hidden_units_range7, hidden_units_range8]

    train_scores, test_scores = validation_curve(estimator=estimator, X=X, y=y, param_name='regressor__hidden_units', param_range=hidden_units_range, cv=10, n_jobs=num_resources)
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    PlotValidationCurve(hidden_units_range, train_mean, train_std, test_mean, test_std)

def PlotValidationCurve(param_range, train_mean, train_std, test_mean, test_std):
    #todo: automatically save plot to disk for exportation
    #plot data

    plot_params = [item[0] for item in param_range]
    plot.plot(plot_params, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plot.fill_between(plot_params, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')

    plot.plot(plot_params, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plot.fill_between(plot_params, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')

    #show plot
    plot.grid()
    plot.xscale('log', basex=2)#since hidden units are all powers of 2
    plot.xlabel('Parameter: Number of hidden units')
    plot.ylabel('Accuracy')
    plot.legend(loc='lower right')
    plot.ylim([0, 1.0])
    plot.show()


def GridSearch(estimator, X, y, X_test, y_test, whichDevice ='AWS'):
    hiddenUnitsGridList = [[32], [64], [128], [256], [512], [1024], [2048], [4096]]
    # hiddenUnitsGridList += [item*2 for item in hiddenUnitsGridList]
    hiddenUnitsGridList += [item * 2 for item in hiddenUnitsGridList] + [item * 3 for item in hiddenUnitsGridList] + \
                           [item * 4 for item in hiddenUnitsGridList]
    print (hiddenUnitsGridList)
    if whichDevice == 'AWS':
        batchSize = [16]
        learningRate = [0.0001]
        optimizers = ['Adagrad', 'SGD', 'Adam',  'RMSProp']
        for i in range (1, 11):
            batchSize.append(batchSize[-1]*2)#final element == 16*2^10 = 16384
            learningRate.append(learningRate[-1]*2)#final element == 1*10^-4*2^10 = .1024

        param_grid = {'regressor__steps': [100, 1000, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000, 1000000],
                      'regressor__hidden_units':hiddenUnitsGridList,
                      'regressor__learning_rate': learningRate,
                      'regressor__batch_size': batchSize,
                      'regressor__optimizer':optimizers,
                      'regressor__activation': ['nn.relu', 'nn.tanh', 'nn.sigmoid']
                      }#total grid space
    elif whichDevice == 'MBP2011':
        param_grid = {'regressor__steps': [100, 1000, 10000],
                      'regressor__hidden_units': hiddenUnitsGridList,
                      'regressor__learning_rate': [0.0001, 0.001, 0.01, 0.1],
                      'regressor__batch_size': [32, 64, 128, 256, 512, 1024, 2048, 4096],
                      'regressor__optimizer': ['Adagrad', 'SGD', 'Adam']
                      }  # 2011 # 1
    elif whichDevice == 'MBP2011_':
        param_grid = {'regressor__steps': [100, 1000, 10000],
                      'regressor__hidden_units': hiddenUnitsGridList,
                      'regressor__learning_rate': [0.0001, 0.001, 0.01, 0.1],
                      'regressor__batch_size': [32, 64, 128, 256, 512, 1024, 2048, 4096],
                      'regressor__optimizer': ['RMSProp']  # ,'regressor__activation_fn': ['relu', 'tanh', 'sigmoid']
                      }  # 2011 # 2
    elif whichDevice == 'MBP2014':
        steps = [10000]
        learningRate = [0.0001]
        for i in range (1, 175):
            steps += [steps[-1]+200] #step in increments of 200

        for i in range (1, 20):
            learningRate += [learningRate[-1]+ 0.0001]
        param_grid = {'steps': steps,
                      'hidden_units': [[32], [64], [128], [256], [300], [350], [400], [450], [512], [750], [1024]],
                      'learning_rate': learningRate,
                      'optimizer': ['Adam']
                      }#2014
    elif whichDevice == 'MBP2014_Scaled':
        steps = [10000]
        learningRate = [0.0001]
        for i in range (1, 175):
            steps += [steps[-1]+200] #step in increments of 200

        for i in range (1, 20):
            learningRate += [learningRate[-1]+ 0.0001]
        param_grid = {'regressor__steps': steps,
                      'regressor__hidden_units':[[32], [64], [128], [256], [300], [350], [400], [450], [512], [750], [1024]],
                      'regressor__learning_rate': learningRate,
                      'regressor__optimizer': ['Adam'],
                      }#2014
    else:
        print("Error: Please specify valid device for GridSearch")
        exit(-10)
        param_grid ={}#silence compiler warning
    fit_params = {'X': X, 'y':y, 'logdir': ''}
    num_resources = -1#engage all available CPUs and GPUs
    grid = grid_search.GridSearchCV(estimator=estimator, param_grid=param_grid,
                                    n_jobs=num_resources, verbose=1, iid=False)
    grid.fit(X=X, y=y)
    #todo: edit grid search to log tensorboard info?
    print ("Best Parameters: {params}".format(params = grid.best_params_))
    print("Best Score: {score}".format(score=grid.best_score_))
    print ("Best Estimator: {estimator}".format(estimator=grid.best_estimator_))
    pprint.pprint("All Parameters And Associated Scores: {all_scores}".format(all_scores=grid.grid_scores_))
    pprint.pprint("Using Scorer: {scorer}".format(scorer = grid.scorer_))

    # Predict and score
    score = metrics.mean_squared_error(grid.predict(X_test), y_test)
    print('Best Estimator MSE on Holdout Set: {0:f}'.format(score))

def AssignDevice(path):
    if path == r'/Users/teofilozosa/PycharmProjects/BreakthroughANN/':
        device = 'MBP2011_'
    elif path == r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/':
        device = 'MBP2014'
    elif path == r'/Users/Home/PycharmProjects/BreakthroughANN/':
        device = 'MBP2011'
    else:
        device = 'AWS'#todo: configure AWS path
    return device


def AssignPath(deviceName ='Workstation'):
    if  deviceName == 'MBP2011_':
       path =  r'/Users/teofilozosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2014':
       path = r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2011':
       path = r'/Users/Home/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'Workstation':
        path =r'G:\TruncatedLogs\PythonDatasets\Datastructures\NumpyArrays\4DArraysHDF5(RxCxF)WBPOEWmBm'
    else:
        path = ''#todo:error checking
    return path


def LoadXAndy(path):
    files = [file_name for file_name in utils.find_files(path, "*.hdf5")]
    X, y = ([] for i in range (2))
    for file in files:  # add training examples and corresponding labels from each file to associated arrays
        training_data = h5py.File(file, 'r')
        X.extend(training_data['X'][:])
        y.extend(training_data['y'][:])
        training_data.close()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  # b x r x c x f & label


def LoadXAndy_1to1(path):
    X = pickle.load(open(path + r'ValueNetRankBinary/NPDataSets/WBPOE/XMatrixByRankBinaryFeaturesWBPOEBiasNoZero.p', 'rb'))
    y = pickle.load(open(path + r'ValueNetRankBinary/NPDataSets/WBPOE/yVectorByRankBinaryFeaturesWBPOEBiasNoZero.p', 'rb'))
    return X, y

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
                                 initializer=tf.constant_initializer(0.01))#karpathy: for relu, 0.00 ensures all relus fire in the beginning
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
            shape=[n_features, 155], # 1 x 64 filter in, 155 classes out
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        bias = tf.get_variable(
            name='bias',
            shape=[155],
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

config = run_config.RunConfig(num_cores=-1)
inputPath = AssignPath()
device = AssignDevice(inputPath)
X, y = LoadXAndy(inputPath)

# X[i] is a 3D matrix corresponding to label at y[i]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
    test_size=0.0001, random_state=42)#sametrain/test split every time


# Adam Optimizer
# Num Filters: 192
# Num Hidden Layers: 5
# Learning Rate: 0.0014 && 0.0015
# Batch Size: 128
# # of Epochs: 5

file = open(os.path.join(inputPath, r'ExperimentLogs', 'AdamNumFiltersNumLayersNON-TFSumCrossEntropy01212017_He_weightsWBPOEWmBm.txt'), 'a')
# file = sys.stdout

for num_hidden in [i for i in range(12,13)]:
    for n_filters in [
                        16, 32, 64,
                       128,
                      192]:
        for learning_rate in [
            0.001,
            # 0.0011, 0.0012, 0.0013,
            #                   0.0014, 0.0015
        ]:
            reset_default_graph()
            batch_size = 128

            #build graph
            X = tf.placeholder(tf.float32, [None, 8, 8, 8])
            # TODO: consider reshaping for C++ input; could also put it into 3d matrix on the fly, ex. if player == board[i][j], X[n][i][j] = [1, 0, 0]
            y = tf.placeholder(tf.float32, [None, 155])
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
            # cost = tf.nn.softmax_cross_entropy_with_logits(outer_layer, y)
            # # alternative implementation
            # cost = tf.reduce_mean(cost) #used in MNIST tensorflow

            #kadenze cross_entropy cost function
            cost = -tf.reduce_sum(y * tf.log(y_pred + 1e-12))


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
            X_train1, X_valid, y_train1, y_valid = model_selection.train_test_split(X_train, y_train,
                                                                                    test_size=512,
                                                                                    random_state=random.randint(1, 1024))  # keep validation outside
            for epoch_i in range(n_epochs):
                X_train1, y_train1 = shuffle(X_train1, y_train1,
                                                   random_state=random.randint(1, 1024))  # reshuffling of training set at each epoch

                X_train_batches, y_train_batches = utils.batch_split(X_train1, y_train1, batch_size)

                startTime = time.time()  #start timer

                #train model
                for i in range (0, len(X_train_batches)):
                    sess.run(optimizer, feed_dict={
                        X: X_train_batches[i],
                        y: y_train_batches[i]
                    })

                    # show stats at every 1/10th interval of epoch
                    if (i+1)%(len(X_train_batches)//10)==0:
                        loss = sess.run(cost, feed_dict={
                            X: X_train_batches[i],
                            y: y_train_batches[i]
                            })
                        accuracy_score = sess.run(accuracy, feed_dict={
                                       X: X_valid,
                                       y: y_valid
                                       })
                        print("Loss: {}".format(loss), end="\n", file=file)
                        # print("Loss Reduced Mean: {}".format(sess.run(tf.reduce_mean(loss))), end="\n", file=file)
                        # print("Loss Reduced Sum: {}".format(sess.run(tf.reduce_sum(loss))), end="\n", file=file)
                        print('Interval {interval} of 10 Accuracy: {accuracy}'.format(
                            interval=(i+1)//(len(X_train_batches)//10),
                            accuracy=accuracy_score), end="\n", file=file)


                #show accuracy at end of epoch
                print ('Epoch {epoch_num} Accuracy: {accuracy_score}'.format(
                    epoch_num=epoch_i+1,
                    accuracy_score=sess.run(accuracy,
                                   feed_dict={
                                       X: X_valid,
                                       y: y_valid
                                   })), end="\n", file=file)

                #show example of what network is predicting vs the move oracle
                y_pred_vector =sess.run(y_pred, feed_dict={X:[X_valid[0]]})
                print("Sample Predicted Probabilities = "
                      "\n{y_pred}"
                      "\nPredicted vs. Actual Move = "
                      "\nIf white: {y_pred_white} vs. {y_act_white}"
                      "\nIf black: {y_pred_black} vs. {y_act_black}".format(
                        y_pred=y_pred_vector,
                        y_pred_white=utils.move_lookup(np.argmax(y_pred_vector), 'White'),
                        y_pred_black=utils.move_lookup(np.argmax(y_pred_vector), 'Black'),
                        y_act_white=utils.move_lookup(np.argmax(y_valid[0]), 'White'),
                        y_act_black=utils.move_lookup(np.argmax(y_valid[0]), 'Black')),
                    end="\n", file=file)

                print("\nMinutes between epochs: {time}".format(time=(time.time() - startTime) / (60)), end="\n", file=file)

            # Print final test accuracy:
            print("Final Test Accuracy: {}".format(sess.run(accuracy,
                           feed_dict={
                               X: X_test,
                               y: y_test
                           })), end="\n", file=file)
            sess.close()
file.close()
# feature_columns = []
# for i in range (0, 256):
#     column = tf.contrib.layers.sparse_column_with_hash_bucket("{number}".format(number=i), hash_bucket_size=2, dtype=tf.int32)
#     sparse_column = tf.contrib.layers.embedding_column(sparse_id_column=column, dimension=2)
#     # sparse_column = tf.contrib.layers.sparse_column_with_integerized_feature("{number}".format(number=i), bucket_size=2, dtype=tf.int32)
#     #feature_columns = [tf.contrib.layers.one_hot_column(sparse_column)]
#     feature_columns.append(sparse_column)
#
#
# # Build 3 layer DNN with 10, 20, 10 units respectively.
# classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
#                                             hidden_units=[1024, 2056, 1024],
#                                             n_classes=155,#including no move
#                                             activation_fn=tf.nn.relu,
#                                             #model_dir="/tmp/breakthrough_model_Adam",
#                                             optimizer=tf.train.AdamOptimizer(learning_rate=0.000000001))
#
# # Fit model.
# classifier.fit(x=X_train,
#                y=y_train,
#                steps=2000,
#                batch_size=2056)
#
# # Evaluate accuracy.
# accuracy_score = classifier.evaluate(x=X_test,y=y_test)["accuracy"]
# print('Accuracy: {0:f}'.format(accuracy_score))
#

# 't' is [
# [[1, 1, 1], [2, 2, 2]],
# [[3, 3, 3], [4, 4, 4]]
# ]


