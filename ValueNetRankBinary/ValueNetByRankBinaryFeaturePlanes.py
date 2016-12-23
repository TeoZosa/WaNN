
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Tools.CustomRegressionNetwork as CustomRegressionNetwork
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import  pprint
import tensorflow as tf
from tensorflow.contrib import skflow
from tensorflow.python.ops import nn
from tensorflow.python.framework import  dtypes
from sklearn import datasets, cross_validation, metrics, grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.learning_curve import learning_curve, validation_curve
from tensorflow.contrib.learn.python.learn.estimators import run_config
import time
from scipy import stats ,integrate
import seaborn as sns

sns.set(color_codes=True)


import pickle
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
    K_folds = cross_validation.KFold(numTrainingExamples, K, shuffle=False,)

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
    scores = cross_validation.cross_val_score(estimator=learningModel,
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
    hidden_units_range8 = [4056]*num_hidden_layers
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

def BuildRegressionNetwork(num_hidden_units = 2048, num_hidden_layers = 2, activation = nn.relu):
    if num_hidden_layers == 0:
        #passed in hidden units list which has heterogenous hidden layers
        hidden_layers = num_hidden_units
    else:
        # fully connected NN with all hidden layers containing the same number of hidden units.
        hidden_layers = [num_hidden_units] * num_hidden_layers

    # if activation == 'relu':#defailt activation function
    #     # estimator = skflow.TensorFlowDNNRegressor(hidden_units=hidden_layers,
    #     #                                  steps=100000, optimizer='Adam', learning_rate=0.1,
    #     #                                  batch_size=32)  # changed to Adam as cursory literature search shows better results
    #     estimator = CustomRegressionNetwork.TensorFlowDNNRegressor(hidden_units=hidden_layers,
    #                                                                steps=100000, optimizer='Adam', learning_rate=0.1,
    #                                                                batch_size=32, activation=nn.relu)
    # elif activation =='sigmoid':
    #     estimator = CustomRegressionNetwork.TensorFlowDNNRegressor(hidden_units=hidden_layers,
    #                                      steps=100000, optimizer='Adam', learning_rate=0.1,
    #                                      batch_size=32, activation=nn.sigmoid)
    # elif activation =='tanh':
    #     estimator = CustomRegressionNetwork.TensorFlowDNNRegressor(hidden_units=hidden_layers,
    #                                      steps=100000, optimizer='Adam', learning_rate=0.1,
    #                                      batch_size=32, activation=nn.tanh)
    # else:
    #     "Error: invalid activation function"
    return CustomRegressionNetwork.TensorFlowDNNRegressor(hidden_units=hidden_layers,
                                                      steps=500000, optimizer='Adam', learning_rate=0.1,
                                                      batch_size=32, activation=activation)

def PrintTable(X, y):
    columns = []
    #generate column names
    for dimension in ['Player', 'Opponent', 'Empty']:
        for i in range (1, 9):
            for char in 'abcdefgh':
                position = 'Position: ' + char + str(i) + ' (' + dimension +')'
                columns.append(position)
    columns.append('Player Turn')
    columns.append('Outcome')
    combined_matrix = np.column_stack((X, y))
    frame = pd.DataFrame(combined_matrix, columns=columns)
    pprint.pprint(frame.tail())
    WriteNPArrayToCSV(combined_matrix)
    WriteDataFrameToXLSX(frame)


def WriteDataFrameToXLSX(frame, file_name ='output.xlsx'):#human readable table
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    frame.to_excel(writer, 'Sheet1')
    writer.save()

def WriteDataFrameToCSV(frame, file_name ='output.csv'):
    frame.to_csv(file_name)

    checkFrame = pd.read_csv(file_name, header = None)
    pprint.pprint(checkFrame)

def WriteNPArrayToCSV(array, file_name='output.csv'):
    np.savetxt(file_name, array, delimiter=',', fmt='%1i')

def ReadCSVfile(file_name='BinaryFeaturePlanesDataset.csv', columns=None):#to import data back in.
    frame = pd.read_csv(file_name, header=None, names=columns)
    return frame
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
def AssignPath(deviceName ='AWS'):
    if  deviceName == 'MBP2011_':
       path =  r'/Users/teofilozosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2014':
       path = r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2011':
       path = r'/Users/Home/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'AWS':
        path =''#todo: configure AWS path
    else:
        path = ''#todo:error checking
    return path
def LoadXAndy(path):
    X = pickle.load(open(path + r'ValueNetRankBinary/NPDatasets/WBPOE/XMatrixByRankBinaryFeaturesWBPOEBiasNoZero.p', 'rb'))
    y = pickle.load(open(path + r'ValueNetRankBinary/NPDatasets/WBPOE/yVectorByRankBinaryFeaturesWBPOEBiasNoZero.p', 'rb'))
    return X, y
def LoadXAndy_1to1(path):
    X = pickle.load(open(path + r'ValueNetRankBinary/NPDatasets/WBPOE/XMatrixByRankBinaryFeaturesWBPOEBiasNoZero.p', 'rb'))
    y = pickle.load(open(path + r'ValueNetRankBinary/NPDatasets/WBPOE/yVectorByRankBinaryFeaturesWBPOEBiasNoZero.p', 'rb'))
    return X, y
# Step #50000, epoch #1, avg. train loss: 0.32184 MSE: 0.664601
# Step #142500, epoch #4, avg. train loss: 0.18429
# Step #142600, epoch #4, avg. train loss: 0.18319
# Step #142700, epoch #4, avg. train loss: 0.18157
#TODO: construct CNN and select models.
#TODO: excise code to be run in main script.
   #main script
inputPath = AssignPath('MBP2014')
device = AssignDevice(inputPath)
X, y = LoadXAndy(inputPath)# row vector X[i] has scalar outcome y[i]
# X_1, y_1 = LoadXAndy_1to1(inputPath)

# writePath = inputPath+'Models/WBPOE/Skflow-1to1AdamNoScalingReluDropout'
# writePath1 = inputPath+'Models/WBPOE/Skflow-1to1AdamNoScalingSoftplusDropout'
# writePath2 = inputPath+'Models/WBPOE/Skflow-1to1AdamNoScalingSoftsignDropout'
# writePath3 = inputPath+'Models/WBPOE/Skflow-1to1SGDNoScalingRelu0009'
# writePath4 = inputPath+'Models/WBPOE/Skflow-1to1SGDNoScalingSoftplusDropout'
# writePath5 = inputPath+'Models/WBPOE/Skflow-1to1SGDNoScalingSoftsignDropout'

#Train/validation/test split or reducing number of training examples
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
#     test_size=0, random_state=42)#change random state?
# X_1train, X_1test, y_1train, y_1test = cross_validation.train_test_split(X_1, y_1,
#     test_size=0.1, random_state=42)#change random state?
#for final run
X, y = shuffle(X, y, random_state= 777)

# 't' is [
# [[1, 1, 1], [2, 2, 2]],
# [[3, 3, 3], [4, 4, 4]]
# ]
config = run_config.RunConfig( num_cores=-1)

#print (len(np.asarray(X[0], np.int8)[0]))

# regressor3 = skflow.TensorFlowDNNRegressor(hidden_units=[400],
#                                            optimizer='Adam', learning_rate=0.0009, steps=20000,
#                                            batch_size=256)  # AlphaGo uses SGD
# GridSearch(regressor3, X_train, y_train, X_test, y_test, whichDevice="MBP2014")

# regressor2 = skflow.TensorFlowDNNRegressor(hidden_units=[400, 400],
#                                            optimizer='Adam', learning_rate=0.0019, steps=20000,
#                                            batch_size=256)
# scaler = ('scaler', StandardScaler())
# DNNR = ('regressor', regressor2)
# pipe_DNNR = Pipeline([scaler, DNNR])
# GridSearch(pipe_DNNR, X_train, y_train, X_test, y_test, whichDevice="MBP2014_Scaled")
# pipe_DNNR.fit(X_1train, y_1train)
#

#regressor3.fit(X, y, logdir=inputPath+r"ValueNetRankBinary/12IntegrationReluAdam1x400_eta0009")
optimizer = tf.train.AdamOptimizer()
regressor3 = skflow.DNNRegressor(hidden_units=[400], model_dir=inputPath+r'/finalIntegration', optimizer=tf.train.AdamOptimizer())
regressor3.fit(X, y, batch_size=256, steps=20000)
timerObj = time.perf_counter()
#print("{predict} vs {actual} in {time} seconds.".format(actual = y_train[0], predict = regressor3.predict(X_train[0]), time= time.perf_counter()-timerObj))
# Dist = sns.distplot(regressor3.predict(X))
# plot.show(Dist)
#score = metrics.mean_squared_error(regressor3.predict(X_test), y_test)
#print('#Relu Adam 0to1 Best Estimator MSE on Holdout Set: {0:f}'.format(score))
# print(regressor3.get_variable_value('hiddenlayer_0/weights'))
#
# # writeOut = np.asarray(regressor3.get_variable_value('hiddenlayer_0/weights'))
# # WriteNPArrayToCSV(writeOut)
# weights = regressor3.get_variable_value('hiddenlayer_0/weights')
# weightList = []
# for weight in weights:
#     weightList.append(weight)
# pprint.pprint(weightList)
# biases = regressor3.get_variable_value('hiddenlayer_0/bias')
# biasList = []
# for bias in biases:
#     biasList.append(bias)
# pprint.pprint(biasList)
# outputFile = open("hiddenUnitWeights.txt", 'w')
# outputFile.write(str(weightList))
# outputFile.close()
# outputFile = open("HiddenUnitBias.txt", 'w')
# outputFile.write(str(biasList))
# outputFile.close()
# weights = regressor3.get_variable_value('dnn_logit/weights')
# weightList = []
# for weight in weights:
#     weightList.append(weight)
# pprint.pprint(weightList)
# biases = regressor3.get_variable_value('dnn_logit/bias')
# biasList = []
# for bias in biases:
#     biasList.append(bias)
# pprint.pprint(biasList)
# outputFile = open("dnn_logitWeights.txt", 'w')
# outputFile.write(str(weightList))
# outputFile.close()
# outputFile = open("dnn_logitBias.txt", 'w')
# outputFile.write(str(biasList))
# outputFile.close()
# biases = regressor3.get_variable_value(r'centered_bias_weight')
# biasList = []
# for bias in biases:
#     biasList.append(bias)
# pprint.pprint(biasList)
# outputFile = open("centeredBias.txt", 'w')
# outputFile.write(str(biasList))
# outputFile.close()
# regressor3.save('12Int')

#WriteToDisk(inputPath, regressor)



# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     checkpoint = tf.train.get_checkpoint_state(inputPath+r"ValueNetRankBinary/12IntegrationReluAdam1x400_eta0009")
#     if checkpoint and checkpoint.model_checkpoint_path:
#         saver.restore(sess, inputPath+r"ValueNetRankBinary/12IntegrationReluAdam1x400_eta0009")
#         print("model Loaded")
#     else:
#         print("no checkpoint found")
#     sess.

  ####ON Player opponent empty dataset
#Skflow Adam 50k steps, batch size 32, 4096x2 hidden units Best Estimator MSE on Holdout Set: 0.164951
#ReluScaled Adam Best Estimator MSE on Holdout Set: 0.178168
#Softplus Adam Best Estimator MSE on Holdout Set: 0.165375
#softsign Adam Best Estimator MSE on Holdout Set: 0.280076
#reluNoScaling Adam Best Estimator MSE on Holdout Set: 0.165375

#Relu Adam 0to100 Best Estimator MSE on Holdout Set: 1777.574707
#Sofplus Adam 0to100 Best Estimator MSE on Holdout Set: 1661.274536
#SoftSign Adam 0to100 Best Estimator MSE on Holdout Set: 1694.384521

#WBEPOE Bias [4096, 4096] Dropout p =0.5
#Relu Adam -1to1 Best Estimator MSE on Holdout Set: 0.163439[2048, 2048]
#Sofplus Adam -1to1 Best Estimator MSE on Holdout Set: 0.163352
#SoftSign Adam -1to1 Best Estimator MSE on Holdout Set: 43.901260
#Relu SGD -1to1 Best Estimator MSE on Holdout Set: 0.178721



#Relu SGD -1to1 Best Estimator MSE on Holdout Set: 0.149449 [4096] a = 0.0007



#BiasWBPOE...

#Relu Adagrad -1to1 Best Estimator MSE on Holdout Set: 0.144799
#Relu Adam -1to1 Best Estimator MSE on Holdout Set: 0.082600!!
#Relu SGD -1to1 Best Estimator MSE on Holdout Set: 0.085655 #at 1024x1 hidden units!

#WBPOENoBias
#Relu Adam 0to1 Best Estimator MSE on Holdout Set: 0.069541 #at 1024x1 hidden units! 0to1 with no bias
