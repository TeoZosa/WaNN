
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import datasets, cross_validation, metrics
from sklearn import preprocessing

from tensorflow.contrib import skflow
import pickle

def K_FoldValidation(XMatrix, yVector, numFolds):
    numTrainingExamples = len(X)
    K = numFolds
    if K < 2:
        print("Error, K must be greater than or equal to 2")
        exit(-10)
    elif K > numTrainingExamples:
        print("Error, K must be less than or equal to the number of training examples")
        exit(-11)
    K_folds = cross_validation.KFold(numTrainingExamples, K, shuffle=False,)

    for k, (train_index, test_index) in enumerate(K_folds):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Fit
        regressor.fit(X_train, y_train)

        # Predict and score
        score = metrics.mean_squared_error(regressor.predict(X_test), y_test)
        print('Iteration {0:f} MSE: {1:f}'.format(k+1, score))

def StratifiedK_FoldValidation(XMatrix, yVector, learningModel, numFolds):
    num_cores = -1#use all cores
    scores = cross_validation.cross_val_score(estimator=learningModel,
                                              X=XMatrix, y=yVector, cv = numFolds, n_jobs=num_cores)
    print('CV accuracy scores: {score}'.format(score = scores))


# Load dataset
pathToCheck = r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/'
#row vector X[i] has scalar outcome y[i]
X = pickle.load(open(pathToCheck+r'XMatrixByWinRatio.p', 'rb'))
y = pickle.load(open(pathToCheck+r'yVectorByWinRatio.p', 'rb'))

# # Split dataset into train / test
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
#     test_size=0.2, random_state=42)


# Build m layers fully connected with n units per layer.
hiddenUnits = 2048
numLayers = 2
hiddenLayers = [hiddenUnits]*numLayers
regressor = skflow.TensorFlowDNNRegressor(hidden_units=hiddenLayers,
    steps=100000, learning_rate=0.01, batch_size=32)

# Step #50000, epoch #1, avg. train loss: 0.32184 MSE: 0.664601
# Step #142500, epoch #4, avg. train loss: 0.18429
# Step #142600, epoch #4, avg. train loss: 0.18319
# Step #142700, epoch #4, avg. train loss: 0.18157

#Cross-Validate model performance
StratifiedK_FoldValidation(XMatrix=X, yVector=y, learningModel=regressor, numFolds=10)

regressorPath = pathToCheck + r'WinRatioValueNet/'
regressor.save(regressorPath)
