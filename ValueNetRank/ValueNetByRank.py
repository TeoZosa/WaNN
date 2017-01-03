
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plot
from tensorflow.contrib import skflow
from sklearn import DataSets, cross_validation, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.learning_curve import learning_curve
import pickle

# Load dataset
pathToCheck = r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/'
#row vector X[i] has scalar outcome y[i]
X = pickle.load(open(pathToCheck+r'XMatrixByRank.p', 'rb'))
y = pickle.load(open(pathToCheck+r'yVectorByRank.p', 'rb'))

# Split dataset into train / test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
    test_size=0.2, random_state=42)

# Build m layers fully connected with n units per layer.
hiddenUnits = 2048
numLayers = 2
hiddenLayers = [hiddenUnits]*numLayers
regressor = skflow.TensorFlowDNNRegressor(hidden_units=hiddenLayers,
     steps=50000, learning_rate=0.1, batch_size=32)
# scaler = ('scl', StandardScaler())
# DNNR = ('regressor', skflow.TensorFlowDNNRegressor(hidden_units=hiddenLayers,
#     steps=50000, learning_rate=0.1, batch_size=32))
# pipe_DNNR = Pipeline([scaler,DNNR])

#@ 32 x 8, Step #50000, epoch #13, avg. train loss: 0.22394 MSE: 0.485589
#@ 256 x 8Step #50000, epoch #13, avg. train loss: 0.07400 MSE: 0.242250
#@ 2048, 2048; Step #50000, epoch #13, avg. train loss: 0.04630; MSE: 0.177363
#@ 4096, Step #50000, epoch #13, avg. train loss: 0.10870 MSE: 0.277146
#@ 4096, 4096, Step #50000, epoch #13, avg. train loss: 0.04518MSE: 0.170503

# Fit
regressor.fit(X_train, y_train)

# Predict and score
score = metrics.mean_squared_error(regressor.predict(X_test), y_test)

print('MSE: {0:f}'.format(score))

regressorPath = pathToCheck + r'RankValueNet/'
regressor.save(regressorPath)
