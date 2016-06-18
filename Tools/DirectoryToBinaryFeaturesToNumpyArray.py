import pickle

import NumpyArray as n2a
import SortPlayerListByRank as sl

from PlayersToDataStructure import PlayerDataDirectoryToAnalysisFormatWBPOEBinaryFeaturePlanes as pd


def TestScript(Xtest, ytest):
    XMatrix = pickle.load(open('' + r'XMatrixByRankBinaryFeatures.p', 'rb'))
    yVector = pickle.load(open('' + r'yVectorByRankBinaryFeatures.p', 'rb'))
    assert (Xtest.tolist() == XMatrix.tolist() and ytest.tolist() == yVector.tolist())
#process data
playerList = []
path = r'/Users/TeofiloZosa/BreakthroughData/AutomatedData/'
pd.ProcessDirectoryOfBreakthroughFiles(path, playerList)
#sort list
playerList = sl.SortByRank(playerList)
fileName = r'PlayerDataBinaryFeaturesDatasetSorted.p'
#check list
#cl.TestOriginalVsMirrorStates(playerList) #the only test currently for this state list
#convert to Numpy data structures
filter = n2a.AssignFilter(fileName)
X, y = n2a.generateArrayByFilter(playerList, filter)
writePath = '' #cwd
#n2a.WriteNPArrayToDisk(writePath, X, y, filter)
#todo: check to see if this script functions properly
TestScript(X, y)