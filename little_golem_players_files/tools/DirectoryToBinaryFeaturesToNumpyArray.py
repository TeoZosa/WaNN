import pickle

from tools import numpy_array as n2a
from little_golem_players_files.tools import SortPlayerListByRank as sl
from little_golem_players_files.PlayersToDataStructure import \
    PlayerDataDirectoryToAnalysisFormatWBPOEBinaryFeaturePlanes as pd


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
fileName = r'PlayerDataBinaryFeaturesDataSetsorted.p'
#check list
#cl.TestOriginalVsMirrorStates(playerList) #the only test currently for this state list
#convert to Numpy data structures
filter = n2a.assign_filter(fileName)
X, y = n2a.filter_training_examples_and_labels(playerList, filter)
writePath = '' #cwd
#n2a.write_np_array_to_disk(writePath, X, y, filter)
#todo: check to see if this script functions properly
TestScript(X, y)