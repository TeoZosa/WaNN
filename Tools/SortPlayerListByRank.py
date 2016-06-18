import pickle #serialize the data structure
import pprint
pathToCheck = r'/Users/TeofiloZosa/BreakthroughData/AutomatedData/'
def AllPlayersData(path):
    outputFile = open(path + r'PlayerDataPythonSorted1.p', 'wb')
    playerList = pickle.load(open(path+r'PlayerDataPython.p', 'rb'))
    return outputFile, playerList
def AnalysisFormatData(path):
    outputFile = open(path + r'PlayerDataPythonDatasetSorted.p', 'wb')
    playerList = pickle.load(open(path + r'PlayerDataPythonDataset.p', 'rb'))
    return outputFile, playerList
def AnalysisFormatBinaryFeaturesData(path):
    outputFile = open(path + r'PlayerDataBinaryFeaturesDatasetSorted.p', 'wb')
    playerList = pickle.load(open(path + r'PlayerDataBinaryFeaturesDataset.p', 'rb'))
    return outputFile, playerList
def AnalysisFormatBinaryFeaturesWBEData(path):
    outputFile = open(path + r'PlayerDataBinaryFeaturesWBEDatasetSorted.p', 'wb')
    playerList = pickle.load(open(path + r'PlayerDataBinaryFeaturesWBEDataset.p', 'rb'))
    return outputFile, playerList
def AnalysisFormatBinaryFeaturesWBPOEData(path):
    outputFile = open(path + r'PlayerDataBinaryFeaturesWBPOEDatasetSorted.p', 'wb')
    playerList = pickle.load(open(path + r'PlayerDataBinaryFeaturesWBPOEDataset.p', 'rb'))
    return outputFile, playerList
def AllPlayersDataNoSpurriousGames(path):
    outputFile = open(path + r'PlayerDataPythonNonSpurriousGamesSorted.p', 'wb')
    playerList = pickle.load(open(path + r'PlayerDataPythonNonSpurriousGames.p', 'rb'))
    return outputFile, playerList
def SortByRank(playerList):
    newList = sorted(playerList, key=lambda k: k['Rank'], reverse=True)
    return newList
def WriteToDisk(playerList, outputFile):
    pickle.dump(playerList, outputFile)

outputFile, playerList = AnalysisFormatBinaryFeaturesWBPOEData(pathToCheck)
playerList = SortByRank(playerList)
WriteToDisk(playerList, outputFile)
