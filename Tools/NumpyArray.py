import  pickle
import numpy as np
import pandas as pd
from random import shuffle
import  pprint
import copy

#TODO: redo logic and paths after directory restructuring
def WriteNPArrayToDisk(path, X, y, filterType, NNType):
    if filterType == r'Rank':
        XMatrix = open(path + r'XMatrixByRank.p', 'wb')
        pickle.dump(X, XMatrix)
        yVector = open(path + r'yVectorByRank.p', 'wb')
        pickle.dump(y, yVector)
    elif filterType == r'Win Ratio':
        XMatrix = open(path + r'XMatrixByWinRatio.p', 'wb')
        pickle.dump(X, XMatrix)
        yVector = open(path + r'yVectorByWinRatio.p', 'wb')
        pickle.dump(y, yVector)
    elif filterType == r'Binary Rank':
        XMatrix = open(path + r'ValueNetRankBinary/NPDatasets/WBPOE/XMatrixByRankBinaryFeaturesWBPOEBiasNoZero.p', 'wb')
        pickle.dump(X, XMatrix)
        yVector = open(path + r'ValueNetRankBinary/NPDatasets/WBPOE/yVectorByRankBinaryFeaturesWBPOEBiasNoZero.p', 'wb')
        pickle.dump(y, yVector)
    elif filterType == r'Self-Play':
        if NNType == 'Policy':
          XMatrix = open(path + r'XMatrixSelfPlayPOEBias.p', 'wb')
          pickle.dump(X, XMatrix)
          yVector = open(path + r'yVectorSelfPlayPOEBias.p', 'wb')
          pickle.dump(y, yVector)
        else:#value net
          XMatrix = open(path + r'XMatrixSelfPlayWBPOEBiasNoZero.p', 'wb')
          pickle.dump(X, XMatrix)
          yVector = open(path + r'yVectorSelfPlayWBPOEBiasNoZero.p', 'wb')
          pickle.dump(y, yVector)
    else:
        print ("Error: You must specify a valid Filter")
def FilterByRank(playerList):
    X = []
    for player in playerList:
        if player['Rank'] >=2000:#Good players only
            for game in player['Games']:
                states = game['BoardStates']['States']
                mirrorStates = game['BoardStates']['MirrorStates']
                assert len(states) == len(mirrorStates)
                for i in range (0, len(states)):
                  X.append(states[i])
                  X.append(mirrorStates[i])
    print ('# of States If We Filter by Rank: {states}'.format(states = len(X)))
    return X
def FilterByWinRatio(playerList):
    X = []
    for player in playerList:
        if player['Wins'] > 0 and player['Wins']/(player['Wins']+player['Losses']) >= .8:  # >=80% win rate
            for game in player['Games']:
                states = game['BoardStates']['States']
                mirrorStates = game['BoardStates']['MirrorStates']
                assert len(states) == len(mirrorStates)
                for i in range(0, len(states)):
                    X.append(states[i])
                    X.append(mirrorStates[i])
    print ('# of States If We Filter by Win Ratio: {states}'.format(states = len(X)))
    return X

def FilterForSelfPlay(selfPlayDataList, NNType):
    X = []
    if NNType == 'Policy':
      for serverNodeByDate in selfPlayDataList:
          for selfPlayLog in serverNodeByDate:
              for game in selfPlayLog['Games']:
                  states = game['BoardStates']['PlayerPOV']
                  mirrorStates = game['MirrorBoardStates']['PlayerPOV']
                  assert len(states) == len(mirrorStates)
                  for i in range(0, len(states)):
                      X.append(states[i])
                      X.append(mirrorStates[i])
      print('# of States for Self-Play Policy Net: {states}'.format(states=len(X)))
    else:
      for serverNodeByDate in selfPlayDataList:
          for selfPlayLog in serverNodeByDate:
              for game in selfPlayLog['Games']:
                  states = game['BoardStates']['States']
                  mirrorStates = game['MirrorBoardStates']['States']
                  assert len(states) == len(mirrorStates)
                  for i in range(0, len(states)):
                      X.append(states[i])
                      X.append(mirrorStates[i])
      print('# of States for Self-Play Value Net: {states}'.format(states=len(X)))
    return X

def SplitArraytoXMatrixAndYVector(arrayToSplit, convertY = True):
    X = []
    y = []
    if (convertY == True):
        ##Much better MSE performance
        for trainingExample in arrayToSplit:# 0 & 1 binary outcomes; more useful so we can view NN output as a probability?
            X.append(trainingExample[0]+ ([1]*1))#1 bias node
            if (trainingExample[1] == 1):
                y.append(trainingExample[1])
            elif (trainingExample[1] == -1):
                y.append(0)
            else:
                print("Error: y[i] should be 1 or -1")
                exit(-11)
    else:
        for trainingExample in arrayToSplit:# -1 & 1 binary outcomes; better if we want NN output to be in terms of cost/reward?
            X.append(trainingExample[0] + ([1]*1))#1 bias node
            # X.append(trainingExample[0])
            y.append(trainingExample[1])
    return X, y


def SplitArraytoXMatrixAndYTransitionVector(arrayToSplit):# TODO: will have to redo Numpy code
    X = []
    y = []
    for trainingExample in arrayToSplit:  # probability space for transitions
        X.append(trainingExample[0] + ([1] * 64))  # 1 bias plane
        y.append(trainingExample[2])#transition vector
    return X, y

def generateArray(playerListDataFriendly, filter, NNType):
        if filter =='Win Ratio':
            X = FilterByWinRatio(playerListDataFriendly)
        elif filter =='Rank':
            X = FilterByRank(playerListDataFriendly)
        elif filter =='Self-Play':
            X = FilterForSelfPlay(playerListDataFriendly, NNType)
        else:
            X=[]
            print('Invalid Filter Specified')
            exit(-2)
        # Xcopy = copy.deepcopy(X)
        # shuffle(X)  # to break symmetry/prevent overfitting
        # assert (Xcopy != X)
        #split into X array and y vectors
        #
        if NNType =='Policy':
            splitX, y = SplitArraytoXMatrixAndYTransitionVector(X)
        else: # Value Net
            splitX, y = SplitArraytoXMatrixAndYVector(X, convertY= True)
        # X = np.matrix(splitX, dtype=np.int8)

        # transform to m x 64 numpy array; 8-bit ints since legal values are -1, 0, 1 or 0, 1
        splitX = np.matrix(splitX, dtype=np.int8)
        # transform to m x 1 numpy array; 8-bit ints since legal values are -1, 1 or 0, 1
        y = np.array(y, dtype=np.int8)
        # test to make sure y values map to corresponding row vectors in X
        # GenerateCSV(splitX)
        # GenerateCSV(y, isX=False)

        # for i in range(0, len(y)):
        #     compare = splitX[i].tolist()
        #     assert (X[i][0] == compare[0])
        #     assert (X[i][1] == y[i] or y[i] == 0)
        return splitX, y
def GenerateCSV(X, isX = True):
    if isX == True:
     np.savetxt(r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/ValueNetRankBinary/NPDatasets/WBPOEUnshuffledBinaryFeaturePlanesWBPOETrainingExamples.csv', X, delimiter=',', fmt='%1i')
    else:
        np.savetxt(
            r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/ValueNetRankBinary/NPDatasets/WBPOE/UnshuffledBinaryFeaturePlanesWBPOETrainingExampleOutcomes.csv',
            X, delimiter=',', fmt='%1i')
def SplitListInHalf(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]
def PreprocessXLSX(X):
    XLSX_X = []
    X = FilterByRank(X)
    for x in X:  # more efficient numpy version somewhere
        XLSX_X.append(x[0] + [x[1]])
    XLSX_X1, XLSX_X2 = SplitListInHalf(XLSX_X)
    XLSX_X1 = np.matrix(XLSX_X1, dtype=np.int8)
    XLSX_X2 = np.matrix(XLSX_X2, dtype=np.int8)
    #data too large, bug causes a fail to write unless we split the data
    GenerateXLSX(XLSX_X1, which=1)
    GenerateXLSX(XLSX_X2, which=2)
def GenerateXLSX(X, which=1):
    columns = []
    # generate column names
    for dimension in ['White', 'Black', 'Player', 'Opponent', 'Empty']:
        for i in range(1, 9):
            for char in 'abcdefgh':
                position = 'Position: ' + char + str(i) + ' (' + dimension + ')'
                columns.append(position)
    columns.append('White\'s Move Preceded This State')
    columns.append('Outcome')
    frame = pd.DataFrame(X, columns=columns)
    if which==1:
        writer = pd.ExcelWriter(r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/ValueNetRankBinary/NPDatasets/WBPOE/UnshuffledBinaryFeaturePlanesDataset1.xlsx', engine='xlsxwriter')
    else:
        writer = pd.ExcelWriter(r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/ValueNetRankBinary/NPDatasets/WBPOE/UnshuffledBinaryFeaturePlanesDataset2.xlsx', engine='xlsxwriter')

    frame.to_excel(writer, 'Sheet1')
    writer.save()
def AssignFilter(fileName):
    if fileName == r'PlayerDataPythonDatasetSorted.p':
        filter = r'Rank'
    elif fileName == r'PlayerDataBinaryFeaturesWBPOEDatasetSorted.p':
        filter = r'Binary Rank'
    else:
        filter = r'UNDEFINED'
    return filter
def AssignPath(deviceName ='AWS'):
    if  deviceName == 'MBP2011_':
       path =  r'/Users/teofilozosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2014':
       path = r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2011':
       path = r'/Users/Home/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'Workstation':
        path =''
    else:
        path = ''#todo:error checking
    return path

def SelfPlayDriver(filter, NNType, path, fileName):
    file = open(path + fileName, 'r+b')
    playerListDataFriendly = pickle.load(file)
    file.close()
    X, y = generateArray(playerListDataFriendly, filter, NNType)
    writePath = path+"NumpyArrays\\"+fileName[0:len(fileName)-2]
    WriteNPArrayToDisk(writePath, X, y, filter, NNType)
