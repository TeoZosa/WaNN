import pickle
import numpy as np
import pandas as pd
import warnings
import h5py
import os

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
        XMatrix = open(path + r'ValueNetRankBinary/NPDataSets/WBPOE/XMatrixByRankBinaryFeaturesWBPOEBiasNoZero.p', 'wb')
        pickle.dump(X, XMatrix)
        yVector = open(path + r'ValueNetRankBinary/NPDataSets/WBPOE/yVectorByRankBinaryFeaturesWBPOEBiasNoZero.p', 'wb')
        pickle.dump(y, yVector)
    elif filterType == r'Self-Play':
        if NNType == 'Policy':
          h5f = h5py.File(path + r'POEBias.hdf5', 'w', driver='core')
          h5f.create_dataset(r'X', data=X)
          h5f.create_dataset(r'y', data=y)
          h5f.close()
          # XMatrix = open(path + r'XMatrixSelfPlayPOEBias.npy', 'wb')
          # np.save(XMatrix, X)
          # yVector = open(path + r'yVectorSelfPlayPOEBias.npy', 'wb')
          # np.save(yVector, y)
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
    print ('# of States If We Filter by Rank: {states}'.format(states=len(X)))
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

def filter_for_self_play(self_play_data, NNType):
    training_data = []
    # if NNType == 'Policy':
    for self_play_log in self_play_data:
      for game in self_play_log['Games']:
          states = game['BoardStates']['PlayerPOV']
          mirror_states = game['MirrorBoardStates']['PlayerPOV']
          assert len(states) == len(mirror_states)
          for i in range(0, len(states)):
              training_data.append(states[i])
              training_data.append(mirror_states[i])
      print('# of States for Self-Play {NNType} Net: {states}'.format(states=len(training_data), NNType=NNType))
    # else:  # value net
    #   for selfPlayLog in selfPlayDataList:
    #       for game in selfPlayLog['Games']:
    #           states = game['BoardStates']['States']
    #           mirrorStates = game['MirrorBoardStates']['States']
    #           assert len(states) == len(mirrorStates)
    #           for i in range(0, len(states)):
    #               X.append(states[i])
    #               X.append(mirrorStates[i])
    #   print('# of States for Self-Play Value Net: {states}'.format(states=len(X)))
    return training_data

def SplitArraytoXMatrixAndYOutcomeVector(arrayToSplit, convertY = True):
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


def split_data_to_training_examples_and_labels_for_CNN(array_to_split, NNType):
    training_examples = []
    labels = []
    for training_example in array_to_split:  # probability space for transitions
        formatted_example, formatted_label = split_training_examples_and_labels(training_example, NNType)
        training_examples.append(formatted_example)
        labels.append(formatted_label)  # transition number
    return training_examples, labels

def split_training_examples_and_labels(training_example, NNType):
    formatted_example = format_training_example(training_example[0])
    if NNType == 'Policy':
         formatted_label = label_for_policy(label=training_example[2])
    else: # Value Net
         formatted_label = label_for_value(label=training_example[1])
    return formatted_example, formatted_label

def format_training_example(training_example):
    formatted_example = []
    for plane in training_example:
        formatted_example += plane  # flatten 2d matrix
    formatted_example += [1] * 64  # 1 bias plane
    formatted_example = np.reshape(np.array(formatted_example, dtype=np.float32),
                                   (len(formatted_example) // 64, 8, 8))  # feature_plane x row x co)
    for i in range(0, len(formatted_example)):
        formatted_example[i] = formatted_example[
            i].transpose()  # transpose (row x col) to get feature_plane x col x row
    formatted_example = formatted_example.transpose()  # transpose to get proper dimensions: row x col  x feature plane
    return formatted_example
    
def label_for_policy(label, one_hot_indexes=False):
    # if one_hot_indexes:
    #     # x still a 1d array; must stay one-hotted to be reshaped properly
    #     formatted_example = np.array([index for index, has_piece in enumerate(formatted_example) if has_piece == 1], dtype=np.float32)
    #     # if we just want the transition index vs a pre-one-hotted vector
    #     formatted_label = label.index(1)  # assumes no errors, y has a single value of 1
    formatted_label = np.array(label, dtype=np.float32)  # One hot transition vector
    return formatted_label

def label_for_value(label): 
    if label == -1:
        formatted_label = 0
    elif label == 1:
        formatted_label = label
    else:
        formatted_label = None
        print("Error: y[i] should be 1 or -1")
        exit(-11)
    return formatted_label

    


def SplitArraytoXMatrixAndYTransitionVectorCNN(arrayToSplit):  # only for boards with pd dataframe
    warnings.warn("Only for use with PD Dataframe data; "
                  "Removed in favor of performing conversion later in the pipeline. "
                  "Else, earlier stages of pipeline will be computationally expensive, "
                  "memory intensive, and require large amounts of disk space "
                  , DeprecationWarning)
    X = []
    y = []
    for trainingExample in arrayToSplit:  # probability space for transitions
        x = []
        for plane in trainingExample[0]:
            x.append(plane.as_matrix()) #convert pd dataframe to np array
        board_dimensions = (8, 8)
        x.append(np.ones(board_dimensions, dtype=np.int32)) # 1 bias plane
        one_hot_transitions = None
        for transition in range(0, len(trainingExample[2])):
            if trainingExample[2][transition] == 1:
                one_hot_transitions = transition
                break
        if one_hot_transitions == None:  # no move
            # one_hot_transitions = -1  # tf.one_hot will return a vector of all 0s
            one_hot_transitions = 155  # DNNClassifier => 155 == no move category
        y.append(one_hot_transitions)  # transition vector
        X.append(np.array(x, dtype=np.int32))
    return X, y

def generateArray(player_list, filter, NNType):
        if filter =='Win Ratio':
            training_data = FilterByWinRatio(player_list)
        elif filter =='Rank':
            training_data = FilterByRank(player_list)
        elif filter =='Self-Play':
            training_data = filter_for_self_play(player_list, NNType)
        else:
            training_data= None
            print('Invalid Filter Specified')
            exit(-2)       
        training_examples, labels = split_data_to_training_examples_and_labels_for_CNN(training_data, NNType)
        training_examples = np.array(training_examples, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        return training_examples, labels
def GenerateCSV(X, isX = True):
    if isX == True:
     np.savetxt(r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/ValueNetRankBinary/NPDataSets/WBPOEUnshuffledBinaryFeaturePlanesWBPOETrainingExamples.csv', X, delimiter=',', fmt='%1i')
    else:
        np.savetxt(
            r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/ValueNetRankBinary/NPDataSets/WBPOE/UnshuffledBinaryFeaturePlanesWBPOETrainingExampleOutcomes.csv',
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
        writer = pd.ExcelWriter(r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/ValueNetRankBinary/NPDataSets/WBPOE/UnshuffledBinaryFeaturePlanesDataset1.xlsx', engine='xlsxwriter')
    else:
        writer = pd.ExcelWriter(r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/ValueNetRankBinary/NPDataSets/WBPOE/UnshuffledBinaryFeaturePlanesDataset2.xlsx', engine='xlsxwriter')

    frame.to_excel(writer, 'Sheet1')
    writer.save()
def AssignFilter(fileName):
    if fileName == r'PlayerDataPythonDataSetsorted.p':
        filter = r'Rank'
    elif fileName == r'PlayerDataBinaryFeaturesWBPOEDataSetsorted.p':
        filter = r'Binary Rank'
    else:
        filter = r'UNDEFINED'
    return filter
def AssignPath(deviceName ='Workstation'):
    if  deviceName == 'MBP2011_':
       path =  r'/Users/teofilozosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2014':
       path = r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'MBP2011':
       path = r'/Users/Home/PycharmProjects/BreakthroughANN/'
    elif deviceName == 'Workstation':
        path ='G:\TruncatedLogs\PythonDataSets\DataStructures'
    else:
        path = ''#todo:error checking
    return path

def SelfPlayDriver(filter, NNType, path, fileName):
    file = open(os.path.join(path, fileName), 'r+b')
    playerListDataFriendly = pickle.load(file)
    file.close()
    X, y = generateArray(playerListDataFriendly, filter, NNType)
    writePath = os.path.join(path,"NumpyArrays",fileName[0:-len(r'DataPython.p')])
    WriteNPArrayToDisk(writePath, X, y, filter, NNType)
