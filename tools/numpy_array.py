import pickle
import numpy as np
import pandas as pd
import warnings
import h5py
import os
import random

#TODO: redo logic and paths after directory restructuring
def write_np_array_to_disk(path, X, y, filterType, NNType):
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
        XMatrix = open(path + r'value_net_rank_binary/NPDataSets/WBPOE/XMatrixByRankBinaryFeaturesWBPOEBiasNoZero.p', 'wb')
        pickle.dump(X, XMatrix)
        yVector = open(path + r'value_net_rank_binary/NPDataSets/WBPOE/yVectorByRankBinaryFeaturesWBPOEBiasNoZero.p', 'wb')
        pickle.dump(y, yVector)
    elif filterType == r'Self-Play':
        h5f = h5py.File(path + r'POEBias.hdf5', 'w', driver='core')
        h5f.create_dataset(r'X', data=X)
        h5f.create_dataset(r'y', data=y)
        h5f.close()
    else:
        print ("Error: You must specify a valid Filter")

def filter_training_examples_and_labels(player_list, filter, NNType='ANN'):
    if filter == 'Win Ratio':
        training_data = filter_by_win_ratio(player_list)
    elif filter == 'Rank':
        training_data = filter_by_rank(player_list)
    elif filter == 'Self-Play':
        training_data = filter_for_self_play(player_list, NNType)
    else:
        training_data = None
        print('Invalid Filter Specified')
        exit(-2)
    training_examples, labels = split_data_to_training_examples_and_labels_for_CNN(training_data, NNType)
    return np.array(training_examples, dtype=np.float32), np.array(labels, dtype=np.float32)

def filter_by_rank(playerList):
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

def filter_by_win_ratio(playerList):
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

def filter_for_self_play(self_play_data, NNType, game_stage='End'):
    training_data = []
    if NNType == 'Policy':
        for self_play_log in self_play_data:
            for game in self_play_log['Games']:
              states = game['BoardStates']['PlayerPOV']
              mirror_states = game['MirrorBoardStates']['PlayerPOV']
              assert len(states) == len(mirror_states)
              for i in range(0, len(states)):
                  training_data.append(states[i])
                  training_data.append(mirror_states[i])
    elif NNType == 'Value':
        for self_play_log in self_play_data:
            for game in self_play_log['Games']: #TODO: experiment with the randomness
                game_length = len(game['BoardStates']['PlayerPOV'])
                if game_stage == None:
                    num_random_states = game_length // 10  # 10% of the states of each game
                    states = random.sample(game['BoardStates']['PlayerPOV'], num_random_states)
                    mirror_states = random.sample(game['MirrorBoardStates']['PlayerPOV'], num_random_states)
                else:
                    if game_stage == 'Start':
                        # Start-Game Value Net
                        start = 0
                        end = game_length // 3
                    elif game_stage == 'Mid':
                        # Mid-Game Value Net
                        start = game_length//3
                        end = (game_length // 3) * 2
                    elif game_stage == 'End':
                        # End-Game Value Net
                        start = (game_length//3) * 2
                        end = game_length
                    temp_states = game['BoardStates']['PlayerPOV'][start:end]
                    temp_mirror_states = game['MirrorBoardStates']['PlayerPOV'][start:end]
                    num_random_states = len(temp_states)//4 #25% of moves
                    #TODO: sample some random # of states in corresponding 3rd of data
                    #if num_random_states == len(states), mixes state order to decorrelate NN training examples
                    states = random.sample(temp_states, num_random_states)
                    mirror_states = random.sample(temp_mirror_states, num_random_states)
                training_data.extend(states)
                training_data.extend(mirror_states)

    else:
        print('Invalid NN for self-play')
        exit(-1)
    print('# of States for Self-Play {NNType} Net: {states}'.format(states=len(training_data), NNType=NNType))
    return training_data

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
         formatted_label = label_for_policy(transition_vector=training_example[2])
    else: # Value Net
         formatted_label = label_for_value(win=training_example[1])
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
    
def label_for_policy(transition_vector, one_hot_indexes=False):
    # if one_hot_indexes:
    #     # x still a 1d array; must stay one-hotted to be reshaped properly
    #     formatted_example = np.array([index for index, has_piece in enumerate(formatted_example) if has_piece == 1], dtype=np.float32)
    #     # if we just want the transition index vs a pre-one-hotted vector
    #     formatted_label = label.index(1)  # assumes no errors, y has a single value of 1
    formatted_label = np.array(transition_vector, dtype=np.float32)  # One hot transition vector
    return formatted_label

def label_for_value(win):
    if win:
        formatted_label = 1
        complement = 0
    else:
        formatted_label = 0
        complement = 1
    return np.array([formatted_label, complement], dtype=np.float32)

    


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


def GenerateCSV(X, isX = True):
    if isX == True:
     np.savetxt(r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/value_net_rank_binary/NPDataSets/WBPOEUnshuffledBinaryFeaturePlanesWBPOETrainingExamples.csv', X, delimiter=',', fmt='%1i')
    else:
        np.savetxt(
            r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/value_net_rank_binary/NPDataSets/WBPOE/UnshuffledBinaryFeaturePlanesWBPOETrainingExampleOutcomes.csv',
            X, delimiter=',', fmt='%1i')
def SplitListInHalf(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]
def PreprocessXLSX(X):
    XLSX_X = []
    X = filter_by_rank(X)
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
        writer = pd.ExcelWriter(r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/value_net_rank_binary/NPDataSets/WBPOE/UnshuffledBinaryFeaturePlanesDataset1.xlsx', engine='xlsxwriter')
    else:
        writer = pd.ExcelWriter(r'/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/value_net_rank_binary/NPDataSets/WBPOE/UnshuffledBinaryFeaturePlanesDataset2.xlsx', engine='xlsxwriter')

    frame.to_excel(writer, 'Sheet1')
    writer.save()
def assign_filter(fileName):
    if fileName == r'PlayerDataPythonDataSetsorted.p':
        filter = r'Rank'
    elif fileName == r'PlayerDataBinaryFeaturesWBPOEDataSetsorted.p':
        filter = r'Binary Rank'
    else:
        filter = r'UNDEFINED'
    return filter
def assign_path(deviceName ='Workstation'):
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

def self_player_driver(filter, NNType, path, fileName):
    file = open(os.path.join(path, fileName), 'r+b')
    player_list = pickle.load(file)
    file.close()
    training_examples, labels = filter_training_examples_and_labels(player_list, filter, NNType)
    write_path = os.path.join(path,"NumpyArrays",'4DArraysHDF5(RxCxF)POE{NNType}Net3rdThird25%DataSet'.format(NNType=NNType), fileName[0:-len(r'DataPython.p')])
    write_np_array_to_disk(write_path, training_examples, labels, filter, NNType)
