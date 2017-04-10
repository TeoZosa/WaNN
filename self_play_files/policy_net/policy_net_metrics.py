import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
import h5py
from monte_carlo_tree_search.MCTS import MCTS, NeuralNetsCombined, NeuralNet
from tools import utils

def load_examples_and_labels(path):
    files = [file_name for file_name in utils.find_files(path, "*.hdf5")]
    examples, labels = ([] for i in range (2))
    for file in files:  # add training examples and corresponding labels from each file to associated arrays
        training_data = h5py.File(file, 'r')
        examples.extend(training_data['X'][:])
        labels.extend(training_data['y'][:])
        training_data.close()
    return np.array(examples, dtype=np.float32), np.array(labels, dtype=np.float32)  # b x r x c x f & label



def print_prediction_statistics(examples, labels, file_to_write, policy_net, color=None):
    correct_prediction_position = [0] * 155
    correct_rank_count = []
    not_in_top_10 = 0

    num_top_moves = 10
    labels_predictions = policy_net.evaluate(examples, color, already_converted=True)
    for example_num in range (0, len(labels_predictions)):
        correct_move_index = np.argmax(labels[example_num])
        predicted_move_index = np.argmax(labels_predictions[example_num])
        top_n_indexes = sorted(range(len(labels_predictions[example_num])), key=lambda i: labels_predictions[example_num][i], reverse=True)[
                        :
                        # num_top_moves
                        ]
        if (correct_move_index in top_n_indexes):
            rank_in_prediction = top_n_indexes.index(correct_move_index) # 0 indexed
            correct_prediction_position[rank_in_prediction] += 1
            correct_rank_count.append(rank_in_prediction+1)
            if correct_move_index != predicted_move_index:
                print("Incorrect prediction. Correct move was ranked {}".format(rank_in_prediction+1), file=file_to_write)
                top_move_predicted_prob = labels_predictions[example_num][predicted_move_index] * 100
                correct_move_predicted_prob = labels_predictions[example_num][correct_move_index] * 100
                print("Predicted move probability = %{pred_prob}. \n"
                      "Correct move probability = %{correct_prob}\n"
                      "Difference = %{prob_diff}\n".format(pred_prob=top_move_predicted_prob,
                                                           correct_prob=correct_move_predicted_prob,
                                                           prob_diff= top_move_predicted_prob-correct_move_predicted_prob),
                      file=file_to_write)
            else:
                print("Correct prediction.",
                      file=file_to_write)
                correct_move_predicted_prob = labels_predictions[example_num][correct_move_index] * 100
                print("Correct move probability = %{correct_prob}   "
                      "Difference = %{prob_diff}\n".format(correct_prob=correct_move_predicted_prob,
                                                           prob_diff=100 - correct_move_predicted_prob),
                      file=file_to_write)

            # in_top_n = True
        else:
            # rank_in_prediction = in_top_n = False
            not_in_top_10 += 1


    total_predictions = sum(correct_prediction_position) + not_in_top_10
    percent_in_top = 0
    for i in range (0, len(correct_prediction_position)):
        rank_percent = (correct_prediction_position[i]*100)/total_predictions
        print("Correct move was in predicted rank {num} slot = %{percent}".format(num=i+1, percent=rank_percent), file=file_to_write)
        percent_in_top += rank_percent
        print("Percent in top {num} predictions = %{percent}\n".format(num=i + 1, percent=percent_in_top), file=file_to_write)
    print("Percentage of time not in predictions = {}".format((not_in_top_10*100)/total_predictions), file=file_to_write)

    # values, base = np.histogram(correct_rank_count, bins=155)
    # # evaluate the cumulative
    # cumulative = np.cumsum(values)
    # # plot the cumulative function
    # plt.plot(base[:-1], cumulative, c='red')
    #
    # plt.show()



    # top_n_white_moves = list(map(lambda index: utils.move_lookup_by_index(index, 'White'), top_n_indexes))
    # top_n_black_moves = list(map(lambda index: utils.move_lookup_by_index(index, 'Black'), top_n_indexes))
    # print("Sample Predicted Probabilities = "
    #       "\n{y_pred}"
    #       "\nIn top {num_top_moves} = {in_top_n}"
    #       "\nTop move's rank in prediction = {move_rank}"
    #       "\nTop {num_top_moves} moves ranked first to last = "
    #       "\nif White: \n {top_white_moves}"
    #       "\nif Black: \n {top_black_moves}"
    #       "\nActual Move = "
    #       "\nif White: {y_act_white}"
    #       "\nif Black: {y_act_black}".format(
    #     y_pred=labels_predictions,
    #     in_top_n=in_top_n,
    #     num_top_moves=num_top_moves,
    #     move_rank=rank_in_prediction,
    #     top_white_moves=top_n_white_moves,
    #     top_black_moves=top_n_black_moves,
    #     y_pred_white=utils.move_lookup_by_index(predicted_move_index, 'White'),
    #     y_pred_black=utils.move_lookup_by_index(predicted_move_index, 'Black'),
    #     y_act_white=utils.move_lookup_by_index(correct_move_index, 'White'),
    #     y_act_black=utils.move_lookup_by_index(correct_move_index, 'Black')),
    #     end="\n", file=file_to_write)

dual_nets = True

if dual_nets:
    policy_net = NeuralNetsCombined()
    for color in ['White', 'Black']:
        for game_stage in ['1stThird', '2ndThird', '3rdThird', 'AllThird']:
            input_path = r'G:\TruncatedLogs\PythonDatasets\Datastructures\NumpyArrays\{net_type}\{features}\4DArraysHDF5(RxCxF){features}{net_type}{game_stage}{color}'.format(features='POE', net_type='PolicyNet', game_stage=game_stage, color=color)
            training_examples, training_labels = load_examples_and_labels(os.path.join(input_path, r'TrainingData'))
            # training_example_batches, training_label_batches = utils.batch_split(training_examples, training_labels, 16384)
            with open(os.path.join(r'..', r'policy_net_model', r'metrics', '04102017PolicyNetPredictionMetrics{game_stage}{color}.txt'.format(game_stage=game_stage, color=color)), 'w') as output_file:
                print_prediction_statistics(training_examples, training_labels, output_file, policy_net, color)

else:
    policy_net = NeuralNet
    for game_stage in ['1stThird', '2ndThird', '3rdThird', 'AllThird']:
        input_path = r'G:\TruncatedLogs\PythonDatasets\Datastructures\NumpyArrays\{net_type}\{features}\4DArraysHDF5(RxCxF){features}{net_type}{game_stage}'.format(
            features='POE', net_type='PolicyNet', game_stage=game_stage)
        training_examples, training_labels = load_examples_and_labels(os.path.join(input_path, r'TrainingData'))
        # training_example_batches, training_label_batches = utils.batch_split(training_examples, training_labels, 16384)
        with open(os.path.join(r'..', r'policy_net_model', r'metrics',
                               '04102017PolicyNetPredictionMetrics{}.txt'.format(game_stage)), 'w') as output_file:
            print_prediction_statistics(training_examples, training_labels, output_file, policy_net)
