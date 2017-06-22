import numpy as np
from scipy import stats
from statsmodels.stats.proportion import multinomial_proportions_confint, proportion_confint
from scipy.stats import multinomial
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os
from time import time
import random
import h5py
from monte_carlo_tree_search.MCTS import MCTS, NeuralNetsCombined_128, NeuralNet
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

def generate_prediction_statistics(examples, labels, file_to_write, policy_net, color=None):


    correct_prediction_position = [0] * 155
    correct_position_in_10_percent_of_top = [0] * 155
    correct_rank_count = []
    not_in_top_n = 0
    incorrect_position_in_10_percent_of_top = [0] * 155

    num_top_moves = 10

    #TODO: hit rate / false positives if we include all this rank, within top % etc.
    times = []

    # for example in examples:
    #     start_time = time()
    #     policy_net.evaluate([example], color, already_converted=True)
    #     time_length = time()-start_time
    #     times.append(time_length)
    # print(sum(times)/len(times))
    #
    # time_examples = examples[:1]
    # start_time = time()
    # labels_predictions = policy_net.evaluate(time_examples, color, already_converted=True)
    # time_length = time() - start_time
    # print(time_length)
    #
    # time_examples = examples[:1]
    # start_time = time()
    # labels_predictions = policy_net.evaluate(time_examples, color, already_converted=True)
    # time_length = time() - start_time
    # print(time_length)
    #
    # time_examples = examples[:4]
    # start_time = time()
    # labels_predictions = policy_net.evaluate(time_examples, color, already_converted=True)
    # time_length = time() - start_time
    # print(time_length)
    #
    # time_examples = examples[:16]
    # start_time = time()
    # labels_predictions = policy_net.evaluate(time_examples, color, already_converted=True)
    # time_length = time() - start_time
    # print(time_length)
    #
    # time_examples = examples[:64]
    # start_time = time()
    # labels_predictions = policy_net.evaluate(time_examples, color, already_converted=True)
    # time_length = time() - start_time
    # print(time_length)
    #
    # time_examples = examples[:256]
    # start_time = time()
    # labels_predictions = policy_net.evaluate(time_examples, color, already_converted=True)
    # time_length = time() - start_time
    # print(time_length)
    #
    # time_examples = examples[:1024]
    # start_time = time()
    # labels_predictions = policy_net.evaluate(time_examples, color, already_converted=True)
    # time_length = time() - start_time
    # print(time_length)
    #
    # time_examples = examples[:4096]
    # start_time = time()
    # labels_predictions = policy_net.evaluate(time_examples, color, already_converted=True)
    # time_length = time() - start_time
    # print(time_length)
    #
    # time_examples = examples[:16384]
    # start_time = time()
    # labels_predictions = policy_net.evaluate(time_examples, color, already_converted=True)
    # time_length = time() - start_time
    # print(time_length)
    #
    # exit(11)
    labels_predictions = policy_net.evaluate(examples, color, already_converted=True)

    for example_num in range(0, len(labels_predictions)):
        correct_move_index = np.argmax(labels[example_num])
        predicted_move_index = np.argmax(labels_predictions[example_num])
        top_n_indexes = sorted(range(len(labels_predictions[example_num])),
                               key=lambda i: labels_predictions[example_num][i], reverse=True)[
                        :
                        # num_top_moves
                        ]
        if (correct_move_index in top_n_indexes):
            rank_in_prediction = top_n_indexes.index(correct_move_index)  # 0 indexed
            correct_prediction_position[rank_in_prediction] += 1
            correct_rank_count.append(rank_in_prediction + 1)

            top_move_predicted_prob = labels_predictions[example_num][predicted_move_index] * 100
            correct_move_predicted_prob = labels_predictions[example_num][correct_move_index] * 100
            if top_move_predicted_prob - correct_move_predicted_prob < n_percent_of_top:
                correct_position_in_10_percent_of_top[rank_in_prediction] += 1

            for i in range(0, len(labels_predictions[example_num])):
                move_prob = labels_predictions[example_num][i]*100
                if move_prob != top_move_predicted_prob and move_prob!= correct_move_predicted_prob:
                    if top_move_predicted_prob-move_prob < n_percent_of_top:
                        incorrect_position_in_10_percent_of_top[top_n_indexes.index(i)] += 1
                # if correct_move_index != predicted_move_index:
                #     print("Incorrect prediction. Correct move was ranked {}".format(rank_in_prediction+1), file=file_to_write)
                #     top_move_predicted_prob = labels_predictions[example_num][predicted_move_index] * 100
                #     correct_move_predicted_prob = labels_predictions[example_num][correct_move_index] * 100
                #     print("Predicted move probability = %{pred_prob}. \n"
                #           "Correct move probability = %{correct_prob}\n"
                #           "Difference = %{prob_diff}\n".format(pred_prob=top_move_predicted_prob,
                #                                                correct_prob=correct_move_predicted_prob,
                #                                                prob_diff= top_move_predicted_prob-correct_move_predicted_prob),
                #           file=file_to_write)
                # else:
                #     print("Correct prediction.",
                #           file=file_to_write)
                #     correct_move_predicted_prob = labels_predictions[example_num][correct_move_index] * 100
                #     print("Correct move probability = %{correct_prob}   "
                #           "Difference = %{prob_diff}\n".format(correct_prob=correct_move_predicted_prob,
                #                                                prob_diff=100 - correct_move_predicted_prob),
                #           file=file_to_write)

                # in_top_n = True
        else:
            # rank_in_prediction = in_top_n = False
            not_in_top_n += 1

    return correct_prediction_position, correct_position_in_10_percent_of_top, incorrect_position_in_10_percent_of_top, not_in_top_n

def print_prediction_statistics(correct_prediction_position, correct_position_in_10_percent_of_top, incorrect_position_in_10_percent_of_top, not_in_top_n, file_to_write):
    total_predictions = sum(correct_prediction_position) + not_in_top_n
    total_incorrect_top_10_predictions = sum(incorrect_position_in_10_percent_of_top)
    incorrect_prediction_position = [abs(y) for y in [sum(x) for x in zip(correct_prediction_position, [-total_predictions]*len(correct_prediction_position))]]


    print("Length of Training Set = {num_examples}\n".format(num_examples=total_predictions),
          file=file_to_write)

    ranks_count_greater_than_5 = 0
    for i in range(0, len(correct_prediction_position)):
        if correct_prediction_position[i] >= 5:
            ranks_count_greater_than_5 = i + 1
        else:
            break
    conf_ints = multinomial_proportions_confint(correct_prediction_position[
                                                :ranks_count_greater_than_5])  # tests only valid if counts > 5; logs show all greater than 21 <5
    percent_in_top = 0
    num_in_top = 0
    num_if_included = 0
    rank_percent_array = [0] * 155
    num_total_trials = total_predictions
    for i in range(0, len(correct_prediction_position)):
        rank_percent = (correct_prediction_position[i] * 100) / total_predictions
        print("Correct move was in predicted rank {num} slot = %{percent}".format(num=i + 1, percent=rank_percent),
              file=file_to_write)
        # incorrect_rank_percent = (incorrect_prediction_position[i] * 100) / total_predictions
        # print("incorrect move was in predicted rank {num} slot = %{percent}".format(num=i + 1, percent=incorrect_rank_percent),
        #       file=file_to_write)
        rank_percent_array[i] = rank_percent
        percent_in_top += rank_percent
        num_in_top += correct_prediction_position[i]
        num_if_included += correct_position_in_10_percent_of_top[i]
        if i < ranks_count_greater_than_5:
            lower = conf_ints[i][0]
            upper = conf_ints[i][1]
            distance = (upper - lower) / 2
            center = rank_percent / 100  # upper-distance
            upper_diff = upper - center
            lower_diff = center - lower
            print("Mean= {center}%; \n"
                  "95% CI {lower}% to {upper}% \n"
                  "+{upper_diff}%; -{lower_diff}%; \n"
                  "+- {avg_diff}%\n".format(center=center * 100, lower=lower * 100, upper=upper * 100,
                                            upper_diff=upper_diff * 100, lower_diff=lower_diff * 100,
                                            avg_diff=distance * 100), file=file_to_write)

        print("Percent in top {num} predictions = %{percent}".format(num=i + 1, percent=percent_in_top),
              file=file_to_write)
        multi = False
        if multi:
            if i < ranks_count_greater_than_5:
                # frame cumulative prob as one category vs rest.
                cum_correct = correct_prediction_position[
                              i:ranks_count_greater_than_5]  # copy this rank and qualified ranks ahead
                cum_correct[0] = num_in_top  # replace this rank val with cumulative rank val
                conf_int_cum = multinomial_proportions_confint(cum_correct)
                lower_cum = conf_int_cum[0][0]
                upper_cum = conf_int_cum[0][1]

                distance_cum = (upper_cum - lower_cum) / 2
                center_cum = percent_in_top / 100  # upper_cum-distance_cum
                upper_diff_cum = upper_cum - center_cum
                lower_diff_cum = center_cum - lower_cum
                print("Mean = {center}% \n"
                      "95% CI {lower}% to {upper}% \n"
                      "+{upper_diff}%; -{lower_diff}%; \n"
                      "+- {avg_distance}%\n".format(center=center_cum * 100, lower=lower_cum * 100,
                                                    upper=upper_cum * 100,
                                                    upper_diff=upper_diff_cum * 100, lower_diff=lower_diff_cum * 100,
                                                    avg_distance=distance_cum * 100), file=file_to_write)
        else:
            # frame as a binomial, either is in top n or it isn't.
            lower_cum, upper_cum = proportion_confint(num_in_top, num_total_trials)
            distance_cum = (upper_cum - lower_cum) / 2
            center_cum = percent_in_top / 100  # upper_cum-distance_cum
            upper_diff_cum = upper_cum - center_cum
            lower_diff_cum = center_cum - lower_cum
            print("Mean = {center}% \n"
                  "95% CI {lower}% to {upper}% \n"
                  "+{upper_diff}%; -{lower_diff}%; \n"
                  "+- {avg_distance}%\n".format(center=center_cum * 100, lower=lower_cum * 100, upper=upper_cum * 100,
                                                upper_diff=upper_diff_cum * 100, lower_diff=lower_diff_cum * 100,
                                                avg_distance=distance_cum * 100), file=file_to_write)

        percent_in_top_10 = (correct_position_in_10_percent_of_top[i] * 100) / max(1, correct_prediction_position[i])
        print("Predicted rank {num} slot is correct and within {n_percent}% of top move = %{percent} of the time".format(num=i + 1,n_percent = n_percent_of_top,
                                                                                                 percent=percent_in_top_10),
              file=file_to_write)

        num_correct = correct_position_in_10_percent_of_top[i]
        num_trials = correct_prediction_position[i]
        if num_trials > 0:
            lower_10p, upper_10p = proportion_confint(num_correct, num_trials)
            # lower_10p = binom_conf_int_10p[0]
            # upper_10p = binom_conf_int_10p[1]
            distance_10p = (upper_10p - lower_10p) / 2
            center_10p = percent_in_top_10 / 100  # upper_10p-distance_10p
            upper_diff_10p = upper_10p - center_10p
            lower_diff_10p = center_10p - lower_10p
            print("Mean = {center}% \n"
                  "95% CI {lower}% to {upper}% \n"
                  "+{upper_diff}%; -{lower_diff}%; \n"
                  "+- {avg_distance}%\n".format(center=center_10p * 100, lower=lower_10p * 100, upper=upper_10p * 100,
                                                upper_diff=upper_diff_10p * 100, lower_diff=lower_diff_10p * 100,
                                                avg_distance=distance_10p * 100), file=file_to_write)

        percent_in_top_10 = (incorrect_position_in_10_percent_of_top[i] * 100) / max(1,
                                                                                     incorrect_prediction_position[i])

        print(
            "Predicted rank {num} slot is incorrect and within {n_percent}% of top move = %{percent} of the time".format(
                num=i + 1, n_percent=n_percent_of_top,
                percent=percent_in_top_10),
            file=file_to_write)

        num_incorrect = incorrect_position_in_10_percent_of_top[i]
        num_trials = incorrect_prediction_position[i]
        if num_trials > 0:
            lower_10p, upper_10p = proportion_confint(num_incorrect, num_trials)
            # lower_10p = binom_conf_int_10p[0]
            # upper_10p = binom_conf_int_10p[1]
            distance_10p = (upper_10p - lower_10p) / 2
            center_10p = percent_in_top_10 / 100  # upper_10p-distance_10p
            upper_diff_10p = upper_10p - center_10p
            lower_diff_10p = center_10p - lower_10p
            print("Mean = {center}% \n"
                  "95% CI {lower}% to {upper}% \n"
                  "+{upper_diff}%; -{lower_diff}%; \n"
                  "+- {avg_distance}%\n".format(center=center_10p * 100, lower=lower_10p * 100, upper=upper_10p * 100,
                                                upper_diff=upper_diff_10p * 100, lower_diff=lower_diff_10p * 100,
                                                avg_distance=distance_10p * 100), file=file_to_write)

        times_rank_in_top_10 = correct_position_in_10_percent_of_top[i]+incorrect_position_in_10_percent_of_top[i]

        likelihood_top_10_is_correct = (correct_position_in_10_percent_of_top[i] * 100) / max(1, times_rank_in_top_10)
        print("Including within {n_percent}% probability from rank {num} slot hit rate = %{percent}".format(num=i + 1,n_percent=n_percent_of_top,
                                                                                                           percent=likelihood_top_10_is_correct),
              file=file_to_write)

        likelihood_top_10_is_incorrect = (incorrect_position_in_10_percent_of_top[i] * 100) / max(1, times_rank_in_top_10)
        print("Including within {n_percent}% probability from rank {num} slot false-positive rate = %{percent}".format(num=i + 1,n_percent=n_percent_of_top,
                                                                                                 percent=likelihood_top_10_is_incorrect),
              file=file_to_write)


        
        percent_including_top_10 = num_if_included * 100 / total_predictions
        print(
            "Percent if we include rank {num} predictions with probability difference <{n_percent} = %{percent}".format(num=i + 1,n_percent=n_percent_of_top,
                                                                                                             percent=percent_including_top_10),
            file=file_to_write)

        # frame as a binomial, either is in top n or it isn't.
        lower_within_10_p, upper_within_10_p = proportion_confint(num_if_included, num_total_trials)
        distance_within_10_p = (upper_within_10_p - lower_within_10_p) / 2
        center_within_10_p = percent_including_top_10 / 100  # upper_within_10_p-distance_within_10_p
        upper_diff_within_10_p = upper_within_10_p - center_within_10_p
        lower_diff_within_10_p = center_within_10_p - lower_within_10_p
        print("Mean = {center}% \n"
              "95% CI {lower}% to {upper}% \n"
              "+{upper_diff}%; -{lower_diff}%; \n"
              "+- {avg_distance}%\n".format(center=center_within_10_p * 100, lower=lower_within_10_p * 100,
                                            upper=upper_within_10_p * 100,
                                            upper_diff=upper_diff_within_10_p * 100,
                                            lower_diff=lower_diff_within_10_p * 100,
                                            avg_distance=distance_within_10_p * 100), file=file_to_write)
        print("\n----------------------------------------------------------------------\n", file=file_to_write)



    print("Percentage of time not in predictions = {}".format((not_in_top_n * 100) / total_predictions),
          file=file_to_write)
    # values, base = np.histogram(correct_rank_count, bins=155)
    # # evaluate the cumulative
    # cumulative = np.cumsum(values)
    # # plot the cumulative function
    # plt.plot(base[:-1], cumulative, c='red')
    #
    # plt.show()

n_percent_of_top = 100



averaged = False

policy_net = NeuralNetsCombined_128()
# policy_net = NeuralNet()
for color in ['White', 'Black']:
    for game_stage in [
        # '1stThird',
        # '2ndThird',
        # '3rdThird',
        'AllThird'
    ]:
        input_path = r'G:\TruncatedLogs\PythonDatasets\Datastructures\NumpyArrays\{net_type}\{features}\4DArraysHDF5(RxCxF){features}{net_type}{game_stage}{color}'.format(features='POE', net_type='PolicyNet', game_stage=game_stage, color=color)
        valid_examples, valid_labels = load_examples_and_labels(os.path.join(input_path, r'ValidationData'))
        training_examples, training_labels = load_examples_and_labels(os.path.join(input_path, r'TrainingData'))
        print (len(training_examples))
        break
        # training_example_batches, training_label_batches = utils.batch_split(training_examples, training_labels, 16384)
        if averaged:
            if color=='White':
                correct_prediction_position_white, correct_position_in_10_percent_of_top_white, incorrect_position_in_10_percent_of_top_white, not_in_top_n_white \
                    = generate_prediction_statistics(valid_examples, valid_labels, sys.stdout, policy_net, color)
            elif color =='Black':
                correct_prediction_position_black, correct_position_in_10_percent_of_top_black, incorrect_position_in_10_percent_of_top_black, not_in_top_n_black \
                    = generate_prediction_statistics(valid_examples, valid_labels, sys.stdout, policy_net, color)
        else:
            with open(os.path.join(r'..', r'policy_net_model', r'metrics', '05012017_065Net_VALID_PolicyNetPredictionMetrics{n_percent}Prob_{game_stage}{color}.txt'.format(n_percent=n_percent_of_top,game_stage=game_stage, color=color)), 'w') as output_file:
                print("Length of Test Set = {num_examples}\n".format(num_examples=len(valid_examples)),
                      file=output_file)
                correct_prediction_position, correct_position_in_10_percent_of_top, incorrect_position_in_10_percent_of_top,not_in_top_n = \
                    generate_prediction_statistics(training_examples, training_labels, output_file, policy_net, color)
                # correct_prediction_position, correct_position_in_10_percent_of_top, incorrect_position_in_10_percent_of_top,not_in_top_n = \
                #     generate_prediction_statistics(valid_examples, valid_labels, output_file, policy_net, color)

                print_prediction_statistics(correct_prediction_position, correct_position_in_10_percent_of_top, incorrect_position_in_10_percent_of_top, not_in_top_n, output_file)
if averaged:
    with open(os.path.join(r'..', r'policy_net_model', r'metrics',
                           '05012017_065Net_VALID_PolicyNetPredictionMetrics{n_percent}Prob_{game_stage}Averaged.txt'.format(n_percent=n_percent_of_top,
                               game_stage=game_stage)), 'w') as output_file:
        correct_prediction_position = [sum(x) for x in
                                       zip(correct_prediction_position_white, correct_prediction_position_black)]
        correct_position_in_10_percent_of_top = [sum(x) for x in zip(correct_position_in_10_percent_of_top_white,
                                                                     correct_position_in_10_percent_of_top_black)]
        incorrect_position_in_10_percent_of_top = [sum(x) for x in zip(incorrect_position_in_10_percent_of_top_white,
                                                                     incorrect_position_in_10_percent_of_top_black)]
        not_in_top_n = not_in_top_n_white + not_in_top_n_black
        print_prediction_statistics(correct_prediction_position, correct_position_in_10_percent_of_top, incorrect_position_in_10_percent_of_top,not_in_top_n, output_file)

