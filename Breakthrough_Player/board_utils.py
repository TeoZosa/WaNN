import copy

import pandas as pd
import numpy as np
# from Breakthrough_Player.policy_net_utils import call_policy_net

def generate_policy_net_moves(game_board, player_color):
    board_representation = convert_board_to_2d_matrix_POE(game_board, player_color)
    # return call_policy_net(board_representation)

def convert_board_to_2d_matrix_POE(game_board, player_color):
    one_hot_board = np.array([generate_binary_vector(game_board, player_color=player_color,
                                                     what_to_filter='Player'),  # [0] player
                     generate_binary_vector(game_board, player_color=player_color,
                                            what_to_filter='Opponent'),  # [1] opponent
                     generate_binary_vector(game_board, player_color=player_color,
                                            what_to_filter='Empty'),  #[2] empty
                     generate_binary_vector(game_board, player_color=player_color,
                                            what_to_filter='Bias')], dtype=np.float32)  # [3] bias
    one_hot_board = one_hot_board.ravel() #1d board

    formatted_example = np.reshape(np.array(one_hot_board, dtype=np.float32),
                                   (len(one_hot_board) // 64, 8, 8))  # feature_plane x row x col

    for i in range(0, len(formatted_example)):
        formatted_example[i] = formatted_example[
            i].transpose()  # transpose (row x col) to get feature_plane x col x row
    formatted_example = formatted_example.transpose()  # transpose to get proper dimensions: row x col  x feature plane

    return np.array(formatted_example, dtype=np.float32)

#Note: not the same as generate_binary_vector in self_play_logs_to_datastructures;
def generate_binary_vector(state, player_color, what_to_filter):
    bias = 1
    if player_color == 'Black':
        state = reflect_board_state(state) #reversed representation for black to ensure POV representation
    panda_board = pd.DataFrame(state).transpose().sort_index(ascending=False).iloc[2:]  # human-readable board
    if what_to_filter == 'White' or what_to_filter == 'Black':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 1,
                'b': 0},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 1}}
    elif what_to_filter == 'Player':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 1,
                'b': 0},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 1}}
    elif what_to_filter == 'Opponent':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 1},
            'Black': {
                'e': 0,
                'w': 1,
                'b': 0}}
    elif what_to_filter == 'Empty':
        what_to_filter_dict = {
            'White': {
                'e': 1,
                'w': 0,
                'b': 0},
            'Black': {
                'e': 1,
                'w': 0,
                'b': 0}}
    elif what_to_filter == 'Capture Move':
        what_to_filter_dict = {
            'White': {
                'e': 1,
                'w': 1,
                'b': 1},
            'Black': {
                'e': 1,
                'w': 1,
                'b': 1}}
    elif what_to_filter == 'Non-Capture Move':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 0},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 0}}
    elif what_to_filter == 'Moves From':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 1,
                'Player Capture To': 0,
                'Player Capture From': 0},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 1,
                'Player Capture To': 0,
                'Player Capture From': 0}}
    elif what_to_filter == 'Moves To':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 1,
                'Player Move From': 0,
                'Player Capture To': 0,
                'Player Capture From': 0},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 1,
                'Player Move From': 0,
                'Player Capture To': 0,
                'Player Capture From': 0}}
    elif what_to_filter == 'Captures From':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 0,
                'Player Capture To': 0,
                'Player Capture From': 1},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 0,
                'Player Capture To': 0,
                'Player Capture From': 1}}
    elif what_to_filter == 'Captures To':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 0,
                'Player Capture To': 1,
                'Player Capture From': 0},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 0,
                'Player Capture To': 1,
                'Player Capture From': 0}}
    elif what_to_filter == 'Bias':  # duplicate across 64 positions since CNN needs same dimensions
        what_to_filter_dict = {
            'White': {
                'e': bias,
                'w': bias,
                'b': bias},
            'Black': {
                'e': bias,
                'w': bias,
                'b': bias}}
    else:
        print("Error, generate_binary_vector needs a valid argument to filter")
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 0,
                'Player Capture To': 0,
                'Player Capture From': 0},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 0,
                'Player Capture To': 0,
                'Player Capture From': 0}}
    for row in sorted(state):
        if row != is_white_index and row != white_move_index:  # don't touch these indexes
            for column in sorted(state[row]):  # needs to be sorted to traverse dictionary in lexicographical order
                binary_vector.append(what_to_filter_dict[player_color][state[row][column]])
    return binary_vector


def initial_game_board():
    empty = 'e'
    white = 'w'
    black = 'b'
    return {
        10: -1,  # (-1 for initial state, 0 if black achieved state, 1 if white achieved state)
        # equivalent to 0 if white's move, 1 if black's move
        9: 1,  # is player_color white
        8: {'a': black, 'b': black, 'c': black, 'd': black, 'e': black, 'f': black, 'g': black, 'h': black},
        7: {'a': black, 'b': black, 'c': black, 'd': black, 'e': black, 'f': black, 'g': black, 'h': black},
        6: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
        5: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
        4: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
        3: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
        2: {'a': white, 'b': white, 'c': white, 'd': white, 'e': white, 'f': white, 'g': white, 'h': white},
        1: {'a': white, 'b': white, 'c': white, 'd': white, 'e': white, 'f': white, 'g': white, 'h': white}
    }

#Note: not the same as move_piece in self_play_logs_to_datastructures; is white index now changes as we are sharing a board,
#so player U opponent = self_play_log_board with is_white_index changing based on who owns the board
def move_piece(board_state, move, whose_move):
    empty = 'e'
    white_move_index = 10
    is_white_index = 9
    move = move.split('-')
    _from = move[0].lower()
    to = move[1].lower()
    next_board_state = copy.deepcopy(board_state)  # edit copy of board_state; don't need this for breakthrough_player?
    next_board_state[int(to[1])][to[0]] = next_board_state[int(_from[1])][_from[0]]
    next_board_state[int(_from[1])][_from[0]] = empty
    if whose_move == 'White':
        next_board_state[white_move_index] = 1
        next_board_state[is_white_index] = 0 #next move isn't white's
    else:
        next_board_state[white_move_index] = 0
        next_board_state[is_white_index] = 1 #since black made this move, white makes next move
    return next_board_state

#Note: not the same as reflect_board_state in self_play_logs_to_datastructures;
def reflect_board_state(state):  # since black needs to have a POV representation
    semi_reflected_state = mirror_board_state(state)
    reflected_state = copy.deepcopy(semi_reflected_state)
    reflected_state[1] = semi_reflected_state[8]
    reflected_state[2] = semi_reflected_state[7]
    reflected_state[3] = semi_reflected_state[6]
    reflected_state[4] = semi_reflected_state[5]
    reflected_state[5] = semi_reflected_state[4]
    reflected_state[6] = semi_reflected_state[3]
    reflected_state[7] = semi_reflected_state[2]
    reflected_state[8] = semi_reflected_state[1]
    return reflected_state

#Note: not the same as mirror_board_state in self_play_logs_to_datastructures;
def mirror_board_state(state):  # helper method for reflect_board_state
    mirror_state = copy.deepcopy(state)  # edit copy of board_state
     # the board state; state[1] is the win or loss value, state [2] is the transition vector
    is_white_index = 9
    white_move_index = 10
    for row in sorted(state):
        if row != is_white_index and row != white_move_index:  # these indexes don't change
            for column in sorted(state[row]):
                if column == 'a':
                    mirror_state[row]['h'] = state[row][column]
                elif column == 'b':
                    mirror_state[row]['g'] = state[row][column]
                elif column == 'c':
                    mirror_state[row]['f'] = state[row][column]
                elif column == 'd':
                    mirror_state[row]['e'] = state[row][column]
                elif column == 'e':
                    mirror_state[row]['d'] = state[row][column]
                elif column == 'f':
                    mirror_state[row]['c'] = state[row][column]
                elif column == 'g':
                    mirror_state[row]['b'] = state[row][column]
                elif column == 'h':
                    mirror_state[row]['a'] = state[row][column]
    return mirror_state