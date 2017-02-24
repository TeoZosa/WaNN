import copy
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.framework.ops import reset_default_graph
from Breakthrough_Player.policy_net_utils import call_policy_net

def generate_policy_net_moves(game_board, player_color):
    board_representation = convert_board_to_2d_matrix_POE(game_board, player_color)
    return call_policy_net(board_representation)

def convert_board_to_2d_matrix_POE(game_board, player_color):
    one_hot_board = [generate_binary_vector(game_board, player_color=player_color,
                                            who_to_filter='Player'),  # [0] player
                     generate_binary_vector(game_board, player_color=player_color,
                                            who_to_filter='Opponent'),  # [1] opponent
                     generate_binary_vector(game_board, player_color=player_color,
                                            who_to_filter='Empty'), #[2] empty
                     generate_binary_vector(game_board, player_color=player_color,
                                           who_to_filter='Bias')]  # [3] bias
    one_hot_board = np.array(one_hot_board, dtype=np.float32)
    one_hot_board = one_hot_board.ravel() #1d board
    # # ensure at most 1 bit is on at each board position for player/opponent/empty
    # assert ((one_hot_board[0] ^ one_hot_board[1] ^ one_hot_board[2]).all().all() and
    #             not(one_hot_board[0] & one_hot_board[1] & one_hot_board[2]).all().all())

    formatted_example = np.reshape(np.array(one_hot_board, dtype=np.float32),
                                   (len(one_hot_board) // 64, 8, 8))  # feature_plane x row x col

    for i in range(0, len(formatted_example)):
        formatted_example[i] = formatted_example[
            i].transpose()  # transpose (row x col) to get feature_plane x col x row
    formatted_example = formatted_example.transpose()  # transpose to get proper dimensions: row x col  x feature plane

    return np.array(formatted_example, dtype=np.float32)

def generate_binary_vector(state, player_color, who_to_filter):
    bias = 1
    if player_color == 'Black':
        state = reflect_board_state(state) #reversed representation for black to ensure POV representation
    panda_board = pd.DataFrame(state).transpose().sort_index(ascending=False).iloc[2:]  # human-readable board
    if who_to_filter == 'White':
        who_dict = {
            'e': 0,
            'w': 1,
            'b': 0}
    elif who_to_filter == 'Black':
        who_dict = {
            'e': 0,
            'w': 0,
            'b': 1}
    elif who_to_filter == 'Player':
        if player_color == 'White':
            who_dict = {
                'e': 0,
                'w': 1,
                'b': 0}
        elif player_color == 'Black':
            who_dict = {
                'e': 0,
                'w': 0,
                'b': 1}
        else:
            print("error in convertBoard")
            exit(-190)
    elif who_to_filter == 'Opponent':
        if player_color == 'White':
            who_dict = {
                'e': 0,
                'w': 0,
                'b': 1}
        elif player_color == 'Black':
            who_dict = {
                'e': 0,
                'w': 1,
                'b': 0}
        else:
            print("error in convertBoard")
            exit(-190)
    elif who_to_filter == 'Empty':
        who_dict = {
            'e': 1,
            'w': 0,
            'b': 0}
    elif who_to_filter == 'Bias':  # duplicate across 64 positions since CNN needs same dimensions
        who_dict = {
            'e': bias,
            'w': bias,
            'b': bias}
    else:
        print("Error, generate_binary_vector needs a valid argument to filter")
    return np.array(panda_board.apply(lambda x: x.apply(lambda y: who_dict[y]))).ravel()#features x col x row

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

def generate_transition_vector(to, _from, player_color):
    # probability distribution over the 154 possible (vs legal) moves from the POV of the player.
    # Reasoning: six center columns where, if a piece was present, it could move one of three ways.
    # A piece in one of the two side columns can move one of two ways.
    # Since nothing can move in the farthest row, there are only seven rows of possible movement.
    # => (2*2*7) + (6*3*7) = 154
    # ==> 154 element vector of all 0s sans the 1 for the transition that was actually made.
    # i.e. a1-a2 (if White) == h8-h7 (if Black) =>
    # row 0 (closest row), column 0(farthest left)
    # moves to
    # row +1, column 0
    # <=> transition[0] = 1, transition[1:len(transition)] = 0

    # Notes: when calling NN, just reverse board state if black and decode output with black's table
    #1/14/17: since NN sees sorted r x c, we actually need to reverse if white..

    from_column = _from[0]
    to_column = to[0]
    from_row = int(_from[1])
    to_row = int(to[1])
    # ex if white and from_column is b => 1*3; moves starting from b are [2] or [3] or [4];
    column_offset = (ord(from_column) - ord('a')) * 3
    if player_color == 'Black':
        row_offset = (to_row - 1) * 22  # 22 possible moves per row
        assert (row_offset == (from_row - 2) * 22)  # double check
        index = 153 - (
        ord(to_column) - ord(from_column) + column_offset + row_offset)  # 153 reverses the board for black
    else:
        row_offset = (from_row - 1) * 22  # 22 possible moves per row
        assert (row_offset == (to_row - 2) * 22)  # double check
        index = ord(to_column) - ord(from_column) + column_offset + row_offset
    transition_vector = [0] * 155 #  last index is the no move index
    transition_vector[index] = 1
    return transition_vector

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