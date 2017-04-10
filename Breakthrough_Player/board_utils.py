import copy
from tools import utils
import numpy as np
import sys
import random
import pandas as pd
from Breakthrough_Player.policy_net_utils import call_policy_net, instantiate_session, instantiate_session_both, instantiate_session_black, instantiate_session_white

def generate_policy_net_moves(game_board, player_color):
    board_representation = utils.convert_board_to_2d_matrix_POEB(game_board, player_color)
    return call_policy_net(board_representation)

def get_NN(player_color=None):
    if player_color == 'White':
        return instantiate_session_white()
    elif player_color == 'Black':
        return instantiate_session_black()
    else:
        return instantiate_session_both()

def generate_policy_net_moves_batch(game_nodes, batch_size=16384):

    board_representations = [utils.convert_board_to_2d_matrix_POEB(node.game_board, node.color) for node in game_nodes]
    inference_batches = utils.batch_split_no_labels(board_representations, batch_size)
    output = []
    for batch in inference_batches:
        output.extend(call_policy_net(batch))
    return output


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
    if move[2] == '-':
        move = move.split('-')
    else:
        move = move.split('x')#x for wanderer captures.
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

def get_random_move(game_board, player_color):
    possible_moves = enumerate_legal_moves(game_board, player_color)
    random_move = random.sample(possible_moves, 1)[0]
    move = random_move['From'] + '-' + random_move['To']
    return move

def print_board(game_board, file=sys.stdout):
    new_piece_map = {
        'w': 'w',
        'b': 'b',
        'e': '_'
    }
    print((pd.DataFrame(game_board).transpose().sort_index(ascending=False).iloc[2:])# human-readable board
          .apply(lambda x: x.apply(lambda y: new_piece_map[y])), file=file)  # transmogrify e's to _'s

def check_legality(game_board, move):
    if move[2] == '-':
        move = move.split('-')
    else:
        move = move.split('x')
    move_from = move[0].lower()
    move_to = move[1].lower()
    player_color_index = 9
    is_white = 1
    if game_board[player_color_index] == is_white:
        player_color = 'White'
    else:
        player_color = 'Black'
    legal_moves = enumerate_legal_moves(game_board, player_color)
    for legal_move in legal_moves:
        if move_from == legal_move['From'].lower() and move_to == legal_move['To'].lower():
            return True#return True after finding the move in the list of legal moves
    return False# if move not in list of legal moves, return False

def check_legality_efficient(game_board, move):
    move = move.split('-')
    move_from = move[0].lower()
    move_to = move[1].lower()

    move_from_column = move_from[0]
    move_from_row = move_from[1]

    move_to_column = move_to[0]
    move_to_row = move_to[1]
    piece = game_board[move_from_row][move_from_column]

    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rows = ['1', '2', '3', '4', '5', '6', '7', '8']
    player_color_index = 9
    is_white = 1
    if game_board[player_color_index] == is_white:
        player_color = 'White'
    else:
        player_color = 'Black'
    moves = [check_left_diagonal_move(game_board, move_from_row, move_from_column, player_color),
             check_forward_move(game_board, move_from_row, move_from_column, player_color),
             check_right_diagonal_move(game_board, move_from_row, move_from_column, player_color)]
    if abs(ord(move_to_column) - ord(move_from_column)) > 1:
        return False

    if player_color == 'White':
        if move_to_row < move_from_row:
            return False
        if piece !='w':
            return False

    else:
        if move_to_row > move_from_row:
            return False
        if piece !='b':
            return False

    legal_moves = enumerate_legal_moves(game_board, player_color)
    for legal_move in legal_moves:
        if move_from == legal_move['From'].lower() and move_to == legal_move['To'].lower():
            return True#return True after finding the move in the list of legal moves
    return False# if move not in list of legal moves, return False


def check_legality_MCTS(game_board, move):
    if move.lower() == 'no-move':
        return False
    split_move = move.split('-')
    move_from = split_move[0].lower()
    move_to = split_move[1].lower()

    move_from_column = move_from[0]
    move_from_row = int(move_from[1])

    move_to_column = move_to[0]
    move_to_row = int(move_to[1])
    piece = game_board[move_from_row][move_from_column]

    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rows = ['1', '2', '3', '4', '5', '6', '7', '8']
    player_color_index = 9
    is_white = 1
    if game_board[player_color_index] == is_white:
        player_color = 'White'
    else:
        player_color = 'Black'

    if abs(ord(move_to_column) - ord(move_from_column)) > 1: #check for correct columns
        return False
    if player_color == 'White':
        if move_to_row < move_from_row:#check for correct direction
            return False
        if piece != 'w':#check for correct piece
            return False
    else:
        if move_to_row > move_from_row: #check for correct direction
            return False
        if piece != 'b':#check for correct piece
            return False

    moves = [check_left_diagonal_move(game_board, move_from_row, move_from_column, player_color),
             check_forward_move(game_board, move_from_row, move_from_column, player_color),
             check_right_diagonal_move(game_board, move_from_row, move_from_column, player_color)]
    moves = list(map(lambda x: convert_move_dict_to_move(x), moves ))

    if move.lower() in moves:
        return True
    else:
        return False  # if move not in list of legal moves, return False

def convert_move_dict_to_move(move):
    if not move is None:
        move =  move['From'] + r'-' + move['To']
    return move
def get_best_move(game_board, policy_net_output):
    player_color_index = 9
    is_white = 1
    if game_board[player_color_index] == is_white:
        player_color = 'White'
    else:
        player_color = 'Black'
    ranked_move_indexes = sorted(range(len(policy_net_output[0])), key=lambda i: policy_net_output[0][i], reverse=True)
    legal_moves = enumerate_legal_moves(game_board, player_color)
    legal_move_indexes = convert_legal_moves_into_policy_net_indexes(legal_moves, player_color)
    for move in ranked_move_indexes:#iterate over moves from best to worst and pick the first legal move; will terminate before loop ends
        if move in legal_move_indexes:
            return utils.move_lookup_by_index(move, player_color)

def get_one_of_the_best_moves(game_board, policy_net_output): #if we want successful stochasticity in pure policy net moves
    player_color_index = 9
    is_white = 1
    if game_board[player_color_index] == is_white:
        player_color = 'White'
    else:
        player_color = 'Black'
    ranked_move_indexes = sorted(range(len(policy_net_output[0])), key=lambda i: policy_net_output[0][i], reverse=True)
    legal_moves = enumerate_legal_moves(game_board, player_color)
    legal_move_indexes = convert_legal_moves_into_policy_net_indexes(legal_moves, player_color)
    best_moves = []
    best_move = None
    for move in ranked_move_indexes:#iterate over moves from best to worst and pick the first legal move =? best legal move; will terminate before loop ends
        if move in legal_move_indexes:
            best_move = move
    best_move_val = policy_net_output[best_move] #to get value to compare
    for move in ranked_move_indexes:#do it again to get array of best moves
        if move in legal_move_indexes:
            move_val = policy_net_output[move]
            if move_val/best_move_val > 0.9: #if within 10% of best_move_val
                best_moves.append(move)
    a_best_move = random.sample(best_moves, 1)[0]
    return utils.move_lookup_by_index(a_best_move, player_color)

def enumerate_legal_moves(game_board, player_color):
    if player_color == 'White':
        player = 'w'
    else: #player_color =='Black':
        player = 'b'
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    legal_moves = []
    for row in range(1, 9):  # rows 1-8; if white is in row 8 or black is in row 1, game over should have been declared
        for column in columns:
            if game_board[row][column] == player:
                legal_moves.extend(get_possible_move(game_board, row, column, player_color))
    return legal_moves

def get_possible_move(game_board, row, column, player_color):
    possible_moves = []

    left_diagonal_move = check_left_diagonal_move(game_board, row, column, player_color)
    if left_diagonal_move is not None:
        possible_moves.append(left_diagonal_move)

    forward_move = check_forward_move(game_board, row, column, player_color)
    if forward_move is not None:
        possible_moves.append(forward_move)

    right_diagonal_move = check_right_diagonal_move(game_board, row, column, player_color)
    if right_diagonal_move is not None:
        possible_moves.append(right_diagonal_move)

    return possible_moves

def check_left_diagonal_move(game_board, row, column, player_color):
    white = 'w'
    black = 'b'
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    farthest_left_column = columns[0]
    move = None
    if column != farthest_left_column:  # check for left diagonal move only if not already at far left
        left_diagonal_column = columns[columns.index(column) - 1]
        _from = column + str(row)
        if player_color == 'White':
            row_ahead = row + 1
            # if isinstance(game_board, int) or isinstance(game_board[row_ahead], int):
            #     True
            if game_board[row_ahead][left_diagonal_column] != white:  # can move left diagonally if black or empty there
                to = left_diagonal_column + str(row_ahead)
                move = {'From': _from, 'To': to}
        else: #player_color == 'Black'
            row_ahead = row - 1
            if game_board[row_ahead][left_diagonal_column] != black:  # can move left diagonally if white or empty there
                to = left_diagonal_column + str(row_ahead)
                move = {'From': _from, 'To': to}
    return move

def check_forward_move(game_board, row, column, player_color):
    empty = 'e'
    move = None
    _from = column + str(row)
    if player_color == 'White':
        farthest_row = 8
        if row != farthest_row: #shouldn't happen anyways
            row_ahead = row + 1
            if game_board[row_ahead][column] == empty:
                to = column + str(row_ahead)
                move = {'From': _from, 'To': to}
    else: # player_color == 'Black'
        farthest_row = 1
        if row != farthest_row: #shouldn't happen
            row_ahead = row - 1
            if game_board[row_ahead][column] == empty:
                to = column + str(row_ahead)
                move = {'From': _from, 'To': to}
    return move

def check_right_diagonal_move(game_board, row, column, player_color):
    white = 'w'
    black = 'b'
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    farthest_right_column = columns[len(columns) - 1]
    move = None
    if column != farthest_right_column:  # check for right diagonal move only if not already at far right
        right_diagonal_column = columns[columns.index(column) + 1]
        _from = column + str(row)
        if player_color == 'White':
            row_ahead = row + 1
            if game_board[row_ahead][right_diagonal_column] != white:  # can move right diagonally if black or empty there
                to = right_diagonal_column + str(row_ahead)
                move = {'From': _from, 'To': to}
        else: #player_color == 'Black'
            row_ahead = row - 1
            if game_board[row_ahead][right_diagonal_column] != black:  # can move right diagonally if white or empty there
                to = right_diagonal_column + str(row_ahead)
                move = {'From': _from, 'To': to}
    return move

def convert_legal_moves_into_policy_net_indexes(legal_moves, player_color):
    return [move.index(1) for move in list(map(lambda move:
                    utils.generate_transition_vector(move['To'], move['From'], player_color), legal_moves))]


# check for gameover (white in row 8; black in row 1); check in between moves
def game_over (game_board):
    white = 'w'
    black = 'b'
    black_home_row = game_board[8]
    white_home_row = game_board[1]
    if white in black_home_row.values():
        return True, 'White'
    elif black in white_home_row.values():
        return True, 'Black'
    else:
        return False, None
