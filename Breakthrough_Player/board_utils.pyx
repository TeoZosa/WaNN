#cython: language_level=3, boundscheck=False

from tools.utils import move_lookup_by_index, index_lookup_by_move
import sys
import random
import pandas as pd
from bottleneck import argpartition
from itertools import chain

def debug_piece_arrays():
    white_pieces = ['a6', 'c5', 'b4', 'f4', 'f3', 'g4', 'd2', 'd1', 'c1', 'b1', 'f1', 'g1']
    black_pieces = ['b8', 'c8', 'f8', 'g8', 'c7', 'c6', 'e6', 'f6', 'g6', 'f5', 'c4', 'e3']
    return white_pieces, black_pieces
def debug_game_board():
    empty = 'e'
    white = 'w'
    black = 'b'
    return {
        10: -1,  # (-1 for initial state, 0 if black achieved state, 1 if white achieved state)
        # equivalent to 0 if white's move, 1 if black's move
        9: 1,  # is player_color white
        8: {'a': empty, 'b': black, 'c': black, 'd': empty, 'e': empty, 'f': black, 'g': black, 'h': empty},
        7: {'a': empty, 'b': empty, 'c': black, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
        6: {'a': white, 'b': empty, 'c': black, 'd': empty, 'e': black, 'f': black, 'g': black, 'h': empty},
        5: {'a': empty, 'b': empty, 'c': white, 'd': empty, 'e': empty, 'f': black, 'g': empty, 'h': empty},
        4: {'a': empty, 'b': white, 'c': black, 'd': empty, 'e': empty, 'f': white, 'g': white, 'h': empty},
        3: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': black, 'f': white, 'g': empty, 'h': empty},
        2: {'a': empty, 'b': empty, 'c': empty, 'd': white, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
        1: {'a': empty, 'b': white, 'c': white, 'd': white, 'e': empty, 'f': white, 'g': white, 'h': empty}
    }

cpdef dict copy_game_board(dict game_board):
    """'
     Returns a copy of the input game board
    ''"""#
    return {
        10:game_board[10],  # (-1 for initial state, 0 if black achieved state, 1 if white achieved state)
        9: game_board[9],  # is player_color white
        8: {'a': game_board[8]['a'], 'b': game_board[8]['b'], 'c': game_board[8]['c'], 'd':game_board[8]['d'], 'e': game_board[8]['e'], 'f':game_board[8]['f'], 'g': game_board[8]['g'], 'h': game_board[8]['h']},
        7: {'a': game_board[7]['a'], 'b': game_board[7]['b'], 'c': game_board[7]['c'], 'd':game_board[7]['d'], 'e': game_board[7]['e'], 'f':game_board[7]['f'], 'g': game_board[7]['g'], 'h': game_board[7]['h']},
        6: {'a': game_board[6]['a'], 'b': game_board[6]['b'], 'c': game_board[6]['c'], 'd':game_board[6]['d'], 'e': game_board[6]['e'], 'f':game_board[6]['f'], 'g': game_board[6]['g'], 'h': game_board[6]['h']},
        5: {'a': game_board[5]['a'], 'b': game_board[5]['b'], 'c': game_board[5]['c'], 'd':game_board[5]['d'], 'e': game_board[5]['e'], 'f':game_board[5]['f'], 'g': game_board[5]['g'], 'h': game_board[5]['h']},
        4: {'a': game_board[4]['a'], 'b': game_board[4]['b'], 'c': game_board[4]['c'], 'd':game_board[4]['d'], 'e': game_board[4]['e'], 'f':game_board[4]['f'], 'g': game_board[4]['g'], 'h': game_board[4]['h']},
        3: {'a': game_board[3]['a'], 'b': game_board[3]['b'], 'c': game_board[3]['c'], 'd':game_board[3]['d'], 'e': game_board[3]['e'], 'f':game_board[3]['f'], 'g': game_board[3]['g'], 'h': game_board[3]['h']},
        2: {'a': game_board[2]['a'], 'b': game_board[2]['b'], 'c': game_board[2]['c'], 'd':game_board[2]['d'], 'e': game_board[2]['e'], 'f':game_board[2]['f'], 'g': game_board[2]['g'], 'h': game_board[2]['h']},
        1: {'a': game_board[1]['a'], 'b': game_board[1]['b'], 'c': game_board[1]['c'], 'd':game_board[1]['d'], 'e': game_board[1]['e'], 'f':game_board[1]['f'], 'g': game_board[1]['g'], 'h': game_board[1]['h']},
    }


def initial_piece_arrays():
    """'
     Returns the initial piece arrays
    ''"""#
    cdef list white_pieces = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2',
                         'h2', ]
    cdef list black_pieces = ['a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7', 'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8',
                         'h8', ]
    return white_pieces, black_pieces


#Note: not the same as move_piece in self_play_logs_to_datastructures; is white index now changes as we are sharing a board,
#so player U opponent = self_play_log_board with is_white_index changing based on who owns the board
def move_piece(board_state, move, whose_move):
    """'
     Returns a copy of the input game board where the move has been made
    ''"""#
    empty = 'e'
    white_move_index = 10
    is_white_index = 9
    if move[2] == '-':
        move = move.split('-')
    else:
        move = move.split('x')#x for wanderer captures.
    _from = move[0].lower()
    to = move[1].lower()
    next_board_state = copy_game_board(board_state)#copy.deepcopy(board_state)  # edit copy of board_state; don't need this for breakthrough_player?
    next_board_state[int(to[1])][to[0]] = next_board_state[int(_from[1])][_from[0]]
    next_board_state[int(_from[1])][_from[0]] = empty
    if whose_move == 'White':
        next_board_state[white_move_index] = 1
        next_board_state[is_white_index] = 0 #next move isn't white's
    else:
        next_board_state[white_move_index] = 0
        next_board_state[is_white_index] = 1 #since black made this move, white makes next move
    assert (board_state != next_board_state)

    return next_board_state


cpdef move_piece_update_piece_arrays(dict board_state, str move, str whose_move):
    """'
     Returns a copy of the input game board where the move has been made, 
     which pieces to add/remove from the player's piece array, 
     and if an opponent's move must also be removed (capture move)
    ''"""#
    cdef:
        list move_list
        str empty = 'e'
        int white_move_index = 10
        int is_white_index = 9
        # str _from
        # str to
        str to_position
        str player_piece_to_remove
        str player_piece_to_add
        dict next_board_state
    remove_opponent_piece = False
    move_list = move.split('-')
    # if move[2] == '-':
    #     pass
    # else:
    #     move = move.split('x')#x for wanderer captures.
    _from = move_list[0]
    to = move_list[1]
    next_board_state = copy_game_board(board_state)# edit copy of board_state; don't need this for breakthrough_player?
    to_position =  next_board_state[int(to[1])][to[0]]
    player_piece_to_remove = _from #this will always become empty
    player_piece_to_add = to
    next_board_state[int(to[1])][to[0]] = next_board_state[int(_from[1])][_from[0]]
    next_board_state[int(_from[1])][_from[0]] = empty
    if whose_move == 'White':
        next_board_state[white_move_index] = 1
        next_board_state[is_white_index] = 0 #next move isn't white's
        if to_position == 'b':
            remove_opponent_piece = True
    else:
        next_board_state[white_move_index] = 0
        next_board_state[is_white_index] = 1 #since black made this move, white makes next move
        if to_position == 'w':
            remove_opponent_piece = True
    return next_board_state, player_piece_to_add, player_piece_to_remove, remove_opponent_piece

def move_piece_update_piece_arrays_in_place(board_state, move, whose_move):
    """'
     Returns the input game board where the move has been made (mutates the input game board), 
     which pieces to add/remove from the player's piece array, 
     and if an opponent's move must also be removed (capture move)
    ''"""#
    empty = 'e'
    remove_opponent_piece = False
    white_move_index = 10
    is_white_index = 9
    if move[2] == '-':
        move = move.split('-')
    else:
        move = move.split('x')#x for wanderer captures.
    _from = move[0].lower()
    to = move[1].lower()
    next_board_state = board_state
    to_position =  next_board_state[int(to[1])][to[0]]
    player_piece_to_remove = _from #this will always become empty
    player_piece_to_add = to
    next_board_state[int(to[1])][to[0]] = next_board_state[int(_from[1])][_from[0]]
    next_board_state[int(_from[1])][_from[0]] = empty
    if whose_move == 'White':
        next_board_state[white_move_index] = 1
        next_board_state[is_white_index] = 0 #next move isn't white's
        if to_position == 'b':
            remove_opponent_piece = True
    else:
        next_board_state[white_move_index] = 0
        next_board_state[is_white_index] = 1 #since black made this move, white makes next move
        if to_position == 'w':
            remove_opponent_piece = True
    return next_board_state, player_piece_to_add, player_piece_to_remove, remove_opponent_piece

def get_random_move(game_board, player_color):
    """'
     Returns a random move for player_color given the input game_board 
    ''"""#
    possible_moves = enumerate_legal_moves(game_board, player_color)
    random_move = random.sample(possible_moves, 1)[0]
    move = random_move['From'] + '-' + random_move['To']
    return move

def print_board(game_board, file=sys.stdout):
    """'
     Prints the input game board 
    ''"""#
    new_piece_map = {
        'w': 'w',
        'b': 'b',
        'e': '_'
    }
    print((pd.DataFrame(game_board).transpose().sort_index(ascending=False).iloc[2:])# human-readable board
          .apply(lambda x: x.apply(lambda y: new_piece_map[y])), file=file)  # transmogrify e's to _'s

def check_legality(game_board, move):
    """'
     Returns whether a move is legal on the given game board by checking if it is in the set of all legal moves
    ''"""#
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
        if legal_move[2] == '-':
            legal_move = legal_move.split('-')
        else:
            legal_move = legal_move.split('x')
        if move_from == legal_move[0].lower() and move_to == legal_move[1].lower():
            return True#return True after finding the move in the list of legal moves
    return False# if move not in list of legal moves, return False

def check_legality_MCTS(game_board, move):
    """'
     Returns whether a move is legal on the given game board by checking it against rules of the game
    ''"""#
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
             check_right_diagonal_move(game_board, move_from_row, move_from_column, player_color)] #either move as a string or None

    if move.lower() in moves:
        return True
    else:
        return False  # if move not in list of legal moves, return False

def get_best_move(game_board, policy_net_output):
    """'
     Returns the best policy net move 
    ''"""#
    player_color_index = 9
    is_white = 1
    if game_board[player_color_index] == is_white:
        player_color = 'White'
    else:
        player_color = 'Black'
    ranked_move_indexes = get_top_children(policy_net_output[0])#sorted(range(len(policy_net_output[0])), key=lambda i: policy_net_output[0][i], reverse=True) #inefficient
    legal_moves = enumerate_legal_moves(game_board, player_color)
    legal_move_indexes = convert_legal_moves_into_policy_net_indexes(legal_moves)
    for move in ranked_move_indexes:#iterate over moves from best to worst and pick the first legal move; will terminate before loop ends
        if move in legal_move_indexes:
            return move_lookup_by_index(move, player_color)

def get_top_children(NN_output, num_top=0):
    """'
     Returns a sorted list (of partitions of size num_top) of the NN output, from best to worst. 
     NOTE: if num_top less than 2 or greater than size NN_output, returns a fully sorted list (num_top = 1)
    ''"""#
    if 1 < num_top <len(NN_output):
        partial_sort = argpartition(NN_output, NN_output.size-num_top+1)[-num_top:]
        full_sort = partial_sort[NN_output[partial_sort].argsort()][::-1]
    else:
        num_top = len(NN_output)
        full_sort = NN_output.argsort()[::-1][:num_top]

    return full_sort

def enumerate_legal_moves(game_board, player_color):
    """'
     Returns a list of legal moves for player_color given a game_board by iterating over entire board
    ''"""#
    if player_color == 'White':
        player = 'w'
    else: #player_color =='Black':
        player = 'b'#
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    legal_moves = []
    for row in range(1, 9):  # rows 1-8; if white is in row 8 or black is in row 1, game over should have been declared
        for column in columns:
            if game_board[row][column] == player:
                for possible_move in get_possible_moves(game_board, row, column, player_color):
                    if possible_move is not None:
                        legal_moves.append(possible_move)
    return legal_moves

cpdef list enumerate_legal_moves_using_piece_arrays(str player_color, dict game_board, list player_pieces):
    """'
     Returns a list of legal moves for player_color given a game board and player_color's piece arrays. 
     Iterates over the piece arrays (as opposed to iterating over the entire game board) for efficiency. 
    ''"""#
    return [move for move in chain.from_iterable(
        [[check_left_diagonal_move(game_board, int(piece[1]), piece[0], player_color),
                                check_forward_move(game_board, int(piece[1]), piece[0], player_color),
                                check_right_diagonal_move(game_board, int(piece[1]), piece[0], player_color)] for piece in player_pieces]) if move is not None]


cpdef list get_possible_moves(dict game_board, int row, str column, str player_color):
    """'
     Returns a list of legal moves for player_color given a game board and player_color's piece arrays. 
     Iterates over the piece arrays (as opposed to iterating over the entire game board) for efficiency. 
    ''"""#
    return [check_left_diagonal_move(game_board, row, column, player_color),
                                check_forward_move(game_board, row, column, player_color),
                                check_right_diagonal_move(game_board, row, column, player_color)]

cpdef check_left_diagonal_move(dict game_board, int row, str column, str player_color):
    """'
     Returns the left diagonal move if it is legal
     else, returns None 
    ''"""#
    cdef dict left_columns = {'b':'a', 'c':'b', 'd':'c', 'e':'d', 'f':'e', 'g':'f', 'h':'g'}
    move = None
    if column != 'a':  # check for left diagonal move only if not already at far left
        left_diagonal_column = left_columns[column]
        _from = ''.join((column, str(row)))
        if player_color == 'White':
            row_ahead = row + 1
            # if isinstance(game_board, int) or isinstance(game_board[row_ahead], int):
            #     True
            if game_board[row_ahead][left_diagonal_column] != 'w':  # can move left diagonally if black or empty there
                to = ''.join((left_diagonal_column, str(row_ahead)))
                move = '-'.join((_from, to))
        else: #player_color == 'Black'
            row_ahead = row - 1
            if game_board[row_ahead][left_diagonal_column] != 'b':  # can move left diagonally if white or empty there
                to = ''.join((left_diagonal_column, str(row_ahead)))
                move = '-'.join((_from, to))
    return move

cpdef check_forward_move(dict game_board, int row, str column, str player_color):
    """'
     Returns the forward move if it is legal
     else, returns None 
    ''"""#
    move = None
    _from = ''.join((column, str(row)))
    if player_color == 'White':
        farthest_row = 8
        if row != farthest_row: #shouldn't happen anyways
            row_ahead = row + 1
            if game_board[row_ahead][column] == 'e':
                to = ''.join((column,str(row_ahead)))
                move = '-'.join((_from, to))

    else: # player_color == 'Black'
        farthest_row = 1
        if row != farthest_row: #shouldn't happen
            row_ahead = row - 1
            if game_board[row_ahead][column] == 'e':
                to = ''.join((column,str(row_ahead)))
                move = '-'.join((_from, to))

    return move

cpdef check_right_diagonal_move(dict game_board, int row, str column, str player_color):
    """'
     Returns the right diagonal move if it is legal
     else, returns None 
    ''"""#
    cdef dict right_columns= {'a':'b', 'b':'c', 'c':'d', 'd':'e', 'e':'f', 'f':'g', 'g':'h'}
    move = None
    if column != 'h':  # check for right diagonal move only if not already at far right
        right_diagonal_column = right_columns[column]
        _from = ''.join((column, str(row)))
        if player_color == 'White':
            row_ahead = row + 1
            if game_board[row_ahead][right_diagonal_column] != 'w':  # can move right diagonally if black or empty there
                to = ''.join((right_diagonal_column,str(row_ahead)))
                move = '-'.join((_from, to))
        else: #player_color == 'Black'
            row_ahead = row - 1
            if game_board[row_ahead][right_diagonal_column] != 'b':  # can move right diagonally if white or empty there
                to = ''.join((right_diagonal_column,str(row_ahead)))
                move = '-'.join((_from, to))
    return move

def convert_legal_moves_into_policy_net_indexes(legal_moves):
    """'
     Returns the legal moves as policy net indexes
    ''"""#
    return [index_lookup_by_move(move) for move in legal_moves]

def game_over (game_board):
    """'
     Returns whether the game_board is a terminal state aka gameover (white in row 8; black in row 1)
     If True, also returns the winner color
    ''"""#
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
