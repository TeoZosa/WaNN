from Breakthrough_Player import board_utils
from tools import utils
import pandas as pd
import random

def play_game(player_is_white):
    game_board = board_utils.initial_game_board()
    gameover = False
    move_number = 0
    winner_color = None
    while not gameover:
        print_board(game_board)
        move, color_to_move = get_move(game_board, player_is_white, move_number)
        print_move(move, color_to_move)
        game_board = board_utils.move_piece(game_board, move, color_to_move)
        gameover, winner_color = game_over(game_board)
        move_number += 1
    print("Game over. {} wins".format(winner_color))

def get_move(game_board, player_is_white, move_number):
    if move_number % 2 == 0:  # white's turn
        color_to_move = 'White'
        move = get_whites_move(game_board, player_is_white)
    else:  # black's turn
        color_to_move = 'Black'
        move = get_blacks_move(game_board, player_is_white)
    return move, color_to_move

def get_whites_move(game_board, player_is_white):
    move = None
    if player_is_white:
        player_move_legal = False
        while not player_move_legal:
            move = input('Enter your move: ')
            player_move_legal = check_legality(game_board, move)
            if not player_move_legal:
                print("Error, illegal move. Please enter a legal move.")
    else:#get policy net move
        ranked_moves = board_utils.generate_policy_net_moves(game_board, 'White')
        move = get_best_move(game_board, ranked_moves)
        # move = get_random_move(game_board, 'White')
    return move

def print_move(move, color_to_move):
    print('{color} move: {move}'.format(color=color_to_move, move=move))


def get_blacks_move(game_board, player_is_white):
    move = None
    if player_is_white:#get policy net move
        ranked_moves = board_utils.generate_policy_net_moves(game_board, 'Black')
        move = get_best_move(game_board, ranked_moves)
        # move = get_random_move(game_board, 'Black')
    else:
        player_move_legal = False
        while not player_move_legal:
            move = input('Enter your move: ')
            player_move_legal = check_legality(game_board, move)
            if not player_move_legal:
                print("Error, illegal move. Please enter a legal move.")
    return move

def get_random_move(game_board, player_color):
    possible_moves = enumerate_legal_moves(game_board, player_color)
    random_move = random.sample(possible_moves, 1)[0]
    move = random_move['From'] + '-' + random_move['To']
    return move

def print_board(game_board):
    new_piece_map = {
        'w': 'w',
        'b': 'b',
        'e': '_'
    }
    print((pd.DataFrame(game_board).transpose().sort_index(ascending=False).iloc[2:])# human-readable board
          .apply(lambda x: x.apply(lambda y: new_piece_map[y])))  # transmogrify e's to _'s

def check_legality(game_board, move):
    move = move.split('-')
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
            return utils.move_lookup(move, player_color)


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
