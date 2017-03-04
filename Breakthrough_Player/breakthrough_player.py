from Breakthrough_Player.board_utils import print_board, move_piece, game_over, \
    initial_game_board, check_legality, generate_policy_net_moves, get_best_move
from monte_carlo_tree_search.MCTS import MCTS


def play_game(player_is_white):
    print("Teo's Fabulous Breakthrough Player")
    game_board = initial_game_board()
    gameover = False
    move_number = 0
    winner_color = None
    while not gameover:
        print_board(game_board)
        move, color_to_move = get_move(game_board, player_is_white, move_number)
        print_move(move, color_to_move)
        game_board = move_piece(game_board, move, color_to_move)
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
        # ranked_moves = .generate_policy_net_moves(game_board, 'White')
        # move = get_best_move(game_board, ranked_moves)

        move = MCTS (game_board, 'White')
        # move = get_random_move(game_board, 'White')
    return move

def print_move(move, color_to_move):
    print('{color} move: {move}'.format(color=color_to_move, move=move))


def get_blacks_move(game_board, player_is_white):
    move = None
    if player_is_white:#get policy net move
        ranked_moves = generate_policy_net_moves(game_board, 'Black')
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

