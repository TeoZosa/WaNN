from Breakthrough_Player.board_utils import print_board, move_piece, game_over, \
    initial_game_board, check_legality, generate_policy_net_moves, get_best_move, get_random_move
from monte_carlo_tree_search.MCTS import MCTS
import sys
from multiprocessing import  Process, pool, Pool

class NoDaemonProcess(Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(pool.Pool):  # make a special class to allow for an inner process pool
    Process = NoDaemonProcess

def play_game(player_is_white, file_to_write=sys.stdout):
    print("Teo's Fabulous Breakthrough Player")
    game_board = initial_game_board()
    gameover = False
    move_number = 0
    winner_color = None
    web_visualizer_link = r'http://www.trmph.com/breakthrough/board#8,'
    while not gameover:
        print_board(game_board)
        move, color_to_move = get_move(game_board, player_is_white, move_number)
        print_move(move, color_to_move)
        game_board = move_piece(game_board, move, color_to_move)
        move = move.split(r'-')
        web_visualizer_link = web_visualizer_link + move[0] + move[1]
        gameover, winner_color = game_over(game_board)
        move_number += 1
    print_board(game_board, file=file_to_write)
    print("Game over. {} wins".format(winner_color), file=file_to_write)
    print("Visualization link = {}".format(web_visualizer_link), file=file_to_write)
    return winner_color

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
        ranked_moves = generate_policy_net_moves(game_board, 'White')
        move = get_best_move(game_board, ranked_moves)

        # move = get_random_move(game_board, 'White')
    return move

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

def self_play_game(white_player, black_opponent, depth_limit, time_to_think, file_to_write=sys.stdout, MCTS_log_file=sys.stdout):
    white_MCTS_tree = MCTS(depth_limit, time_to_think, white_player, MCTS_log_file)
    black_MCTS_tree = MCTS(depth_limit, time_to_think, black_opponent, MCTS_log_file)

    print("{} Vs. Policy".format(black_opponent), file=file_to_write)
    game_board = initial_game_board()
    gameover = False
    move_number = 0
    winner_color = None
    web_visualizer_link = r'http://www.trmph.com/breakthrough/board#8,'
    while not gameover:
        white_MCTS_tree.height = move_number
        black_MCTS_tree.height = move_number
        print_board(game_board, file=file_to_write)
        move, color_to_move = get_move_self_play(game_board, white_player, move_number, black_opponent, white_MCTS_tree, black_MCTS_tree)
        print_move(move, color_to_move, file_to_write)
        game_board = move_piece(game_board, move, color_to_move)
        move = move.split(r'-')
        web_visualizer_link = web_visualizer_link + move[0] + move[1]
        gameover, winner_color = game_over(game_board)
        move_number += 1
    print_board(game_board, file=file_to_write)
    print("Game over. {} wins".format(winner_color), file=file_to_write)
    print("Visualization link = {}".format(web_visualizer_link), file=file_to_write)
    return winner_color

def get_move_self_play(game_board, white_player, move_number, black_opponent, white_MCTS_tree, black_MCTS_tree):
    if move_number % 2 == 0:  # white's turn
        color_to_move = 'White'
        move = get_whites_move_self_play(game_board, white_player, move_number, white_MCTS_tree)
    else:  # black's turn
        color_to_move = 'Black'
        move = get_blacks_move_self_play(game_board, black_opponent, move_number, black_MCTS_tree)
    return move, color_to_move

def get_whites_move_self_play(game_board, white_player, move_number, white_MCTS_tree):
    color_to_move = 'White' # explicitly declared here since this is only for white
    if white_player == 'Random' or white_player == 'Policy':
        move = get_player_move(game_board, color_to_move, white_player, white_MCTS_tree)
    else:#
        if move_number <= 4:
            move = get_random_move(game_board, color_to_move)
        else:
            move = get_player_move(game_board, color_to_move, white_player, white_MCTS_tree)
    return move

def get_blacks_move_self_play(game_board, black_opponent, move_number, black_MCTS_tree):
    color_to_move = 'Black' # explicitly declared here since this is only for black
    if black_opponent == 'Random' or black_opponent == 'Policy':# get policy opponent move
        move = get_player_move(game_board, color_to_move, black_opponent, black_MCTS_tree)
    else:
        if move_number <= 4:
            move = get_random_move(game_board, color_to_move)
        else:
            move = get_player_move(game_board, color_to_move, black_opponent, black_MCTS_tree)
    return move

def get_player_move(game_board, color_to_move, player, MCTS_tree):
    if player == 'Random':
        move = get_random_move(game_board, color_to_move)
    elif player == 'Expansion MCTS' or player == 'EBFS MCTS' or player == 'Policy':
        move = MCTS_tree.evaluate(game_board, color_to_move)
    elif player == 'BFS MCTS':  # BFS to depth MCTS
        move = MCTS_move_multithread(game_board, color_to_move, MCTS_tree)
    return move

def MCTS_move_multithread(game_board, player_color, white_MCTS_tree):
    # to dealloc memory upon thread close;
    # if we call directly without forking a new process, ,
    # keeps memory until next call, causing huge bloat before it can garbage collect
    # if we increase processes, we get root-level parallelism (must pull out best move)

    pool_func = MyPool
    with pool_func(processes=1) as processes:
        move = processes.starmap(white_MCTS_tree.evaluate, [[game_board, player_color]])[0]
        processes.close()
        processes.join()
    return move

def print_move(move, color_to_move, file=sys.stdout):
    print('{color} move: {move}\n'.format(color=color_to_move, move=move), file=file)