from Breakthrough_Player.board_utils import print_board, move_piece, game_over, \
    initial_game_board, check_legality, generate_policy_net_moves, get_best_move, get_random_move
from monte_carlo_tree_search.MCTS import MCTS_BFS_to_depth_limit, MCTS_with_expansions
import sys
import threading
from multiprocessing import  Process, pool
class NoDaemonProcess(Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(pool.Pool):  # Had to make a special class to allow for an inner process pool
    Process = NoDaemonProcess

class myThread (threading.Thread):
   def __init__(self, threadID, game_board, color):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.game_board = game_board
      self.color= color
      self.move = None
   def run(self):
      print ("Starting " + self.threadID)
      self.move = MCTS_BFS_to_depth_limit(self.name, self.game_board, self.color)
      print ("Exiting " + self.threadID)

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
        # ranked_moves = generate_policy_net_moves(game_board, 'White')
        # move = get_best_move(game_board, ranked_moves)

        move = MCTS_BFS_to_depth_limit (game_board, 'White')
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

def self_play_game(player_is_white, policy_opponent='Expansion MCTS', file_to_write=sys.stdout):
    print("MCTS_BFS_to_depth_limit Vs. Policy", file=file_to_write)
    game_board = initial_game_board()
    gameover = False
    move_number = 0
    winner_color = None
    while not gameover:
        print_board(game_board, file=file_to_write)
        move, color_to_move = get_move_self_play(game_board, player_is_white, move_number, policy_opponent)
        print_move(move, color_to_move, file_to_write)
        game_board = move_piece(game_board, move, color_to_move)
        gameover, winner_color = game_over(game_board)
        move_number += 1
    print("Game over. {} wins".format(winner_color), file=file_to_write)
    return winner_color

def get_move_self_play(game_board, player_is_white, move_number, policy_opponent):
    if move_number % 2 == 0:  # white's turn
        color_to_move = 'White'
        move = get_whites_move_self_play(game_board, player_is_white, policy_opponent)
    else:  # black's turn
        color_to_move = 'Black'
        move = get_blacks_move_self_play(game_board, player_is_white, policy_opponent)
    return move, color_to_move

def get_whites_move_self_play(game_board, player_is_white, policy_opponent):
    color_to_move = 'White' # explicitly declared here since this is only for white
    if player_is_white:#get policy net move
        ranked_moves = generate_policy_net_moves(game_board, color_to_move)
        move = get_best_move(game_board, ranked_moves)
    else:# get MCTS type or random
        move = get_policy_opponent_move(game_board, color_to_move, policy_opponent)
    return move

def get_blacks_move_self_play(game_board, player_is_white, policy_opponent):
    color_to_move = 'Black' # explicitly declared here since this is only for black
    if player_is_white:#get MCTS_BFS_to_depth_limit
       move = get_policy_opponent_move(game_board, color_to_move, policy_opponent)
    else:#get policy net move
        ranked_moves = generate_policy_net_moves(game_board, color_to_move)
        move = get_best_move(game_board, ranked_moves)
    return move

def get_policy_opponent_move(game_board, color_to_move, policy_opponent):
    if policy_opponent == 'random':
        move = get_random_move(game_board, color_to_move)
    elif policy_opponent == 'Expansion MCTS':
        move = MCTS_expansions_move(game_board, color_to_move)
    else:  # BFS to depth MCTS
        move = MCTS_move_multithread(game_board, color_to_move)
    return move

def MCTS_move_multithread(game_board, player_color):
    # to dealloc memory upon thread close;
    # if we call directly without forking a new process, ,
    # keeps memory until next call, causing huge bloat before it can garbage collect
    # if we increase processes, we get root-level parallelism (must pull out best move)
    processes = MyPool(processes=1)
    move = processes.starmap(MCTS_BFS_to_depth_limit, [[game_board, player_color]])[0]
    processes.close()
    processes.join()
    return move

def MCTS_expansions_move(game_board, player_color):
    return MCTS_with_expansions(game_board, player_color)

def print_move(move, color_to_move, file=sys.stdout):
    print('{color} move: {move}'.format(color=color_to_move, move=move), file=file)