from Breakthrough_Player.board_utils import print_board, move_piece, game_over, \
    initial_game_board, check_legality, generate_policy_net_moves, get_best_move, get_random_move, get_NN
from monte_carlo_tree_search.MCTS import MCTS, NeuralNetsCombined
import sys
import os
from multiprocessing import  Process, pool, Pool
import pexpect
from pexpect.popen_spawn import PopenSpawn
import pickle


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

def play_game_vs_wanderer(white_player, black_opponent, depth_limit=1, time_to_think=10, file_to_write=sys.stdout, MCTS_log_file=sys.stdout, root=None, game_num=-1):




    if black_opponent == 'Wanderer':
        wanderer_color = 'Black'
        WaNN_color = 'White'
        wand_player = black_opponent
        WaNN_player = white_player


    else:
        wanderer_color = 'White'
        WaNN_color = 'Black'
        wand_player = white_player
        WaNN_player = black_opponent
    OSX_input_engine = '/Users/TeofiloZosa/Clion/Breakthrough/BreakthroughInput/bin/Release/BreakthroughInput'
    OSX_wanderer_executable = r'/Users/TeofiloZosa/Clion/Breakthrough/BreakthroughCurrent/bin/Release/BreakthroughCurrent'

    open_input_engine = r'C:\Users\damon\PycharmProjects\BreakthroughANN\BreakthroughInput.exe'

    wanderer_executable = r'C:\Users\damon\PycharmProjects\BreakthroughANN\BreakthroughCurrent.exe'
    ttt = r'--ttt=10'
    full_command = open_input_engine + ' ' + WaNN_color + ' ' + wanderer_executable + ' ' + ttt
    # wanderer = pexpect.spawn(open_input_engine,  args= [color_to_be, wanderer_executable, ttt])
    # wanderer = PopenSpawn(full_command, cwd=r'C:\Users\damon\PycharmProjects\BreakthroughANN' )
    wanderer = PopenSpawn([open_input_engine, WaNN_color, wanderer_executable, ttt])
    wanderer_MCTS_tree = MCTS(depth_limit, time_to_think, wanderer_color, wand_player, MCTS_log_file, wanderer)
    wanderer.log_file = sys.stdout

    policy_net = NeuralNetsCombined()
    computer_MCTS_tree = MCTS(depth_limit, time_to_think, WaNN_color, WaNN_player, MCTS_log_file, policy_net)
    computer_MCTS_tree.game_num = game_num
    if root is not None:
        computer_MCTS_tree.selected_child = root

    print("{white} Vs. {black}".format(white=white_player, black=black_opponent), file=file_to_write)

    game_board = initial_game_board()
    gameover = False
    move_number = 0
    winner_color = None
    web_visualizer_link = r'http://www.trmph.com/breakthrough/board#8,'
    while not gameover:
        print_board(game_board, file=file_to_write)
        computer_MCTS_tree.height = wanderer_MCTS_tree.height = move_number
        move, color_to_move = get_move(game_board, white_player, black_opponent, move_number, computer_MCTS_tree, wanderer_MCTS_tree, file_to_write)
        print_move(move, color_to_move, file_to_write)
        game_board = move_piece(game_board, move, color_to_move)
        if move[2] == r'-':
            move = move.split(r'-')
        else:
            move = move.split(r'x')
        if (color_to_move == 'Black' and wanderer_color =='Black') or (color_to_move == 'White' and wanderer_color =='White'):
            move_to_send = move[0]+'-' +move[1]
            computer_MCTS_tree.last_opponent_move = move_to_send
        web_visualizer_link = web_visualizer_link + move[0] + move[1]
        gameover, winner_color = game_over(game_board)
        move_number += 1
    print_board(game_board, file=file_to_write)
    print("Game over. {} wins".format(winner_color), file=file_to_write)
    print("Visualization link = {}".format(web_visualizer_link), file=file_to_write)

#TODO: consider permutations of winning subsequences? cut down on space/ have playbook. edit distance? as current subsequence gets farther from old subsequence, stop trying to match moves?

    final_policy_move = computer_MCTS_tree.selected_child
    while final_policy_move.parent is not None:#root doesn't need this updated so it's fine to not update after
        if winner_color == WaNN_color:
            final_policy_move.wins_down_this_tree +=1
        else:
            final_policy_move.losses_down_this_tree += 1
        final_policy_move = final_policy_move.parent

    # output_file = open(r'G:\TruncatedLogs\PythonDataSets\DataStructures\GameTree\FreshRootTrueWinLossFieldBlackPrototype04082017{}.p'.format(str(0)), 'wb')
    # # online reinforcement learning: resave the root at each new game (if it was kept, values would have backpropagated)
    # pickle.dump(final_policy_move, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    # output_file.close()

    # final_policy_move = computer_MCTS_tree.selected_child
    # while final_policy_move.parent is not None:
    #     final_policy_move = final_policy_move.parent

    if wanderer_MCTS_tree is not None:
        wanderer_MCTS_tree.policy_net.sendline('quit')
    return winner_color


def play_game(white_player, black_opponent, depth_limit=1, time_to_think=10, file_to_write=sys.stdout, MCTS_log_file=open(os.devnull, 'w')):
    policy_net = NeuralNetsCombined()
    computer_MCTS_tree = MCTS(depth_limit, time_to_think, white_player, MCTS_log_file, policy_net)

    print("Teo's Fabulous Breakthrough Player")
    game_board = initial_game_board()
    gameover = False
    move_number = 0
    winner_color = None
    web_visualizer_link = r'http://www.trmph.com/breakthrough/board#8,'
    while not gameover:
        print_board(game_board)
        move, color_to_move = get_move(game_board, white_player, black_opponent, move_number, computer_MCTS_tree, None, file_to_write)
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

def get_move(game_board, white_player, black_opponent, move_number, computer_MCTS_tree, wanderer_MCTS_tree, file_to_write):
    if move_number % 2 == 0:  # white's turn
        color_to_move = 'White'
        move = get_whites_move(game_board, white_player, computer_MCTS_tree, wanderer_MCTS_tree, file_to_write)
    else:  # black's turn
        color_to_move = 'Black'
        move = get_blacks_move(game_board, black_opponent, computer_MCTS_tree, wanderer_MCTS_tree, file_to_write)
    return move, color_to_move

def get_whites_move(game_board, white_player, computer_MCTS_tree, wanderer_MCTS_tree, file_to_write):
    move = None
    color_to_move = 'White'
    if white_player == 'Human':
        move = get_human_move(game_board)
    elif white_player == 'Wanderer':
        move = get_player_move(game_board, color_to_move, "Wanderer", wanderer_MCTS_tree)
        if not check_legality(game_board, move):
            print("Illegal move by Wanderer", file=file_to_write)
    else:#get policy net move
        move = get_player_move(game_board, color_to_move, computer_MCTS_tree.MCTS_type, computer_MCTS_tree)
        if wanderer_MCTS_tree is not None:
            wanderer_MCTS_tree.policy_net.sendline(move)
    return move

def get_blacks_move(game_board, black_opponent, computer_MCTS_tree, wanderer_MCTS_tree, file_to_write):
    move = None
    color_to_move = 'Black'
    if black_opponent == 'Human':
        move = get_human_move(game_board)
    elif black_opponent == 'Wanderer':
        move = get_player_move(game_board, color_to_move, "Wanderer", wanderer_MCTS_tree)
        if not check_legality(game_board, move):
            print("Illegal move by Wanderer", file=file_to_write)
    else:#get policy net move
        move = get_player_move(game_board, color_to_move, computer_MCTS_tree.MCTS_type, computer_MCTS_tree)
        if wanderer_MCTS_tree is not None:
            wanderer_MCTS_tree.policy_net.sendline(move)
    return move

def get_human_move(game_board):
    player_move_legal = False
    while not player_move_legal:
        move = input('Enter your move: ')
        player_move_legal = check_legality(game_board, move)
        if not player_move_legal:
            print("Error, illegal move. Please enter a legal move.")
    return move

def self_play_game(white_player, black_opponent, depth_limit, time_to_think, file_to_write=sys.stdout, MCTS_log_file=sys.stdout):
    #Initialize policy net; use single net instance for both players
    policy_net = NeuralNetsCombined()

    white_MCTS_tree = MCTS(depth_limit, time_to_think, white_player, MCTS_log_file, policy_net)
    black_MCTS_tree = MCTS(depth_limit, time_to_think, black_opponent, MCTS_log_file, policy_net)

    print("{white} Vs. {black}".format(white=white_player, black=black_opponent), file=file_to_write)
    game_board = initial_game_board()
    gameover = False
    move_number = 0
    winner_color = None
    web_visualizer_link = r'http://www.trmph.com/breakthrough/board#8,'
    while not gameover:
        white_MCTS_tree.height = black_MCTS_tree.height = move_number
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
    elif player == 'Expansion MCTS' \
        or player == 'Expansion MCTS Pruning' \
        or player == 'Expansion MCTS Post-Pruning' \
        or player == 'EBFS MCTS' \
        or player == 'MCTS Asynchronous'\
        or player == 'Wanderer'\
        or player == 'Policy':
        move = MCTS_tree.evaluate(game_board, color_to_move)
    elif player == 'BFS MCTS':  # BFS to depth MCTS
        move = MCTS_move_multithread(game_board, color_to_move, MCTS_tree)
    return move

def MCTS_move_multithread(game_board, player_color, white_MCTS_tree):
    # to dealloc memory upon thread close;
    # if we call directly without forking a new process, ,
    # keeps memory until next call, causing huge bloat before it can garbage collect
    # we can't actually use multiple cores as multiple tensorflow sessions throw errors

    pool_func = MyPool
    with pool_func(processes=1) as processes:
        move = processes.starmap(white_MCTS_tree.evaluate, [[game_board, player_color]])[0]
        processes.close()
        processes.join()
    return move

def print_move(move, color_to_move, file=sys.stdout):
    print('{color} move: {move}\n'.format(color=color_to_move, move=move), file=file)