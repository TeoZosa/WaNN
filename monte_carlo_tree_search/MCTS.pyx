#cython: language_level=3, boundscheck=False

from monte_carlo_tree_search.expansion_MCTS_functions import MCTS_with_expansions
#from monte_carlo_tree_search.BFS_MCTS_functions import MCTS_BFS_to_depth_limit
from Breakthrough_Player.board_utils import get_best_move
from Breakthrough_Player.policy_net_utils import instantiate_session_both, instantiate_session,  instantiate_session_both_128
from tools.utils import convert_board_to_2d_matrix_POEB, batch_split_no_labels
import re
import numpy as np
cimport numpy as np
DTYPE = np.float32
ctypedef np.float_t DTYPE_t

class MCTS(object):
    #Option B: Traditional MCTS with expansion using policy net to generate prior values and prune tree
    # start with root and put in NN queue, (level 0)
    # while time to think,
    # 1. MCTS search to find the best move
    # 2. When we reach a leaf node, expand, evaluate with policy net, prune and update prior values on children
    # 3. keep searching to desired depth (final depth = depth at expansion + depth_limit)
    # 4. do random rollouts. repeat 1.

    def __init__(self, depth_limit, time_to_think, color, MCTS_type, MCTS_log_file, neural_net):
        self.time_to_think = time_to_think
        self.depth_limit = depth_limit
        self.selected_child = None
        self.previous_selected_child = None
        self.last_opponent_move = None
        self.MCTS_type = MCTS_type
        self.log_file = MCTS_log_file
        self.height = 0
        self.policy_net = neural_net
        self.game_num = -1
        self.color = color

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    def evaluate(self, game_board, player_color, background_search=False):
        previous_child = self.selected_child
        if self.MCTS_type == 'EBFS MCTS' \
            or self.MCTS_type == 'Expansion MCTS' \
            or self.MCTS_type == 'Expansion MCTS Pruning' \
            or self.MCTS_type ==  'MCTS Asynchronous'\
            or self.MCTS_type == 'Expansion MCTS Post-Pruning': #pruning plus depth expansions      #NOTE: good in theory, untested
            if background_search:
                _, move = MCTS_with_expansions(game_board, player_color, self.time_to_think, self.depth_limit, self.previous_selected_child, self.last_opponent_move, self.height-1, self.log_file, self.MCTS_type, self.policy_net, self.game_num)
            else:
                self.previous_selected_child = previous_child
                self.selected_child, move = MCTS_with_expansions(game_board, player_color, self.time_to_think, self.depth_limit, previous_child, self.last_opponent_move, self.height, self.log_file, self.MCTS_type, self.policy_net, self.game_num)
        elif self.MCTS_type == 'Policy':
            ranked_moves = self.policy_net.evaluate(game_board, player_color)
            move = get_best_move(game_board, ranked_moves)
        elif self.MCTS_type == 'Wanderer':
            # sleep(self.time_to_think)
            if self.color == 'White':
                move_regex = re.compile(r".*play\sw\s([a-h]\d.[a-h]\d).*",
                                        re.IGNORECASE)
                self.policy_net.expect('play w.*', timeout=200)

                move = self.policy_net.before.decode('utf-8') + self.policy_net.after.decode('utf-8')
                print(move, file=self.log_file)
                move = move_regex.search(move)
                if move is not None:
                    move = move.group(1)
            else:
                move_regex = re.compile(r".*play\sb\s([a-h]\d.[a-h]\d).*",
                                        re.IGNORECASE)
                self.policy_net.expect('play b.*', timeout=200)

                move = self.policy_net.before.decode('utf-8') + self.policy_net.after.decode('utf-8')
                print(move, file=self.log_file)
                move = move_regex.search(move)
                if move is not None:
                    move = move.group(1)

        return move



class NeuralNetsCombined():

    #initialize the Neural Net (Only works with combined net for now)
    def __init__(self):
        self.sess, self.output_white, self.input_white, self.output_black, self.input_black= instantiate_session_both()

    #evaluate a list of game nodes or a game board directly (must pass in player_color in the latter case)
    def evaluate(self, game_nodes, player_color=None, already_converted=False):
        cdef:
            list moves = []
            np.ndarray output
            list board_representations
            list inference_batches
            int batch_size = 16384

        if not already_converted:
            if player_color is not None: #1 board + 1 color from direct policy net call
                board_representations = [convert_board_to_2d_matrix_POEB(game_nodes, player_color)]
            else:
                board_representations = [convert_board_to_2d_matrix_POEB(node['game_board'], node['color']) for node in
                                     game_nodes]
        else:
            board_representations = game_nodes
        if player_color is None and not already_converted:
            player_color = game_nodes[0]['color']

        if len(board_representations) > batch_size:
            inference_batches = batch_split_no_labels(board_representations, batch_size)
        else:
            inference_batches = [board_representations]

        if player_color == 'White':
            y_pred = self.output_white
            X = self.input_white
        else:
            y_pred = self.output_black
            X = self.input_black
        if len(inference_batches) >1:
            for batch in inference_batches:
                predicted_moves = self.sess.run(y_pred, feed_dict={X: batch})
                moves.extend(predicted_moves)
            output = np.ndarray(moves, dtype =DTYPE)
         else:
            output = self.sess.run(y_pred, feed_dict={X: inference_batches[0]})
        return output

class NeuralNetsCombined_128():

    #initialize the Neural Net (Only works with combined net for now)
    def __init__(self):
        self.sess, self.output_white, self.input_white, self.output_black, self.input_black= instantiate_session_both_128()

    #evaluate a list of game nodes or a game board directly (must pass in player_color in the latter case)
    def evaluate(self, game_nodes, player_color=None, already_converted=False):
        cdef:
            list moves = []
            np.ndarray output
            list board_representations
            list inference_batches
            int batch_size = 16384
        if not already_converted:
            if player_color is not None: #1 board + 1 color from direct policy net call
                board_representations = [convert_board_to_2d_matrix_POEB(game_nodes, player_color)]
            else:
                board_representations = [convert_board_to_2d_matrix_POEB(node['game_board'], node['color']) for node in
                                     game_nodes]
        else:
            board_representations = game_nodes
        if player_color is None and not already_converted:
            player_color = game_nodes[0]['color']
        if len(board_representations) > batch_size:
            inference_batches = batch_split_no_labels(board_representations, batch_size)
        else:
            inference_batches = [board_representations]
        if player_color == 'White':#Leave as is for WaNN as White. White == Winner, Black == All
            y_pred = self.output_white
            X = self.input_white
        else:
            y_pred = self.output_black
            X = self.input_black
        # self.sess.close()
        # self.sess, self.output, self.input = get_NN(player_color)

        if len(inference_batches) >1:
            for batch in inference_batches:
                predicted_moves = self.sess.run(y_pred, feed_dict={X: batch})
                moves.extend(predicted_moves)
            output = np.ndarray(moves, dtype =DTYPE)
        else:
            output = self.sess.run(y_pred, feed_dict={X: inference_batches[0]})
        return output

class NeuralNet():

    #initialize the Neural Net (only 1 as of 03/10/2017)
    def __init__(self):
        self.sess, self.output, self.input = instantiate_session()

    #evaluate a list of game nodes or a game board directly (must pass in player_color in the latter case)
    def evaluate(self, game_nodes, player_color=None, already_converted=False):
        if not already_converted:
            if player_color is not None: #1 board + 1 color from direct policy net call
                board_representations = [convert_board_to_2d_matrix_POEB(game_nodes, player_color)]
            else:
                board_representations = [convert_board_to_2d_matrix_POEB(node['game_board'], node['color']) for node in
                                     game_nodes]
        else:
            board_representations = game_nodes
        batch_size = 16384
        inference_batches = batch_split_no_labels(board_representations, batch_size)
        output = []
        for batch in inference_batches:
            predicted_moves = self.sess.run(self.output, feed_dict={self.input: batch})
            output.extend(predicted_moves)
        return output

    def __enter__(self):
        return self

    #close the tensorflow session when we are done.
    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()
