from monte_carlo_tree_search.expansion_MCTS_functions import MCTS_with_expansions
from monte_carlo_tree_search.BFS_MCTS_functions import MCTS_BFS_to_depth_limit
from Breakthrough_Player.board_utils import get_best_move, get_NN
from tools.utils import convert_board_to_2d_matrix_POEB, batch_split_no_labels
import re


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

    def evaluate(self, game_board, player_color):
        previous_child = self.selected_child
        if self.MCTS_type == 'EBFS MCTS': #pruning plus depth expansions      #NOTE: good in theory, untested
            # depth_limit = self.height // 20 + self.depth_limit  # check 1 level deeper every 10 moves. Will never go over 4 levels deeper since EMCTS stops doing that at height 40
            self.selected_child, move = MCTS_with_expansions(game_board, player_color, self.time_to_think, self.depth_limit, previous_child, self.last_opponent_move, self.height, self.log_file, self.MCTS_type, self.policy_net, self.game_num)
        elif self.MCTS_type == 'Expansion MCTS' \
            or self.MCTS_type == 'Expansion MCTS Pruning' \
            or self.MCTS_type ==  'MCTS Asynchronous'\
            or self.MCTS_type == 'Expansion MCTS Post-Pruning': #pruning with no depth expansions       #NOTE: seemed to be good initially
            self.selected_child, move = MCTS_with_expansions(game_board, player_color, self.time_to_think, self.depth_limit, previous_child, self.last_opponent_move, self.height, self.log_file, self.MCTS_type, self.policy_net, self.game_num)
        elif self.MCTS_type == 'BFS MCTS':
            self.selected_child, move = MCTS_BFS_to_depth_limit(game_board, player_color, self.time_to_think, self.depth_limit, previous_child, self.log_file, self.policy_net)
        elif self.MCTS_type == 'Policy':
            ranked_moves = self.policy_net.evaluate(game_board, player_color)
            move = get_best_move(game_board, ranked_moves)
        elif self.MCTS_type == 'Wanderer':
            if self.color == 'White':
                move_regex = re.compile(r".*play\sw\s([a-h]\d.[a-h]\d).*",
                                        re.IGNORECASE)
                self.policy_net.expect('play w.*')

                move = self.policy_net.before.decode('utf-8') + self.policy_net.after.decode('utf-8')
                print(move, file=self.log_file)
                move = move_regex.search(move).group(1)
            else:
                move_regex = re.compile(r".*play\sb\s([a-h]\d.[a-h]\d).*",
                                        re.IGNORECASE)
                self.policy_net.expect('play b.*')

                move = self.policy_net.before.decode('utf-8') + self.policy_net.after.decode('utf-8')
                print(move, file=self.log_file)
                move = move_regex.search(move).group(1)

        return move


class NeuralNet():

    #initialize the Neural Net (only 1 as of 03/10/2017)
    def __init__(self):
        self.sess, self.output_white, self.input_white, self.output_black, self.input_black= get_NN()

    #evaluate a list of game nodes or a game board directly (must pass in player_color in the latter case)
    def evaluate(self, game_nodes, player_color=None, already_converted=False):
        if not already_converted:
            if player_color is not None: #1 board + 1 color from direct policy net call
                board_representations = [convert_board_to_2d_matrix_POEB(game_nodes, player_color)]
            else:
                board_representations = [convert_board_to_2d_matrix_POEB(node.game_board, node.color) for node in
                                     game_nodes]
        else:
            board_representations = game_nodes
        node_color = game_nodes[0].color
        batch_size = 16384
        inference_batches = batch_split_no_labels(board_representations, batch_size)
        output = []
        if node_color == 'White':
            y_pred = self.output_white
            X = self.input_white
        else:
            y_pred = self.output_black
            X = self.input_black
        # self.sess.close()
        # self.sess, self.output, self.input = get_NN(player_color)

        for batch in inference_batches:
            predicted_moves = self.sess.run(y_pred, feed_dict={X: batch})
            output.extend(predicted_moves)
        return output

    def __enter__(self):
        return self

    #close the tensorflow session when we are done.
    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

# class NeuralNet():
#
#     #initialize the Neural Net (only 1 as of 03/10/2017)
#     def __init__(self):
#         self.sess, self.output_white, self.input_black = get_NN()
#
#     #evaluate a list of game nodes or a game board directly (must pass in player_color in the latter case)
#     def evaluate(self, game_nodes, player_color=None, already_converted=False):
#         if not already_converted:
#             if player_color is not None: #1 board + 1 color from direct policy net call
#                 board_representations = [convert_board_to_2d_matrix_POEB(game_nodes, player_color)]
#             else:
#                 board_representations = [convert_board_to_2d_matrix_POEB(node.game_board, node.color) for node in
#                                      game_nodes]
#         else:
#             board_representations = game_nodes
#         batch_size = 16384
#         inference_batches = batch_split_no_labels(board_representations, batch_size)
#         output = []
#         for batch in inference_batches:
#             predicted_moves = self.sess.run(self.output_white, feed_dict={self.input_black: batch})
#             output.extend(predicted_moves)
#         return output
#
#     def __enter__(self):
#         return self
#
#     #close the tensorflow session when we are done.
#     def __exit__(self, exc_type, exc_value, traceback):
#         self.sess.close()
