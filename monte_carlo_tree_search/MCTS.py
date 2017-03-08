from monte_carlo_tree_search.expansion_MCTS_functions import MCTS_with_expansions
from monte_carlo_tree_search.BFS_MCTS_functions import MCTS_BFS_to_depth_limit
from Breakthrough_Player.board_utils import generate_policy_net_moves, get_best_move


class MCTS(object):
    #Option B: Traditional MCTS with expansion using policy net to generate prior values and prune tree
    # start with root and put in NN queue, (level 0)
    # while time to think,
    # 1. MCTS search to find the best move
    # 2. When we reach a leaf node, expand, evaluate with policy net, prune and update prior values on children
    # 3. keep searching to desired depth (final depth = depth at expansion + depth_limit)
    # 4. do random rollouts. repeat 1.

    def __init__(self, depth_limit, time_to_think, MCTS_type, MCTS_log_file):
        self.time_to_think = time_to_think
        self.depth_limit = depth_limit
        self.selected_child = None
        self.MCTS_type = MCTS_type
        self.log_file = MCTS_log_file

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    def evaluate(self, game_board, player_color):
        previous_child = self.selected_child
        if self.MCTS_type == 'Expansion MCTS' or self.MCTS_type == 'EBFS MCTS':
            self.selected_child, move = MCTS_with_expansions(game_board, player_color, self.time_to_think, self.depth_limit, previous_child, self.log_file, self.MCTS_type)
        elif self.MCTS_type == 'BFS MCTS':
            self.selected_child, move = MCTS_BFS_to_depth_limit(game_board, player_color, self.time_to_think, self.depth_limit, previous_child, self.log_file)
        elif self.MCTS_type == 'Policy':
            ranked_moves = generate_policy_net_moves(game_board, player_color)
            move = get_best_move(game_board, ranked_moves)
        return move

