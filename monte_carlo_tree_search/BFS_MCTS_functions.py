from monte_carlo_tree_search.TreeNode import TreeNode
from monte_carlo_tree_search.tree_search_utils import choose_UCT_move, \
    update_values_from_policy_net, get_UCT, randomly_choose_a_winning_move, choose_UCT_or_best_child, SimulationInfo, update_win_status_from_children
from monte_carlo_tree_search.tree_builder import build_game_tree, visit_single_node_and_expand, random_eval
from tools.utils import move_lookup_by_index
from Breakthrough_Player.board_utils import print_board
import time
import sys

#Option A: BFS expansion
# start with root and put in NN queue, (level 0)
# enumerate legal moves,
# make all legal moves to generate new nodes (level 1) and put those in queue with opponent color,
# 1. search sequentially to desired depth,
# 2. run policy net JUST ONCE HERE when we get to bottom of tree,
# 3. keep searching to bottom of tree where we do random rollouts.

def MCTS_BFS_to_depth_limit(game_board, player_color, time_to_think=1000, depth_limit=5, previous_move=None, log_file=sys.stdout, policy_net=None):
    with SimulationInfo(log_file) as sim_info:
        # wanderer = 93,650k nodes  6GB
        #this = 270k nodes 50 GB..
        start_time = time.time()
        root = TreeNode(game_board, player_color, None, None)
        sim_info.game_tree = build_game_tree(player_color, 0, [root], depth_limit)
        print("Number of Tree Nodes = {nodes} in {time} seconds".format(nodes=len(sim_info.game_tree), time=time.time()-start_time))
        update_values_from_policy_net(sim_info.game_tree)

        while (time.time()- start_time < time_to_think ):
            run_BFS_MCTS_simulation(sim_info)

        print("seconds taken: {}".format(time.time() - start_time))
        best_move = move_lookup_by_index(randomly_choose_a_winning_move(root).index, player_color)
        print("For {player_color}, best move is {move}\n".format(player_color=player_color, move=best_move),file=sim_info.file)
        sim_info.file.close()
    return root, best_move

def run_BFS_MCTS_simulation(sim_info):
    root = sim_info.game_tree[0]
    BFS_MCTS_game(root)
    if sim_info.counter % 400 == 0:  # log every 400th simulation
        print_simulation_statistics(sim_info)
    sim_info.counter += 1

def BFS_MCTS_game(root):
    if root.children is None:
        #bottom of tree; random rollout
       random_eval(root)
    else:
        move = choose_UCT_move(root)
        BFS_MCTS_game(move)
