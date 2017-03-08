from monte_carlo_tree_search.TreeNode import TreeNode
from monte_carlo_tree_search.tree_search_utils import choose_UCT_move, \
    update_values_from_policy_net, get_UCT, randomly_choose_a_winning_move, choose_UCT_or_best_child, SimulationInfo
from monte_carlo_tree_search.tree_builder import visit_single_node_and_expand, random_rollout, update_win_status_from_children, expand_child_nodes_wrt_NN
from tools.utils import move_lookup_by_index
from Breakthrough_Player.board_utils import print_board
import time
import sys

# Expansion MCTS: Traditional MCTS with expansion using policy net to generate prior values
# start with root and put in NN queue, (level 0)
# while time to think,
# 1. MCTS search to find the best move
# 2. When we reach a leaf node, expand, evaluate with policy net, and update prior values on children
# 3. keep searching to desired depth (final depth = depth at expansion + depth_limit)
# 4. do random rollouts. repeat 1.

#OR

# EBFS MCTS: Hybrid BFS MCTS with expansion to desired depth using policy net to generate prior values
# since policy prunes children inherently(O = num_top_moves^depth), makes continued expansion tractable.
# Therefore, we can expand tree nodes prior to MCTS to take advantage of batch NN processing
#
# start with root and put in NN queue, (level 0)
# while time to think,
# 1. MCTS search to find the best move
# 2. When we reach a leaf node, continually expand to desired depth (final depth = depth at expansion + depth_limit),
#   evaluate with policy net, and update prior values on children
# 3. Find the best unexpanded descendant
# 4. do a random rollout. repeat 1.

#TODO: for root-level parallelism here, add stochasticity to UCT constant?

#for production, remove simulation info logging code
def MCTS_with_expansions(game_board, player_color, time_to_think,
                         depth_limit, previous_move, move_number, log_file=sys.stdout, MCTS_Type='EBFS MCTS'):
    with SimulationInfo(log_file) as sim_info:

        sim_info.root = root = assign_root(game_board, player_color, previous_move, move_number)
        start_time = time.time()
        done = False
        while time.time() - start_time < time_to_think and not done:
            done = run_MCTS_with_expansions_simulation(root, depth_limit, start_time, sim_info)
        best_child = randomly_choose_a_winning_move(root)
        best_move = move_lookup_by_index(best_child.index, player_color)
        print_best_move(player_color, best_move, sim_info)
    return best_child, best_move #save this root to use next time

def assign_root(game_board, player_color, previous_move, move_number):
    if previous_move is None: #no tree to reuse
        root = TreeNode(game_board, player_color, None, None, move_number)
    else:
        root = None
        for child in previous_move.children: #check if we can reuse tree
            if child.game_board == game_board:
                root = child
                root.parent = None # separate new root from old parent reference
                break
        if root is None: #can't reuse tree
            root = TreeNode(game_board, player_color, None, None, move_number)
        previous_move.children = None #to dealloc unused tree
    return  root

def run_MCTS_with_expansions_simulation(root, depth_limit, start_time, sim_info):
    play_MCTS_game_with_expansions(root, 0, depth_limit, sim_info, 0)
    if sim_info.counter  % 5 == 0:  # log every 5th simulation
        print_expansion_statistics(sim_info, start_time)
    sim_info.prev_game_tree_size = len(sim_info.game_tree)
    sim_info.counter += 1
    print_forced_win(root.win_status, sim_info)
    if root.win_status is None:
        return False
    else:
        return True

def play_MCTS_game_with_expansions(root, depth, depth_limit, sim_info, this_height):
    if root.gameover is False: #terminates at end-of-game moves.
        # could check for win_status, but UCT methods will filter moves and MCTS will end if we have guaranteed wins/losses
        if root.children is None: #reached non-game ending leaf node
            expand_leaf_node(root, depth, depth_limit, sim_info, this_height)
        else:#keep searching tree
            select_best_child(root, depth, depth_limit, sim_info, this_height)

def expand_leaf_node(root, depth, depth_limit, sim_info, this_height):
    if depth < depth_limit:
        expand_and_select(root, depth, depth_limit, sim_info, this_height)
    else:  # reached depth limit
        play_simulation(root, sim_info, this_height)
        # return here


def expand_and_select(node, depth, depth_limit, sim_info, this_height):
    #TODO: check; this should work for both. depth_limit 1 = Expansion MCTS with pre-pruning, depth_limit > 1 = EBFS MCTS
    expand(node, depth, depth_limit, sim_info, this_height)
    if node.win_status is None:  # if we don't know the win status, search deeper
        select_unexpanded_child(node, depth_limit, depth_limit, sim_info, this_height)
            # since NN expansion went depth_limit deeper, this will just make it end up at a rollout
    else:
        # hacky fix: if node expanded child is a winner/loser, won't prune children and doesn't initialize all visits => buggy UCT values
        for child in node.children:
            if child.visits == 0:
                child.visits = 1
                # else: return here; #don't bother checking past sub-tree if we already know there is a guaranteed win/loss

#TODO: reimplement postpruning? at least near end of game? i.e. if height > 40?
def expand(node, depth, depth_limit, sim_info, this_height):
    # TODO: keep expanding single node to depth limit? or set depth = depth_limit so it does a rollout after expansion?
    if node.height > 40: #expand one at a time and prune after checking immediate wins/losses
        expand_node(node, depth, depth_limit, sim_info, this_height)
    else: # batch expansion and preprune (misses potential instant gameovers?)
        expand_and_update_children_wrt_NN(node, depth, depth_limit, sim_info, this_height)


def select_unexpanded_child(node, depth, depth_limit, sim_info, this_height):
    move = choose_UCT_move(node)
    # fact: since this node was previously unexpanded, all subsequent nodes will be unexpanded
    # => will increment depth each subsequent call
    play_MCTS_game_with_expansions(move, depth, depth_limit, sim_info,
                                   this_height + 1)  # search until depth limit

def expand_node(node, depth, depth_limit, sim_info, this_height):
    visit_single_node_and_expand([node, node.color])
    update_values_from_policy_net([node])

def expand_and_update_children_wrt_NN(node, depth, depth_limit, sim_info, this_height):
    expand_child_nodes_wrt_NN([node], depth,
                              depth_limit, sim_info)  # searches to a depth to take advantage of NN batch processing
    log_expanded_node(node, this_height, sim_info)

def log_expanded_node(node, this_height, sim_info):
    # put expanded node and height into game tree
    if node.parent is None:
        node.height = this_height
    else:
        node.height = node.parent.height+1

def play_simulation(root, sim_info, this_height):
    random_rollout(root)
    log_max_tree_height(sim_info, this_height)

def log_max_tree_height(sim_info, this_height):
    if this_height > sim_info.game_tree_height:  # keep track of max tree height
        sim_info.game_tree_height = this_height

def select_best_child(node, depth, depth_limit, sim_info, this_height):
    if node.win_status is None:  # if False => all kids are winners, if True => some kid is a loser
        update_win_status_from_children(node)  # to check for forced wins/losses
    move = choose_UCT_or_best_child(node) #if child is a leaf, chooses policy net's top choice
    play_MCTS_game_with_expansions(move, depth, depth_limit, sim_info, this_height + 1)

def print_simulation_statistics(sim_info):
    print("Monte Carlo Game {iteration}\n"
          "Played at root   Height {height}:    UCT = {uct}     wins = {wins}       visits = {visits}\n".format(
        height = sim_info.root.height, uct=0, wins=sim_info.root.wins, visits=sim_info.root.visits,
        iteration=sim_info.counter+1), file=sim_info.file)
    print_board(sim_info.root.game_board, sim_info.file)
    print("\n", file=sim_info.file)
    for i in range(0, len(sim_info.game_tree)):
        node_parent = sim_info.game_tree[i].parent
        if node_parent is None:
            UCT = 0
        else:
            UCT = get_UCT(sim_info.game_tree[i], node_parent.visits)
        print("Node {i} Height {height}:    UCT = {uct}     wins = {wins}       visits = {visits}".format(
            i=i, height=sim_info.game_tree[i].height, uct=UCT, wins=sim_info.game_tree[i].wins, visits=sim_info.game_tree[i].visits),
            file=sim_info.file)
        print_board(sim_info.game_tree[i].game_board, sim_info.file)
        print("\n", file=sim_info.file)

def print_expansion_statistics(sim_info, start_time):
    print_simulation_statistics(sim_info)
    print("Number of Tree Nodes added in simulation {counter} = "
          "{nodes} in {time} seconds\n"
          "Current tree height = {height}".format(counter=sim_info.counter+1,
                                                  nodes=len(sim_info.game_tree) - sim_info.prev_game_tree_size,
                                                  time=time.time() - start_time,
                                                  height=sim_info.game_tree_height), file=sim_info.file)
def print_best_move(player_color, best_move, sim_info):
    print("For {player_color}, best move is {move}\n".format(player_color=player_color, move=best_move),
          file=sim_info.file)


def print_forced_win(root_win_status, sim_info):
    if root_win_status is True:
        print('I have forced a win!', file=sim_info.file)
    elif root_win_status is False:
        print(r'I\'m doomed!', file=sim_info.file)