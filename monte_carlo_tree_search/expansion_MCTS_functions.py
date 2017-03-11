from monte_carlo_tree_search.TreeNode import TreeNode
from monte_carlo_tree_search.tree_search_utils import choose_UCT_move, update_value_from_policy_net_async,\
    update_values_from_policy_net, get_UCT, randomly_choose_a_winning_move, choose_UCT_or_best_child, SimulationInfo
from monte_carlo_tree_search.tree_builder import visit_single_node_and_expand, random_rollout,  expand_descendants_to_depth_wrt_NN
from tools.utils import move_lookup_by_index
from Breakthrough_Player.board_utils import print_board
import time
import sys
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool
import threading

# Expansion MCTS: Traditional MCTS with expansion using policy net to generate prior values
# start with root and put in NN queue, (level 0)
# while time to think,
# 1. MCTS search to find the best move
# 2. When we reach a leaf node, expand, evaluate with policy net, and update prior values on children
# 3. keep searching to desired depth (final depth = depth at expansion + depth_limit)
# 4. do random rollouts. repeat 1.
# Note: if this is the case,

#OR

# Expansion MCTS + pruning (EBFS MCTS with depth_limit=1): Traditional MCTS with expansion using policy net to generate prior values and pruning of top kids.
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

NN_queue_lock = threading.Lock()#making sure the queue is accessed one at a time
async_update_lock = threading.Lock() #for expansions, updates, and selecting a child
ready = threading.Condition()
NN_input_queue = []

#for production, remove simulation info logging code
def MCTS_with_expansions(game_board, player_color, time_to_think,
                         depth_limit, previous_move, move_number, log_file=sys.stdout, MCTS_Type='EBFS MCTS', policy_net=None):
    with SimulationInfo(log_file) as sim_info:

        sim_info.root = root = assign_root(game_board, player_color, previous_move, move_number, sim_info)
        start_time = time.time()
        done = False
        if MCTS_Type == 'Expansion MCTS Pruning':
            pruning = True
        else:
            pruning = False
        thread1 = ThreadPool(processes=3)
        num_processes = 10
        thread2 = ThreadPool(processes=num_processes)#TODO check this
        while time.time() - start_time < time_to_think and not done:
            NN_args = [done, pruning, policy_net]
            MCTS_args = [[root,depth_limit, start_time, sim_info, MCTS_Type, policy_net] * num_processes]
            # thread1.apply_async(async_node_updates, NN_args)
            # result = thread2.apply_async(run_MCTS_with_expansions_simulation, (root,
            #                                                               depth_limit, start_time, sim_info, MCTS_Type, policy_net))
            # done = result.get()
            thread2.map_async(run_MCTS_with_expansions_simulation, MCTS_args)
            done = run_MCTS_with_expansions_simulation(MCTS_args[0])
            # done = done.get()[0]
        best_child = randomly_choose_a_winning_move(root)
        best_move = move_lookup_by_index(best_child.index, player_color)

        print_best_move(player_color, best_move, sim_info)
    return best_child, best_move #save this root to use next time

def assign_root(game_board, player_color, previous_move, move_number, sim_info):
    if previous_move is None or previous_move.children is None: #no tree to reuse
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

def append_subtree(node, sim_info): #DFS
    if node.children is not None:
        for child in node.children:
            sim_info.game_tree.append(child)
            append_subtree(child, sim_info)

def run_MCTS_with_expansions_simulation(args): #change back to starmap?
    # root, depth_limit, start_time, sim_info, MCTS_Type, policy_net = arg for arg in args
    root = args[0]
    depth_limit = args[1]
    start_time = args[2]
    sim_info = args[3]
    MCTS_Type = args[4]
    policy_net = args[5]
    play_MCTS_game_with_expansions(root, 0, depth_limit, sim_info, 0, MCTS_Type, policy_net)
    with async_update_lock: #so prints aren't interleaved
        if sim_info.counter  % 5 == 0:  # log every 5th simulation
            print_expansion_statistics(sim_info, start_time)
        sim_info.prev_game_tree_size = len(sim_info.game_tree)
        sim_info.counter += 1
        print_forced_win(root.win_status, sim_info)
    if root.win_status is True: # if we have a guaranteed win, we are done
        return True
    else:#just because a good opponent has a good reply for every move doesn't mean all opponents will;
        #fix this, since the MCTS is running for both colors,
        # this just returns for the entire TTT since the UCT is hardcoded to give back a winning move at each depth
        return False

def play_MCTS_game_with_expansions(root, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net):
    #todo: should we really assume opponent is as smart as we are and not check subtrees of nodes with win statuses?
    if root.gameover is False: #terminates at end-of-game moves #03/10/2017 originally had it stop at "guaranteed" wins/losses
        # but it prematurely stops search even though opponent may not be as smart; might as well let it keep searching just in case
        if root.children is None: #reached non-game ending leaf node
            expand_leaf_node(root, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net)
        else:#keep searching tree
            select_best_child(root, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net)

def expand_leaf_node(root, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net):
    if depth < depth_limit:
        expand_and_select(root, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net)
    else:  # reached depth limit
        play_simulation(root, sim_info, this_height)
        # return here


def expand_and_select(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net):
    #TODO: check; this should work for both. depth_limit 1 = Expansion MCTS with pre-pruning, depth_limit > 1 = EBFS MCTS
    expand(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net)
    sim_info.game_tree.append(node)
    # if node.win_status is None:  # if we don't know the win status, search deeper
    if MCTS_Type == 'EBFS MCTS': # since NN expansion went depth_limit deeper, this will just make it end up at a rollout
        depth = depth_limit
    # else:
    #     sim_info.game_tree.append(node)
    select_unexpanded_child(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net)

    # else:
    #     # hacky fix: if node expanded child is a winner/loser, won't prune children and doesn't initialize all visits => buggy UCT values
    #     for child in node.children:
    #         if child.visits == 0:
    #             child.visits = 1
    #             # else: return here; #don't bother checking past sub-tree if we already know there is a guaranteed win/loss

#TODO: EBFS  MCTS class increases depth limit as game progresses
def expand(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net):
    if MCTS_Type == 'Expansion MCTS': #Classic MCTS
        #EMCTS, expanding one at a time and only pruning children that lead to loss for parent node (03/09/17 no pruning at all since may lead to a childless root)
        expand_node_and_update_children(node, depth, depth_limit, sim_info, policy_net, this_height)
    elif MCTS_Type == 'Expansion MCTS Post-Pruning' or MCTS_Type == 'Expansion MCTS Pruning': #03/07/2017 best performance?
        #EMCTS, expanding one at a time and POST pruning children that lead to loss for parent node or not in top NN
        expand_node_and_update_children(node, depth, depth_limit, sim_info, this_height, policy_net, pruning=True)
        #TODO: bother with making this work with async updating?
    else: #EBFS MCTS, pre-pruning and expanding in batches to depth limit
        # TODO: keep expanding single node to depth limit? or set depth = depth_limit so it does a rollout after expansion?
        if node.height > 40:
            #batch expansion and prune children not in top NN picks AFTER checking for immediate wins/losses
            without_enumerating = False
        else:
            # batch expansion only on children in top NN picks
            #(misses potential instant gameovers not in top NN picks, so only call early in game)
            without_enumerating = True

        expand_descendants_to_depth_wrt_NN([node], without_enumerating, depth,depth_limit,
                                                sim_info, async_update_lock, policy_net)  # searches to a depth to take advantage of NN batch processing


def select_unexpanded_child(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net):
    with async_update_lock:
        move = node
        while move.children is not None: #get furthest descendant; happens if another thread expanded this node before we had a chance
            move = choose_UCT_move(move)
            this_height += 1

    # fact: since this node was previously unexpanded, all subsequent nodes will be unexpanded
    # => will increment depth each subsequent call
    play_MCTS_game_with_expansions(move, depth+1, depth_limit, sim_info,
                                   this_height + 1, MCTS_Type, policy_net)  # search until depth limit

def expand_node_and_update_children(node, depth, depth_limit, sim_info, this_height, policy_net, pruning=False): #use if we want traditional MCTS
    if not node.expanded: #in case we're multithreading and we ended up with the same node to expand
        without_enumerating = False
        expand_descendants_to_depth_wrt_NN([node], without_enumerating, depth, depth_limit, sim_info, async_update_lock, policy_net) #prepruning
            # visit_single_node_and_expand([node, node.color])
            # sim_info.game_tree.append(node)
            # with NN_queue_lock: #for async updates
            #     NN_input_queue.append(node)
            # update_values_from_policy_net([node], policy_net, pruning) #for single-threading expansions with POST pruning;

def async_node_updates(done, pruning, policy_net): #thread enters here
        if len (NN_input_queue) > 0:
            with NN_queue_lock:
                thread = threading.local() #we don't need to pass this in, right?
                thread.batch_examples = NN_input_queue.copy()
                NN_input_queue.clear() #reset the queue to empty
            update_value_from_policy_net_async(thread.batch_examples, async_update_lock, policy_net)


def play_simulation(root, sim_info, this_height):
    random_rollout(root)
    log_max_tree_height(sim_info, this_height)

def log_max_tree_height(sim_info, this_height):
    if this_height > sim_info.game_tree_height:  # keep track of max tree height
        sim_info.game_tree_height = this_height

def select_best_child(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net):
    with async_update_lock: #make sure it's not being updated asynchronously
        move = choose_UCT_or_best_child(node) #if child is a leaf, chooses policy net's top choice
    play_MCTS_game_with_expansions(move, depth, depth_limit, sim_info, this_height + 1, MCTS_Type, policy_net)

def print_simulation_statistics(sim_info):#TODO: try not calling this and see if this is what is slowing down the program
    print("Monte Carlo Game {iteration}\n"
          "Played at root   Height {height}:    Player = {color}    UCT = {uct}     wins = {wins}       visits = {visits}\n".format(
        height = sim_info.root.height, color=sim_info.root.color, uct=0, wins=sim_info.root.wins, visits=sim_info.root.visits,
        iteration=sim_info.counter+1), file=sim_info.file)
    print_board(sim_info.root.game_board, sim_info.file)
    print("\n", file=sim_info.file)
    for i in range(0, len(sim_info.game_tree)):
        node_parent = sim_info.game_tree[i].parent
        if node_parent is None:
            UCT = 0
        else:
            UCT = get_UCT(sim_info.game_tree[i], node_parent.visits)
        print("Node {i} Height {height}:    Player = {color}    UCT = {uct}     wins = {wins}       visits = {visits}".format(
            i=i, height=sim_info.game_tree[i].height, color=sim_info.game_tree[i].color, uct=UCT, wins=sim_info.game_tree[i].wins, visits=sim_info.game_tree[i].visits),
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