from monte_carlo_tree_search.TreeNode import TreeNode
from monte_carlo_tree_search.tree_search_utils import choose_UCT_move, \
    get_UCT, randomly_choose_a_winning_move, choose_UCT_or_best_child, SimulationInfo, random_rollout
from monte_carlo_tree_search.tree_builder import visit_single_node_and_expand, random_eval,  expand_descendants_to_depth_wrt_NN
from tools.utils import move_lookup_by_index
from Breakthrough_Player.board_utils import print_board
import time
import sys
import random
from multiprocessing.pool import ThreadPool
import threading

# Expansion MCTS with tree saving and (optional) tree parallelization:
# Traditional MCTS with expansion using policy net to generate prior values
# start with root and expand with NN, (level 0)
# while time to think,
# 1. MCTS search to find the best move
# 2. When we reach a leaf node, expand, evaluate with policy net, and update prior values on children
# 3. keep searching to desired depth (final depth = depth at expansion + depth_limit)
# 4. do random rollouts. repeat 1.


#OR
# Expansion MCTS with pruning, tree saving and (optional) tree parallelization:
# Traditional MCTS with expansion using policy net to generate prior values and pruning of unpromising children..
# start with root and expand with NN, (level 0)
# while time to think,
# 1. MCTS search to find the best move
# 2. When we reach a leaf node, expand, evaluate with policy net, and update prior values on children
# 3. keep searching to desired depth (final depth = depth at expansion + depth_limit)
# 4. do random rollouts. repeat 1.
#BEST SO FAR

#OR
# EBFS MCTS: Hybrid BFS MCTS with expansion to desired depth using policy net to generate prior values
# since we can use policy net to prune children, makes continued expansion tractable.
# (O = num_moves_over_threshold ^depth)
# Therefore, we can expand tree nodes in batches during MCTS to take advantage of batch NN processing
#
# start with root and put in NN queue, (level 0)
# while time to think,
# 1. MCTS search to find the best move
# 2. When we reach a leaf node, continually expand to desired depth (final depth = depth at expansion + depth_limit),
#   evaluate with policy net, and update prior values on children
# 3. Find the best unexpanded descendant
# 4. do a random rollout. repeat 1.
#not so good (but tested prior to more fine-grained pruning or multithreading)


#OR
#
# MCTS with asynchronous policy net updates and pruning:
# normal MCTS while another thread waits for policy net output and updates/prunes accordingly.
# policy net output updater thread expands tree nodes in batches (whenever the queue has some element, it runs)
#
# start with root and put in NN queue, (level 0)
# while time to think,
# 1. MCTS search to find the best move
# 2. When we reach a leaf node, expand, add to policy net queue for evaluation
# 2.a Separate thread evaluates policy net queue and updates/prunes children
# 3. keep searching to desired depth (final depth = depth at expansion + depth_limit)
# 4. do random rollouts. repeat 1.

# Pretty bad performance, the normal MCTS-running thread is mostly useless as python isn't fast enough for it to
# move away from dumb moves. Maybe if we had multiple threads doing policy net updates it wouldn't be so bad?
# But even then, MCTS may be taking processor cycle doing useless work whereas tree parallelization completely avoids that
# (we can't multiprocess since tensorflow complains about interleaving calls to NNs from separate processes)


NN_queue_lock = threading.Lock()#making sure the queue is accessed one at a time
async_update_lock = threading.Lock() #for expansions, updates, and selecting a child
NN_input_queue = [] #for expanded nodes that need to be evaluated asynchronously

#for production, remove simulation info logging code
def MCTS_with_expansions(game_board, player_color, time_to_think,
                         depth_limit, previous_move, move_number, log_file=sys.stdout, MCTS_Type='Expansion MCTS Pruning', policy_net=None):
    with SimulationInfo(log_file) as sim_info:
        sim_info.root = root = assign_root(game_board, player_color, previous_move, move_number, sim_info)
        sim_info.start_time = start_time = time.time()
        sim_info.time_to_think = time_to_think
        done = False
        if MCTS_Type == 'Expansion MCTS Pruning':
            pruning = True
        else:
            pruning = False
        thread1 = ThreadPool(processes=3)
        num_processes = 100
        thread2 = ThreadPool(processes=num_processes)
        while time.time() - start_time < time_to_think and not done:
            if MCTS_Type ==  'MCTS Asynchronous':
                NN_args = [done, pruning, sim_info, policy_net]
                thread1.apply_async(async_node_updates, NN_args)
            else:
                MCTS_args = [[root,depth_limit, time_to_think, sim_info, MCTS_Type, policy_net, start_time] * num_processes]
                thread2.map_async(run_MCTS_with_expansions_simulation, MCTS_args)
                done = run_MCTS_with_expansions_simulation(MCTS_args[0]) #have the main thread return done
        thread2.close()
        thread2.join()
        thread1.close()
        thread1.join()
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
    time_to_think = args[2]
    sim_info = args[3]
    MCTS_Type = args[4]
    policy_net = args[5]
    start_time = args[6]
    play_MCTS_game_with_expansions(root, 0, depth_limit, sim_info, 0, MCTS_Type, policy_net, start_time, time_to_think)
    with async_update_lock: #so prints aren't interleaved
        if sim_info.counter  % 5 == 0:  # log every 5th simulation
            print_expansion_statistics(sim_info, sim_info.start_time)
        sim_info.prev_game_tree_size = len(sim_info.game_tree)
        sim_info.counter += 1
        print_forced_win(root.win_status, sim_info)
    if root.win_status is True: # if we have a guaranteed win, we are done
        return True
    else:#just because a good opponent has a good reply for every move doesn't mean all opponents will;
        #fix this, since the MCTS is running for both colors,
        # this just returns for the entire TTT since the UCT is hardcoded to give back a winning move at each depth
        return False

def play_MCTS_game_with_expansions(root, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think):
    #todo: should we really assume opponent is as smart as we are and not check subtrees of nodes with win statuses?
    if root.gameover is False: #terminates at end-of-game moves #03/10/2017 originally had it stop at "guaranteed" wins/losses
        # but it prematurely stops search even though opponent may not be as smart; might as well let it keep searching just in case
        if root.children is None: #reached non-game ending leaf node
            expand_leaf_node(root, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think)
        else:#keep searching tree
            #TODO: 03/18/2017 this only executes in the beginning of MCTS, the "play till end of game" selection greedily chooses best child
            # select_best_child(root, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think) #normal recursive
            #while to keep UCTing until we find a good kid
            select_UCT_child(root, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think)


def expand_leaf_node(root, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think):
    if depth < depth_limit:
        expand_and_select(root, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think)
    else:  # reached depth limit
        play_simulation(root, sim_info, this_height)
        # return here


def expand_and_select(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think):
    expand(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net)
    #TODO: separate the in-tree depth limit vs the batch expansion depth limit.
    # if MCTS_Type == 'EBFS MCTS': # since NN expansion went depth_limit deeper, this will just make it end up at a rollout
    #     depth = depth_limit

    # select_unexpanded_child(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think)

        #TODO: 03/18/2017 10 PM greedy rollouts. Can also do eval from here if we don't want to do random rollouts
    greedy_rollout(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net)

    UCT_rollout(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think)

    random_rollout_EOG(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net)

    ##TODO: 03/18/2017 1 PMrandom rollouts to eval at unexpanded node
    # play_simulation(node, sim_info, this_height)

#TODO: EBFS  MCTS class increases depth limit as game progresses
def expand(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net):
    if MCTS_Type =='MCTS Asynchronous':
        expand_node_async(node)
    else:
        if MCTS_Type == 'Expansion MCTS' or \
                MCTS_Type == 'Expansion MCTS Post-Pruning':
            pre_pruning = False
            depth = depth_limit
        elif MCTS_Type == 'Expansion MCTS Pruning': #03/07/2017 best performance?
            pre_pruning = True
            depth = depth_limit
        #depth = depth_limit makes NN expansion run on only the node given (in-line MCTS)

        else: #EBFS MCTS, pre-pruning and expanding in batches to depth limit
            if node.height > 40:
                #batch expansion and prune children not in top NN picks AFTER checking for immediate wins/losses
                pre_pruning = False
            else:
                # batch expansion only on children in top NN picks
                #(misses potential instant gameovers not in top NN picks, so only call early in game)
                pre_pruning = True
        expand_node_and_update_children(node, depth, depth_limit, sim_info, this_height, policy_net, pre_pruning)

def select_unexpanded_child(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think):
    select_best_child(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think)

            #TODO: 03/18/2017 UCT to find child, expand, run game to end by batch expanding, choosing best move, repeat

    # fact: since this node was previously unexpanded, all subsequent nodes will be unexpanded
    # => will increment depth each subsequent call
    # play_MCTS_game_with_expansions(move, depth+1, depth_limit, sim_info,
    #                                this_height + 1, MCTS_Type, policy_net, start_time, time_to_think)  # search until depth limit

def greedy_rollout(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net):
    while node.gameover is False:
        with async_update_lock:  # make sure it's not being updated asynchronously
            while node.children is not None:
                node = randomly_choose_a_winning_move(node)  # if child is a leaf, chooses policy net's top choice
                this_height += 1
        expand(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net)

def UCT_rollout(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think):
    while node.gameover is False:
        with async_update_lock:  # make sure it's not being updated asynchronously
            while node.children is not None:
                node = choose_UCT_or_best_child(node, start_time, time_to_think)  # if child is a leaf, chooses policy net's top choice
                this_height += 1
        expand(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net)

def random_rollout_EOG(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net):
    while node.gameover is False:
        with async_update_lock:  # make sure it's not being updated asynchronously
            while node.children is not None:
                node = random.sample(node.children, 1)  # if child is a leaf, chooses policy net's top choice
                this_height += 1
        expand(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net)

def expand_node_and_update_children(node, depth, depth_limit, sim_info, this_height, policy_net, pre_pruning=False): #use if we want traditional MCTS
    if not node.expanded: #in case we're multithreading and we ended up with the same node to expand
        if pre_pruning:
            without_enumerating = True
        else:
            without_enumerating = False
        expand_descendants_to_depth_wrt_NN([node], without_enumerating, depth, depth_limit, sim_info, async_update_lock, policy_net) #prepruning

def expand_node_async(node):
    with NN_queue_lock:  # for async updates
        NN_input_queue.append(node)
    visit_single_node_and_expand([node, node.color])

def async_node_updates(done, pruning, sim_info, policy_net): #thread enters here
        if len (NN_input_queue) > 0:
            without_enumerating = True
            depth = depth_limit = 0
            with NN_queue_lock:
                thread = threading.local() #we don't need to pass this in, right?
                thread.batch_examples = NN_input_queue.copy()
                NN_input_queue.clear() #reset the queue to empty
            # update_value_from_policy_net_async(thread.batch_examples, async_update_lock, policy_net)
            expand_descendants_to_depth_wrt_NN(thread.batch_examples, without_enumerating, depth, depth_limit, sim_info, async_update_lock, policy_net)


def select_UCT_child(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think):
    with async_update_lock: #make sure it's not being updated asynchronously
        while node.children is not None:
            this_height += 1
            node = choose_UCT_or_best_child(node, start_time, time_to_think) #if child is a leaf, chooses policy net's top choice
    play_MCTS_game_with_expansions(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think)

def select_best_child(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think):
    with async_update_lock: #make sure it's not being updated asynchronously
        while node.children is not None:
            node = randomly_choose_a_winning_move(node) #if child is a leaf, chooses policy net's top choice
            this_height += 1
    play_MCTS_game_with_expansions(node, depth, depth_limit, sim_info, this_height, MCTS_Type, policy_net, start_time, time_to_think)

def play_simulation(root, sim_info, this_height):
    # random_eval(root)
    log_max_tree_height(sim_info, this_height)
    random_rollout(root)

# def play_random_game(root, sim_info, this_height):
#
#     # if game_saving_move(root):#TODO fix this
#     #     # game saving move
#     #     game_saving_move_exists = True
#     # else:
#     #     game_saving_move_exists = False
#     #
#     # for child in root.children:
#     # # game winning move(s)
#     #     if child.win_status is False:
#     #         return child
#     #     elif game_saving_move_exists:






    # win threat
    # prevent win threat

#     if friend behind and to the left of of you
# if friend behind and tot he right

    #     if enemy front and to the left of of you
    # if enemy front and tot he right
        #     if friend behind and to the left of of you
        # if friend behind and tot he right
# def game_saving_move(root):
#     #note if multiple game saving moves exist, only one is returned.
#     white = 'w'
#     black = 'b'
#     if root.color == 'White':
#         black_threat_row = root.game_board[2]
#         if black in black_threat_row.values() :
#             return True, list(black_threat_row.keys())[list(black_threat_row.values()).index(black)]
#         else:
#             return False, None
#     else:
#         white_threat_row = root.game_board[7]
#         if white in white_threat_row.values():
#             return True, list(white_threat_row.keys())[list(white_threat_row.values()).index(white)]
#         else:
#             return False, None





def log_max_tree_height(sim_info, this_height):
    if this_height > sim_info.game_tree_height:  # keep track of max tree height
        sim_info.game_tree_height = this_height

def print_simulation_statistics(sim_info):#TODO: try not calling this and see if this is what is slowing down the program
    root = sim_info.root
    start_time = sim_info.start_time
    time_to_think = sim_info.time_to_think
    print("Monte Carlo Game {iteration}\n"
          "Played at root   Height {height}:    Player = {color}    UCT = {uct}     wins = {wins}       visits = {visits}\n".format(
        height = root.height, color=root.color, uct=0, wins=root.wins, visits=root.visits,
        iteration=sim_info.counter+1), file=sim_info.file)
    print_board(sim_info.root.game_board, sim_info.file)
    print("\n", file=sim_info.file)
    #to see best reply
    if sim_info.root.children is not None:
        best_child = randomly_choose_a_winning_move(root)
        best_move =  move_lookup_by_index(best_child.index, root.color)
        best_child_UCT = get_UCT(best_child, root.visits, start_time, time_to_think)
        print("Current (Random) Best Move   {best_move}\n"
              "Height {height}:    Player = {color}    UCT = {uct}     wins = {wins}       visits = {visits}\n".format(
            best_move=best_move,
            height=best_child.height, color=best_child.color, uct=best_child_UCT, wins=best_child.wins,
            visits=best_child.visits), file=sim_info.file)
        print_board(best_child.game_board, sim_info.file)
        #to see predicted best counter
        if best_child.children is not None:
            best_counter_child = randomly_choose_a_winning_move(best_child)
            best_counter_move = move_lookup_by_index(best_counter_child.index, best_child.color)
            best_counter_child_UCT = get_UCT(best_counter_child, best_child.visits, start_time, time_to_think)
            print("Current Best Move (Random) Best Counter Move   {best_counter_move}\n"
                  "Height {height}:    Player = {color}    UCT = {uct}     wins = {wins}       visits = {visits}\n".format(
                best_counter_move=best_counter_move,
                height=best_counter_child.height, color=best_counter_child.color, uct=best_counter_child_UCT, wins=best_counter_child.wins,
                visits=best_counter_child.visits), file=sim_info.file)
            print_board(best_counter_child.game_board, sim_info.file)
            print("All Counter Moves",file=sim_info.file)
            for counter_child in best_child.children:
                counter_move = move_lookup_by_index(counter_child.index, best_child.color)
                counter_child_UCT = get_UCT(counter_child, best_child.visits, start_time, time_to_think)
                print("Counter Move   {counter_move}    UCT = {uct}     wins = {wins}       visits = {visits}\n".format(
                    counter_move=counter_move,
                    uct=counter_child_UCT,
                    wins=counter_child.wins,
                    visits=counter_child.visits), file=sim_info.file)
        else:
            print("No Counter Moves explored",file=sim_info.file)
    else:
        print("No Moves explored", file=sim_info.file)



    print("\n", file=sim_info.file)

    for i in range(0, len(sim_info.game_tree)):
        node_parent = sim_info.game_tree[i].parent
        if node_parent is None:
            UCT = 0
        else:
            UCT = get_UCT(sim_info.game_tree[i], node_parent.visits, start_time, time_to_think)
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