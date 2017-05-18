#cython: language_level=3, boundscheck=False

from monte_carlo_tree_search.TreeNode import TreeNode
from monte_carlo_tree_search.tree_search_utils import get_UCT, randomly_choose_a_winning_move, choose_UCT_or_best_child, \
    SimulationInfo, increment_threads_checking_node, decrement_threads_checking_node, transform_wrt_overwhelming_amount, backpropagate_num_checked_children, update_win_status_from_children
from monte_carlo_tree_search.tree_builder import expand_descendants_to_depth_wrt_NN, init_new_root
from tools.utils import move_lookup_by_index, initial_game_board
from Breakthrough_Player.board_utils import print_board, initial_piece_arrays, debug_game_board, debug_piece_arrays
from time import time
from sys import stdout
from math import ceil
import random
from multiprocessing.pool import ThreadPool
from threading import Lock, local, current_thread


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


NN_queue_lock = Lock()#making sure the queue is accessed one at a time
async_update_lock = Lock() #for expansions, updates, and selecting a child
NN_input_queue = [] #for expanded nodes that need to be evaluated asynchronously
#for production, remove simulation info logging code
def MCTS_with_expansions(game_board, player_color, time_to_think,
                         depth_limit, previous_move, last_opponent_move, move_number, log_file=stdout, MCTS_Type='Expansion MCTS Pruning', policy_net=None, game_num = -1):

    sim_info = SimulationInfo(log_file)
    time_to_think = time_to_think+1 #takes 1 second for first eval (io needs to warm up?)with threads, takes 1.125 times longer to finish up
    # sim_info['root'] = root = assign_root_reinforcement_learning(game_board, player_color, previous_move, last_opponent_move,  move_number, policy_net, sim_info)
    sim_info['game_num'] = game_num
    sim_info['time_to_think'] = time_to_think
    sim_info['start_time'] = time()

    sim_info['root'] = root = assign_root_reinforcement_learning(game_board, player_color, previous_move, last_opponent_move,  move_number, policy_net, sim_info)
    sim_info['main_pid'] = current_thread().name
    # if move_number == 0 or move_number == 1:
    #     time_to_think = 60
    # if root['height']>= 60: #still missing forced wins
    #     time_to_think = 6000



    #TODO: prune the tree later once it gets too big? Does this matter since current search will only check a subtree without a win status?
    # i.e. from height 60 (where all children are enumerated), if a node is a guaranteed loss_init for parent, remove?
    # ex if True, remove all but one loser child (since we only need one route to win)
    #if False

    def prune_red_herrings_from_parents_with_win_status(root): #if calling this, stop reexpanding nodes at height 60 with less than their number of legal children
        unvisited_queue = [root]
        while len (unvisited_queue)>0:
            node = unvisited_queue.pop()
            if node['height']>=60 and not node['gameover']:
                if node['win_status']is True: #pick all losing children
                    losing_children = []
                    for child in node['children']:
                        if child['win_status'] is False:
                            losing_children.append(child)
                    node['children']= losing_children#some forced win
                    unvisited_queue.extend(losing_children)
                    #note: this may be bad if path to loss_init is so long and you don't kkno
                # elif node['win_status']is False: #pick one out of all of your losing children.
                #     winning_child = None
                #     for child in node['children']:
                #         if child['win_status'] is True:
                #             winning_child = child
                #             break
                #     node['children']= [winning_child] #some doomed move
                #     unvisited_queue.append(winning_child)
                else: #win_status is False or None => some kids may be True, some may be None; check them all.
                    if node['children']is not None:
                        unvisited_queue.extend(node['children'])
            else:
                if node['children']is not None:
                    unvisited_queue.extend(node['children'])
    # if move_number == 0:
    #     reinit_gameover_values(root)

    if root['subtree_checked']:# or root['win_status']is True
        done = True
    else:
        done = False
    if MCTS_Type == 'Expansion MCTS Pruning':
        pruning = True
    else:
        pruning = False


    num_processes = 10
    parallel_search_threads = ThreadPool(processes=num_processes)

    sim_info['start_time'] = start_time = time()
    MCTS_args = [[root, depth_limit,  sim_info,  policy_net, start_time, time_to_think]] * int(num_processes)
    #BFS tree to reset node check semaphore
    reset_thread_flag(root)

    while time() - start_time < time_to_think and not done:
        if MCTS_Type ==  'MCTS Asynchronous':
            NN_args = [done, pruning, sim_info, policy_net]
            # async_NN_update_threads.apply_async(async_node_updates, NN_args)
        else:
            parallel_search_threads.starmap_async(run_MCTS_with_expansions_simulation, MCTS_args)
            done = run_MCTS_with_expansions_simulation(root,depth_limit, sim_info,  policy_net, start_time, time_to_think) #have the main thread return done
    search_time = time() - start_time
    parallel_search_threads.terminate()
    # parallel_search_threads.close()
    parallel_search_threads.join()
    true_search_time = time() - start_time





    # async_NN_update_threads.close()
    # async_NN_update_threads.join()



    best_child = randomly_choose_a_winning_move(root, max(0, game_num))

    # if (move_number == 0 or move_number == 1) and game_num > -1: #save each iteration. If training makes it worse, we can revert back to a better version and change search parameters.
    #     top_root = root
    #     while top_root['parent']is not None:
    #         top_root = top_root['parent']
    #
    #     output_file = open(r'G:\TruncatedLogs\PythonDataSets\DataStructures\GameTree\04102017DualNets10secsDepth80_TrueWinLossFieldBlack{}.p'.format(str(game_num)), 'wb')
    #     #online reinforcement learning: resave the root at each new game (if it was kept, values would have backpropagated)
    #     pickle.dump(top_root, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    #     output_file.close()
    #     print("done saving root in seconds = ")
    #     print(search_time)

    # if done: #subtree checked or has a win status: prune off unnecessary subtrees after height 60
    #     prune_red_herrings_from_parents_with_win_status(root)# note:

    print_forced_win(root['win_status'],sim_info)
    print("Time spent searching tree = {}".format(search_time), file=log_file)
    print("Time spent searching tree = {}".format(search_time))


    print("Time waiting for threads to close = {}".format(true_search_time - search_time), file=log_file)
    print("Total Time = {}".format(true_search_time), file=log_file)

    print("Time waiting for threads to close = {}".format(true_search_time - search_time))
    print("Total Time = {}".format(true_search_time))


    best_move = get_best_move(best_child, player_color, move_number, aggressive=None)
    print_expansion_statistics(sim_info, time())

    print_best_move(player_color, best_move, sim_info)
    print(best_move)

    if best_child['parent'] is not None:#separate for backgroundsearch
        parent = best_child['parent']
        parent['children'] = [best_child]
        parent['other_children'] =None
        parent['reexpanded_already']=True


    best_child['reexpanded_already'] =True
    if best_child['children'] is not None and best_child['other_children'] is not None:
        best_child['children'].extend(best_child['other_children']) # should never reappend anewroot

        #refresh win status since we unpruned the children
        parent = best_child['parent']
        if parent is not None:
            parent['win_status'] = None
            parent['subtree_checked'] = False
        best_child['win_status'] =None

        update_win_status_from_children(best_child)


        #refresh sub tree checked since we unpruned children
        best_child['subtree_checked'] =False
        backpropagate_num_checked_children(best_child)

        best_child['other_children'] =None


    return best_child, best_move #save this root to use next time

def reset_thread_flag(root):
    unvisited_queue = [root]
    while len(unvisited_queue) > 0:
        node = unvisited_queue.pop()
        node['threads_checking_node']= 0
        node['subtree_being_checked']= False
        node['num_children_being_checked']= 0
        node['being_checked']= False
        if node['children']is not None:
            unvisited_queue.extend(node['children'])


def assign_root_reinforcement_learning(game_board, player_color, previous_move, last_opponent_move, move_number, policy_net, sim_info):
    # if move_number == 0 and version > 0:
    #     input_file = open(r'G:\TruncatedLogs\PythonDataSets\DataStructures\GameTree\AgnosticRoot{}.p'.format(str(version-1)),
    #                            'r+b')
    #     root = pickle.load(input_file)
    #     input_file.close()
    white_pieces, black_pieces = initial_piece_arrays()
    if move_number == 0:
        if previous_move is not None: #saved root
            root = previous_move
        else:#saving a new board
            # print("WARNING: INITIALIZING A NEW BOARD TO BE SAVED", file=sim_info['file'])
            # answer = input("WARNING: INITIALIZING A NEW BOARD TO BE SAVED, ENTER \'yes\' TO CONTINUE")
            # if answer.lower() != 'yes':
            #     exit(-10)
            # debug_height = 54
            # debug_white_pieces, debug_black_pieces = debug_piece_arrays()
            # root = TreeNode(debug_game_board(), debug_white_pieces, debug_black_pieces, player_color, 0, None, debug_height)

            root = TreeNode(game_board, white_pieces, black_pieces, player_color, 0, None, move_number)
    elif move_number == 1: #WaNN is black
        if previous_move is None: #saving new board
            # print("WARNING: INITIALIZING A NEW BOARD TO BE SAVED", file=sim_info['file'])
            # answer = input("WARNING: INITIALIZING A NEW BOARD TO BE SAVED, ENTER \'yes\' TO CONTINUE")
            # if answer.lower() != 'yes':
            #     exit(-10)
            previous_move = TreeNode(initial_game_board(), white_pieces, black_pieces, 'White', 0, None, 0) #initialize a dummy start board state to get white's info
            root = init_new_root(last_opponent_move, game_board, player_color, previous_move, policy_net, sim_info,
                                 async_update_lock)
            print("Initialized and appended new subtree", file=sim_info['file'])
        else:
            root = None
            if previous_move['children'] is not None:
                for child in previous_move['children']:  # check if we can reuse tree
                    if child['game_board'] ==game_board:
                        root = child
                        print("Reused old tree", file=sim_info['file'])
                        break
            if root is None:  # no subtree; append new subtree to old parent
                root = init_new_root(last_opponent_move, game_board, player_color, previous_move, policy_net, sim_info,
                                     async_update_lock)
                print("Initialized and appended new subtree", file=sim_info['file'])


    else:#should have some tree to reuse or append a new one
        root = None
        if previous_move['children'] is not None:
            for child in previous_move['children']: #check if we can reuse tree
                if child['game_board'] ==game_board:
                    root = child
                    print("Reused old tree", file=sim_info['file'])
                    break
        if root  is None: #no subtree; append new subtree to old parent
            root = init_new_root(last_opponent_move, game_board, player_color, previous_move, policy_net, sim_info, async_update_lock)
            print("Initialized and appended new subtree", file=sim_info['file'])

    #if terminating threads didn't end correctly, clean this up on the way down.
    root['being_checked']= False
    root['threads_checking_node']= 0
    root['num_children_being_checked']= 0
    root['subtree_being_checked']= False

    #can't do this anymore if we are background searching
    # #break off tree so python can garbage collect from memory
    # if root['parent']is not None:
    #     root['parent']['children'] =None
    # root['parent']= None #garbage collect
    if root['parent']is not None:
        parent = root['parent']
        if parent['parent'] is not None:
            grandparent = parent['parent']
            if grandparent['parent'] is not None:
                great_grandparent = grandparent['parent']
                great_grandparent['children'] = None
                grandparent['parent'] = None
    return  root

def get_best_move(best_child, player_color, move_number, aggressive=None, on=False):
    if on:
        opening_moves = ['g2-f3', 'b2-c3', 'a2-b3', 'h2-g3']
        if aggressive is not None:  # not agnostic
            if aggressive:
                opening_moves.extend(['d2-d3', 'e2-e3'])

            else:  # defensive
                opening_moves.extend(['a1-b2', 'h1-g2'])

        this_move_number = ceil(move_number/2)
        if this_move_number < len (opening_moves):
            if move_number %2 == 0: #white to move
               best_move = opening_moves[this_move_number]
            else:
                best_move = move_lookup_by_index(best_child['index'],player_color)
        else:
                best_move = move_lookup_by_index(best_child['index'],player_color)
    else:
        best_move = move_lookup_by_index(best_child['index'],player_color)
    return best_move

def get_num_nodes_in_tree(root):
    checked = []
    unchecked = [root]
    while len(unchecked)>0:
        node = unchecked.pop()
        if node['children']is not None:
            unchecked.extend(node['children'])
        checked.append(node)
    return len(checked)

cpdef run_MCTS_with_expansions_simulation(dict root,int depth_limit, dict sim_info,  policy_net, start_time, int time_to_think ):
    if not root['subtree_checked']:
        play_MCTS_game_with_expansions(root, 0, depth_limit, sim_info,  policy_net, start_time, time_to_think)
        sim_info['counter'] += 1
        if root['subtree_checked']: #necessary for end game moves since 1-step lookahead makes the search stop prematurely
            done = True
        else:
            done = False
    else:
        done = True
    return done





cdef play_MCTS_game_with_expansions(dict root, int depth, int depth_limit, dict sim_info,  policy_net, start_time, int time_to_think):
    if root['gameover']is False: #terminates at end-of-game moves
        if root['children']is None : #reached non-game ending leaf node
            with async_update_lock:
                if root['threads_checking_node']<=0:
                    increment_threads_checking_node(root)
                    abort = False
                else:
                    abort = True
            if not abort:
                expand_leaf_node(root, depth, depth_limit, sim_info,  policy_net, start_time, time_to_think)
        else:#keep searching tree

            select_UCT_child(root, depth, depth_limit, sim_info,  policy_net, start_time, time_to_think)


cdef void expand_leaf_node(dict root, int depth, int depth_limit, dict sim_info,  policy_net, start_time, int time_to_think):
    if depth < depth_limit:
        # expand_and_select(root, depth, depth_limit, sim_info,  policy_net, start_time, time_to_think)
        expand_descendants_to_depth_wrt_NN([root], depth, depth_limit, sim_info, async_update_lock, policy_net) #prepruning
    else:  # reached depth limit
        decrement_threads_checking_node(root)

cdef void select_UCT_child(dict node, int depth, int depth_limit, dict sim_info, policy_net, start_time, int time_to_think):
    with async_update_lock: #make sure it's not being updated asynchronously
        while node is not None and node['children']is not None:
            node['threads_checking_node']= 0 #clean this up on the way down
            # if sim_info['counter'] > 500 and len(sim_info['game_tree']) == 0:
            #     breakpoint = True
            # if node['parent']is not None:
            #     node['parent']['subtree_being_checked'] =False
            node = choose_UCT_or_best_child(node, start_time, time_to_think, sim_info) #if child is a leaf, chooses policy net's top choice
    if node is not None:
        play_MCTS_game_with_expansions(node, depth, depth_limit, sim_info,  policy_net, start_time, time_to_think)



cpdef void print_simulation_statistics(dict sim_info):#TODO: try not calling this and see if this is what is slowing down the program
    cdef:
        dict root = sim_info['root']
        int time_to_think = sim_info['time_to_think']
    overwhelming_on = False
    start_time = sim_info['start_time']

    wins, visits, true_wins, true_losses = transform_wrt_overwhelming_amount(root, overwhelming_on)
    print("Monte Carlo Game {iteration}\n"
          "Played at root   Height {height}:    Player = {color}    UCT = {uct}     wins = {wins}       visits = {visits}      true_wins = {true_wins}     true_losses = {true_losses}    prob = %{prob}    win = {win_status}\n".format(
        height = root['height'], color=root['color'], uct=0, wins=wins, visits=visits,true_wins=true_wins,true_losses=true_losses,
        iteration=sim_info['counter']+1, prob=(root['UCT_multiplier']-1)*100, win_status = root['win_status']),file=sim_info['file'])
    print_board(sim_info['root']['game_board'],sim_info['file'])
    print("\n", file=sim_info['file'])
    #to see best reply
    if root['children']is not None:
        best_child = randomly_choose_a_winning_move(root, sim_info['game_num'])
        best_move =  move_lookup_by_index(best_child['index'],root['color'])
        best_child_UCT = get_UCT(best_child, root['visits'], start_time, time_to_think,sim_info)
        wins, visits, true_wins, true_losses = transform_wrt_overwhelming_amount(best_child, overwhelming_on)

        print("Current (Random) Best Move   {best_move}\n"
              "Height {height}:    Player = {color}    UCT = {uct}     wins = {wins}       visits = {visits}      true_wins = {true_wins}     true_losses = {true_losses}   prob = %{prob}    win = {win_status}\n".format(
            best_move=best_move,
            height=best_child['height'], color=best_child['color'], uct=best_child_UCT, wins=wins,true_wins=true_wins,true_losses=true_losses,
            visits=visits, prob=(best_child['UCT_multiplier']-1)*100, win_status = best_child['win_status']),file=sim_info['file'])
        print_board(best_child['game_board'],sim_info['file'])

        print("All Moves", file=sim_info['file'])
        for child in root['children']:
            move = move_lookup_by_index(child['index'],root['color'])

            child_UCT = get_UCT(child, root['visits'], start_time, time_to_think,sim_info)
            wins, visits, true_wins, true_losses = transform_wrt_overwhelming_amount(child, overwhelming_on)

            print(" Move   {move}    UCT = {uct}     wins = {wins}       visits = {visits}      true_wins = {true_wins}     true_losses = {true_losses}    prob = %{prob}    win = {win_status}\n".format(
                move=move,
                uct=child_UCT,
                wins=wins, true_wins=true_wins, true_losses=true_losses,
                visits=visits, prob=(child['UCT_multiplier']-1)*100, win_status = child['win_status']),file=sim_info['file'])
            print_board(child['game_board'],sim_info['file'])

        #to see predicted best counter
        if best_child['children'] is not None:
            best_counter_child = randomly_choose_a_winning_move(best_child, sim_info['game_num'])
            best_counter_move = move_lookup_by_index(best_counter_child['index'],best_child['color'])
            best_counter_child_UCT = get_UCT(best_counter_child, best_child['visits'], start_time, time_to_think,sim_info)
            wins, visits, true_wins, true_losses = transform_wrt_overwhelming_amount(best_counter_child, overwhelming_on)

            print("Current Best Move (Random) Best Counter Move   {best_counter_move}\n"
                  "Height {height}:    Player = {color}    UCT = {uct}     wins = {wins}       visits = {visits}      true_wins = {true_wins}     true_losses = {true_losses}    prob = %{prob}    win = {win_status}\n".format(
                best_counter_move=best_counter_move, true_wins=true_wins, true_losses=true_losses,
                height=best_counter_child['height'], color=best_counter_child['color'],uct=best_counter_child_UCT,wins=wins,
                visits=visits, prob=(best_counter_child['UCT_multiplier']-1)*100, win_status = best_counter_child['win_status']),file=sim_info['file'])
            print_board(best_counter_child['game_board'],sim_info['file'])


            print("All Counter Moves",file=sim_info['file'])
            for counter_child in best_child['children']:
                counter_move = move_lookup_by_index(counter_child['index'],best_child['color'])

                counter_child_UCT = get_UCT(counter_child, best_child['visits'], start_time, time_to_think,sim_info)
                wins, visits, true_wins, true_losses = transform_wrt_overwhelming_amount(counter_child, overwhelming_on)

                print("Counter Move   {counter_move}    UCT = {uct}     wins = {wins}       visits = {visits}      true_wins = {true_wins}     true_losses = {true_losses}    prob = %{prob}    win = {win_status}\n".format(
                    counter_move=counter_move,
                    uct=counter_child_UCT, true_wins=true_wins, true_losses=true_losses,
                    wins=wins,
                    visits=visits, prob=(counter_child['UCT_multiplier']-1)*100, win_status = counter_child['win_status']),file=sim_info['file'])
                print_board(counter_child['game_board'],sim_info['file'])

        else:
            print("No Counter Moves explored",file=sim_info['file'])
    else:
        print("No Moves explored", file=sim_info['file'])



    print("\n", file=sim_info['file'])
    k = 1
    while len(sim_info['eval_times'])>0:
        total = 0
        count = 0
        new_eval_times = []
        for eval_stat in sim_info['eval_times']:
            if eval_stat[0] == k:
                total+= eval_stat[1]
                count+=1
            else:
                new_eval_times.append(eval_stat)
        if count > 0:
            avg_time = total/count
            print("approximately {time} ms to evaluate {num_nodes} node (out of {trials})".format(time=avg_time, num_nodes=k, trials=count), end='\n', file=sim_info['file'])
        sim_info['eval_times'] = new_eval_times
        k+=1

    # get_debug_nodes_in_tree(root['best_child'], sim_info)
    # exit(22)


    # for i in range(0, len(sim_info['game_tree'])):
    #     node_parent = sim_info['game_tree'][i]['parent']
    #     if node_parent is None:
    #         UCT = 0
    #     else:
    #         UCT = get_UCT(sim_info['game_tree'][i], node_parent['visits'], start_time, time_to_think, sim_info)
    #     print("Node {i} Height {height}:    Player = {color}    UCT = {uct}     wins = {wins}       visits = {visits}      true_wins = {true_wins}       true_losses = {true_losses} checked = {checked}".format(
    #         i=i, height=sim_info['game_tree'][i]['height'], color=sim_info['game_tree'][i]['color'], uct=UCT, wins=sim_info['game_tree'][i]['wins'], visits=sim_info['game_tree'][i]['visits'], checked=sim_info['game_tree'][i]['subtree_checked']),
    #         file=sim_info['file'])
    #     print_board(sim_info['game_tree'][i]['game_board'], sim_info['file'])
    #     print("\n", file=sim_info['file'])

def get_debug_nodes_in_tree(root, sim_info):
    while root['children'] is not None:
        best_child = randomly_choose_a_winning_move(root, 1)
        counter_move = move_lookup_by_index(best_child['index'], best_child['color'])

        best_child_UCT = get_UCT(best_child, best_child['visits'], sim_info['start_time'], sim_info['time_to_think'], sim_info)
        wins, visits, true_wins, true_losses = transform_wrt_overwhelming_amount(best_child, False)
        print(
            "Winning Move   {counter_move}    UCT = {uct}     wins = {wins}       visits = {visits}      true_wins = {true_wins}     true_losses = {true_losses}    prob = %{prob}    win = {win_status}\n".format(
                counter_move=counter_move,
                uct=best_child_UCT, true_wins=true_wins, true_losses=true_losses,
                wins=wins,
                visits=visits, prob=(best_child['UCT_multiplier'] - 1) * 100,
                win_status=best_child['win_status']), file=sim_info['file'])
        print_board(best_child['game_board'], sim_info['file'])
        root = best_child

def print_expansion_statistics(sim_info, start_time):
    print_simulation_statistics(sim_info)
    tree_size = get_num_nodes_in_tree(sim_info['root'])
    print("Number of Tree Nodes added in simulation {counter} = "
          "{nodes}. Time to print this statistic =  {time} seconds\n"
          "Tree rooted at height = {height} Number of nodes in Tree = {tree_size}".format(counter=sim_info['counter']+1,
                                                    tree_size=tree_size,
                                                  nodes=len(sim_info['game_tree']) #- sim_info['prev_game_tree_size']
                                                  ,
                                                  time=time() - start_time,
                                                  height=sim_info['root']['height']),file=sim_info['file'])
def print_best_move(player_color, best_move, sim_info):
    print("For {player_color}, best move is {move}\n".format(player_color=player_color, move=best_move),
          file=sim_info['file'])


def print_forced_win(root_win_status, sim_info):
    if root_win_status is True:
        print('I have forced a win!', file=sim_info['file'])
    elif root_win_status is False:
        print(r'I\'m doomed!', file=sim_info['file'])
