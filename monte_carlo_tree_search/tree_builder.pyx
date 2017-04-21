#cython: language_level=3, boundscheck=False

from Breakthrough_Player.board_utils import game_over, enumerate_legal_moves_using_piece_arrays, move_piece_update_piece_arrays
from tools.utils import index_lookup_by_move, move_lookup_by_index
from monte_carlo_tree_search.TreeNode import TreeNode
from Breakthrough_Player.board_utils import  check_legality_MCTS
from monte_carlo_tree_search.tree_search_utils import update_tree_losses, update_tree_wins, \
    get_top_children, update_child, set_win_status_from_children, random_rollout, update_win_status_from_children, eval_child, SimulationInfo, rollout_and_eval_if_parent_at_depth
from multiprocessing import Pool, Process, pool
from math import ceil
from threading import Lock, current_thread
from sys import stdout
from time import time
import random


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



def build_game_tree(player_color, depth, unvisited_queue, depth_limit): #first-pass BFS to enumerate all the concrete nodes we want to keep track of/run through policy net
    if depth < depth_limit: # play game at this root to depth limit
       visited_queue = visit_to_depth_limit(player_color, depth, unvisited_queue, depth_limit)
    else: #reached depth limit;
       update_bottom_of_tree(unvisited_queue)
       visited_queue = [] #don't return bottom of tree so it doesn't run inference on these nodes
    return visited_queue


def visit_to_depth_limit(player_color, depth, unvisited_queue, depth_limit):

    unvisited_children = visit_all_nodes_and_expand_multithread(unvisited_queue, player_color)
    visited_queue = unvisited_queue  # all queue members have now been visited

    if len(unvisited_children) > 0:  # if children to visit
        # visit children
        opponent_color = get_opponent_color(player_color)
        visited_queue.extend(build_game_tree(opponent_color, depth + 1, unvisited_children,
                                             depth_limit)) #TODO: is recursion taking too long/too much stack space?
        # else: game over taken care of in visit
    return visited_queue


def update_bottom_of_tree(unvisited_queue):#don't do this as it will mark the bottom as losses
    #NN will take care of these wins.
    for node in unvisited_queue:  # bottom of tree, so percolate visits to the top
        random_rollout(node)
        #don't return bottom of tree so it doesn't run inference on these nodes
    # visited_queue = unvisited_queue
    # return visited_queue

def get_opponent_color(player_color):
    if player_color == 'White':
        opponent_color = 'Black'
    else:
        opponent_color = 'White'
    return opponent_color

def visit_all_nodes_and_expand_single_thread(unvisited_queue, player_color):
    unvisited_children = []
    for this_root_node in unvisited_queue:  # if empty=> nothing to visit;
       unvisited_child_nodes= visit_single_node_and_expand([this_root_node, player_color])
       unvisited_children.extend(unvisited_child_nodes)
    return unvisited_children

def visit_all_nodes_and_expand_multithread(unvisited_queue, player_color):
    unvisited_children = []
    arg_lists = [[node, player_color] for node in unvisited_queue]
    processes = Pool(processes=7)#prevent threads from taking up too much memory before joining
    unvisited_children_separated = processes.map(visit_single_node_and_expand, arg_lists)  # synchronized with unvisited queue
    processes.close()
    processes.join()
    for i in range (0, len(unvisited_children_separated)):#does this still outweigh just single threading?
        child_nodes = unvisited_children_separated[i]
        parent_node = arg_lists[i][0]
        for child in child_nodes:
            child.parent = parent_node
        parent_node.children = child_nodes
        unvisited_children.extend(child_nodes)
    return unvisited_children

def visit_single_node_and_expand(node_and_color):
    node = node_and_color[0]
    unvisited_children = expand_node(node)
    return unvisited_children #necessary if multiprocessing from above

def visit_single_node_and_expand_no_lookahead(node_and_color):
    node = node_and_color[0]
    node_color = node_and_color[1]
    unvisited_children = []
    game_board = node.game_board
    is_game_over, winner_color = game_over(game_board)
    if is_game_over:  # only useful at end of game
        set_game_over_values(node, node_color, winner_color)
    else:  # expand node, adding children to parent
        unvisited_children = expand_node(node)
    return unvisited_children

def expand_node(parent_node, rollout=False):
    children_as_moves = enumerate_legal_moves_using_piece_arrays(parent_node)
    child_nodes = []
    children_win_statuses = []
    for child_as_move in children_as_moves:  # generate children
        move = child_as_move['From'] + r'-' + child_as_move['To']
        child_node = init_child_node_and_board(move, parent_node)
        check_for_winning_move(child_node, rollout) #1-step lookahead for gameover
        children_win_statuses.append(child_node.win_status)
        child_nodes.append(child_node)
    set_win_status_from_children(parent_node, children_win_statuses)
    if parent_node.children is None: #if another thread didn't expand and update this node
       parent_node.children = child_nodes
    parent_node.expanded = True
    return child_nodes


def expand_descendants_to_depth_wrt_NN(unexpanded_nodes, without_enumerating, depth, depth_limit, sim_info, lock, policy_net): #nodes are all at the same depth
    # Prunes child nodes to be the NN's top predictions or instant gameovers.
    # expands all nodes at a depth to the depth limit to take advantage of GPU batch processing.
    # This step takes time away from MCTS in the beginning, but builds the tree that the MCTS will use later on in the game.
    original_unexpanded_node = unexpanded_nodes[0]
    # original_unexpanded_node.
    entered_once = False
    over_time = not( time() - sim_info.start_time < sim_info.time_to_think)
    while depth < depth_limit and not over_time:
        entered_once = True
        # if len (unexpanded_nodes) > 0:
        #     if unexpanded_nodes[0] is None:
        #         True

        unexpanded_nodes = list(filter(lambda x: ((x.children is None or x.reexpanded) and not x.gameover ), unexpanded_nodes)) #redundant
        if len(unexpanded_nodes) > 0 and len(unexpanded_nodes)<2056: #if any nodes to expand;
            if sim_info.root is not None: #aren't coming from reinitializing a root
                if sim_info.main_pid == current_thread().name:
                    depth_limit = 1 #main thread expands once and exits to call other threads
                # elif unexpanded_nodes[0].height > 80:
                #     # if len(unexpanded_nodes) == 1:
                #     #     print("1 element to expand\n"
                #     #           "move = {move} at height {height}\n"
                #     #           "threads checking this node = {threads}\n".format(threads=unexpanded_nodes[0].threads_checking_node,height=unexpanded_nodes[0].height,move=move_lookup_by_index(unexpanded_nodes[0].index, get_opponent_color(unexpanded_nodes[0].color))))
                #     # else:
                #     #     print(len(unexpanded_nodes))
                #     depth_limit = 128
                # elif unexpanded_nodes[0].height > 70:
                #     depth_limit = 16
                elif unexpanded_nodes[0].height > 50:
                    depth_limit = 800
                    without_enumerating = False
                elif unexpanded_nodes[0].height > 40:
                    depth_limit = 800
                    without_enumerating = True
                else:
                    depth = 4
                    without_enumerating = True


            # the point of multithreading is that other threads can do useful work while this thread blocks from the policy net calls
            NN_output = policy_net.evaluate(unexpanded_nodes)

            if len(unexpanded_nodes) >= 10:
                unexpanded_children, over_time = offload_updates_to_separate_process(unexpanded_nodes, without_enumerating, depth, depth_limit, NN_output, sim_info, lock)
            else:
                unexpanded_children, over_time = do_updates_in_the_same_process(unexpanded_nodes, without_enumerating, depth, depth_limit, NN_output, sim_info, lock)
            if over_time:
                for child in unexpanded_children: #unset flag for children we were planning to check
                    child.threads_checking_node = 0

            unexpanded_nodes = unexpanded_children

            depth += 1

        else:
            depth = depth_limit #done if no nodes to expand or too many nodes to expand
            for parent in unexpanded_nodes:
                parent.threads_checking_node = 0
    if not entered_once:
        for parent in unexpanded_nodes:
            parent.threads_checking_node -=1

def offload_updates_to_separate_process(unexpanded_nodes, without_enumerating, depth, depth_limit, NN_output, sim_info, lock):
    grandparents = []
    unexpanded_children = []
    over_time = False
    with lock:
        if unexpanded_nodes[0].threads_checking_node <= 1 and (unexpanded_nodes[0].children is None or unexpanded_nodes[0].reexpanded):  # this shouldn't happen anyway? if it was doing a batch expansion, it came from a node that wasn't expanded twice?
            abort = False
        else:
            abort = True
            unexpanded_nodes[0].threads_checking_node -= 1

    if not abort:
        for parent in unexpanded_nodes:
            grandparents.append(parent.parent)
            # parent.parent_reexpanded_already = parent.parent.reexpanded_already
            # parent.winning_siblings = parent.parent.winning_kids
            parent.parent = None
        process = Pool(processes=1)
        expanded_parents, unexpanded_children, over_time = process.map(update_parents_for_process, [
            [without_enumerating, unexpanded_nodes, NN_output, depth, depth_limit, sim_info.start_time,
             sim_info.time_to_think]])[0]
        process.close()  # sleeps here so other threads can work
        process.join()
        with lock:
            for i in range(0, len(expanded_parents)):  # reattach these copies to tree; if adjacent parents have the same grandparent => duplicate parent in associated index in array
                parent = expanded_parents[i]
                grandparent = grandparents[i]
                reattach_parent_to_grandparent(grandparent, parent)
                parent.threads_checking_node = 0
            for child in unexpanded_children:
                if child.gameover is True: #child color is a loser
                    if child.parent is not None:
                        if child.parent.parent is not None:
                            overwhelming_amount = 65536
                            update_tree_losses(child.parent.parent, overwhelming_amount) #if grandchild lost, parent won, grandparent lost
                else:
                    rollout_and_eval_if_parent_at_depth(child, 1)  # backprop eval
            for grandparent in grandparents:
                if not grandparent.reexpanded: #else this value will change later
                    update_win_status_from_children(grandparent)  # since parents may have been updated, grandparent must check these copies; maybe less duplication to do it here than in expanded parents loop?
                    backpropagate_num_checked_children(grandparent)
            sim_info.game_tree.extend(expanded_parents)
    return unexpanded_children, over_time

def do_updates_in_the_same_process(unexpanded_nodes, without_enumerating, depth, depth_limit, NN_output, sim_info, lock):
    unexpanded_children = []
    over_time = False
    for i in range(0, len(unexpanded_nodes)):
        parent = unexpanded_nodes[i]
        if time() - sim_info.start_time < sim_info.time_to_think:  # stop updating parents as soon as we go over our time to think
            with lock:
                if parent.threads_checking_node <= 1 and (parent.children is None or parent.reexpanded):
                    if parent.threads_checking_node <= 0:
                        parent.threads_checking_node = 1
                    abort = False
                    parent.being_checked = True
                else:
                    parent.threads_checking_node -= 1
                    abort = True
            if not abort:
                children = update_parent(without_enumerating, parent, NN_output[i], sim_info, lock)
                children_to_consider = []
                if depth < depth_limit - 1:  # if we are allowed to enter the outermost while loop again
                    best_child = None
                    # for child in children:
                    #     child.threads_checking_node += 1
                    # children_to_consider = children
                    with lock:
                        if parent.height % 2 == 1:  # black move
                            best_child = None
                            # for child in children:
                            #     child.threads_checking_node += 1
                            # children_to_consider = children
                            for child in children: #walk down children sorted by NN probability
                                if child.win_status is None and child.threads_checking_node <=0:
                                    best_child = child
                                    break
                            # if parent.best_child is not None:
                            #     if parent.best_child.win_status is None:
                            #         best_child = parent.best_child
                            #     else:
                            #
                            #     # best_child.visited = True #? seemed to be doing fine without it
                            # else:
                            #     best_child_val = 1
                            #     best_child = None
                            #     for child in children:
                            #         child_win_rate = child.wins / max(1, child.visits)
                            #         if child_win_rate <= best_child_val:
                            #             best_child = child
                            #             best_child_val = child_win_rate
                            if best_child is not None:
                                best_child.threads_checking_node += 1
                                children_to_consider = [best_child]
                        else:  # white move
                            for child in children:
                                child.threads_checking_node += 1
                            children_to_consider = children

                unexpanded_children.extend(children_to_consider)
        else:
            parent.threads_checking_node = 0  # have to iterate over all parents to set this
            over_time = True
    return unexpanded_children, over_time
def reattach_parent_to_children(parent, children):
    for child in children:
        child.parent = parent
        eval_child(child)
        check_for_winning_move(child)

def reattach_parent_to_grandparent(grandparent, parent):
    for i in range(0, len(grandparent.children)):
        if parent.game_board == grandparent.children[i].game_board:
            grandparent.children[i] = parent
            parent.parent = grandparent
            if parent.win_status is True:
               grandparent.winning_kids += 1

            if grandparent.color == 'Black' and \
                grandparent.winning_kids == len(grandparent.children) and \
                not grandparent.reexpanded_already:

                grandparent.reexpanded = True
            break


def update_parent(without_enumerating_children, parent, NN_output, sim_info, lock):
    pruned_children = []
    with lock:  # Lock after the NN update and check if we still need to update the parent
        if parent.children is None or parent.reexpanded:
            abort = False
            if not parent.expanded:  # if an expansion wasn't already attempted
                sim_info.game_tree.append(parent)
            parent.expanded = True
        else:
            abort = True
            parent.threads_checking_node -=1
    if not abort:  # if the node hasn't already been updated by another thread
        pruned_children = get_pruned_children(without_enumerating_children, parent, NN_output, sim_info,lock)
    return pruned_children

def get_pruned_children(without_enumerating_children, parent, NN_output, sim_info,lock):
    if without_enumerating_children:
        pruned_children = update_and_prune(parent, NN_output, sim_info,lock)
    else:
        pruned_children = enumerate_update_and_prune(parent, NN_output, sim_info,lock)
    return pruned_children

def update_parents_for_process(args):#when more than n parents, have a separate process do the updating
    without_enumerating = args[0]
    unexpanded_nodes = args[1]
    NN_output = args[2]
    depth = args[3]
    depth_limit = args[4]
    start_time = args[5]
    time_to_think = args[6]
    sim_info = SimulationInfo(stdout)
    sim_info.do_eval = False
    lock = Lock()

    unexpanded_children = []
    over_time = False
    for i in range(0, len(unexpanded_nodes)):
        parent = unexpanded_nodes[i]
        if time() - start_time < time_to_think:  # stop updating parents as soon as we go over our time to think
            with lock:
                if parent.threads_checking_node <= 1 and (parent.children is None or parent.reexpanded):
                    if parent.threads_checking_node <= 0:
                        parent.threads_checking_node = 1
                    abort = False
                    parent.being_checked = True
                else:
                    parent.threads_checking_node -= 1
                    abort = True
            if not abort:

                children = update_parent(without_enumerating, parent, NN_output[i], sim_info, lock)
                children_to_consider = []
                if depth < depth_limit - 1:  # if we are allowed to enter the outermost while loop again
                    # for child in children:
                    #     child.threads_checking_node += 1
                    # children_to_consider = children
                    if parent.height % 2 == 1:  # black move
                        best_child = None
                        for child in children:  # walk down children sorted by NN probability
                            if child.win_status is None and child.threads_checking_node <=0:
                                best_child = child
                                break
                        if best_child is not None:
                            best_child.threads_checking_node += 1
                            children_to_consider = [best_child]
                    else:
                        for child in children:
                            child.threads_checking_node += 1
                        children_to_consider = children

                unexpanded_children.extend(children_to_consider)
        else:
            parent.threads_checking_node = 0  # have to iterate over all parents to set this
            over_time = True
    return unexpanded_nodes, unexpanded_children, over_time

def update_and_prune(parent, NN_output,sim_info, lock):
    pruned_children = []
    children_win_statuses = []

    # comparing to child val should reduce this considerably,
    # yet still allows us to call parent function from a top-level asynchronous tree updater
    num_to_check_for_legality = get_num_children_to_consider(parent)
    ranks_to_consider = num_to_check_for_legality + 2 #just in case some top choices are illegal moves
    top_children_indexes = get_top_children(NN_output, ranks_to_consider)
    best_child_val, best_rank = get_best_child_val(parent, NN_output, top_children_indexes)

    if num_to_check_for_legality == 1: #since this happens more often than not, make it a dedicated block
        move = move_lookup_by_index(top_children_indexes[best_rank], parent.color)
        pruned_child, child_win_status = get_cached_child(parent, move, NN_output, top_children_indexes,
                                                                  best_child_val, best_rank, sim_info, lock, num_to_check_for_legality, aggressive=None)
        pruned_children.append(pruned_child)
        children_win_statuses.append(child_win_status)
        pruned_children = assign_pruned_children(parent, pruned_children, children_win_statuses, lock)
    else:
        top_children_indexes = top_children_indexes[best_rank:]#start from the first best legal move
        for child_index in top_children_indexes:
            move = move_lookup_by_index(child_index, parent.color)  # turn the child indexes into moves
            if check_legality_MCTS(parent.game_board, move):
                pruned_child, child_win_status = get_cached_child(parent, move, NN_output, top_children_indexes,
                                                                  best_child_val, best_rank, sim_info, lock, num_to_check_for_legality, aggressive=None)

                if pruned_child is not None:
                    pruned_children.append(pruned_child)
                    children_win_statuses.append(child_win_status)
        pruned_children = assign_pruned_children(parent, pruned_children, children_win_statuses, lock)
    return pruned_children

def enumerate_update_and_prune(parent, NN_output,sim_info, lock):
    #Less efficient since we enumerate all children first.
    pruned_children = []
    children_win_statuses = []
    children_as_moves = enumerate_legal_moves_using_piece_arrays(parent)
    num_legal_moves = len(children_as_moves)
    ranks_to_consider = num_legal_moves + 5 #just in case some top choices are illegal moves
    top_children_indexes = get_top_children(NN_output, ranks_to_consider)

    best_child_val, best_rank = get_best_child_val(parent, NN_output, top_children_indexes)
    if parent.children is None:
        predicate = True
    else:
        predicate = num_legal_moves != len(parent.children)
    if predicate:#if we don't already have all legal children
        for child_as_move in children_as_moves:
            move = child_as_move['From'] + r'-' + child_as_move['To']
            pruned_child, child_win_status = get_cached_child(parent, move, NN_output, top_children_indexes, best_child_val, best_rank, sim_info, lock, num_legal_moves, aggressive=None)
            if pruned_child is not None:
                pruned_children.append(pruned_child)
                children_win_statuses.append(child_win_status)
        pruned_children = assign_pruned_children(parent, pruned_children, children_win_statuses, lock)
    #else: parent already has all of its legal children
    return pruned_children

def get_cached_child(parent, move, NN_output, top_children_indexes, best_child_val, best_rank,  sim_info, lock, num_legal_moves, aggressive=None, on = False):
    if on:
        opening_move = True
        opening_moves = ['g2-f3', 'b2-c3', 'a2-b3', 'h2-g3']

        if aggressive is not None: #not agnostic
            if aggressive:
                opening_moves.extend(['d2-d3', 'e2-e3'])

            else: #defensive
                opening_moves.extend(['a1-b2', 'h1-g2'])
        move_number = ceil(parent.height/2)
        if move_number < len(opening_moves): #or parent.height <20
             # : #opening moves
            if parent.height %2 == 0: #white's move
                if move.lower() == opening_moves[move_number]:
                    pruned_child, child_win_status = get_pruned_child(parent, move, NN_output, top_children_indexes,
                                                                      best_child_val, best_rank,sim_info, lock, num_legal_moves,
                                                                      opening_move)
                else:#don't check moves we know we aren't going to make
                    pruned_child = child_win_status = None
            else: #black's move
                pruned_child, child_win_status = get_pruned_child(parent, move, NN_output, top_children_indexes,
                                                                  best_child_val, best_rank,
                                                                  sim_info, lock, num_legal_moves, opening_move)
        else: #non-opening move
            opening_move = False
            pruned_child, child_win_status = get_pruned_child(parent, move, NN_output, top_children_indexes, best_child_val, best_rank,
                                                               sim_info, lock, num_legal_moves, opening_move)
    else:
        opening_move = False
        pruned_child, child_win_status = get_pruned_child(parent, move, NN_output, top_children_indexes, best_child_val, best_rank,
                                                          sim_info, lock, num_legal_moves, opening_move)

    return pruned_child, child_win_status

def get_best_child_val(parent, NN_output, top_children_indexes):
    best_child_val = 0
    rank = 0 # sometimes top prediction isn't the highest ranked.
    for top_child_index in top_children_indexes:  # find best legal child value
        top_move = move_lookup_by_index(top_child_index, parent.color)
        if check_legality_MCTS(parent.game_board, top_move):
            best_child_val = NN_output[top_child_index]
            break
        else:
            rank += 1
    return best_child_val, rank

# def get_pruning_filter_by_height(parent, child, top_children_indexes, child_val, best_child_val, best_rank, opening_move=False):
#     if parent.height < 80 and not opening_move and not parent.reexpanded:
#         # or if root, top3?
#
#         if parent.height % 2 == 0:  # white moves
#
#             if False:
#                 num_top_to_consider = best_rank +  1
#                 top_n_children = top_children_indexes[best_rank:num_top_to_consider]
#                 predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#             # elif parent.height in num_to_consider_dict.keys():
#             #     num_top_to_consider = best_rank +  num_to_consider_dict[parent.height]
#             #     top_n_children = top_children_indexes[:num_top_to_consider]
#             #     predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#
#             # 70-79
#             elif parent.height >= 70:
#                 num_top_to_consider = best_rank +  3
#                 top_n_children = top_children_indexes[best_rank:num_top_to_consider]
#                 predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#
#             # 2?   65-69      #??
#             elif parent.height < 70 and parent.height >= 65:
#                 num_top_to_consider = best_rank +  2  # else play with child val threshold?
#                 top_n_children = top_children_indexes[best_rank:num_top_to_consider]
#                 predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#
#             # 4?   60-64
#             # elif parent.height < 65 and parent.height >= 60:
#             #     num_top_to_consider = best_rank +  4  # else play with child val threshold?
#             #     top_n_children = top_children_indexes[:num_top_to_consider]
#             #     predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#
#             # 7 or 4?   50-59
#             # elif parent.height < 60 and parent.height >= 55:
#             #     num_top_to_consider = best_rank +  4 #else play with child val threshold?
#             #     top_n_children = top_children_indexes[:num_top_to_consider]
#             #     predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#
#             # 4?   40-64
#             elif parent.height < 65 and parent.height >= 40:
#                 num_top_to_consider = best_rank +  3  # else play with child val threshold?
#                 top_n_children = top_children_indexes[best_rank:num_top_to_consider]
#                 predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#
#             # 5    40-49
#             # elif parent.height < 50 and parent.height >= 40:
#             #     num_top_to_consider = best_rank +  5 #else play with child val threshold?
#             #     top_n_children = top_children_indexes[:num_top_to_consider]
#             #     predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#
#             # 2    30-39 #TODO: put this in at next test: at log 85 doomed itself when it thought it found a forced win because it didn't see the rank 2 reply
#             # elif parent.height < 40 and parent.height >= 30:
#             #     num_top_to_consider = best_rank +  2  # else play with child val threshold?
#             #     top_n_children = top_children_indexes[:num_top_to_consider]
#             #     predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#
#             # 2    20-29
#             # elif parent.height < 30 and parent.height >= 20 and not parent.height < 20 :
#             #     num_top_to_consider = best_rank +  2 #else play with child val threshold?
#             #     top_n_children = top_children_indexes[:num_top_to_consider]
#             #     predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#             else:
#                 num_top_to_consider = best_rank +  1  # else play with child val threshold?
#                 top_n_children = top_children_indexes[best_rank:num_top_to_consider]
#                 predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#                 # predicate = True
#         else:  # black's move
#             if False:
#                 num_top_to_consider = best_rank +  1
#                 top_n_children = top_children_indexes[best_rank:num_top_to_consider]
#                 predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#                 # 3? 2?  61-69
#             elif parent.height < 70 and parent.height >= 61:
#                 num_top_to_consider = best_rank +  2  # else play with child val threshold? WaNN missed an easy win and thought it was doomed at height 53
#                 top_n_children = top_children_indexes[best_rank:num_top_to_consider]
#                 predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#                 # 52-60 #3?
#             elif parent.height < 61 and parent.height >= 52:
#                 num_top_to_consider = best_rank +  3  # else play with child val threshold? WaNN missed an easy win and thought it was doomed at height 53
#                 top_n_children = top_children_indexes[best_rank:num_top_to_consider]
#                 predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#
#                 # 3?   40-51 #2?
#             elif parent.height < 52 and parent.height >= 40:  # 40?
#                 num_top_to_consider = best_rank +  2  # else play with child val threshold? missed an easy win and thought it was doomed at 2 height 53
#                 top_n_children = top_children_indexes[best_rank:num_top_to_consider]
#                 predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#             else:
#                 num_top_to_consider = best_rank +  1  # else play with child val threshold?
#                 top_n_children = top_children_indexes[best_rank:num_top_to_consider]
#                 predicate = child.index in top_n_children or child_val > .30 or best_child_val - child_val < .10
#     else:
#         if parent.reexpanded:
#             num_top_to_consider = best_rank + 5  # else play with child val threshold?
#             top_n_children = top_children_indexes[best_rank:num_top_to_consider]
#             predicate = child.index in top_n_children or best_child_val - child_val < .10
#         else:
#             predicate = True
#     return predicate
def get_num_children_to_consider(parent):
    height = parent.height

    if height < 80:
        # or if root, top3?
        if parent.reexpanded:
            num_top_to_consider = 10
        else:
            if height % 2 == 0:  # white moves
                if height >= 70:
                    num_top_to_consider =  3
                # 2?   65-69      #??
                elif height < 70 and height >= 65:
                    num_top_to_consider =  3  # else play with child val threshold?
                # # # 4?   60-64
                elif height < 65 and height >= 60:
                    num_top_to_consider =   3  # else play with child val threshold
                # 7 or 4?   50-59
                elif height < 60 and height >= 50:
                    num_top_to_consider =   5 #else play with child val threshold?
                # 5    40-49
                elif height < 50 and height >= 40:
                    num_top_to_consider =   4 #else play with child val threshold?
                # 2    30-39
                elif height < 40 and height >= 30:
                    num_top_to_consider =   4  # else play with child val threshold?
                # 2    20-29
                elif height < 30 and height >= 20:
                    num_top_to_consider =   4 #else play with child val threshold?
                 # 2    0-19
                elif height < 20 and height >= 0:
                    num_top_to_consider =   3 #else play with child val threshold?
                else: #2?
                    num_top_to_consider =  1  # else play with child val threshold?
            else:
                    # 3?   61-69
                if height >= 70:
                    num_top_to_consider =  1  # else play with child val threshold? WaNN missed an easy win and thought it was doomed at height 53
                    # 52-60
                elif height < 70 and height >= 61:#2?
                    num_top_to_consider =  1#2  # else play with child val threshold? WaNN missed an easy win and thought it was doomed at height 53
                    # 52-60
                elif height < 61 and height >= 52:
                    num_top_to_consider =  1#3  # else play with child val threshold? WaNN missed an easy win and thought it was doomed at height 53
                    # 3?   40-51
                elif height < 52 and height >= 40:  # 40?
                    num_top_to_consider =  1#2  # else play with child val threshold? missed an easy win and thought it was doomed at 2 height 53
                elif height < 20 and height >= 0:
                    num_top_to_consider =   1#2 #else play with child val threshold?
                else:
                    num_top_to_consider =  1#1 # else play with child val threshold?
    else:

        num_top_to_consider = 999
    return num_top_to_consider

def get_pruning_filter_by_height_ttt120(parent, child, top_children_indexes, child_val, best_child_val, best_rank,
                                 opening_move=False):
    num_top_to_consider = get_num_children_to_consider(parent)

    if num_top_to_consider == 999:
        predicate = True
    else:
        top_n_children = top_children_indexes[best_rank:num_top_to_consider]
        predicate = child.index in top_n_children or child_val > best_child_val/1.2#or best_child_val - child_val < .10
    return predicate


def get_pruned_child(parent, move, NN_output, top_children_indexes, best_child_val, best_rank, sim_info, lock, num_legal_children, opening_move=False):
    pruned_child = None
    child = init_child_node_and_board(move, parent)
    abort = False
    if parent.children is not None:
        for old_child in parent.children:
            if old_child.game_board == child.game_board:
                abort = True
                break
    if not abort:
        check_for_winning_move(child)  # 1-step lookahead for gameover

        child_val = NN_output[child.index]

        if child.gameover is False:  # update only if not the end of the game
            if child_val == best_child_val:  # rank 1 child
                child.parent.best_child = child  # mark as node to expand first
            # opens up the tree to more lines of play if best child sucks to begin with.
            # in the worst degenerate case where best_val == ~4.5%, will include all children which is actually pretty justified.

            predicate = get_pruning_filter_by_height_ttt120(parent, child, top_children_indexes, child_val, best_child_val, best_rank)

            if predicate:  # absolute value not necessary ; if #1 or over threshold or within 10% of best child

                update_child(child, NN_output, top_children_indexes, num_legal_children, sim_info)
                pruned_child = child  # always keeps top children who aren't losses for parent
        else:  # if it has a win status and not already in NN choices, keep it (should always be a game winning node)
            pruned_child = child
            # child.expanded = True  # don't need to check it any more ; redundant
            sim_info.game_tree.append(child)
    return pruned_child, child.win_status

def assign_pruned_children(parent, pruned_children, children_win_statuses, lock):
    with lock:
        parent.being_checked = False
        parent.threads_checking_node = 0
        if len(pruned_children) > 0:
            if parent.children is None:
                parent.children = pruned_children
                #need to do this once to get win status for parent
                reexpanding_grandparent = True
                set_win_status_from_children(parent, children_win_statuses, new_subtree=False, reexpanding_grandparent=reexpanding_grandparent)

                #true reexpanding grandparent value
                reexpanding_grandparent = mark_doomed_for_reexpansion(parent)
                if reexpanding_grandparent:
                    backpropagate_num_checked_children(parent, reexpansion=False, reexpanding_grandparent=reexpanding_grandparent)
                else: #we aren't reexpanding grandparent, backprop true values
                    set_win_status_from_children(parent, children_win_statuses)
                    backpropagate_num_checked_children(parent)
                # prune_from_win_status(parent)
            else: #had children but we are now adding all children past a certain height
                # new_children_added = False
                # new_children = []
                # for new_child in pruned_children:
                #     duplicate_game_board = False
                #     for child in parent.children:
                #         if new_child.game_board == child.game_board:
                #             duplicate_game_board = True
                #     if not duplicate_game_board:
                #         new_children.append(new_child)
                #         new_children_added = True
                # if new_children_added:
                reset_reexpansion_flag(parent)
                parent.children.extend(pruned_children) #since we moved the duplicate check to a previous function, pruned children should only have new children
                backpropagate_num_checked_children(parent)
                update_win_status_from_children(parent)
                    # prune_from_win_status(parent)
            pruned_children = parent.children
            for child in parent.children:
                child.sibling_count = len(parent.children)
            parent.children.sort(key=lambda x: x.UCT_multiplier, reverse=True)#sort them by probability
    return pruned_children

def mark_doomed_for_reexpansion(parent):
    reexpanding_grandparent = False
    if parent.win_status is True and parent.parent is not None: #potentially doomed grandparent
         #if we are not at the end of the game or threaded out
        parent.parent.winning_kids +=1
        if parent.color == r'White' and \
           parent.height <=80 and \
           parent.parent.winning_kids == parent.sibling_count and \
           not parent.parent.reexpanded_already:
               # print("doomed marked for reexpansion")
               parent.parent.reexpanded = True
               reexpanding_grandparent = True

    return reexpanding_grandparent

def reset_reexpansion_flag(parent):
    parent.reexpanded = False
    parent.reexpanded_already = True



def prune_from_win_status(node, height_cutoff = 0):
    unvisited_queue = [node]
    parent = node.parent
    while len(unvisited_queue) >0:
        node = unvisited_queue.pop()
        if node.height >= height_cutoff and not node.gameover:
            if node.win_status is True:  # pick all losing children
                losing_children = []
                for child in node.children:
                    if child.win_status is False:
                        losing_children.append(child)
                node.children = losing_children  # all forced wins
                # unvisited_queue.extend(node.children)
    #         else:
    #             if node.children is not None:
    #                 # unvisited_queue.extend(node.children)
    # #
    # while parent is not None and parent.height >= height_cutoff:#tell parents to prune too
    #     if parent.win_status is True:
    #         losing_children = []
    #         for child in parent.children:
    #             if child.win_status is False:
    #                 losing_children.append(child)
    #         parent.children = losing_children  # all forced wins
    #         parent = parent.parent



def backpropagate_num_checked_children(node, reexpansion=False, reexpanding_grandparent=False):
    while node is not None:
        set_num_checked_children(node)
        if reexpansion:
            if node.num_children_checked != len(node.children):
               node.subtree_checked = False #may have to reupdate parent trees
            else:
               node.subtree_checked = True
            node = node.parent
        else:
            if node.num_children_checked ==  len(node.children): #all children have been searched; since this is going backwards, will only return true if recursively true
                node.subtree_checked = True
                if reexpanding_grandparent:
                    node = None
                else:
                    node = node.parent
            else:
                node = None

def set_num_checked_children(node): #necessary for gameover kids: wouldn't have reported that they are expanded since another node may also be
    count = 0
    for child in node.children:
        if child.subtree_checked:
            count+= 1
    node.num_children_checked = count

def init_child_node_and_board(child_as_move, parent):
    child_color = get_opponent_color(parent.color)
    child_index = index_lookup_by_move(child_as_move)

    child_board, player_piece_to_add, player_piece_to_remove, remove_opponent_piece\
        = move_piece_update_piece_arrays(parent.game_board, child_as_move, parent.color)

    child_white_pieces, child_black_pieces \
        = update_piece_arrays(parent, player_piece_to_add, player_piece_to_remove, remove_opponent_piece)
    return TreeNode(child_board, child_white_pieces, child_black_pieces, child_color, child_index, parent, parent.height + 1)

def update_piece_arrays(parent, player_piece_to_add, player_piece_to_remove, remove_opponent_piece):
    # slicing is the fastest way to make a copy
    if parent.color == 'White':
        player_pieces = parent.white_pieces[:]
        opponent_pieces = parent.black_pieces[:]
        child_white_pieces = player_pieces
        child_black_pieces = opponent_pieces
    else:
        player_pieces = parent.black_pieces[:]
        opponent_pieces = parent.white_pieces[:]
        child_white_pieces = opponent_pieces
        child_black_pieces = player_pieces
    player_pieces[player_pieces.index(player_piece_to_remove)] = player_piece_to_add
    if remove_opponent_piece:
        opponent_pieces.remove(player_piece_to_add)
    return child_white_pieces, child_black_pieces

def init_new_root(new_root_as_move, new_root_game_board, player_color, new_root_parent, policy_net, sim_info, lock): #online reinforcement learning if wanderer made a move that wasn't in the tree



    if new_root_parent.children is None:# in case parent was never expanded
        print("New root's parent had no children", file=sim_info.file)

        new_root_parent.threads_checking_node = 1
        expand_descendants_to_depth_wrt_NN([new_root_parent], False, 0, 1, sim_info, lock,
                                           policy_net)

        if new_root_parent.children is None: #still none means the pruning was too aggressive; still append this child.
            print("Pruning too aggressive: New root's parent still has no children after expansion", file=sim_info.file)

            # new_root_index = index_lookup_by_move(new_root_as_move)
            # new_root = TreeNode(new_root_game_board, player_color, new_root_index, new_root_parent,
            #                     new_root_parent.height + 1)

            new_root = init_child_node_and_board(new_root_as_move, new_root_parent)

            NN_output = policy_net.evaluate([new_root_parent])[0]
            top_children_indexes = get_top_children(NN_output)
            update_child(new_root, NN_output, top_children_indexes,
                         0, sim_info)  # update prior values on the node NOTE: never a game over
            rank = list(top_children_indexes).index(new_root.index)
            print("PRUNING INVESTIGATION: new root's probability = %{prob} Rank = {rank}".format(
                prob=((new_root.UCT_multiplier - 1) * 100), rank=rank + 1),
                file=sim_info.file)
            new_root_parent.children = [new_root]

        else: #now expanded with some children; check if new root is one of them
            now_in_parent = False
            duplicate_child = None
            for child in new_root_parent.children:
                if child.game_board == new_root_game_board:
                    print("Pruning Fine: New root is now in parent's children after expansion",
                          file=sim_info.file)

                    duplicate_child = child
                    now_in_parent  = True

            if not now_in_parent:
                print("Pruning too aggressive: New root's parent still did not have the actual new root after expansion",
                      file=sim_info.file)

                # new_root_index = index_lookup_by_move(new_root_as_move)
                # new_root = TreeNode(new_root_game_board, player_color, new_root_index, new_root_parent,
                #                     new_root_parent.height + 1)

                new_root = init_child_node_and_board(new_root_as_move, new_root_parent)

                NN_output = policy_net.evaluate([new_root_parent])[0]
                top_children_indexes = get_top_children(NN_output)
                update_child(new_root, NN_output, top_children_indexes,
                             len(new_root_parent.children), sim_info)  # update prior values on the node NOTE: never a game over
                rank = list(top_children_indexes).index(new_root.index)
                print("PRUNING INVESTIGATION: new root's probability = %{prob} Rank = {rank}".format(
                    prob=((new_root.UCT_multiplier - 1) * 100), rank=rank + 1),
                    file=sim_info.file)
                new_root_parent.children.append(new_root) #attach new root to parent
            else:
                new_root = duplicate_child

    else:#has kids but new_root wasn't in them.
        print("Pruning investigation: New root's parent did not have new root after initial expansion",
              file=sim_info.file)
        # new_root_index = index_lookup_by_move(new_root_as_move)
        #
        # new_root = TreeNode(new_root_game_board, player_color, new_root_index, new_root_parent,
        #                     new_root_parent.height + 1)

        new_root = init_child_node_and_board(new_root_as_move, new_root_parent)

        NN_output = policy_net.evaluate([new_root_parent])[0]
        top_children_indexes = get_top_children(NN_output)
        update_child(new_root, NN_output, top_children_indexes,
                     len(new_root_parent.children), sim_info)  # update prior values on the node NOTE: never a game over
        rank = list(top_children_indexes).index(new_root.index)
        print("PRUNING INVESTIGATION: new root's probability = %{prob} Rank = {rank}".format(
            prob=((new_root.UCT_multiplier - 1) * 100), rank=rank + 1),
            file=sim_info.file)
        new_root_parent.children.append(new_root)  # attach new root to parent

    backpropagate_num_checked_children(new_root_parent, True)
    update_win_status_from_children(new_root_parent, new_subtree=True)
    return new_root

# def reexpand_doomed_root():
#     def enumerate_update_and_prune(parent, NN_output, sim_info, lock):
#         # Less efficient since we enumerate all children first.
#         pruned_children = []
#         children_win_statuses = []
#         top_children_indexes = get_top_children(NN_output)
#
#         best_child_val, best_rank = get_best_child_val(parent, NN_output, top_children_indexes)
#
#         children_as_moves = enumerate_legal_moves_using_piece_arrays(parent.game_board, parent.color)
#         num_legal_moves = len(children_as_moves)
#         if parent.children is None:
#             predicate = True
#         else:
#             predicate = num_legal_moves != len(parent.children)
#             child_indexes = list(map(lambda x: x.index, parent.children))
#         if predicate:  # if we don't already have all legal children
#             for child_as_move in children_as_moves:
#                 move = child_as_move['From'] + r'-' + child_as_move['To']
#                 move_as_index = index_lookup_by_move(move)
#                 if move_as_index in child_indexes:
#
#
#                 pruned_child, child_win_status = get_cached_child(parent, move, NN_output, top_children_indexes,
#                                                                   best_child_val, best_rank, sim_info, lock,
#                                                                   num_legal_moves, aggressive=None)
#
#                 if pruned_child is not None:
#                     if parent.children is not None:
#                         for child in parent.children:
#                             if child.game_board == pruned_child.game_board:
#                                 pruned_child = child
#                                 child_win_status = child.win_status
#                                 break
#                     pruned_children.append(pruned_child)
#                     children_win_statuses.append(child_win_status)
#             assign_pruned_children(parent, pruned_children, children_win_statuses, lock)
#         # else: parent already has all of its legal children
#         return pruned_children


def check_for_winning_move(child_node, rollout=False):
    is_game_over, winner_color = game_over(child_node.game_board)
    if is_game_over:  # one step lookahead see if children are game over before any NN updates
        set_game_over_values(child_node, child_node.color, winner_color, rollout)

def set_game_over_values(node, node_color, winner_color, rollout=False):
    node.gameover = True
    node.expanded = True
    node.subtree_checked = True
    node.visited = True
    #
    if rollout is False:
        overwhelming_amount = 65536# is this value right? technically true and will draw parent towards siblings of winning moves
        #but will make it too greedy when choosing a best move; maybe make best move be conservative? choose safest child?
    else:
        overwhelming_amount = 1
    if node.parent.children is None:
        # overwhelming_amount = 10000

        if winner_color == node_color:
            node.wins = 0 #overwhelming amount replaces this
            update_tree_wins(node, overwhelming_amount) #draw agent towards subtree
            node.win_status = True
        else:
            node.wins = 0 # this node will never win;
            update_tree_losses(node, overwhelming_amount) #keep agent away from subtree and towards subtrees of the same level
            node.win_status = False
    #else: had children already and this is coming from a reexpansion; don't send back duplicate gameover values

def reset_game_over_values(node):
    if node.gameover is True:
        if node.win_status is False:
            update_tree_wins(node, -(node.visits-1))
        elif node.win_status is True:
            update_tree_losses(node, -(node.wins-1))

def true_random_rollout_EOG_thread_func(node):
    depth = 0
    while not node.gameover and depth <= 4:
        if node.children is None:
            expand_node(node, rollout=True)
            node.expanded = False  # not a true expansion
        while node.children is not None:
            node = random.sample(node.children, 1)[0]  #
            depth += 1
    random_rollout(node)
    return node

def true_random_rollout_EOG(node):
    # with Pool(processes=10) as process:
    #     outcome_node = process.apply_async(true_random_rollout_EOG_thread_func, [node])
    #     outcome_node = outcome_node.get()
    outcome_node = true_random_rollout_EOG_thread_func(node)
    # if outcome_node.color == node.color:
    #     if outcome_node.win_status is True:
    #         update_tree_wins(node, 1)
    #     elif outcome_node.win_status is False:
    #         update_tree_losses(node, 1)
    # else:
    #     if outcome_node.win_status is True:
    #         update_tree_losses(node, 1)
    #     elif outcome_node.win_status is False:
    #         update_tree_wins(node, 1)