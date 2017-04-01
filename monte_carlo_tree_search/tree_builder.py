from Breakthrough_Player.board_utils import game_over, enumerate_legal_moves, move_piece
from tools.utils import index_lookup_by_move, move_lookup_by_index
from monte_carlo_tree_search.TreeNode import TreeNode
from Breakthrough_Player.board_utils import  check_legality_MCTS
from monte_carlo_tree_search.tree_search_utils import update_tree_losses, update_tree_wins, \
    get_top_children, update_child, set_win_status_from_children, random_rollout, update_win_status_from_children, eval_child, SimulationInfo
from multiprocessing import Pool, Process, pool
from math import ceil
from threading import Thread, Lock
from sys import stdout
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
    children_as_moves = enumerate_legal_moves(parent_node.game_board, parent_node.color)
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
    while depth < depth_limit:
        # if len (unexpanded_nodes) > 0:
        #     if unexpanded_nodes[0] is None:
        #         True

        unexpanded_nodes = list(filter(lambda x: (x.children is None and not x.gameover ), unexpanded_nodes)) #redundant
        if len(unexpanded_nodes) > 0: #if any nodes to expand;
            # the point of multithreading is that other threads can do useful work while this thread blocks from the policy net calls
            NN_output = policy_net.evaluate(unexpanded_nodes)

            # process = Pool(processes=1)
            # NN_output = process.map(generate_policy_net_moves_batch,[unexpanded_nodes])
            # process.join()
            # process.close()

            unexpanded_children = []
            best_val = 1
            best_nodes = []
            for i in range(0, len(unexpanded_nodes)):
                parent = unexpanded_nodes[i]
                with lock:
                    if  parent.threads_checking_node <= 1 and parent.children is None :
                        if parent.threads_checking_node <=0:
                            parent.threads_checking_node = 1
                        abort = False
                        parent.being_checked = True
                    else:
                        parent.threads_checking_node -=1
                        abort = True
                if not abort:
                    parent_win_rate = parent.wins/max(1, parent.visits)
                    if parent_win_rate <= best_val:#grandparent would've picked the parent with the lowest wins
                        best_nodes.append(parent)
                        best_val = parent_win_rate
                    unexpanded_children = update_parent(without_enumerating, parent, NN_output[i], sim_info, lock)

                        # unexpanded_children.extend(update_parent(without_enumerating, parent, NN_output[i], sim_info, lock))
                    # process = Pool(processes=1)
                    # grandparent = parent.parent
                    # parent.parent = None #sever the reference to the rest of the tree so it isn't copied over
                    #
                    # unexpanded_children_and_game_tree = process.map(update_parent_for_process,[[without_enumerating, parent, NN_output[i]]])[0]
                    # process.close()
                    # process.join()
                    # with lock:
                    #     parent.being_checked = False
                    #     parent.threads_checking_node -=1
                    #     parent.parent = grandparent #reattach rest of tree
                    #     #since this is a copy, now have to redo all values
                    #     unexpanded_children = unexpanded_children_and_game_tree[0]
                    #     if unexpanded_children is not None:
                    #         if len(unexpanded_children) > 0 and parent.children is None:
                    #             if not parent.expanded:  # if an expansion wasn't already attempted
                    #                 sim_info.game_tree.append(parent)
                    #             parent.expanded = True #should've been set below
                    #             parent.children = unexpanded_children
                    #             reattach_parent_to_children(parent, parent.children)
                    #             backpropagate_num_checked_children(parent)
                    #             update_win_status_from_children(parent)
                    #             sim_info.game_tree.extend(unexpanded_children_and_game_tree[1])



            for i in range(len(best_nodes) - 1, -1, -1):  # from best to worst, find children to expand
                if best_nodes[i].children is not None:
                    best_child = best_nodes[i].best_child
                    if best_child is not None:
                        unexpanded_children = [best_nodes[i].best_child]
                    else:
                        best_child_val = 1
                        best_child = None
                        for child in best_nodes[i].children:
                            child_win_rate = child.wins/max(1, child.visits)
                            if child_win_rate <= best_child_val:
                                best_child = child
                                best_child_val = child_win_rate
                        unexpanded_children = [best_child] #check the best one
                    break
            unexpanded_nodes = unexpanded_children

            depth += 1
            # if depth < depth_limit-1: #keep expanding; offset makes it so depth_limit = 1 => Normal Expansion MCTS
            #     expand_descendants_to_depth_wrt_NN(unexpanded_children, without_enumerating, depth + 1, depth_limit, sim_info, lock, policy_net)
                # return here
                #return here
            if len(unexpanded_nodes)==0:
                prune_from_win_status(parent)
        else:
            depth = depth_limit #done if no nodes to expand

def reattach_parent_to_children(parent, children):
    for child in children:
        child.parent = parent
        eval_child(child)
        check_for_winning_move(child)



def update_parent(without_enumerating_children, parent, NN_output, sim_info, lock):
    pruned_children = []
    with lock:  # Lock after the NN update and check if we still need to update the parent
        if parent.children is not None:
            abort = True
        else:
            abort = False
            if not parent.expanded: #if an expansion wasn't already attempted
               sim_info.game_tree.append(parent)
            parent.expanded = True
    if not abort:  # if the node hasn't already been updated by another thread
        if without_enumerating_children:
            pruned_children = update_and_prune(parent, NN_output, sim_info,lock)
        else:
            pruned_children = enumerate_update_and_prune(parent, NN_output, sim_info,lock)
    return pruned_children

def update_parent_for_process(args):
    without_enumerating_children = args[0]
    parent = args[1]
    NN_output = args[2]
    sim_info = SimulationInfo(stdout)
    lock = Lock()
    pruned_children = []
    with lock:  # Lock after the NN update and check if we still need to update the parent
        if parent.children is not None or not parent.expanded:
            abort = True
        else:
            abort = False
            if not parent.expanded: #if an expansion wasn't already attempted
               sim_info.game_tree.append(parent)
            parent.expanded = True
    if not abort:  # if the node hasn't already been updated by another thread
        if without_enumerating_children:
            pruned_children = update_and_prune(parent, NN_output, sim_info,lock)
        else:
            pruned_children = enumerate_update_and_prune(parent, NN_output, sim_info,lock)
    return pruned_children, sim_info.game_tree

def update_and_prune(parent, NN_output,sim_info, lock):
    pruned_children = []
    children_win_statuses = []

    # comparing to child val should reduce this considerably,
    # yet still allows us to call parent function from a top-level asynchronous tree updater
    num_to_check_for_legality = 30
    top_children_indexes = get_top_children(NN_output, num_to_check_for_legality)

    best_child_val = get_best_child_val(parent, NN_output, top_children_indexes)
    for child_index in top_children_indexes:
        move = move_lookup_by_index(child_index, parent.color)  # turn the child indexes into moves
        if check_legality_MCTS(parent.game_board, move):
            pruned_child, child_win_status = get_cached_child(parent, move, NN_output, top_children_indexes,
                                                              best_child_val, sim_info, lock, num_to_check_for_legality, aggressive=None)

            if pruned_child is not None:
                pruned_children.append(pruned_child)
                children_win_statuses.append(child_win_status)
    assign_pruned_children(parent, pruned_children, children_win_statuses, lock)
    return pruned_children

def enumerate_update_and_prune(parent, NN_output,sim_info, lock):
    #Less efficient since we enumerate all children first.
    pruned_children = []
    children_win_statuses = []
    top_children_indexes = get_top_children(NN_output)

    best_child_val = get_best_child_val(parent, NN_output, top_children_indexes)

    children_as_moves = enumerate_legal_moves(parent.game_board, parent.color)
    num_legal_moves = len(children_as_moves)
    if parent.children is None:
        predicate = True
    else:
        predicate = num_legal_moves != len(parent.children)
    if predicate:#if we don't already have all legal children
        for child_as_move in children_as_moves:
            move = child_as_move['From'] + r'-' + child_as_move['To']
            pruned_child, child_win_status = get_cached_child(parent, move, NN_output, top_children_indexes, best_child_val, sim_info, lock, num_legal_moves, aggressive=None)
            if pruned_child is not None:
                pruned_children.append(pruned_child)
                children_win_statuses.append(child_win_status)
        assign_pruned_children(parent, pruned_children, children_win_statuses, lock)
    #else: parent already has all of its legal children
    return pruned_children

def get_cached_child(parent, move, NN_output, top_children_indexes, best_child_val, sim_info, lock, num_legal_moves, aggressive=None, on = False):
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
                                                                      best_child_val, sim_info, lock, num_legal_moves,
                                                                      opening_move)
                else:#don't check moves we know we aren't going to make
                    pruned_child = child_win_status = None
            else: #black's move
                pruned_child, child_win_status = get_pruned_child(parent, move, NN_output, top_children_indexes,
                                                                  best_child_val,
                                                                  sim_info, lock, num_legal_moves, opening_move)
        else: #non-opening move
            opening_move = False
            pruned_child, child_win_status = get_pruned_child(parent, move, NN_output, top_children_indexes, best_child_val,
                                                               sim_info, lock, num_legal_moves, opening_move)
    else:
        opening_move = False
        pruned_child, child_win_status = get_pruned_child(parent, move, NN_output, top_children_indexes, best_child_val,
                                                          sim_info, lock, num_legal_moves, opening_move)

    return pruned_child, child_win_status

def get_best_child_val(parent, NN_output, top_children_indexes):
    best_child_val = 0
    for top_child_index in top_children_indexes:  # find best legal child value
        top_move = move_lookup_by_index(top_child_index, parent.color)
        if check_legality_MCTS(parent.game_board, top_move):
            best_child_val = NN_output[top_child_index]
            break
    return best_child_val

def get_pruned_child(parent, move, NN_output, top_children_indexes, best_child_val,sim_info, lock, num_legal_children, opening_move=False):
    pruned_child = None
    child = init_child_node_and_board(move, parent)
    check_for_winning_move(child)  # 1-step lookahead for gameover
    child_val = NN_output[child.index]
    # if child.height>60: #not seeing forced wins, play with pruning at later depths
    #     num_top_to_consider = 2
    # else:
    #     num_top_to_consider = 1
    num_top_to_consider = 1
    top_n_children = top_children_indexes[:num_top_to_consider]

    if child.gameover is False:  # update only if not the end of the game
        if opening_move or parent.height >= 600:#keep all children
            predicate = True
        else: #prune as needed
            predicate = child.index in top_n_children or \
                child_val > .30 or best_child_val - child_val < .10
        #  opens up the tree to more lines of play if best child sucks to begin with.
        # in the worst degenerate case where best_val == ~4.5%, will include all children which is actually pretty justified.
        if predicate:  # absolute value not necessary ; if #1 or over threshold or within 10% of best child
            #TODO: keep all children past a certain height?
            # if child.index in top_n_children:
            #     backprop_win = True
            # else:
            #     backprop_win = False
            # update_child(child, NN_output, top_children_indexes, backprop_win)

            update_child(child, NN_output, top_children_indexes, num_legal_children)
            pruned_child = child  # always keeps top children who aren't losses for parent
    else:  # if it has a win status and not already in NN choices, keep it (should always be a game winning node)
        pruned_child = child
        child.expanded = True  # don't need to check it any more
        sim_info.game_tree.append(child)
    return pruned_child, child.win_status

def assign_pruned_children(parent, pruned_children, children_win_statuses, lock):
    with lock:
        parent.being_checked = False
        parent.threads_checking_node -=1
        if len(pruned_children) > 0:
            if parent.children is None:
                parent.children = pruned_children
                backpropagate_num_checked_children(parent)
                set_win_status_from_children(parent, children_win_statuses)
                prune_from_win_status(parent)
            else: #had children but we are now adding all children past a certain height
                new_children_added = False
                new_children = []
                for new_child in pruned_children:
                    duplicate_game_board = False
                    for child in parent.children:
                        if new_child.game_board == child.game_board:
                            duplicate_game_board = True
                    if not duplicate_game_board:
                        new_children.append(new_child)
                        new_children_added = True
                if new_children_added:
                    parent.children.extend(new_children)
                    backpropagate_num_checked_children(parent, reexpansion=True)
                    set_win_status_from_children(parent, children_win_statuses)
                    prune_from_win_status(parent)


def prune_from_win_status(node, height_cutoff = 600):
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



def backpropagate_num_checked_children(node, reexpansion=False):
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
                node = node.parent
            else:
                node = None

def set_num_checked_children(node): #necessary for gameover kids: wouldn't have reported that they are expanded since another node may also be
    count = 0
    for child in node.children:
        if child.subtree_checked:
            count+= 1
    node.num_children_checked = count

def init_child_node_and_board(child_as_move, parent_node):

    game_board = parent_node.game_board
    parent_color= parent_node.color
    child_color = get_opponent_color(parent_color)
    child_board = move_piece(game_board, child_as_move, parent_color)
    child_index = index_lookup_by_move(child_as_move)
    return TreeNode(child_board, child_color, child_index, parent_node, parent_node.height+1)

def init_new_root(new_root_as_move, new_root_game_board, player_color, new_root_parent, policy_net, sim_info, lock): #online reinforcement learning if wanderer made a move that wasn't in the tree




    if new_root_parent.children is None:# in case parent was never expanded
        print("New root's parent had no children", file=sim_info.file)

        new_root_parent.threads_checking_node = 1
        expand_descendants_to_depth_wrt_NN([new_root_parent], False, 0, 500, sim_info, lock,
                                           policy_net)

        if new_root_parent.children is None: #still none means the pruning was too aggressive; still append this child.
            print("Pruning too aggressive: New root's parent still has no children after expansion", file=sim_info.file)

            new_root_index = index_lookup_by_move(new_root_as_move)
            new_root = TreeNode(new_root_game_board, player_color, new_root_index, new_root_parent,
                                new_root_parent.height + 1)
            NN_output = policy_net.evaluate([new_root_parent])[0]
            top_children_indexes = get_top_children(NN_output)
            update_child(new_root, NN_output, top_children_indexes,
                         len(new_root_parent.children))  # update prior values on the node NOTE: never a game over
            print("PRUNING INVESTIGATION: new root's probability = %{}".format((100 - new_root.wins) / 100),
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

                new_root_index = index_lookup_by_move(new_root_as_move)
                new_root = TreeNode(new_root_game_board, player_color, new_root_index, new_root_parent,
                                    new_root_parent.height + 1)
                NN_output = policy_net.evaluate([new_root_parent])[0]
                top_children_indexes = get_top_children(NN_output)
                update_child(new_root, NN_output, top_children_indexes,
                             len(new_root_parent.children))  # update prior values on the node NOTE: never a game over
                print("PRUNING INVESTIGATION: new root's probability = %{}".format((100 - new_root.wins) ),
                      file=sim_info.file)
                new_root_parent.children.append(new_root) #attach new root to parent
            else:
                new_root = duplicate_child

    else:#has kids but new_root wasn't in them.
        print("Pruning investigation: New root's parent still did not have the actual new root after expansion",
              file=sim_info.file)
        new_root_index = index_lookup_by_move(new_root_as_move)
        new_root = TreeNode(new_root_game_board, player_color, new_root_index, new_root_parent,
                            new_root_parent.height + 1)
        NN_output = policy_net.evaluate([new_root_parent])[0]
        top_children_indexes = get_top_children(NN_output)
        update_child(new_root, NN_output, top_children_indexes,
                     len(new_root_parent.children))  # update prior values on the node NOTE: never a game over
        print("PRUNING INVESTIGATION: new root's probability = %{}".format((100-new_root.wins)),
              file=sim_info.file)
        new_root_parent.children.append(new_root)  # attach new root to parent

    backpropagate_num_checked_children(new_root_parent, True)
    update_win_status_from_children(new_root_parent)
    return new_root

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
        overwhelming_amount = 9999999# is this value right? technically true and will draw parent towards siblings of winning moves
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