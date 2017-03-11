from Breakthrough_Player.board_utils import game_over, enumerate_legal_moves, move_piece
from tools.utils import index_lookup_by_move, move_lookup_by_index
from monte_carlo_tree_search.TreeNode import TreeNode
from Breakthrough_Player.board_utils import generate_policy_net_moves_batch, check_legality_MCTS
import itertools
from monte_carlo_tree_search.tree_search_utils import update_tree_losses, update_tree_wins, \
    update_sum_for_normalization, get_top_children, update_child, set_win_status_from_children
from multiprocessing import Pool
import random



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
        # if len(unvisited_children_separated) == 1:
        #     unvisited_children_separated = unvisited_children_separated[0]
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

def expand_node(parent_node):
    children_as_moves = enumerate_legal_moves(parent_node.game_board, parent_node.color)
    child_nodes = []
    children_win_statuses = []
    for child_as_move in children_as_moves:  # generate children
        move = child_as_move['From'] + r'-' + child_as_move['To']
        child_node = init_child_node_and_board(move, parent_node)
        check_for_winning_move(child_node) #1-step lookahead for gameover
        children_win_statuses.append(child_node.win_status)
        child_nodes.append(child_node)
    set_win_status_from_children(parent_node, children_win_statuses)
    parent_node.children = child_nodes
    parent_node.expanded = True
    return child_nodes

def expand_descendants_to_depth_wrt_NN(unexpanded_nodes, depth, depth_limit, sim_info, lock, policy_net): #nodes are all at the same depth
    # Prunes child nodes to be the NN's top predictions, all nodes at a depth to the depth limit to take advantage
    # of GPU batch processing. This step takes time away from MCTS in the beginning, but builds the tree that the MCTS
    # will use later on in the game.

    # Problem: if instant game over not in top NN indexes, will not be marked. Calling method should use caution
    # when invoking this function at the latter stages of games.

    #don't expand end of game moves or previously expanded nodes (important for tree reuse)
    unexpanded_nodes = list(filter(lambda x: not x.expanded and not x.gameover and x.win_status is None, unexpanded_nodes))

    if len(unexpanded_nodes) > 0: #if any nodes to expand;
        NN_output = policy_net.evaluate(unexpanded_nodes)

        #get top children of unexpanded nodes
        # unexpanded_nodes_top_children_indexes = list(map(get_top_children, NN_output))

        #
        # #turn the child indexes into moves
        # unexpanded_nodes_suggested_children_as_moves = list(map(lambda node_top_children:
        #                                                         list(map(lambda top_child: move_lookup_by_index(top_child, parent_color),
        #                                                             node_top_children)),
        #                                                         unexpanded_nodes_top_children_indexes)) # [[moves i] ... [moves n]  ]
        # #filter the illegal moves (if any)
        # unexpanded_nodes_legal_children_as_moves = list(map(lambda unexpanded_node, children_as_moves:#make sure NN top picks are legal
        #                      list(filter(lambda child_as_move:
        #                             check_legality_MCTS(unexpanded_node.game_board, child_as_move),
        #                             children_as_moves)),
        #                      unexpanded_nodes, unexpanded_nodes_suggested_children_as_moves))
        #
        # #turn the child moves into nodes
        # unexpanded_nodes_children_nodes = list(map(lambda children_as_moves, parent_node:
        #                                            list(map(lambda child_as_move:
        #                                                init_child_node_and_board(child_as_move, parent_node),
        #                                                children_as_moves)),
        #                                            unexpanded_nodes_legal_children_as_moves, unexpanded_nodes))

        # mark parent as expanded, check for wins, update win status, assign children to parents,
        # and update children with NN values
        parent_color = unexpanded_nodes[0].color #fact, will always be the same color since we are looking at the same depth
        parent_height = unexpanded_nodes[0].height
        unexpanded_children = []
        if parent_height < 20:#TODO: dynamically change number of top children we consider based on height i.e. earlier in the game, num = 7, later in the game, num = 5?
            num_top_children = 5
        else: #TODO: maybe use normalized value to prune further? i.e. if rank 0 = 99%, all others are <1%, can we just prune them?
            num_top_children = 5
        #TODO: if we still want to try this, maybe expand num_top_children, but only pass subset to recursive call?
        for i in range(0, len(unexpanded_nodes)):
            parent = unexpanded_nodes[i]
            with lock: #Lock after the NN update
                if parent.expanded:
                    abort = True
                else:
                    abort = False
                    parent.expanded = True

            if not abort:
                top_children_indexes = get_top_children(NN_output[i], num_top_children)
                children = []
                children_win_statuses = []
                sum_for_normalization = update_sum_for_normalization(parent, NN_output[i], top_children_indexes)
                for child_index in top_children_indexes:
                    move = move_lookup_by_index(child_index, parent_color) #turn the child indexes into moves
                    if check_legality_MCTS(parent.game_board, move):
                        child = init_child_node_and_board(move, parent)
                        check_for_winning_move(child)
                        children_win_statuses.append(child.win_status)
                        children.append(child)
                        normalized_value = (NN_output[i][child.index] / sum_for_normalization) * 100
                        if child.gameover is False and child.win_status is None: # update and expand only if not the end of the game or unknown
                            if normalized_value > 0:
                                update_child(child, NN_output[i], top_children_indexes)
                                unexpanded_children.append(child)
                        else:
                            child.expanded = True #since this child has a win_status, don't check it again
                with lock:
                    if len (children) > 0:
                        parent.children = children
                        set_win_status_from_children(parent, children_win_statuses)
                    else:
                        parent.children = None
                    # parent.expanded = True

                sim_info.game_tree.append(parent)

        #below probably isn't safe with multithreading? or will initial filter and/or lock from above take care of it?
        if depth < depth_limit-1: #keep expanding; offset makes it so depth_limit = 1 => Normal Expansion MCTS
            # unexpanded_nodes_children_nodes = list(itertools.chain(*unexpanded_nodes_children_nodes))
            expand_descendants_to_depth_wrt_NN(unexpanded_children, depth + 1, depth_limit, sim_info, lock, policy_net)
        # return here
    #return here

def expand_descendants_to_depth_wrt_NN_midgame(unexpanded_nodes, depth, depth_limit, sim_info, lock, policy_net): #nodes are all at the same depth
    # Prunes child nodes to be the NN's top predictions or instant gameovers. Less efficient since we enumerate all children first.
    # all nodes at a depth to the depth limit to take advantage
    # of GPU batch processing. This step takes time away from MCTS in the beginning, but builds the tree that the MCTS
    # will use later on in the game.

    unexpanded_nodes = list(filter(lambda x: not x.expanded and not x.gameover and x.win_status is None, unexpanded_nodes)) #redundant

    if len(unexpanded_nodes) > 0: #if any nodes to expand;
        NN_output = policy_net.evaluate(unexpanded_nodes) #the point of multithreading is that other threads can do useful work while this thread blocks from the policy net calls
        parent_height = unexpanded_nodes[0].height
        unexpanded_children = []
        if parent_height < 60:#TODO: dynamically change number of top children we consider based on height i.e. middle of the game = 5, later in the game, num = 5?
            num_top_children = 5
        else:
            num_top_children = 5 #should this number get big again since we can potentially find forced wins?
        for i in range(0, len(unexpanded_nodes)):
            parent = unexpanded_nodes[i]
            with lock: #Lock after the NN update
                if parent.expanded:
                    abort = True
                else:
                    abort = False
                    parent.expanded = True
            #TODO: check if removing the lock here causes problems.
            if not abort: #if the node hasn't already been updated by another thread
                children_as_moves = enumerate_legal_moves(parent.game_board, parent.color)
                pruned_children = []
                children_win_statuses = []
                top_children_indexes = get_top_children(NN_output[i], num_top_children)
                sum_for_normalization = update_sum_for_normalization(parent, NN_output[i], top_children_indexes)
                for child_as_move in children_as_moves:
                    move = child_as_move['From'] + r'-' + child_as_move['To']
                    child = init_child_node_and_board(move, parent)
                    check_for_winning_move(child)  # 1-step lookahead for gameover
                    children_win_statuses.append(child.win_status)
                    normalized_value = (NN_output[i][child.index] / sum_for_normalization) * 100
                    if child.gameover is False and child.win_status is None:  # update only if not the end of the game
                        if child.index in top_children_indexes and normalized_value >= 100/num_top_children: #TODO filter children under average normalized value; should correct for ordinal effects?
                            pruned_children.append(child) #always keeps top children who aren't losses for parent
                            update_child(child, NN_output[i], top_children_indexes)
                            unexpanded_children.append(child)
                    else: #if it has a win status and not already in NN choices,, keep it
                        pruned_children.append(child)
                        child.expanded = True#child is a loss for parent, don't need to check it any more
                with lock:
                    if len (unexpanded_children) > 0:
                        parent.children = unexpanded_children
                        set_win_status_from_children(parent, children_win_statuses)
                    else:
                        parent.children = None
                parent.expanded = True
                set_win_status_from_children(parent, children_win_statuses)
                sim_info.game_tree.append(parent)
        #TODO: anneal this in MCTS class so it explores to deeper depth later in game? it isn't seeing forced wins as fast as it should
        if depth < depth_limit-1: #keep expanding; offset makes it so depth_limit = 1 => Normal Expansion MCTS
            expand_descendants_to_depth_wrt_NN_midgame(unexpanded_children, depth + 1, depth_limit, sim_info, lock, policy_net)
            # return here
            #return here

def assign_children(node, children):
    node.children = children

def set_expanded(node):
    node.expanded = True

def init_child_node_and_board(child_as_move, parent_node):
    game_board = parent_node.game_board
    parent_color= parent_node.color
    child_color = get_opponent_color(parent_color)
    child_board = move_piece(game_board, child_as_move, parent_color)
    child_index = index_lookup_by_move(child_as_move)
    return TreeNode(child_board, child_color, child_index, parent_node, parent_node.height+1)

def check_for_winning_move(child_node):
    is_game_over, winner_color = game_over(child_node.game_board)
    if is_game_over:  # one step lookahead see if children are game over before any NN updates
        set_game_over_values(child_node, child_node.color, winner_color)

def set_game_over_values(node, node_color, winner_color):
    node.gameover = True
    overwhelming_amount = 9999999#change this value? it may be too large and influence tree growth in a funny way
    if winner_color == node_color:
        update_tree_wins(node, overwhelming_amount) #draw agent towards move
        node.win_status = True
    else:
        node.wins = 0 # this node will never win;
        update_tree_losses(node, overwhelming_amount) #keep agent away from move
        node.win_status = False

def random_rollout(node):
    amount = 1  # increase to pretend to outweigh NN?
    win = random.randint(0, 1)
    if win == 1:
        update_tree_wins(node, amount)
    else:
        update_tree_losses(node, amount)