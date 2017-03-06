from Breakthrough_Player.board_utils import game_over, enumerate_legal_moves, move_piece
from tools.utils import index_lookup_by_move
from monte_carlo_tree_search.TreeNode import TreeNode
from monte_carlo_tree_search.tree_search_utils import update_tree_losses, update_tree_wins, update_win_statuses
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
        child_node = init_child_node_and_board(parent_node.game_board, move, parent_node.color, parent_node)
        check_for_winning_move(child_node)
        children_win_statuses.append(child_node.win_status)
        child_nodes.append(child_node)
    set_win_status_from_children(parent_node, children_win_statuses)
    parent_node.children = child_nodes
    return child_nodes

def update_win_status_from_children(node):
    win_statuses = get_win_statuses_of_children(node)
    set_win_status_from_children(node, win_statuses)

def get_win_statuses_of_children(node):
    win_statuses = []
    children = node.children
    for child in children:
        win_statuses.append(child.win_status)
    return win_statuses

def set_win_status_from_children(node, children_win_statuses):
    if False in children_win_statuses: #Fact: if any child is false => parent is true
        update_win_statuses(node, True)  # some kid is a loser, I have some game winning move to choose from
    if True in children_win_statuses and not False in children_win_statuses and not None in children_win_statuses:
        update_win_statuses(node, False)#all children winners = node is a loss no matter what
    #else:
       # some kids are winners, some kids are unknown => can't say anything with certainty

def init_child_node_and_board(game_board, child_as_move, parent_color, parent_node):
    child_color = get_opponent_color(parent_color)
    child_board = move_piece(game_board, child_as_move, parent_color)
    child_index = index_lookup_by_move(child_as_move)
    return TreeNode(child_board, child_color, child_index, parent_node)

def check_for_winning_move(child_node):
    is_game_over, winner_color = game_over(child_node.game_board)
    if is_game_over:  # one step lookahead see if children are game over before any NN updates
        set_game_over_values(child_node, child_node.color, winner_color)

def set_game_over_values(node, node_color, winner_color):
    node.gameover = True
    overwhelming_amount = 999999999 #change this value? it may be too large and influence tree growth in a funny way
    if winner_color == node_color:
        update_tree_wins(node, overwhelming_amount) #draw agent towards move
        update_win_statuses(node, True)
    else:
        node.wins = -overwhelming_amount # this node will never win; also sets UCT to be large
        update_tree_losses(node, overwhelming_amount) #keep agent away from move
        update_win_statuses(node, False)

def random_rollout(node):
    amount = 1  # increase to pretend to outweigh NN?
    win = random.randint(0, 1)
    if win == 1:
        update_tree_wins(node, amount)
    else:
        update_tree_losses(node, amount)