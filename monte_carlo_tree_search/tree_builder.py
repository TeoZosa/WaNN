from Breakthrough_Player.board_utils import game_over, enumerate_legal_moves, move_piece
from tools.utils import index_lookup_by_move
from monte_carlo_tree_search.TreeNode import TreeNode
from monte_carlo_tree_search.tree_search_utils import update_tree_visits, update_tree_wins
from multiprocessing import Pool
from multiprocessing.managers import BaseManager

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
                                             depth_limit))
        # else: game over taken care of in visit
    return visited_queue


def update_bottom_of_tree(unvisited_queue):
    #NN will take care of these wins.
    for node in unvisited_queue:  # bottom of tree, so percolate visits to the top
        update_tree_visits(node)
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
    unvisited_children_separated = []
    arg_lists = [[node, player_color] for node in unvisited_queue]
    processes = Pool(processes=5)#prevent threads from taking up too much memory before joining
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
    player_color = node_and_color[1]
    unvisited_children = []
    game_board = node.game_board
    is_game_over, winner_color = game_over(game_board)
    if is_game_over:  # only useful at end of game
        set_game_over_values(node, player_color, winner_color)
    else:  # expand node, adding children to unvisited queue
        parent_node = node
        child_as_moves = enumerate_legal_moves(game_board, player_color)
        child_nodes = []
        for child_as_move in child_as_moves:  # generate grandchildren in order
            move = child_as_move['From'] + r'-' + child_as_move['To']
            child_node = init_child_node_and_board(game_board, move, player_color, parent_node)
            child_nodes.append(child_node)
        parent_node.children = child_nodes
        unvisited_children.extend(child_nodes)
    return unvisited_children# return tuple instead of sharing object


def init_child_node_and_board(game_board, child_as_move, parent_color, parent_node):
    child_color = get_opponent_color(parent_color)
    child_board = move_piece(game_board, child_as_move, parent_color)
    child_index = index_lookup_by_move(child_as_move)
    # child_value = random.randint(0, 100)  # inclusive
    return TreeNode(child_board, child_color, child_index, parent_node)


def set_game_over_values(node, player_color, winner_color):
    node.gameover = True
    overwhelming_amount = 999999999 #should be infinity as this node will always win
    if winner_color == player_color:
        update_tree_wins(node, overwhelming_amount) #draw agent towards move
    else:
        node.wins = 0 # this node will never win
        update_tree_visits(node, overwhelming_amount) #keep agent away from move