#cython: language_level=3, boundscheck=False
# from cpython cimport bool
# from cpython.array cimport array

# class TreeNode(object):
#     def __init__(self, game_board, white_pieces, black_pieces, player_color, index, parent,  height):
#         self['game_board'] = game_board
#         self['color'] = player_color
#         self['index'] = index #if None => root
#         self['parent'] = parent#to pass up visits/wins if None => root
#         self['white_pieces'] = white_pieces
#         self['black_pieces'] = black_pieces
#         self['children'] = None # all kids
#         self['other_children'] = None #separate non_pruned tree
#         self['win_status'] = None
#         self['best_child'] = None #to expand best child first
#         self['subtree_being_checked'] = False
#         self['subtree_checked'] = False
#
#
#         self['reexpanded_already'] = False
#
#
#         self['gameover'] = False
#
#         self['visits'] = 0 #should be equal to sum of all children + num_times it was rolled out
#         self['wins'] = 0 # 0 <= wins <= visits
#
#         self['gameover']_visits = 0
#         self['gameover_wins'] = 0
#
#         self['height'] = height #if root is first state in game MCTS saw
#
#         self['threads_checking_node'] = 0
#
#
#         self['UCT_multiplier'] = 1
#         self['num_children_being_checked'] = 0
#
#
#         self['overwhelming_amount'] = 1#65536
#
#         self['num_to_consider'] = 1
#         self['num_to_keep'] = 1
#         self['times_reexpanded'] = 0


# cdef struct TreeNode:
#     # def __init__(self, game_board, white_pieces, black_pieces, player_color, index, parent,  height):
#     dict game_board = game_board
#     str color = player_color
#     int index = index #if None => root
#     TreeNode parent = parent#to pass up visits/wins if None => root
#     array[str] white_pieces = white_pieces
#     array[str] black_pieces = black_pieces
#     array[TreeNode] children # all kids
#     array[TreeNode] other_children #separate non_pruned tree
#     # enum win_status: 0 = 'None', 1 = 'True', 2 = 'False'
#     TreeNode best_child #to expand best child first
#     bool subtree_being_checked = False
#     bool subtree_checked = False
#
#
#     bool reexpanded_already = False
#
#
#     bool gameover = False
#
#     int visits = 0 #should be equal to sum of all children + num_times it was rolled out
#     int wins = 0 # 0 <= wins <= visits
#
#     int gameover_visits = 0
#     int gameover_wins = 0
#
#     int height = height #if root is first state in game MCTS saw
#
#     int threads_checking_node = 0
#     float UCT_multiplier = 1
#     int num_children_being_checked = 0

    #
    #
    #
    #
    #
    #
    # int overwhelming_amount = 1#65536
    #
    # int num_to_consider = 1
    # int num_to_keep = 1
    # int times_reexpanded = 0

def dict_TreeNode(game_board, white_pieces, black_pieces, player_color, index, parent,  height):
    return {
        'game_board': game_board,
        'white_pieces': white_pieces,
        'black_pieces': black_pieces,
        'color': player_color,
        'index': index,
        'parent': parent,
        'height': height,
        'children': None,
        'other_children': None,
        'win_status': None,
        'best_child': None,
        'subtree_checked': False,
        'subtree_being_checked': False,
        'reexpanded_already': False,
        'gameover': False,
        'visits': 0,
        'wins': 0,
        'gameover_visits': 0,
        'gameover_wins': 0,
        'threads_checking_node': 0,
        'UCT_multiplier': 1,
        'num_children_being_checked': 0,
        'overwhelming_amount': 1,
        'num_to_consider': 1,
        'num_to_keep': 1,
        'times_reexpanded': 0
    }