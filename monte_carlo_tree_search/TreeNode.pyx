#cython: language_level=3, boundscheck=False
from cpython.array cimport array
"""'
TreeNode dict. 
Holds the relevant info for a node in the game tree
''"""
cpdef dict TreeNode(dict game_board, list white_pieces, list black_pieces, str player_color, int index, dict parent, int height):
    cdef: #redundant
        int visits = 0
        int wins = 0
        int gameover_visits = 0
        int gameover_wins = 0
        int threads_checking_node = 0
        double UCT_multiplier = 1
        int num_children_being_checked = 0
        int overwhelming_amount = 1
        int num_to_consider = 0
        int num_to_keep = 999
        int times_reexpanded = 0
        array[dict] children = None
        dict best_child = None
    return {
        'game_board': game_board,
        'white_pieces': white_pieces,
        'black_pieces': black_pieces,
        'color': player_color,
        'index': index,
        'parent': parent,
        'height': height,
        'children': children,
        'other_children': children,
        'win_status': None,
        'best_child': best_child,
        'subtree_checked': False,
        'subtree_being_checked': False,
        'reexpanded_already': False,
        'gameover': False,
        # 'game_saving_move': False,
        'visits': visits,
        'wins': wins,
        'gameover_visits': gameover_visits,
        'gameover_wins': gameover_wins,
        'threads_checking_node': threads_checking_node,
        'UCT_multiplier': UCT_multiplier,
        'num_children_being_checked': num_children_being_checked,
        'overwhelming_amount': overwhelming_amount,
        'num_to_consider': num_to_consider,
        'num_to_keep': num_to_keep,
        'times_reexpanded': times_reexpanded
    }