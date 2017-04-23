#cython: language_level=3, boundscheck=False

class TreeNode(object):
    def __init__(self, game_board, white_pieces, black_pieces, player_color, index, parent, height):
        self.game_board = game_board
        self.color = player_color
        self.index = index #if None => root
        self.parent = parent#to pass up visits/wins; if None => root
        self.children = None # all kids
        self.other_children = None #separate non_pruned tree
        self.gameover = False
        self.visits = 0 #should be equal to sum of all children + num_times it was rolled out
        self.wins = 0 # 0 <= wins <= visits

        self.height = height #if root is first state in game MCTS saw
        self.best_child = None #to expand best child first
        self.expanded = False #if already expanded  for NN evaluation_function in tree reuse
        self.subtree_checked = False
        self.threads_checking_node = 0
        self.visited = False #for best child, visit first if not already visited
        self.win_status = None
        self.UCT_multiplier = 1
        self.sum_for_children_normalization = None #so we don't calculate for each child.
        self.num_children_being_checked = 0
        self.subtree_being_checked = False
        self.rolled_out_from = False
        self.reexpanded = False
        self.reexpanded_already = False

        self.eval_result = 0
        self.sum_rollout_rewards = 0

        self.wins_down_this_tree = 0 #backprop a win down this tree
        self.losses_down_this_tree = 0

        self.white_pieces = white_pieces
        self.black_pieces = black_pieces

class NeuralNetInput(object):
    def __init__(self):
        self.samples = []
        self.results = []