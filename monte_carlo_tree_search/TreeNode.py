class TreeNode(object):
    def __init__(self, game_board, player_color, index, parent, height):
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
        self.visited = False #for best child, visit first if not already visited
        self.win_status = None
        self.UCT_multiplier = 1
        self.sum_for_children_normalization = None #so we don't calculate for each child.

class NeuralNetInput(object):
    def __init__(self):
        self.samples = []
        self.results = []