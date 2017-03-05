import math
class TreeNode(object):
    def __init__(self, game_board, player_color, index, parent):
        self.game_board = game_board
        self.color = player_color
        self.index = index #if None => root
        self.parent = parent#to pass up visits/wins
        self.children = None
        self.gameover = False

        # self.value = 0 #UCT calculated during tree search
        self.visits = 0 #should be equal to sum of all children
        self.wins = 0 # 0 <= wins <= visits

    def get_UCT_value(self, parent_visits):
        return (self.wins / self.visits) + (1.414 * math.sqrt(math.log(parent_visits) / self.visits))
