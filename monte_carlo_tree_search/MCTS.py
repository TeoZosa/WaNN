from Breakthrough_Player.board_utils import generate_policy_net_moves_batch
from monte_carlo_tree_search.TreeNode import TreeNode
from monte_carlo_tree_search.tree_search_utils import update_tree_visits, update_tree_wins
from monte_carlo_tree_search.tree_builder import build_game_tree
from tools.utils import move_lookup_by_index
import random
import time
#TODO: thread pool to play games, play games: ,

#TODO: algorithm: randomly initialize new nodes with some visit count [0, 100],

#Option A:
# start with root and put in NN queue, (level 0)
# enumerate legal moves,
# make all legal moves to generate new nodes (level 1) and put those in queue with opponent color,
# 1. search sequentially to desired depth,
# 2. run policy net JUST ONCE HERE when we get to bottom of tree,
# 3. run rollouts on leaves, update values as we pop back up the stack
# 4. keep searching, updating value based on uct and random rollouts.


def MCTS(game_board, player_color, time_to_think=180, depth_limit=5):
    startTime = time.time()
    root = TreeNode(game_board, player_color, None, None)
    game_tree = build_game_tree(player_color, 0, [root], depth_limit)
    update_values_from_policy_net(game_tree)
    while (time.time()- startTime < time_to_think ):
        MCTS_game(root)
        for i in range (0, len(game_tree)):
            print("Node {i}:\n"
                  "wins = {wins}\n"
                  "visits = {visits}\n".format(i=i, wins=game_tree[i].wins,visits=game_tree[i].visits))
    print("seconds taken: {}".format(time.time() - startTime))
    return move_lookup_by_index(choose_move(root).index, player_color)
    #TODO:change NN to be called asynchronously?



def MCTS_game(root):
    if root.children is None:
        #bottom of tree; random rollout
        win = random.randint(0,1)
        if win == 1:
            update_tree_wins(root)
        else:
            update_tree_visits(root)
    else:
        move = choose_move(root)
        MCTS_game(move)

def choose_move(node):
    parent_visits = node.visits
    best = None
    best_val = 0
    if node.children is not None:
        for child in node.children:
            update_UCT(child, parent_visits)
            if child.value > best_val:
                best = child
                best_val = best.value
    return best



def update_UCT(node, parent_visits):
    node.update_value(parent_visits)


# def launch_policy_net(batch):
#     return list(map(
#         lambda x: board_utils.generate_policy_net_moves(x.state, x.color), batch))

def update_values_from_policy_net(game_tree):
    NN_output = generate_policy_net_moves_batch(game_tree)
    for i in range(0, len(NN_output)):
        parent = game_tree[i]
        if parent.children is not None:
            for child in game_tree[i].children:
                # assert (child.parent is game_tree[i])
                if child.gameover == False: #if deterministically won/lost, don't update anything
                    NN_weighting = num_visits = 1000
                    weighted_wins = int(NN_output[i][child.index]*NN_weighting)
                    child.wins += weighted_wins # is this a good weighting?
                    child.visits += num_visits
                    parent.wins += weighted_wins
                    parent.visits += num_visits