from Breakthrough_Player.board_utils import generate_policy_net_moves_batch, generate_policy_net_moves
import math
import numpy as np

#backpropagate wins
def update_tree_wins(node, amount=1): #visits and wins go together
    node.wins += amount
    node.visits += amount
    parent = node.parent
    if parent is not None: #node win => parent loss
        update_tree_losses(parent, amount)

#backpropagate losses
def update_tree_losses(node, amount=1): # visit and no win = loss
    node.visits += amount
    parent = node.parent
    if parent is not None: #node loss => parent win
        update_tree_wins(parent, amount)

def choose_move(node):
    parent_visits = node.visits
    worst = None
    if node.children is not None:
        worst = node.children[0]
        worst_val = get_UCT(node.children[0], parent_visits)
        for i in range(1, len(node.children)):
            child_value = get_UCT(node.children[i], parent_visits)
            if child_value < worst_val:
                worst = node.children[i]
                worst_val = child_value
    return worst  #because a win for me = a loss for child?

def get_UCT(node, parent_visits):
    return np.float64((node.wins / node.visits) + (1.414 * math.sqrt(math.log(parent_visits) / node.visits)))

def update_values_from_policy_net(game_tree):
    NN_output = generate_policy_net_moves_batch(game_tree)
    for i in range(0, len(NN_output)):
        parent = game_tree[i]
        if parent.children is not None:
            for child in game_tree[i].children:
                # assert (child.parent is game_tree[i])
                if child.gameover == False:  # if deterministically won/lost, don't update anything
                    NN_weighting = 1000
                    weighted_wins = int(NN_output[i][child.index] * NN_weighting)
                    # num_extra_visits = weighted_wins
                    # update_tree_losses(child, num_extra_visits)
                    update_tree_losses(child, weighted_wins)  # say we won (child lost) every time we visited,
                    # or else it may be a high visit count with low win count

def update_single_value_from_policy_net(node):  # multi valued call should work too?
    NN_output = generate_policy_net_moves(node.game_board, node.color)
    for child in node.children:
        assert (child.parent is node)
        if child.gameover is False:  # if deterministically won/lost, don't update anything
            NN_weighting = 1000
            weighted_wins = int(NN_output[child.index] * NN_weighting)
            # num_extra_visits = weighted_wins
            # update_tree_losses(child, num_extra_visits)
            update_tree_losses(child, weighted_wins)  # say we won (child lost) every time we visited,
            # or else it may be a high visit count with low win count