from Breakthrough_Player.board_utils import generate_policy_net_moves_batch, generate_policy_net_moves
import math

def update_tree_wins(node, amount=1): #visits and wins go together
    node.wins += amount
    node.visits += amount
    parent = node.parent
    while parent is not None:
        grandparent = parent.parent
        if grandparent is not None:
            grandparent.wins += amount #since win for child = loss for parent = win for grandparent
        parent.visits += amount #should correctly increment grandparent visits as well
        parent = parent.parent

def update_tree_losses(node, amount=1): # visits and losses go together
    node.visits += amount
    parent = node.parent
    while parent is not None:
        grandparent = parent.parent
        if grandparent is not None:
            grandparent.visits += amount
        parent.wins += amount# since loss for child = win for parent = loss for grandparent
        parent.visits += amount
        parent = parent.parent

def choose_move(node):
    parent_visits = node.visits
    best = None
    best_val = -1  # child may have a 0 uct value if visited too much
    if node.children is not None:
        for child in node.children:
            child_value = get_UCT(child, parent_visits)
            if child_value > best_val:
                best = child
                best_val = child_value
    return best

def get_UCT(node, parent_visits):
    return (node.wins / node.visits) + (1.414 * math.sqrt(math.log(parent_visits) / node.visits))

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
                    update_tree_wins(child, weighted_wins)  # say we won every time we visited,
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
            update_tree_wins(child, weighted_wins)  # say we won every time we visited,
            # or else it may be a high visit count with low win count