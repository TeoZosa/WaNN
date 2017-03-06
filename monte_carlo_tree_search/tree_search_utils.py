from Breakthrough_Player.board_utils import generate_policy_net_moves_batch, generate_policy_net_moves
import math
import numpy as np
import random

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

def choose_UCT_move(node):
    parent_visits = node.visits
    best = None
    if node.children is not None:
        best = node.children[0]
        best_val = get_UCT(node.children[0], parent_visits)
        for i in range(1, len(node.children)):
            child_value = get_UCT(node.children[i], parent_visits)
            if child_value > best_val:
                best = node.children[i]
                best_val = child_value
    return best  #because a win for me = a loss for child?

def randomly_choose_a_winning_move(node):
    best = None
    best_nodes = []
    if node.children is not None:
        best = node.children[0]
        best_val = node.children[0].wins/node.children[0].visits
        for i in range(1, len(node.children)): #find best child
            child = node.children[i]
            child_win_rate = child.wins/child.visits
            if child_win_rate < best_val: #get the child with the lowest win rate
                best = child
                best_val = child_win_rate
            elif child_win_rate == best_val: #if both have equal win rate (i.e. both 0/visits), get the one with the most visits
                if best.visits < child.visits:
                    best = child
                    best_val = child_win_rate

        for child in node.children: #find equally best children
            child_win_rate = child.wins / child.visits
            if child_win_rate == best_val:
                if best.visits == child.visits:
                    best_nodes.append(child)
        #should now have list of equally best children
    return random.sample(best_nodes, 1)[0]  # because a win for me = a loss for child


def get_UCT(node, parent_visits):
    exploration_constant = 1.414 # 1.414 ~ √2
    exploitation_factor = (node.visits - node.wins) / node.visits  # losses / visits of child = wins / visits for parent
    exploration_factor = exploration_constant * math.sqrt(math.log(parent_visits) / node.visits)
    return np.float64(exploitation_factor + exploration_factor)

def update_values_from_policy_net(game_tree):
    NN_output = generate_policy_net_moves_batch(game_tree)
    for i in range(0, len(NN_output)):
        top_children_indexes = get_top_children(NN_output[i])
        parent = game_tree[i]
        if parent.children is not None:
            sum_for_normalization = 0
            for child in parent.children:#iterate over to get the sum
                sum_for_normalization += NN_output[i][child.index]
            for child in parent.children:#iterate again to update values
                # assert (child.parent is game_tree[i]) #for multithreading
                if child.gameover is False:  # update only if not the end of the game
                    update_child(child, NN_output[i], top_children_indexes, sum_for_normalization)


def update_child(child, NN_output, top_children_indexes, sum_for_normalization):
    num_top_children = len(top_children_indexes)
    # TODO: only update if over a certain threshold? or top 10?
    # sometimes majority of children end up being the same weight as precision is lost with rounding
    child_index = child.index
    child_val = NN_output[child_index]
    normalized_value = (child_val / sum_for_normalization) * 100
    if child_index in top_children_indexes:  # weight top moves higher for exploitation
        #  ex. even if all same rounded visits, rank 0 will have visits + 50
        rank = top_children_indexes.index(child_index)
        relative_weighting_offset = (num_top_children - rank)
        relative_weighting = relative_weighting_offset * (num_top_children * 2) #change number??
        weighted_wins = int(normalized_value) + relative_weighting
    else:
        weighted_wins = int(normalized_value)
    update_tree_losses(child, weighted_wins)  # say we won (child lost) every time we visited,
    # or else it may be a high visit count with low win count

def get_top_children(NN_output):
    num_to_consider = 5
    return sorted(range(len(NN_output)), key=lambda k: NN_output[k], reverse=True)[:num_to_consider]