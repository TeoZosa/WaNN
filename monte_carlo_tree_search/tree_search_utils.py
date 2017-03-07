from Breakthrough_Player.board_utils import generate_policy_net_moves_batch, generate_policy_net_moves
import math
import numpy as np
import random
import time
class SimulationInfo():
    def __init__(self, file):
        self.file= file
        self.counter = 1
        self.prev_game_tree_size = 0
        self.game_tree = []
        self.game_tree_height = 0
        self.start_time = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.counter = 1
        self.prev_game_tree_size = 0
        self.game_tree = []
        self.game_tree_height = 0
        self.game_node_height = []  # synchronized with nodes in game_tree
        self.start_time = None

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

#backpropagate guaranteed win status
def update_win_statuses(node, win_status): #is win status really necessary? shouldn't the UCT do the same thing with huge wins/losses?
    node.win_status = win_status
    # loss_status = not win_status
    # parent = node.parent
    # if parent is not None:
    #     update_win_statuses(parent, loss_status)

def choose_UCT_move(node):
    best = None #shouldn't ever return None
    if node.children is not None:
        best = find_best_UCT_child(node)
    return best  #because a win for me = a loss for child?

def choose_UCT_or_best_child(node):
    best = None  # shouldn't ever return None
    if node.best_child is not None: #expand best child if not already previously expanded
        if node.best_child.expanded == False:
            best = node.best_child
            node.best_child.expanded = True
        else:
            best = find_best_UCT_child(node)
    elif node.children is not None: #if this node has children to choose from: should always happen
        best = find_best_UCT_child(node)
    return best  # because a win for me = a loss for child?

def find_best_UCT_child(node):
    parent_visits = node.visits
    best = node.children[0]
    best_val = get_UCT(node.children[0], parent_visits)
    for i in range(1, len(node.children)):
        if not node.children[i].win_status == True: #don't even consider children who will win ; TAKEN CARE OF IN get_UCT??
            child_value = get_UCT(node.children[i], parent_visits)
            if child_value > best_val:
                best = node.children[i]
                best_val = child_value
    return best

def randomly_choose_a_winning_move(node): #for stochasticity: choose among equally successful children
    best_nodes = []
    if node.children is not None:
        win_can_be_forced, best_nodes = check_for_forced_win(node.children)
        if not win_can_be_forced:
            best_nodes = get_best_children(node.children)
    return random.sample(best_nodes, 1)[0]  # because a win for me = a loss for child

def check_for_forced_win(node_children):
    guaranteed_children = []
    forced_win = False
    for child in node_children:
        if child.win_status == False:
            guaranteed_children.append(child)
            forced_win = True
    if len(guaranteed_children) > 0: #TODO: does it really matter which losing child is the best if all are losers?
        guaranteed_children = get_best_children(guaranteed_children)
    return forced_win, guaranteed_children

def get_best_child(node_children):
    best = node_children[0]
    best_val = node_children[0].wins / node_children[0].visits
    for i in range(1, len(node_children)):  # find best child
        child = node_children[i]
        child_win_rate = child.wins / child.visits
        if child_win_rate < best_val:  # get the child with the lowest win rate
            best = child
            best_val = child_win_rate
        elif child_win_rate == best_val:  # if both have equal win rate (i.e. both 0/visits), get the one with the most visits
            if best.visits < child.visits:
                best = child
                best_val = child_win_rate
    return best, best_val

def get_best_children(node_children):#TODO: make sure to not pick winning children?
    best, best_val = get_best_child(node_children)
    best_nodes = []
    for child in node_children:  # find equally best children
        if not child.win_status == True: #don't even consider children who will win
            child_win_rate = child.wins / child.visits
            if child_win_rate == best_val:
                if best.visits == child.visits:
                    best_nodes.append(child)
                    # should now have list of equally best children
    if len(best_nodes) == 0: #all children are winners => checkmated, just make the best move you have
        best_nodes.append(best)
    return best_nodes

def get_UCT(node, parent_visits):
    if node.win_status == True:#taken care of in calling method; this will never run(saves on the computation)
        UCT_multiplier = -999
    elif node.win_status == False:
        UCT_multiplier = 999
    else:
        UCT_multiplier = node.UCT_multiplier
    exploration_constant = 1.414 # 1.414 ~ √2
    exploitation_factor = (node.visits - node.wins) / node.visits  # losses / visits of child = wins / visits for parent
    exploration_factor = exploration_constant * math.sqrt(math.log(parent_visits) / node.visits)
    return np.float64((exploitation_factor + exploration_factor) * UCT_multiplier)

def update_values_from_policy_net(game_tree):
    NN_output = generate_policy_net_moves_batch(game_tree)
    for i in range(0, len(NN_output)):
        top_children_indexes = get_top_children(NN_output[i])
        parent = game_tree[i]
        if parent.children is not None:
            sum_for_normalization = 0
            pruned_children = []
            for child in parent.children:#iterate over to get the sum
                if child.index in top_children_indexes:
                   sum_for_normalization += NN_output[i][child.index]
            for child in parent.children:#iterate again to update values
                if child.index in top_children_indexes:
                    pruned_children.append(child)
                elif child.win_status is False: #if it was a loser (win for parent) and wasn't in NN top index already
                    pruned_children.append(child)
                    # assert (child.parent is game_tree[i]) #for multithreading
                    # TODO: prune children. We don't even consider non top n children, UCT multiplier keeps moves relative priority/ranking
                if child.gameover is False:  # update only if not the end of the game
                    update_child(child, NN_output[i], top_children_indexes, sum_for_normalization)

            parent.children = pruned_children


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
        if rank == 0:
            child.parent.best_child = child #mark as node to expand first
        child.UCT_multiplier  = 1 + normalized_value/10 #prefer to choose NN's top picks as a function of probability returned by NN; policy trajectory stays close to NN trajectory
        relative_weighting_offset = (num_top_children - rank)
      #  relative_weighting = relative_weighting_offset * num_top_children * 2 #change number??
        weighted_wins = int(normalized_value) #+ relative_weighting
    else:
        weighted_wins = int(normalized_value)
    update_tree_losses(child, weighted_wins)  # say we won (child lost) every time we visited,
    # or else it may be a high visit count with low win count



def get_top_children(NN_output):
    num_to_consider = 5
    return sorted(range(len(NN_output)), key=lambda k: NN_output[k], reverse=True)[:num_to_consider]