from Breakthrough_Player.board_utils import generate_policy_net_moves_batch, generate_policy_net_moves
import math
import numpy as np
import random
import time
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import threading
from multiprocessing import Pool

class SimulationInfo():
    def __init__(self, file):
        self.file= file
        self.counter = 0
        self.prev_game_tree_size = 0
        self.game_tree = []
        self.game_tree_height = 0
        self.start_time = None
        self.time_to_think = None
        self.root = None

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
def update_win_statuses(node, win_status):
    node.win_status = win_status
    parent = node.parent
    if parent is not None:#parent may now know if it is a win or loss.
        update_win_status_from_children(parent)

def update_win_status_from_children(node):
    win_statuses = get_win_statuses_of_children(node)
    set_win_status_from_children(node, win_statuses)

def get_win_statuses_of_children(node):
    win_statuses = []
    if node.children is not None:
        for child in node.children:
            win_statuses.append(child.win_status)
    return win_statuses

def set_win_status_from_children(node, children_win_statuses):
    if False in children_win_statuses: #Fact: if any child is false => parent is true
        update_win_statuses(node, True)  # some kid is a loser, I have some game winning move to choose from
    if True in children_win_statuses and not False in children_win_statuses and not None in children_win_statuses:
        update_win_statuses(node, False)#all children winners = node is a loss no matter what
    #else:
       # some kids are winners, some kids are unknown => can't say anything with certainty

def subtract_child_wins_and_visits(node, child): #for pruning tree built by MCTS using async updates
    parent = node
    while parent is not None:
        parent.wins -= child.wins
        parent.visits -= child.visits
        parent = parent.parent

def choose_UCT_move(node, start_time, time_to_think):
    best = None #shouldn't ever return None
    if node.children is not None:
        best = find_best_UCT_child(node, start_time, time_to_think)
    return best  #because a win for me = a loss for child

def choose_UCT_or_best_child(node, start_time, time_to_think):
    best = None  # shouldn't ever return None
    if node.best_child is not None: #return best child if not already previously expanded
        if node.best_child.visited is False:
            best = node.best_child
            node.best_child.visited = True
        else:
            best = find_best_UCT_child(node, start_time, time_to_think)
    elif node.children is not None: #if this node has children to choose from: should always happen
        best = find_best_UCT_child(node, start_time, time_to_think)
    return best  # because a win for me = a loss for child

def find_best_UCT_child(node, start_time, time_to_think):
    parent_visits = node.visits
    best = node.children[0]
    best_val = get_UCT(node.children[0], parent_visits, start_time, time_to_think)
    for i in range(1, len(node.children)):
        #TODO: 03/18/2017 10 PM don't consider children with a win status? seems right?
        if node.children[i].win_status is None: #only consider subtrees where we don't already know what's going to happen
            child_value = get_UCT(node.children[i], parent_visits, start_time, time_to_think)
            if child_value > best_val:
                best = node.children[i]
                best_val = child_value
    return best

def get_UCT(node, parent_visits, start_time, time_to_think):
    # TODO: change this? stops exploring subtrees that dumber opponent may lead to if 1 good move exists
    # Con: may not  preferentially search the tree that the opponent is more likely to pick
    # maybe I should add some value to the UCT multiplier when a win_status is declared? is this necessary due to the overwhelming
    # amount of wins/losses backpropagated?
    # if node.win_status == True:
    #     UCT = -999
    # elif node.win_status == False:
    #     UCT = 999
    if node.visits == 0:
        UCT = 998 #necessary for MCTS without NN initialized values
    else:
        UCT_multiplier = node.UCT_multiplier

        # 03/11/2017 games actually good with this and no division, but annealing seems right?
        # stochasticity_for_multithreading = random.random()  #/2.4 # keep between [0,0.417] => [1, 1.417]; 1.414 ~ √2

        #TODO test this. SEEMS like a good idea since it will explore early and exploit later
        annealed_factor = 1 - ((start_time - time.time())/time_to_think)  #  starts at 1 and trends towards 0 as search proceeds
        # with stochasticity multiplier between .5 and 1 so multithreading sees different values
        stochasticity_for_multithreading = (1- random.random()/2) * annealed_factor

        exploration_constant = 1.0 + stochasticity_for_multithreading # 1.414 ~ √2

        exploitation_factor = (node.visits - node.wins) / node.visits  # losses / visits of child = wins / visits for parent
        exploration_factor = exploration_constant * math.sqrt(math.log(parent_visits) / node.visits)
        UCT = np.float64((exploitation_factor + exploration_factor) * UCT_multiplier) #TODO change back to non-np float? shouldn't have problems with 0 UCT nodes anymore
    return UCT #UCT from parent's POV

#TODO: undiagnosed threading bug.
# Will sometimes be given an expanded root with no children (usually near the end of the game)
def randomly_choose_a_winning_move(node): #for stochasticity: choose among equally successful children
    best_nodes = []
    if node.children is not None:
        win_can_be_forced, best_nodes = check_for_forced_win(node.children)
        if not win_can_be_forced:
            best_nodes = get_best_children(node.children)
    if len(best_nodes) == 0:
        breakpoint = True
    return random.sample(best_nodes, 1)[0]  # because a win for me = a loss for child

def check_for_forced_win(node_children):
    guaranteed_children = []
    forced_win = False
    for child in node_children:
        if child.win_status is False:
            guaranteed_children.append(child)
            forced_win = True
    if len(guaranteed_children) > 0: #TODO: does it really matter which losing child is the best if all are losers?
        guaranteed_children = get_best_children(guaranteed_children)
    return forced_win, guaranteed_children

def get_best_children(node_children):#TODO: make sure to not pick winning children?
    best, best_val = get_best_child(node_children)
    best_nodes = []
    for child in node_children:  # find equally best children
        if not child.win_status == True: #only consider children who will not lead to a win for opponent
            child_win_rate = child.wins / child.visits
            if child_win_rate == best_val:
                if best.visits == child.visits:
                    best_nodes.append(child)
                    # should now have list of equally best children
    if len(best_nodes) == 0: #all children are winners => checkmated, just make the best move you have
        best_nodes.append(best)
    return best_nodes

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

def random_rollout(node):
    move = node
    while move.children is not None:
        move = random.sample(move.children, 1)[0]
    outcome = evaluation_function(node)
    if outcome == 0:
        update_tree_losses(node, 1)
    else:
        update_tree_wins(node, 1)

def evaluation_function(root):
    # row 2:
    # if white piece or (?) no diagonal black pieces in row 3
    weighted_board = {  #50%
        8: {'a': 24, 'b': 24, 'c': 24, 'd': 24, 'e': 24, 'f': 24, 'g': 24, 'h': 24},
        7: {'a': 21, 'b': 23, 'c': 23, 'd': 23, 'e': 23, 'f': 23, 'g': 23, 'h': 21},
        6: {'a': 14, 'b': 22, 'c': 22, 'd': 22, 'e': 22, 'f': 22, 'g': 22, 'h': 14},
        5: {'a': 9, 'b': 15, 'c': 21, 'd': 21, 'e': 21, 'f': 21, 'g': 15, 'h': 19},
        4: {'a': 6, 'b': 9, 'c': 16, 'd': 16, 'e': 16, 'f': 16, 'g': 9, 'h': 6},
        3: {'a': 3, 'b': 5, 'c': 10, 'd': 10, 'e': 10, 'f': 10, 'g': 5, 'h': 3},
        2: {'a': 2, 'b': 3, 'c': 3, 'd': 3, 'e': 3, 'f': 3, 'g': 3, 'h': 2},
        1: {'a': 5, 'b': 28, 'c': 28, 'd': 12, 'e': 12, 'f': 28, 'g': 28, 'h': 5}
    }
    white = 'White'
    black = 'Black'
    is_white_index = 9
    white_move_index = 10
    player_color = root.color
    opponent_dict = { 'White': 'Black',
             'Black': 'White'

    }
    win_weight = 0
    result = 0
    for row in root.game_board:
        if row != is_white_index and row != white_move_index:  # don't touch these indexes
            for col in root.game_board[row]:
                if root.game_board[row][col] == white:
                # if row == 2:
                #     if col == player_color:
                #         win_weight += 1
                #     if
                    result += weighted_board[row][col]
                elif root.game_board[row][col] == black:
                    result -= weighted_board[9-row][col]

    if result > 0:
        if player_color == white:
            return 1
        else:
            return 0
    else:
        if player_color == black:
            return 1
        else:
            return 0

def update_values_from_policy_net(game_tree, policy_net, lock = None, pruning=False): #takes in a list of parents with children,
    # removes children who were guaranteed losses for parent (should be fine as guaranteed loss info already backpropagated)
    # can also prune for not in top NN, but EBFS MCTS with depth_limit = 1 will also do that
    NN_output = policy_net.evaluate(game_tree)
    if lock is None: #make a useless lock so we can have clean code either way
        lock = threading.Lock()
    with lock:
        for i in range(0, len(NN_output)):
            top_children_indexes = get_top_children(NN_output[i], num_top=5)#TODO: play with this number
            parent = game_tree[i]
            sum_for_normalization = update_sum_for_normalization(parent, NN_output, top_children_indexes)
            if parent.children is not None:
                pruned_children = []
                for child in parent.children:#iterate to update values
                    normalized_value = (NN_output[child.index] / sum_for_normalization) * 100
                    if child.gameover is False and normalized_value > 0:# update only if not the end of the game or a reasonable value
                        if child.index in top_children_indexes: #prune
                            pruned_children.append(child)
                            update_child(child, NN_output[i], top_children_indexes)
                        elif not pruning:#if unknown and not pruning, keep
                            pruned_children.append(child)
                            update_child(child, NN_output[i], top_children_indexes)
                    elif child.win_status is not None: #if win/loss for parent, keep
                            pruned_children.append(child)
                        #no need to update child. if it has a win status, either it was expanded or was an instant game over
                parent.children = pruned_children


# def update_value_from_policy_net_async(game_tree, lock, policy_net, thread = None, pruning=False): #takes in a list of parents with children,
#     # removes children who were guaranteed losses for parent (should be fine as guaranteed loss info already backpropagated))
#     # can also prune for not in top NN, but EBFS MCTS with depth_limit = 1 will also do tha
#     # NN_processing = False
#
#     #TODO: figure out if this even needs thread local storage since threads enter separate function calls?
#
#     if thread is None:
#         thread = threading.local()
#     with lock:
#         thread.game_tree = game_tree
#     thread.NN_output = policy_net.evaluate(thread.game_tree)
#
#     #TODO: test thread safety without lock using local variables
#     thread.parent_index = 0
#     # with lock:
#     while thread.parent_index < len(thread.NN_output): #batch of nodes unique to thread in block
#         thread.pruned_children = []
#         thread.top_children_indexes = get_top_children(thread.NN_output[thread.parent_index], num_top=5)#TODO: play with this number
#         thread.parent = thread.game_tree[thread.parent_index]
#         if thread.parent.children is not None:
#             thread.child_index = 0
#             while thread.child_index < len(thread.parent.children):
#                 thread.child = thread.parent.children[thread.child_index]
#                 if thread.child.index in thread.top_children_indexes:
#                     thread.pruned_children.append(thread.child)
#                     if thread.child.gameover is False:  # update only if not the end of the game
#                         with lock:
#                             update_child(thread.child, thread.NN_output[thread.parent_index], thread.top_children_indexes)
#                 elif thread.child.win_status is not None: #if win/loss for thread.parent, keep
#                     thread.pruned_children.append(thread.child)
#                     #no need to update child. if it has a win status, either it was expanded or was an instant game over
#                 elif pruning:
#                     subtract_child_wins_and_visits(thread.parent, thread.child)
#                 elif not pruning:#if unknown and not pruning, keep non-top child
#                     if thread.child.gameover is False:
#                         with lock:
#                             update_child(thread.child, thread.NN_output[thread.parent_index], thread.top_children_indexes)
#                     thread.pruned_children.append(thread.child)
#                 thread.child_index += 1
#         thread.parent.children = thread.pruned_children
#         thread.parent_index += 1

def update_child(child, NN_output, top_children_indexes):
    child_val = NN_output[child.index]
    normalized_value = int(child_val  * 100)
    if child.index in top_children_indexes:  # weight top moves higher for exploitation
        rank = top_children_indexes.index(child.index)
        if rank == 0:
            child.parent.best_child = child #mark as node to expand first
            #TODO: #2 implemented 03102017 9:56 PM consider changing this to + child_val for a lighter UCT multiplier; 9:58 PM play with this number

        #TODO: 03/15/2017 removed based on prof Lorentz input
        child.UCT_multiplier  = 1 + child_val# stay close to policy (net) trajectory by biasing UCT selection of
                                             # NN's top picks as a function of probability returned by NN

    #TODO: 03/11/2017 7:45 AM added this to do simulated random rollouts instead of assuming all losses
    # i.e. if policy chooses child with 30% probability => 30/100 games
    # => randomly decide which of those 30 games are wins and which are losses

    #TODO: 03/15/2017 based on Prof Lorentz input, wins/visits = NNprob/100
    weighted_losses = normalized_value
    weighted_wins = 100-weighted_losses
    update_tree_losses(child, weighted_losses)
    update_tree_wins(child, weighted_wins)

def update_sum_for_normalization(parent, NN_output, child_indexes):
    #if we want to normalize over legal/top children vs relying on NN's softmax
    if parent.sum_for_children_normalization is None:
        parent.sum_for_children_normalization = sum(map(lambda child_index:
                                    NN_output[child_index],
                                                        child_indexes))
    return parent.sum_for_children_normalization

#returns a sorted list (best to worst as evaluated by the NN) of the children as move indexes
def get_top_children(NN_output, num_top=0):
    if num_top == 0:
        num_top = len(NN_output)
    return sorted(range(len(NN_output)), key=lambda k: NN_output[k], reverse=True)[:num_top]