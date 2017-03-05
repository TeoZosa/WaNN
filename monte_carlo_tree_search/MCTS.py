from Breakthrough_Player.board_utils import generate_policy_net_moves_batch, generate_policy_net_moves
from monte_carlo_tree_search.TreeNode import TreeNode
from monte_carlo_tree_search.tree_search_utils import update_tree_visits, update_tree_wins
from monte_carlo_tree_search.tree_builder import build_game_tree, visit_single_node_and_expand
from tools.utils import move_lookup_by_index
import random
import time
import math

#TODO: algorithm: randomly initialize new nodes with some visit count [0, 100],

#Option A:
# start with root and put in NN queue, (level 0)
# enumerate legal moves,
# make all legal moves to generate new nodes (level 1) and put those in queue with opponent color,
# 1. search sequentially to desired depth,
# 2. run policy net JUST ONCE HERE when we get to bottom of tree,
# 3. run rollouts on leaves, update values as we pop back up the stack
# 4. keep searching, updating value based on uct and random rollouts.


def MCTS_BFS_to_depth_limit(game_board, player_color, time_to_think=1000, depth_limit=5):

    # wanderer = 93,650k nodes  6GB
    #this = 270k nodes 50 GB..
    file = open(r'G:\TruncatedLogs\PythonDatasets\03042017_depth_{depth}__timetothink{time_to_think}.txt'.format(
        depth=depth_limit, time_to_think=time_to_think),'a')
    start_time = time.time()
    root = TreeNode(game_board, player_color, None, None)
    game_tree = build_game_tree(player_color, 0, [root], depth_limit)
    print("Number of Tree Nodes = {nodes} in {time} seconds".format(nodes=len(game_tree), time=time.time()-start_time))
    update_values_from_policy_net(game_tree)
    counter = 1
    root = game_tree[0]#tree node gets copied by threads; pull out new root

    while (time.time()- start_time < time_to_think ):
        MCTS_game(root)
        if counter % 1000 == 0:#log every 100th simulation
            print("Monte Carlo Game {iteration}\n".format(iteration=counter), file=file)
            for i in range (0, len(game_tree)):
                node_parent = game_tree[i].parent
                if node_parent is None:
                    UCT = 0
                else:
                    UCT = game_tree[i].get_UCT_value(node_parent.visits)
                print("Node {i}:    UCT = {uct}     wins = {wins}       visits = {visits}".format(
                    i=i, uct= UCT, wins=game_tree[i].wins,visits=game_tree[i].visits),
                    file=file)
        counter += 1
    print("seconds taken: {}".format(time.time() - start_time))
    best_move = move_lookup_by_index(choose_move(root).index, player_color)
    game_tree = []  # so garbage collector lets it go before next move? or multiprocessing again higher up?
    root = None
    print("For {player_color}, best move is {move}\n".format(player_color=player_color, move=best_move), file=file)
    file.close()
    return best_move
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


class simulation_info():
    def __init__(self, file):
        self.file= file
        self.counter = 0
        self.prev_game_tree_size = 0
        self.game_tree = []

def MCTS_with_expansions(game_board, player_color, time_to_think=300, depth_limit=5):
    sim_info = simulation_info(open(r'G:\TruncatedLogs\PythonDatasets\03052017ExpansionMCTS_depth_{depth}__timetothink{time_to_think}.txt'.format(
        depth=depth_limit, time_to_think=time_to_think), 'a'))
    root = TreeNode(game_board, player_color, None, None)
    start_time = time.time()
    while time.time() - start_time < time_to_think:
        run_MCTS_with_expansions_simulation(root, depth_limit, start_time, sim_info)
    best_move = move_lookup_by_index(choose_move(root).index, player_color)
    sim_info.file.close()
    return best_move

def run_MCTS_with_expansions_simulation(root, depth_limit, start_time, sim_info):
    sim_info.game_tree = MCTS_game_with_expansions(root, 0, depth_limit, sim_info.game_tree)
    if sim_info.counter % 1000 == 0:  # log every 1000th simulation
        print("Number of Tree Nodes added in simulation {counter} = "
              "{nodes} in {time} seconds".format(counter=sim_info.counter,
                                                 nodes=len(sim_info.game_tree) - sim_info.prev_game_tree_size,
                                                 time=time.time() - start_time), file=sim_info.file)
    sim_info.prev_game_tree_size = len(sim_info.game_tree)
    sim_info.counter += 1


def MCTS_game_with_expansions(root, depth, depth_limit, expanded_nodes):
    if root.children is None:
        if depth < depth_limit:
            #expand
            visit_single_node_and_expand([root, root.color])
            # update_single_value_from_policy_net(root)
            update_values_from_policy_net([root])
            #put it into game tree
            expanded_nodes.append(root)
            #root should now have children with values
            move = choose_move(root)
            MCTS_game_with_expansions(move, depth+1, depth_limit) #search until depth limit
            #fact: since this node was previously unexpanded, all subsequent nodes will be unexpanded =>
            #will increment depth each subsequent call
        else:
            # reached depth limit; random rollout
            win = random.randint(0, 1)
            if win == 1:
                update_tree_wins(root)
            else:
                update_tree_visits(root)
            return expanded_nodes
    else:
        move = choose_move(root)
        MCTS_game_with_expansions(move, depth, depth_limit)


def choose_move(node):
    parent_visits = node.visits
    best = None
    best_val = -1 #child may have a 0 uct value if visited too much
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
                if child.gameover == False: #if deterministically won/lost, don't update anything
                    NN_weighting = 1000
                    weighted_wins = int(NN_output[i][child.index]*NN_weighting)
                    # num_extra_visits = weighted_wins
                    # update_tree_visits(child, num_extra_visits)
                    update_tree_wins(child, weighted_wins) #say we won every time we visited,
                    # or else it may be a high visit count with low win count

def update_single_value_from_policy_net(node): # multi valued call should work too?
    NN_output = generate_policy_net_moves(node.game_board, node.color)
    for child in node.children:
        assert (child.parent is node)
        if child.gameover is False: #if deterministically won/lost, don't update anything
            NN_weighting = 1000
            weighted_wins = int(NN_output[child.index]*NN_weighting)
            # num_extra_visits = weighted_wins
            # update_tree_visits(child, num_extra_visits)
            update_tree_wins(child, weighted_wins) #say we won every time we visited,
            # or else it may be a high visit count with low win count
