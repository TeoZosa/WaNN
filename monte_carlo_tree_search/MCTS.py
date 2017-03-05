from monte_carlo_tree_search.TreeNode import TreeNode
from monte_carlo_tree_search.tree_search_utils import update_tree_losses, update_tree_wins, choose_move, update_values_from_policy_net
from monte_carlo_tree_search.tree_builder import build_game_tree, visit_single_node_and_expand
from tools.utils import move_lookup_by_index
import random
import time

class SimulationInfo():
    def __init__(self, file):
        self.file= file
        self.counter = 1
        self.prev_game_tree_size = 0
        self.game_tree = []
        self.game_tree_height = 0

#Option A: BFS expansion
# start with root and put in NN queue, (level 0)
# enumerate legal moves,
# make all legal moves to generate new nodes (level 1) and put those in queue with opponent color,
# 1. search sequentially to desired depth,
# 2. run policy net JUST ONCE HERE when we get to bottom of tree,
# 3. keep searching to bottom of tree where we do random rollouts.

def MCTS_BFS_to_depth_limit(game_board, player_color, time_to_think=1000, depth_limit=5):
    sim_info = SimulationInfo(open(r'G:\TruncatedLogs\PythonDatasets\03042017_depth_{depth}__timetothink{time_to_think}.txt'.format(
        depth=depth_limit, time_to_think=time_to_think),'a'))
    # wanderer = 93,650k nodes  6GB
    #this = 270k nodes 50 GB..
    start_time = time.time()
    root = TreeNode(game_board, player_color, None, None)
    sim_info.game_tree = build_game_tree(player_color, 0, [root], depth_limit)
    print("Number of Tree Nodes = {nodes} in {time} seconds".format(nodes=len(sim_info.game_tree), time=time.time()-start_time))
    update_values_from_policy_net(sim_info.game_tree)

    while (time.time()- start_time < time_to_think ):
        run_BFS_MCTS_simulation(sim_info)

    print("seconds taken: {}".format(time.time() - start_time))
    best_move = move_lookup_by_index(choose_move(root).index, player_color)
    print("For {player_color}, best move is {move}\n".format(player_color=player_color, move=best_move),file=sim_info.file)
    sim_info.file.close()
    return best_move

def run_BFS_MCTS_simulation(sim_info):
    root = sim_info.game_tree[0]
    BFS_MCTS_game(root)
    if sim_info.counter % 400 == 0:  # log every 400th simulation
        print("Monte Carlo Game {iteration}\n".format(iteration=sim_info.counter), file=sim_info.file)
        for i in range(0, len(sim_info.game_tree)):
            node_parent = sim_info.game_tree[i].parent
            if node_parent is None:
                UCT = 0
            else:
                UCT = sim_info.game_tree[i].get_UCT_value(node_parent.visits)
            print("Node {i}:    UCT = {uct}     wins = {wins}       visits = {visits}".format(
                i=i, uct=UCT, wins=sim_info.game_tree[i].wins, visits=sim_info.game_tree[i].visits),
                file=sim_info.file)
    sim_info.counter += 1

def BFS_MCTS_game(root):
    if root.children is None:
        #bottom of tree; random rollout
       random_rollout(root)
    else:
        move = choose_move(root)
        BFS_MCTS_game(move)
        
        
#Option B: Traditional MCTS with expansion using policy net to generate prior values
# start with root and put in NN queue, (level 0)
# while time to think,
# 1. MCTS search to find the best move
# 2. When we reach a leaf node, expand, evaluate with policy net, and update prior values on children
# 3. keep searching to desired depth (final depth = depth at expansion + depth_limit)
# 4. do random rollouts. repeat 1.

#TODO: for root-level parallelism here, add stochasticity to UCT constant? 
def MCTS_with_expansions(game_board, player_color, time_to_think=10, depth_limit=5):
    sim_info = SimulationInfo(open(r'G:\TruncatedLogs\PythonDatasets\03052017ExpansionMCTS_depth_{depth}__timetothink{time_to_think}.txt'.format(
        depth=depth_limit, time_to_think=time_to_think), 'a'))
    root = TreeNode(game_board, player_color, None, None)
    start_time = time.time()

    while time.time() - start_time < time_to_think:
        run_MCTS_with_expansions_simulation(root, depth_limit, start_time, sim_info)

    best_move = move_lookup_by_index(choose_move(root).index, player_color)
    sim_info.file.close()
    return best_move

def run_MCTS_with_expansions_simulation(root, depth_limit, start_time, sim_info):
    play_MCTS_game_with_expansions(root, 0, depth_limit, sim_info)
    if sim_info.counter % 10 == 0:  # log every 10th simulation
        print("Number of Tree Nodes added in simulation {counter} = "
              "{nodes} in {time} seconds\n"
              "Current tree height = {height}".format(counter=sim_info.counter,
                                                 nodes=len(sim_info.game_tree) - sim_info.prev_game_tree_size,
                                                 time=time.time() - start_time, height=sim_info.game_tree_height), file=sim_info.file)
    sim_info.prev_game_tree_size = len(sim_info.game_tree)
    sim_info.counter += 1


def play_MCTS_game_with_expansions(root, depth, depth_limit, sim_info, this_height=0):
    if root.children is None:
        if not root.gameover:
            if depth < depth_limit:
                visit_single_node_and_expand([root, root.color])
                update_values_from_policy_net([root])

                #put expanded node into game tree
                sim_info.game_tree.append(root)

                #root should now have children with values
                move = choose_move(root)
                #fact: since this node was previously unexpanded, all subsequent nodes will be unexpanded =>
                #will increment depth each subsequent call
                play_MCTS_game_with_expansions(move, depth + 1, depth_limit, sim_info, this_height + 1) #search until depth limit
            else: # reached depth limit
                random_rollout(root)
                if this_height > sim_info.game_tree_height: #keep track of max tree height
                    sim_info.game_tree_height = this_height
                #return here
        #else: return here
    else:#keep searching tree
        move = choose_move(root)
        play_MCTS_game_with_expansions(move, depth, depth_limit, sim_info, this_height + 1)


def random_rollout(node):
    win = random.randint(0, 1)
    if win == 1:
        update_tree_wins(node)
    else:
        update_tree_losses(node)