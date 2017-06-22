#cython: language_level=3, boundscheck=False

from math import  floor #, sqrt
from libc.math cimport sqrt
# from libc.stdlib cimport rand
from libcpp cimport bool
from random import sample, randint
from operator import itemgetter
from tools.utils import move_lookup_by_index
from Breakthrough_Player.board_utils import  enumerate_legal_moves_using_piece_arrays, game_over, move_piece_update_piece_arrays_in_place, copy_game_board
# from time import time
cimport numpy as np


cpdef dict SimulationInfo(file):
        return {'file': file,
        'counter': 0,
        'prev_game_tree_size': 0,
        'game_tree': [],
        'game_tree_height': 0,
        'start_time': None,
        'time_to_think': None,
        'root': None,
        'do_eval': True,
        'main_pid': -1,
        'root_thread_counter': 0,
        'eval_times': []}


cdef extern from "limits.h":
    double INT_MAX
#backpropagate wins
cpdef void update_tree_wins(dict node, int amount=1, gameover=False): #visits and wins go together
    cdef dict parent
    node['wins'] +=amount
    node['visits'] +=amount
    if gameover:
        node['gameover_wins'] +=amount
        node['gameover_visits'] +=amount
    parent = node['parent']
    if parent is not None: #node win => parent loss_init
        update_tree_losses(parent, amount, gameover)

#backpropagate losses
cpdef void  update_tree_losses(dict node, int amount=1, gameover=False): # visit and no win = loss_init
    cdef dict parent
    node['visits'] +=amount
    if gameover:
        node['gameover_visits'] +=amount
    parent = node['parent']
    if parent is not None: #node loss_init => parent win
        update_tree_wins(parent, amount, gameover)

#backpropagate guaranteed win status
cpdef void update_win_statuses(dict node, win_status, new_subtree=False):
    cdef dict parent

    continue_propagating = True


    if continue_propagating:
        # if win_status is True:
        #     update_tree_wins(node, 1, gameover=True)
        if win_status is False : #and node['color'] == 'Black'
            update_tree_losses(node, 1, gameover=True)
        node['win_status'] =win_status
        parent = node['parent']
        if parent is not None:#parent may now know if it is a win or loss.
            update_win_status_from_children(parent, new_subtree)

cpdef void update_win_status_from_children(dict node, new_subtree=False):
    cdef list win_statuses = get_win_statuses_of_children(node)
    set_win_status_from_children(node, win_statuses, new_subtree)

cdef list get_win_statuses_of_children(dict node):
    cdef:
        list win_statuses = []
        list win_statuses_considering = []
        dict child
        float threshold = 1.001


    # if node['color'] == 'White':
    #     threshold = 1.001
    # else:
    #     threshold = 1.001
    children = node['children']
    if children is not None:
        # win_statuses = [child['win_status'] for child in children]
        considering_append = win_statuses_considering.append
        all_append = win_statuses.append
        for child in children:
            if child['UCT_multiplier'] > threshold:
                considering_append(child['win_status'])
            all_append(child['win_status'])
    if False not in win_statuses and len (win_statuses_considering)>0: #if any children are false, node is true. Else, just consider the ones over threshold probability. If they are True => just say we are doomed.
        win_statuses = win_statuses_considering



    return win_statuses

cdef void set_win_status_from_children(dict node, list children_win_statuses, new_subtree=False):
    if False in children_win_statuses: #Fact: if any child is false => parent is true
        update_win_statuses(node, True, new_subtree)  # some kid is a loser, I have some game winning move to choose from
    elif True in children_win_statuses and not False in children_win_statuses and not None in children_win_statuses:
        update_win_statuses(node, False, new_subtree)#all children winners = node is a loss_init no matter what
    elif new_subtree: #appended from opponent move not already in tree
        update_win_statuses(node, None, new_subtree) #maybe this subtree had an inaccurate win_status before, ensure tree is correct. since this is done upon root assignment, won't take time in the middle of search.
       # some kids are winners, some kids are unknown => can't say anything with certainty


cpdef void backpropagate_num_checked_children(dict node):
    while node is not None:
        set_num_checked_children(node)
        if node['num_children_checked'] ==len(node['children']):
           node['subtree_checked'] = True #may have to reupdate parenttrees
           node = node['parent']
        else:
           node['subtree_checked'] =False
           node=None




cdef void set_num_checked_children(dict node): #necessary for gameover kids: wouldn't have reported that they are expanded since another node may also be
    cdef:
        int count = 0
        dict child
    for child in node['children']:
        if child['subtree_checked']:
            count += 1
    node['num_children_checked'] = count


cdef void _crement_threads_checking_node(dict node, int amount):
    cdef:
        dict parent
        int num_children

    if amount == 0:
        node['threads_checking_node'] =0
    else:
        node['threads_checking_node']+=amount
    parent = node['parent']
    if parent is not None:
        update_num_children_being_checked(parent)
        # previous_status = parent['subtree_being_checked']
        #
        # #if no threads checking node (purportedly)
        # if node['threads_checking_node']<=0: #reset or decremented
        #     parent['subtree_being_checked'] = False
        #     if parent['num_children_being_checked'] > 0:
        #         parent['num_children_being_checked'] -= 1
        # #if exactly one thread checking node (purportedly)
        # elif node['threads_checking_node'] ==1:
        #
        #
        #     if amount > 0:#if it was previously not being checked
        #         parent['num_children_being_checked'] += 1
        #     num_children = len(parent['children'])
        #
        #     #should at most be equal to num children being checked
        #     if num_children <= parent['num_children_being_checked']:
        #         parent['num_children_being_checked'] = num_children
        #         parent['subtree_being_checked'] = True
        #     else:#some child(ren) not being checked
        #         parent['subtree_being_checked'] = False
        #
        #
        # #if the flag changed, may have to update the parent
        # if previous_status != parent['subtree_being_checked'] and parent['parent'] is not None:
        #     update_num_children_being_checked(parent['parent'])


cpdef void increment_threads_checking_node(dict node):
    _crement_threads_checking_node(node, 1)#

cpdef void decrement_threads_checking_node(dict node):
    _crement_threads_checking_node(node, -1)

cpdef void reset_threads_checking_node(dict node):
    _crement_threads_checking_node(node, 0)

cpdef void update_num_children_being_checked(dict node):
    cdef:
        int children_being_checked = 0
        dict child
    if node['children'] is not None:
        # children_being_checked = sum([child['threads_checking_node'] > 0 or child['subtree_being_checked'] for child in node['children']])
        for child in node['children']:
            if (child['threads_checking_node'] > 0 and child['children'] is None) or child['subtree_being_checked'] or child['win_status'] is not None:
                children_being_checked+=1

        node['num_children_being_checked'] = children_being_checked
        previous_status = node['subtree_being_checked']
        if children_being_checked ==len(node['children']):
            node['subtree_being_checked'] =True
        else:
            node['subtree_being_checked'] =False

        #if the flag changed, may have to update the parent
        if previous_status != node['subtree_being_checked'] and node['parent'] is not None:
            update_num_children_being_checked(node['parent'])


def choose_UCT_move(node, start_time, time_to_think, sim_info):
    best = None #shouldn't ever return None
    if node['children'] is not None:
        best = find_best_UCT_child(node, start_time, time_to_think, sim_info)
    return best  #because a win for me = a loss_init for child


cpdef choose_UCT_or_best_child(dict node, start_time, int time_to_think, dict sim_info):
    cdef:
        int num_children = len(node['children'])
        dict best_child
        int true_wins
        double prior_prob
        float ttt_multiplier = time_to_think/10


    best = None  # shouldn't ever return None
    if num_children == 1:
        best = node['children'][0]
    else:
        return find_best_UCT_child(node, start_time, time_to_think, sim_info)
        if node['best_child'] is not None:
            best_child = node['best_child']
            true_wins = best_child['gameover_wins']
            prior_prob = best_child['UCT_multiplier'] -1
            over_LCB = prior_prob>0.35
            under_UCB = prior_prob <0.8
            player_node = node['color'] == sim_info['root']['color']

            if best_child['height'] > 20 and over_LCB and player_node:
                threshold_gameover_wins = true_wins > 5000*ttt_multiplier
            else:
                threshold_gameover_wins= true_wins > 1000*ttt_multiplier

            if threshold_gameover_wins and under_UCB:
                if player_node:
                    if prior_prob > .4:
                        meets_winrate_threshold = true_wins  < best_child['gameover_visits'] *.6 #Keep searching best until it drops down to a 40% WR if it is a high-ish probability node
                    else:
                        meets_winrate_threshold = true_wins < best_child['gameover_visits'] *.5#Keep searching best until it drops down to a 50% WR if it is a low probability node
                else:
                    meets_winrate_threshold = true_wins  <  best_child['gameover_visits']*.33  #Keep searching for opponent if best has a 66% WR
                meets_winrate_threshold = meets_winrate_threshold and over_LCB
            else: #keep searching if it is over a very high probability threshold
                meets_winrate_threshold = True

            best_child_can_be_searched = best_child['win_status'] is None and not best_child['subtree_being_checked'] and best_child['threads_checking_node'] <=0
            if best_child_can_be_searched and meets_winrate_threshold :
                best = best_child
            else:
                best = find_best_UCT_child(node, start_time, time_to_think, sim_info)

        elif node['children'] is not None: #if this node has children to choose from: should alwayshappen
            best = find_best_UCT_child(node, start_time, time_to_think, sim_info)
    return best


cdef find_best_UCT_child(dict node, start_time, int time_to_think, dict sim_info):
    cdef:
        int parent_visits = node['visits']
        list viable_children = []
        float first_threshold
        float second_threshold
        double best_val
    best = None
    append = viable_children.append
    if node['win_status'] is not None:

        viable_children = [child for child in node['children']
                           if (child['win_status'] is None and not child['subtree_being_checked'])
                           and (child['threads_checking_node'] <=0 or child['children'] is not None)]
        if len(viable_children) == 0:#search until all subtrees are checked

            viable_children = [child for child in node['children']
                               if not (child['subtree_checked'] or child['subtree_being_checked'])  #DeMorgan's law
                               and (child['threads_checking_node'] <=0 or child['children'] is not None)]
            if len(viable_children) == 0:#search until all subtrees are checked

                viable_children = [child for child in node['children']
                                   if not (child['subtree_checked'] )  #DeMorgan's law
                                   and (child['threads_checking_node'] <=0 or child['children'] is not None)]

    else:#
        is_WaNN = node['color'] == sim_info['root']['color']

        if is_WaNN:
            if node['height']<60:#restrict search if early in the game. rationale: sometimes switches to lower probability moves with high short-term wins
                best_prob = node['children'][0]['UCT_multiplier']-1
            else:
                best_prob = .01

            if best_prob <.1:#TODO Bugged white was doing good (67% @ 1.1 v 0.1 for the if)
                first_threshold = second_threshold = 1.04 #under 4% moves may have good stats but look whacky
            elif best_prob <.2:
                first_threshold = 1.10
                second_threshold = 1.05
            elif best_prob <.3:
                first_threshold = 1.15
                second_threshold = 1.10
            else:#best prob >= 30%
                first_threshold = 1.20
                second_threshold = 1.15
        else:
            first_threshold = 1.1
            second_threshold= 1.01



        viable_children = [child for child in node['children']
                           if (child['win_status'] is None and not child['subtree_being_checked'])
                           and child['UCT_multiplier'] > first_threshold
                           # and (child['gameover_wins']<=1000 or (child['gameover_wins'] * 2 < child['gameover_visits']))
                           and (child['threads_checking_node'] <=0 or child['children'] is not None)]#and not child['subtree_being_checked']
        if len(viable_children) == 0:

            if node is sim_info['root']: #node is sim_info['root'] #isWaNN
                viable_children = [child for child in node['children']
                                   if (child['win_status'] is None and not child['subtree_being_checked'])
                                   and child['UCT_multiplier'] > first_threshold
                                   and (child['threads_checking_node'] <=0 or child['children'] is not None)]

                if len(viable_children) == 0:

                    viable_children = [child for child in node['children']
                                       if (child['win_status'] is None and not child['subtree_being_checked'] )
                                       and child['UCT_multiplier'] > second_threshold
                                       and (child['gameover_wins']<=1000 or (child['gameover_wins'] * 2 < child['gameover_visits']))
                                       and (child['threads_checking_node'] <=0 or child['children'] is not None)]#and not child['subtree_being_checked']


                    if len(viable_children) == 0:

                        viable_children = [child for child in node['children']
                                           if (child['win_status'] is None and not child['subtree_being_checked'] )
                                           and child['UCT_multiplier'] > second_threshold
                                           and (child['threads_checking_node'] <=0 or child['children'] is not None)]#and not child['subtree_being_checked']

                        if len(viable_children) == 0:#TODO: WILL STALL THE TREE SEARCH IF WE DO EOG BATCH EXPANSIONS WITH B = 1 (THREADS BURROW INTO TREE. SINCE ONLY 1 CHILD => SUBTREE BEING CHECKED) since it needs to determine win statuses of lower prob children to determine win status

                            viable_children = [child for child in node['children']
                                               if (child['win_status'] is None and not child['subtree_being_checked'])
                                               and (child['threads_checking_node'] <=0 or child['children'] is not None)]
            else:
                # viable_children = [child for child in node['children']
                #                            if (child['win_status'] is None and not child['subtree_being_checked'] )
                #                            and child['UCT_multiplier'] > second_threshold
                #                            and (child['threads_checking_node'] <=0 or child['children'] is not None)]#and not child['subtree_being_checked']
                if len(viable_children) == 0:
                    viable_children = [child for child in node['children']
                                       if (child['win_status'] is None and not child['subtree_being_checked'])
                                       and (child['threads_checking_node'] <=0 or child['children'] is not None)]

    if len(viable_children)>0:
        best = max( zip(
                    [get_UCT(child, parent_visits, start_time, time_to_think, sim_info) for child in viable_children],
                     viable_children),
               key=itemgetter(0))[1]

    return best


cpdef double get_UCT(dict node, int parent_visits, float start_time, float time_to_think, dict sim_info):




    # # 03/11/2017 games actually good with this and no division, but annealing seems right?
    # stochasticity_for_multithreading = random.random()  #/2.4 # keep between [0,0.417] => [1, 1.417]; 1.414 ~ √2
    # stoch_for_NN_weighting = 1- (stochasticity_for_multithreading/4)#since this can get so huge in the beginning,
    # maybe bouncing it around will be good so threads don't knock into each other?
    # # #TODO test this. SEEMS like a good idea since it will explore early and exploit later
    # annealed_factor = 1 - (
    #     (start_time - time()) / time_to_think)  # starts at 1 and trends towards 0 as search proceeds
    # # with stochasticity multiplier between .5 and 1 so multithreading sees different values
    # stochasticity_for_multithreading = (1 - random.random() / 2) * max(0.25,
    #                                                                    annealed_factor)  # in printed stats games, probably never got to a low value

    # stochasticity_for_multithreading = (random.random()*4) * max(0.25, annealed_factor) #in printed stats games, probably never got to a low value

    # exploration_constant = 0.1 + stochasticity_for_multithreading # 1.414 ~ √2
    # exploration_constant =  stochasticity_for_multithreading # 1.414 ~ √2
    # if node['parent']['color'] ==sim_info.root['color']:#make black's moveexploremore?
    #     PUCT_exploration_constant = stochasticity_for_multithreading*1000#1000
    # else:
    # cdef double random_number = rand()/INT_MAX
    cdef:
        double PUCT_exploration_constant = 1#(time_to_think/10)**2#*random_number#3*annealed_factor# stochasticity_for_multithreading
    
    
        # norm_wins, norm_visits,true_wins, true_losses = transform_wrt_overwhelming_amount(node, overwhelming_on=False)
        int norm_wins = node['wins']
        int norm_visits = node['visits']
        int true_wins = node['gameover_wins']
    
        int total_gameovers = node['gameover_visits']
        int true_losses = total_gameovers-true_wins
    
        #TODO: problem: exploring lower ranked children when their gameover ratio is favorable. Stops exploring higher ranked children even though their long term ratios may improve and get higher UCT vals
        double NN_prob = max(0.10,(node['UCT_multiplier'] -1))
        double prior_prob_weighting
        double norm_loss_rate
        int norm_losses
        cdef int sibling_visits
        cdef double PUCT_exploration
        cdef double exploitation_factor

    if total_gameovers > 10 and node['color'] != sim_info['root']['color']: #opponent may explore more based on eval results whereas we use EOG values (but maybe that approximates wanderer's eval better anyway?)
        prior_prob_weighting = 1#NN_prob/1#.80 => .08 TODO should I? this may defeat the search a bit.
        norm_loss_rate = (true_losses/ total_gameovers)+ prior_prob_weighting
    else:
        norm_losses = (norm_visits-norm_wins)
        norm_loss_rate = norm_losses/norm_visits

    # PUCT_exploration = PUCT_exploration_constant*NN_prob*(sqrt(max(1, parent_visits-norm_visits))/(1+norm_visits)) #interchange norm visits with node visits
    sibling_visits = max(1, parent_visits-norm_visits)
    PUCT_exploration = calculate_PUCT(PUCT_exploration_constant, NN_prob, sibling_visits, norm_visits)

    exploitation_factor = norm_loss_rate# losses / visits of child = wins / visits for parent

    # cdef double PUCT = (exploitation_factor + PUCT_exploration)
    return exploitation_factor + PUCT_exploration # PUCT from parent's point of view

cdef double calculate_PUCT(double c, double NN_prob, int sibling_visits, int visits):
    return c*NN_prob*(sqrt(sibling_visits)/(1+visits))

def randomly_choose_a_winning_move(node): #for stochasticity: choose among equally successful children
    best_nodes = []
    win_can_be_forced = False
    if node['children'] is not None:
        win_can_be_forced, best_nodes = check_for_forced_win(node['children'])
        if not win_can_be_forced:

            best = choose_best_true_loser(node['children'])
            # best = most_visited(node['children'])
            if best is not None:
               best_nodes.append(best)


            if len(best_nodes) == 0:
                best_nodes = get_best_children(node['children'])
    # if len(best_nodes) == 0:
    #     breakpoint = True
    return best_nodes[0]  # because a win for me = a loss for child

def most_visited(node_children):
    return  max( zip([child['visits'] for child in node_children],
                    node_children),
                 key=itemgetter(0))[1]

def choose_best_true_loser(node_children, second_time=False):
    best_losses = 0
    best_wins = 0
    best_rate = 0
    best_total = 0
    best = None
    filtered_children  = []
    prob_scaling = 3
    best_child = node_children[0]
    action_value_threshold = best_child['gameover_wins']/2#150000
    best_gameover_wins = best_child['gameover_wins']
    top_rank_prob = best_child['UCT_multiplier']

    if top_rank_prob<1.1: #really only happens on the first couple of moves
        first_threshold = second_threshold = 1.04 #under 4% moves may have good stats but look whacky
    elif top_rank_prob <1.2:
        first_threshold = 1.10
        second_threshold = 1.05
    elif top_rank_prob <1.3:
        first_threshold = 1.15
        second_threshold = 1.10
    else:
        first_threshold = 1.20
        second_threshold = 1.15

    # if best_child['height']>60:
    #     first_threshold = 1.10
    #     second_threshold = 1.05
    # if best_gameover_wins >10000 and best_gameover_wins /best_child['gameover_visits']>.80:
    #     first_threshold = second_threshold = 1.04 #under 4% moves may have good stats but look whacky


    #non-losing moves over probability threshold or action-value threshold
    filtered_children = [child for child in node_children if ((child['UCT_multiplier'] > first_threshold or child['gameover_visits']-child['gameover_wins'] > action_value_threshold) and child['win_status'] is not True)]


    if len(filtered_children) ==0:
        filtered_children = [child for child in node_children if ((child['UCT_multiplier'] > second_threshold or child['gameover_visits']-child['gameover_wins'] > action_value_threshold) and child['win_status'] is not True)]

    if len(filtered_children) ==0:
        filtered_children = [child for child in node_children if child['win_status'] is not True]


    if len(filtered_children) > 0:

        best_losses, best = max( zip(
                    [child['gameover_visits']-child['gameover_wins'] for child in filtered_children],
                    filtered_children),
                 key=itemgetter(0))#TODO move this to outside to get max over all children vs just filtered?
        highest_losses = best_losses
        best_rate = best_losses/max(1, best['gameover_visits']) + (best['UCT_multiplier']-1)/prob_scaling


        for child in filtered_children: #sorted by prob
            _, child_visits_norm,true_wins, true_losses = transform_wrt_overwhelming_amount(child, overwhelming_on=False)
            prior_prob_weighting = (child['UCT_multiplier']-1)/prob_scaling#.80 =>.
            total = true_losses + true_wins
            loss_rate = (true_losses/max(1, total)) + prior_prob_weighting

            if loss_rate> best_rate and loss_rate-best_rate>0.05 and true_losses/highest_losses>.30 and child['win_status'] is not True:#  and child['UCT_multiplier'] > prob_threshold
                best = child
                best_rate = loss_rate
    else:
        best = node_children[0]
    return best

def check_for_forced_win(node_children):
    forced_win = False
    guaranteed_children = []
    best_guaranteed_child = None

    for child in node_children:
        if child['gameover']:
            guaranteed_children.append(child)

    if len(guaranteed_children) ==0:

        for child in node_children:
            if child['win_status'] is False:
                guaranteed_children.append(child)
                forced_win = True

        if len(guaranteed_children) > 0: #TODO: does it really matter which losing child is the best if all are losers?
            # guaranteed_children = get_best_children(guaranteed_children, game_num)
            best_guaranteed_child = guaranteed_children[0]#choose_best_true_loser(guaranteed_children)
            if best_guaranteed_child is not None:
                guaranteed_children = [best_guaranteed_child]
    else:
        forced_win = True
    return forced_win, guaranteed_children

def transform_wrt_overwhelming_amount(child, overwhelming_on=False):
    overwhelming_amount = child['overwhelming_amount']
    true_wins = 0
    true_losses = 0

    if child['wins'] >= overwhelming_amount and overwhelming_on:
        true_wins = floor(child['wins'] /overwhelming_amount)
        child_wins_norm = true_wins + (child['wins'] %overwhelming_amount)
    else:
        child_wins_norm = child['wins']
        true_wins = child['gameover_wins']
    if child['visits'] >= overwhelming_amount and overwhelming_on:
        gameover_visits = floor(child['visits'] /overwhelming_amount)
        child_visits_norm = gameover_visits+ (child['visits'] %overwhelming_amount)
        true_losses = gameover_visits - true_wins
    else:
        child_visits_norm = child['visits']
        true_losses = (child['gameover_visits'] -child['gameover_wins'])
    return child_wins_norm, child_visits_norm, true_wins, true_losses


def get_best_children(node_children):#TODO: make sure to not pick winning children?
    best, best_val = get_best_child(node_children)
    # best, best_val = get_best_most_visited_child(node_children)
    best_nodes = []
    NN_scaling_factor = 0#1
    overwhelming_amount = best['overwhelming_amount']

    for child in node_children:  # find equally best children
       # NN_weighting = (NN_scaling_factor*(child['UCT_multiplier'] -1))/(1+child['parent']['visits'])
        NN_weighting = (NN_scaling_factor*(child['UCT_multiplier'] -1))+1

        child_wins_norm, child_visits_norm, true_wins, true_losses = transform_wrt_overwhelming_amount(child, overwhelming_on=False)
        total_gameovers = true_losses+true_wins
        if total_gameovers > 0:
            child_loss_rate = true_losses / total_gameovers
        else:
            child_loss_rate = (child_visits_norm - child_wins_norm) / child_visits_norm


        if not child['win_status'] == True: #only consider children who will not lead to a win foropponent
            if child['visits'] >0:
                child_NN_scaled_loss_rate = child_loss_rate * NN_weighting

                if child_NN_scaled_loss_rate == best_val:
                    if best['visits'] == child['visits']:
                        best_nodes.append(child)
                        # should now have list of equally best children
    if len(best_nodes) == 0: #all children are winners => checkmated, just make the best move you have
        best_nodes.append(best)
    return best_nodes

def get_best_child(node_children, non_doomed = True):
    NN_scaling_factor = 0#1#+(game_num/100)
    k = 0
    if non_doomed:
        while k < len(node_children) and (node_children[k]['visits'] <= 0 or node_children[k]['win_status'] is True) :
            k += 1
    else:
        while k < len(node_children) and node_children[k]['visits'] <= 0:
            k += 1
    if k < len(node_children):
        best = node_children[k]
        overwhelming_amount = best['overwhelming_amount']

        NN_weighting = (NN_scaling_factor*(best['UCT_multiplier'] - 1)) + 1

        best_wins_norm, best_visits_norm, best_true_wins, best_true_losses = transform_wrt_overwhelming_amount(best, overwhelming_on=False)
        best_total_gameovers = best_true_losses + best_true_wins

        if best_total_gameovers>0:
            if best is best['parent']['best_child']:
                alternate_value = NN_weighting/10
            else:
                alternate_value = 0
            best_loss_rate = max(alternate_value, (best_true_losses / best_total_gameovers))
        else:
            best_loss_rate = ((best_visits_norm - best_wins_norm) / best_visits_norm)

        # NN_weighting = (NN_scaling_factor*(node_children[k]['UCT_multiplier'] - 1)) / (1+node_children[k]['parent']['visits'])


        best_val_NN_scaled = best_loss_rate * NN_weighting

        # probability = node_children[k]['UCT_multiplier']-1
        # best_val_NN_scaled = (probability+ best_loss_rate)/(1+probability)

        for i in range(k, len(node_children)):  # find best child
            child = node_children[i]

            # NN_weighting = (NN_scaling_factor*(child['UCT_multiplier'] -1))/(1+child['parent']['visits'])
            NN_weighting = (NN_scaling_factor*(child['UCT_multiplier'] -1))+1

            if non_doomed:
                predicate = child['visits'] > 0 and not child['win_status'] is True
            else:
                predicate = child['visits'] >0
            if predicate: #only consider non-doomed moves

                child_wins_norm, child_visits_norm,true_wins, true_losses = transform_wrt_overwhelming_amount(child, overwhelming_on=False)
                total_gameovers = true_losses + true_wins
                if total_gameovers > 0:
                    if child is child['parent']['best_child']:
                        alternate_value = NN_weighting/10
                    else:
                        alternate_value = 0
                    child_loss_rate = max (alternate_value, (true_losses / total_gameovers))
                else:
                    child_loss_rate = (child_visits_norm - child_wins_norm) / child_visits_norm

                child_NN_scaled_loss_rate = child_loss_rate * NN_weighting

                # probability = (child['UCT_multiplier']-1)
                # child_NN_scaled_loss_rate = (child_loss_rate + probability) / (1+probability)
                #and (child['visits'] / best['visits'] > 0.3) or best_loss_rate <.5)
                if child_NN_scaled_loss_rate > best_val_NN_scaled:  # get the child with the highest loss_init rate
                    if (best['visits'] >= overwhelming_amount and child['visits'] >= overwhelming_amount) or\
                        (best['visits'] < overwhelming_amount and child['visits'] >= overwhelming_amount) or\
                        (best['visits'] < overwhelming_amount and child['visits'] < overwhelming_amount) or\
                        ((best_loss_rate) < .30):
                        # if both have searched to game overs, (best case # 1)
                        # if neither have searched to game overs,
                        # if new child has searched to a gameover and current best hasn't (best case # 2)
                        # if new child hasn't searched to a gameover, best has, but new child has a better loss_init-rate (i.e. current best has a lot of losses, new child looks better)
                        best = child
                        best_val_NN_scaled = child_NN_scaled_loss_rate
                        best_loss_rate = child_loss_rate


                elif child_NN_scaled_loss_rate == best_val_NN_scaled:  # if both have equal win rate (i.e. both 0/visits),

                    if best_val_NN_scaled == 0: #if both are winners insofar as we know, pick the one with the least visits? (since we're more sure we're doomed with the one visited a lot)
                        if best['visits'] > child['visits']:
                            best = child
                            best_val_NN_scaled = child_NN_scaled_loss_rate
                            best_loss_rate = child_loss_rate
                    else: # get the one with the most visits
                        if best['visits'] < child['visits']:
                            best = child
                            best_val_NN_scaled = child_NN_scaled_loss_rate
                            best_loss_rate = child_loss_rate

    else:
        if non_doomed:
            # no children with value or all doomed
            best, best_val_NN_scaled = get_best_child(node_children, non_doomed=False)
        else:
            # no children with value? happens if search is too slow
            best = node_children[0]
            best_val_NN_scaled = 0
    return best, best_val_NN_scaled


def random_rollout(node):
    move = node
    while move['children'] is not None:
        move = sample(move['children'], 1)[0]
    outcome, result = evaluation_function(move)
    if outcome == 0:
        update_tree_losses(move, 1)
    else:
        update_tree_wins(move, 1)
    return result

def real_random_rollout(node, end_of_game=True):
    local_board = copy_game_board(node['game_board'])
    current_color = player_color = node['color']
    local_white_pieces = node['white_pieces'][:]
    local_black_pieces = node['black_pieces'][:]
    gameover, winner_color = game_over(local_board)
    if end_of_game:
        depth_limit = 999
    else:
        depth_limit = 4
    depth = -1
    while not gameover and depth < depth_limit:
        depth +=1
        if current_color =='White':
            player_pieces = local_white_pieces
            opponent_pieces = local_black_pieces
        else:
            player_pieces = local_black_pieces
            opponent_pieces = local_white_pieces

        moves = enumerate_legal_moves_using_piece_arrays(current_color, local_board, player_pieces)
        random_move = sample(moves, 1)[0]
        random_move =  random_move['From'] + r'-' + random_move['To']

        local_board, player_piece_to_add, player_piece_to_remove, remove_opponent_piece = move_piece_update_piece_arrays_in_place(local_board, random_move, current_color)
        update_piece_arrays_in_place(player_pieces, opponent_pieces, player_piece_to_add, player_piece_to_remove, remove_opponent_piece)

        gameover, winner_color = game_over(local_board)
        current_color = get_opponent_color(current_color)
    if gameover:
        if winner_color == player_color:
            outcome =  1
        else:
            outcome =  0
    else:#reached depth limit
        outcome, _ = evaluation_function_nodeless(local_white_pieces, local_black_pieces, player_color)
    return outcome

cpdef str get_opponent_color(str player_color):
    cdef str opponent_color
    if player_color == 'White':
        opponent_color = 'Black'
    else:
        opponent_color = 'White'
    return opponent_color

def update_piece_arrays_in_place(player_pieces, opponent_pieces, player_piece_to_add, player_piece_to_remove, remove_opponent_piece):
    player_pieces[player_pieces.index(player_piece_to_remove)] = player_piece_to_add
    if remove_opponent_piece:
        opponent_pieces.remove(player_piece_to_add)

def rollout(node_to_update, descendant_to_eval):
    outcome, _ = evaluation_function(descendant_to_eval)
    if outcome == 1:
        if node_to_update['color'] == descendant_to_eval['color']: #future board where I am mover
            update_tree_wins(node_to_update)
        else:
            update_tree_losses(node_to_update)
    elif outcome == 0:#loss_init
        if node_to_update['color'] == descendant_to_eval['color']: #future board where I am mover
            update_tree_losses(node_to_update)
        else:
            update_tree_wins(node_to_update)

cdef evaluation_function(dict root):
    cdef:
        dict weighted_board = {  #50%
            8: {'a': 24, 'b': 24, 'c': 24, 'd': 24, 'e': 24, 'f': 24, 'g': 24, 'h': 24},
            7: {'a': 21, 'b': 23, 'c': 23, 'd': 23, 'e': 23, 'f': 23, 'g': 23, 'h': 21},
            6: {'a': 14, 'b': 22, 'c': 22, 'd': 22, 'e': 22, 'f': 22, 'g': 22, 'h': 14},
            5: {'a': 9, 'b': 15, 'c': 21, 'd': 21, 'e': 21, 'f': 21, 'g': 15, 'h': 9},
            4: {'a': 6, 'b': 9, 'c': 16, 'd': 16, 'e': 16, 'f': 16, 'g': 9, 'h': 6},
            3: {'a': 3, 'b': 5, 'c': 10, 'd': 10, 'e': 10, 'f': 10, 'g': 5, 'h': 3},
            2: {'a': 2, 'b': 3, 'c': 3, 'd': 3, 'e': 3, 'f': 3, 'g': 3, 'h': 2},
            1: {'a': 5, 'b': 28, 'c': 28, 'd': 12, 'e': 12, 'f': 28, 'g': 28, 'h': 5}
        }
        str white = 'w'
        str black = 'b'
        int is_white_index = 9
        int white_move_index = 10
        str player_color = root['color']

        int result = 0
        list white_pieces = root['white_pieces']
        list black_pieces = root['black_pieces']
        int row
        str col
        int outcome

    for piece in white_pieces:
        col = piece[0]
        row = int(piece[1])
        result += weighted_board[row][col]
    for piece in black_pieces:
        col = piece[0]
        row = int(piece[1])
        result -= weighted_board[row][col]

    if result > 0:
        if player_color == 'White':
            outcome = 1
        else:
            outcome =  0

    else:
        if player_color == 'Black':
            outcome = 1

        else:
            outcome = 0
    return outcome

def evaluation_function_nodeless(white_pieces, black_pieces, player_color):
    # row 2:
    # if white piece or (?) no diagonal black pieces in row 3
    weighted_board = {  #50%
        8: {'a': 24, 'b': 24, 'c': 24, 'd': 24, 'e': 24, 'f': 24, 'g': 24, 'h': 24},
        7: {'a': 21, 'b': 23, 'c': 23, 'd': 23, 'e': 23, 'f': 23, 'g': 23, 'h': 21},
        6: {'a': 14, 'b': 22, 'c': 22, 'd': 22, 'e': 22, 'f': 22, 'g': 22, 'h': 14},
        5: {'a': 9, 'b': 15, 'c': 21, 'd': 21, 'e': 21, 'f': 21, 'g': 15, 'h': 9},
        4: {'a': 6, 'b': 9, 'c': 16, 'd': 16, 'e': 16, 'f': 16, 'g': 9, 'h': 6},
        3: {'a': 3, 'b': 5, 'c': 10, 'd': 10, 'e': 10, 'f': 10, 'g': 5, 'h': 3},
        2: {'a': 2, 'b': 3, 'c': 3, 'd': 3, 'e': 3, 'f': 3, 'g': 3, 'h': 2},
        1: {'a': 5, 'b': 28, 'c': 28, 'd': 12, 'e': 12, 'f': 28, 'g': 28, 'h': 5}
    }
    white = 'w'
    black = 'b'
    is_white_index = 9
    white_move_index = 10
    # opponent_dict = { 'White': 'Black',
    #          'Black': 'White'
    #
    # }
    win_weight = 0
    result = 0

    for piece in white_pieces:
        col = piece[0]
        row = int(piece[1])
        result += weighted_board[row][col]
    for piece in black_pieces:
        col = piece[0]
        row = int(piece[1])
        result -= weighted_board[row][col]

    if result > 0:
        if player_color == 'White':
            outcome = 1
        else:
            outcome =  0

    else:
        if player_color == 'Black':
            outcome = 1

        else:
            outcome = 0
    return outcome, result

cpdef void update_child(dict child, np.ndarray NN_output, dict sim_info, do_eval=True):
    cdef:
        int child_index = child['index']
        str child_color = child['color']
        dict parent = child['parent']
        int game_over_row
        int caution_row
        int maybe_caution_row
        str enemy_piece
        # str move
        dict previous_game_board
        double child_val
        int normalized_value
        int prior_value_multiplier
        int weighted_losses
        int weighted_wins


    do_update = True

    # if 0< child_index < 21 and parent is not None:#home rows that can capture; check for game-saving moves
    #     if child_color =='White':#parent was black
    #         game_over_row = 7
    #         caution_row = 6
    #         maybe_caution_row = 5
    #         enemy_piece = 'w'
    #     else:
    #         game_over_row = 2
    #         caution_row = 3
    #         maybe_caution_row = 4
    #         enemy_piece = 'b'
    #
    #     move = move_lookup_by_index(child_index, get_opponent_color(child_color))
    #     previous_game_board = parent['game_board']
    #     game_saving_move = enemy_piece == previous_game_board[int(move[4])][move[3]]
    #
    #
    #     if game_saving_move: #
    #         child['gameover_visits'] = 1000
    #         child['gameover_wins'] = 0
    #         child['visits'] = 65536
    #         child['wins'] = 0
    #         child['UCT_multiplier'] = 1+NN_output[child_index] #might mess up search if I don't do this?
    #         child['game_saving_move'] = True
    #         do_update = False
    #     # elif (2 <= child_index <= 7 or 14<= child_index <=19): #if gameover next move and it wasn't a game saving move, it will be marked a loser so we don't need to consider that case anymore
    #     #     # gameover_next_move = enemy_piece in previous_game_board[game_over_row].values()
    #     #     maybe_in_danger = enemy_piece in previous_game_board[caution_row].values()
    #     #     kinda_sorta_maybe_in_danger = enemy_piece in previous_game_board[maybe_caution_row].values()
    #     #     if not maybe_in_danger and not kinda_sorta_maybe_in_danger and child['height'] < 40: #child_color == 'Black' and and child['height'] < 60 not gameover_next_move and
    #     #         child['gameover_visits'] = 9000
    #     #         child['gameover_wins'] = 9000
    #     #         child['visits'] = 65536
    #     #         child['wins'] = 65536
    #     #         child['UCT_multiplier'] = 1.0001 #might mess up search if I don't do this?
    #     #         do_update = False

    if 23 <=child_index <=42 and parent is not None: #Row guarding home that can capture

        if child_color =='White':#parent was black
            enemy_piece = 'w'
        else:
            enemy_piece = 'b'
        move = move_lookup_by_index(child_index, get_opponent_color(child_color))

        previous_game_board = parent['game_board']
        capture_close_to_home = enemy_piece == previous_game_board[int(move[4])][move[3]]
        if capture_close_to_home: #values are ad-hoc, not sure how important this is every time.
            child['gameover_visits'] = 100
            child['gameover_wins'] = 0
            child['visits'] = 100
            child['wins'] = 0
            child['UCT_multiplier'] = 1+NN_output[child_index]# 1+max(NN_output[child_index], 0.60) #might mess up search if I don't do this?
            do_update = False

    if do_update:
        child_val = NN_output[child_index]


        child['UCT_multiplier'] = 1 + (child_val)  # stay close to policy (net) trajectory by biasing UCT selection of
        # NN's top picks as a function of probability returned by NN

        if not child['gameover']:
            normalized_value = int(child_val * 100)

            # TODO: 03/15/2017 removed based on prof Lorentz input
            # if child['color'] =='Black':
            #     child['UCT_multiplier']  = 1 + (child_val*2)# stay close to policy (net) trajectory by biasing UCT selection of
            #                                      # NN's top picks as a function of probability returned by NN
            # else:
            #     child['UCT_multiplier'] = 1 + (child_val)  # stay close to policy (net) trajectory by biasing UCT selection of
            #     # NN's top picks as a function of probability returned by NN



            # TODO: 03/11/2017 7:45 AM added this to do simulated random rollouts instead of assuming all losses
            # i.e. if policy chooses child with 30% probability => 30/100 games
            # => randomly decide which of those 30 games are wins and which are losses

            # TODO: 03/15/2017 based on Prof Lorentz input, wins/visits = NNprob/100
            prior_value_multiplier = 1
            weighted_losses = normalized_value * prior_value_multiplier
            child['visits'] = 100 *prior_value_multiplier
            weighted_wins = ((child['visits'] - weighted_losses) ) #/ child['visits']  # may be a negative number if we are increasing probability **2
            child['wins'] =  min(weighted_wins, child['visits']*.75)#to prevent it from initializing it with a 0 win rate for really low probabilitynodes
            if weighted_wins < child['visits']*.70:
                child['gameover_visits'] =child['visits']
                child['gameover_wins'] =weighted_wins

            if sim_info['do_eval'] and do_eval:
                eval_child(child)

def random_to_depth_rollout(parent_to_update, depth=0): #bad because will not find every depth d child
    descendant_to_eval = parent_to_update #this turns into child to rollout
    descendant_exists = True
    while depth > 0 and descendant_exists:
        if descendant_to_eval['children'] is not None:
            candidates = list(filter(lambda x: x.depth == depth, descendant_to_eval['children']))
            descendant_to_eval = sample(candidates, 1)[0]  # more MCTS-y.
            depth -=1
        else:
            descendant_exists = False
    if depth == 0 :
        rollout(parent_to_update, descendant_to_eval)

 #TODO: better to keep track of subtree height so we can randomly rollout which of the subtrees we know has depth d?

def rollout_and_eval_if_parent_at_depth(descendant_to_eval, depth = 0): #bad because will find every depth d child
    parent_to_update = descendant_to_eval #this turns into parent to update
    parent_exists = True
    while depth > 0 and parent_exists:
        if parent_to_update['parent'] is not None:
            parent_to_update = parent_to_update['parent']
            depth -=1
        else:
            parent_exists = False
    if depth == 0 :
        rollout(parent_to_update, descendant_to_eval)

def eval_child(child):
    if child['height'] >400:
        outcome = real_random_rollout(child, end_of_game=True)
    else:
        outcome = evaluation_function(child)
    if outcome == 1:
        update_tree_losses(child['parent'])
    else:
        update_tree_wins(child['parent'])

cpdef void eval_children(list children):
    cdef:
        int parent_wins = 0
        int parent_losses = 0
        dict parent = children[0]['parent']
        int outcome

    for child in children:
        outcome = evaluation_function(child)
        if outcome == 1:
            parent_losses +=1
        else:
            parent_wins +=1
    update_tree_losses(parent, parent_losses)
    update_tree_wins(parent, parent_wins)

