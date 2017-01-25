import re as re  # regular expressions
import os
import fnmatch  # to retrieve file information from path
import pickle  # serialize the data structure
import mmap  # read entire files into memory for (only for workstation)
import copy
import math
import warnings
from Tools import utils

import pandas as pd
from multiprocessing import Pool, freeze_support

def process_directory_of_breakthrough_files(path):
    player_list = []
    arg_lists = [[path, self_play_games] for self_play_games in utils.find_files(path, '*.txt')]
    freeze_support()
    process_pool = Pool(processes=len(arg_lists))
    player_list.extend(process_pool.starmap(process_breakthrough_file, arg_lists))
    process_pool.close()
    process_pool.join()
    return player_list

def process_breakthrough_file(path, self_play_games):
    file_name = self_play_games[len(path):- len('.txt')]  # trim path & extension
    file_name = file_name.split('_SelfPlayLog')  # \BreakthroughN_SelfPlayLog00-> ['\BreakthroughN',00]
    server_name = str(file_name[0].strip('\\'))  # '\BreakthroughN' -> 'BreakthroughN'
    self_play_log = str(file_name[1]).strip(
                        '(').strip(
                        ')')  # if file_name was originally \BreakthroughN_SelfPlayLog(xy)
    date_range = str(self_play_games[len(r'G:\TruncatedLogs')
                                     + 1:len(path)
                                     - len(r'\selfPlayLogsBreakthroughN')])
    games_list, white_wins, black_wins = format_game_list(self_play_games)
    return {'ServerNode': server_name, 'Self-PlayLog': self_play_log, 'DateRange': date_range, 'Games': games_list,
            'WhiteWins': white_wins, 'BlackWins': black_wins}


def write_to_disk(data_to_write, path):
    write_directory = 'G:\TruncatedLogs\PythonDataSets\DataStructures'
    date = str(path[len(r'G:\TruncatedLogs') + 1
                    :- len(r'\selfPlayLogsBreakthroughN')])
    server_name = path[- len(r'\selfPlayLogsBreakthroughN') + 1
                       :]
    output_file = open(os.path.join(write_directory, (
                         date  # prepend data to name of server
                       + server_name
                       + r'DataPython.p')), 'wb')  # append data qualifier
    pickle.dump(data_to_write, output_file, protocol=pickle.HIGHEST_PROTOCOL)

def format_game_list(self_play_games):
    games = []
    black_win = None
    white_win = None
    end_regex = re.compile(r'.* End')
    move_regex = re.compile(r'\*play (.*)')
    black_win_regex = re.compile(r'Black Wins:.*')
    white_win_regex = re.compile(r'White Wins:.*')
    num_white_wins = 0
    num_black_wins = 0
    file = open(self_play_games, "r+b")  # read in file
    file = mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)  # prot = PROT_READ only in Unix
    # iterate over list of the form:
    # Game N Start
    # ...
    # (Black|White) Wins: \d
    # [Game N End]
    unformatted_move_list = []
    while True:
        line = file.readline().decode('utf-8')  # convert to string
        if line == '':
            break  # EOF
        if move_regex.match(line):  # put plays into move list
            unformatted_move_list.append(move_regex.search(line).group(1))
        elif black_win_regex.match(line):
            black_win = True
            white_win = False
        elif white_win_regex.match(line):
            white_win = True
            black_win = False
        elif end_regex.match(line):
            # Format move list
            move_list, mirror_move_list, original_visualizer_link, mirror_visualizer_link \
                = format_move_lists_and_links(unformatted_move_list)
            white_board_states = generate_board_states(move_list, 'White', white_win)
            white_mirror_board_states = generate_board_states(mirror_move_list, 'White', white_win)
            # self-play => same states, but win under policy for W <=> lose under policy for B
            black_board_states = generate_board_states(move_list, 'Black', black_win)
            black_mirror_board_states = generate_board_states(mirror_move_list, 'Black', black_win)
            if white_win:
                num_white_wins += 1
            elif black_win:
                num_black_wins += 1
            games.append({'Win': white_win,
                          'Moves': move_list,
                          'MirrorMoves': mirror_move_list,
                          'BoardStates': white_board_states,
                          'MirrorBoardStates': white_mirror_board_states,
                          'OriginalVisualizationURL': original_visualizer_link,
                          'MirrorVisualizationURL': mirror_visualizer_link}
                         )  # append new white game
            games.append({'Win': black_win,
                          'Moves': move_list,
                          'MirrorMoves': mirror_move_list,
                          'BoardStates': black_board_states,
                          'MirrorBoardStates': black_mirror_board_states,
                          'OriginalVisualizationURL': original_visualizer_link,
                          'MirrorVisualizationURL': mirror_visualizer_link}
                         )  # append new black game
            unformatted_move_list = []  # reset move_list for next game
            white_win = None  # not necessary; redundant, but good practice
            black_win = None
    file.close()
    return games, num_white_wins, num_black_wins


def initial_state(move_list, player_color, win):
    empty = 'e'
    white = 'w'
    black = 'b'
    if player_color == 'White':
        is_white = 1
    else:
        is_white = 0
    return [
        {
            10: -1,  # (-1 for initial state, 0 if black achieved state, 1 if white achieved state)
            9: is_white,  # is player_color white
            8: {'a': black, 'b': black, 'c': black, 'd': black, 'e': black, 'f': black, 'g': black, 'h': black},
            7: {'a': black, 'b': black, 'c': black, 'd': black, 'e': black, 'f': black, 'g': black, 'h': black},
            6: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
            5: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
            4: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
            3: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
            2: {'a': white, 'b': white, 'c': white, 'd': white, 'e': white, 'f': white, 'g': white, 'h': white},
            1: {'a': white, 'b': white, 'c': white, 'd': white, 'e': white, 'f': white, 'g': white, 'h': white}
        },
        win,
        generate_transition_vector(move_list[0]['White']['To'], move_list[0]['White']['From'],
                                   'White')]  # White's opening move


def generate_board_states(move_list, player_color, win):
    state = initial_state(move_list, player_color, win)
    if player_color == 'White':
        player_POV = [state]
    else:
        player_POV = []
    board_states = {'Win': win, 'States': [state], 'PlayerPOV': player_POV}
    for i in range(0, len(move_list)):
        assert (move_list[i]['#'] == i + 1)
        # structure kept for symmetry
        if isinstance(move_list[i]['White'], dict):  # for self-play, this should always happen.
            if isinstance(move_list[i]['Black'], dict):  # if no black move => white won
                black_transition_vector = generate_transition_vector(move_list[i]['Black']['To'],
                                                                     move_list[i]['Black']['From'], 'Black')
                # can't put black move block in here as it would execute before white's move
            else:
                black_transition_vector = [0] * 154 + [1]
            state = [
                move_piece(state[0], move_list[i]['White']['To'], move_list[i]['White']['From'], whose_move='White'),
                win,
                black_transition_vector]  # Black's response to the generated state
            if player_color == 'Black':  # reflect positions tied to black transitions
                board_states['PlayerPOV'].append(reflect_board_state(state))
            board_states['States'].append(state)
        if isinstance(move_list[i]['Black'], dict):  # if string, then == resign or NIL
            if i + 1 == len(move_list):  # if no next white move => black won
                white_transition_vector = [0] * 154 + [1]  # no white move from the next generated state
            else:
                white_transition_vector = generate_transition_vector(move_list[i + 1]['White']['To'],
                                                                   move_list[i + 1]['White']['From'], 'White')
            state = [
                move_piece(state[0], move_list[i]['Black']['To'], move_list[i]['Black']['From'], whose_move='Black'),
                win,
                white_transition_vector]  # White's response to the generated state
            board_states['States'].append(state)
            if player_color == 'White':
                board_states['PlayerPOV'].append(state)
    # for data transformation; inefficient to essentially compute board states twice, but more error-proof
    board_states = convert_board_states_to_arrays(board_states, player_color)
    return board_states


def generate_transition_vector(to, _from, player_color):
    # probability distribution over the 154 possible (vs legal) moves from the POV of the player.
    # Reasoning: six center columns where, if a piece was present, it could move one of three ways.
    # A piece in one of the two side columns can move one of two ways.
    # Since nothing can move in the farthest row, there are only seven rows of possible movement.
    # => (2*2*7) + (6*3*7) = 154
    # ==> 154 element vector of all 0s sans the 1 for the transition that was actually made.
    # i.e. a1-a2 (if White) == h8-h7 (if Black) =>
    # row 0 (closest row), column 0(farthest left)
    # moves to
    # row +1, column 0
    # <=> transition[0] = 1, transition[1:len(transition)] = 0

    # Notes: when calling NN, just reverse board state if black and decode output with black's table
    #1/14/17: since NN sees sorted r x c, we actually need to reverse if white..

    from_column = _from[0]
    to_column = to[0]
    from_row = int(_from[1])
    to_row = int(to[1])
    # ex if white and from_column is b => 1*3; moves starting from b are [2] or [3] or [4];
    column_offset = (ord(from_column) - ord('a')) * 3  
    if player_color == 'Black':
        row_offset = (to_row - 1) * 22  # 22 possible moves per row
        assert (row_offset == (from_row - 2) * 22)  # double check
        index = 153 - (
        ord(to_column) - ord(from_column) + column_offset + row_offset)  # 153 reverses the board for black
    else:
        row_offset = (from_row - 1) * 22  # 22 possible moves per row
        assert (row_offset == (to_row - 2) * 22)  # double check
        index = ord(to_column) - ord(from_column) + column_offset + row_offset
    transition_vector = [0] * 155 #  last index is the no move index
    transition_vector[index] = 1
    return transition_vector


def mirror_move(move):
    mirrored_move = copy.deepcopy(move)
    white_to = move['White']['To']
    white_from = move['White']['From']
    white_from_column = white_from[0]
    white_to_column = white_to[0]
    white_from_row = int(white_from[1])
    white_to_row = int(white_to[1])
    mirrored_move['White']['To'] = mirror_column(white_to_column) + str(white_to_row)
    mirrored_move['White']['From'] = mirror_column(white_from_column) + str(white_from_row)

    if isinstance(move['Black'], dict):
        black_to = move['Black']['To']
        black_from = move['Black']['From']
        black_from_column = black_from[0]
        black_to_column = black_to[0]
        black_from_row = int(black_from[1])
        black_to_row = int(black_to[1])
        mirrored_move['Black']['To'] = mirror_column(black_to_column) + str(black_to_row)
        mirrored_move['Black']['From'] = mirror_column(black_from_column) + str(black_from_row)
    # else 'Black' == NIL, don't change it
    return mirrored_move


def mirror_column(column_char):
    mirror_dict = {'a': 'h',
                   'b': 'g',
                   'c': 'f',
                   'd': 'e',
                   'e': 'd',
                   'f': 'c',
                   'g': 'b',
                   'h': 'a'
                   }
    return mirror_dict[column_char]


def reflect_board_state(state):  # since black needs to have a POV representation
    semi_reflected_state = mirror_board_state(state)
    reflected_state = copy.deepcopy(semi_reflected_state)
    reflected_state[0][1] = semi_reflected_state[0][8]
    reflected_state[0][2] = semi_reflected_state[0][7]
    reflected_state[0][3] = semi_reflected_state[0][6]
    reflected_state[0][4] = semi_reflected_state[0][5]
    reflected_state[0][5] = semi_reflected_state[0][4]
    reflected_state[0][6] = semi_reflected_state[0][3]
    reflected_state[0][7] = semi_reflected_state[0][2]
    reflected_state[0][8] = semi_reflected_state[0][1]
    return reflected_state


def mirror_board_state(state):  # since a mirror image has the same strategic value
    mirror_state_with_win = copy.deepcopy(state)  # edit copy of board_state
    mirror_state = mirror_state_with_win[0]
    state = state[0]  # the board state; state[1] is the win or loss value, state [2] is the transition vector
    is_white_index = 9
    white_move_index = 10
    for row in sorted(state):
        if row != is_white_index and row != white_move_index:  # these indexes don't change
            for column in sorted(state[row]):
                if column == 'a':
                    mirror_state[row]['h'] = state[row][column]
                elif column == 'b':
                    mirror_state[row]['g'] = state[row][column]
                elif column == 'c':
                    mirror_state[row]['f'] = state[row][column]
                elif column == 'd':
                    mirror_state[row]['e'] = state[row][column]
                elif column == 'e':
                    mirror_state[row]['d'] = state[row][column]
                elif column == 'f':
                    mirror_state[row]['c'] = state[row][column]
                elif column == 'g':
                    mirror_state[row]['b'] = state[row][column]
                elif column == 'h':
                    mirror_state[row]['a'] = state[row][column]
    return mirror_state_with_win

def convert_board_states_to_matrices(board_states, player_color):
    warnings.warn("Removed in favor of performing conversion later in the pipeline. "
                  "Else, this stage of the pipeline will be computationally expensive, "
                  "memory intensive, and require large amounts of disk space "
                  , DeprecationWarning)
    new_board_states = board_states
    POV_states = board_states['PlayerPOV']
    states = board_states['States']
    new_board_states['States'] = []
    new_board_states['PlayerPOV'] = []
    for state in states:
        new_board_states['States'].append(
            convert_board_to_2d_matrix_value_net(state, player_color))  # These can be inputs to value net
    for POV_state in POV_states:
        new_board_states['PlayerPOV'].append(
            convert_board_to_2d_matrix_policy_net(POV_state, player_color))  # These can be inputs to policy net
    return new_board_states

def convert_board_to_2d_matrix_value_net(board_state, player_color):
    state = board_state[0]
    # Do we need White/Black if black is a flipped representation? we are only asking the net "from my POV, these are
    # my pieces and my opponent's pieces what move should I take? If not, the net sees homogeneous data.
    # My fear is seeing black in a flipped representation may mess things up as, if you are white, black being in a
    # lower row is not good, but would the Player/Opponent features cancel this out?

    # if player color == white, player and white states are mirrors; else, player and black states are mirrors
    one_hot_board = [generate_binary_plane(state, player_color=player_color,
                                            who_to_filter='White'),  # [0] white
                     generate_binary_plane(state, player_color=player_color,
                                            who_to_filter='Black'),  # [1] black
                     generate_binary_plane(state, player_color=player_color,
                                            who_to_filter='Player'),  # [2] player
                     generate_binary_plane(state, player_color=player_color,
                                            who_to_filter='Opponent'),  # [3] opponent
                     generate_binary_plane(state, player_color=player_color,
                                            who_to_filter='Empty'),  # [4] empty
                     generate_binary_plane(state, player_color=player_color,
                                           who_to_filter='MoveFlag')]  # [5] flag indicating if the transition came from a white move
    
    # ensure at most 1 bit is on at each board position for white/black/empty
    assert ((one_hot_board[0] ^ one_hot_board[1] ^ one_hot_board[4]).all().all() and
            not (one_hot_board[0] & one_hot_board[1] & one_hot_board[4]).all().all())
    # ensure at most 1 bit is on at each board position for player/opponent/empty
    assert ((one_hot_board[2] ^ one_hot_board[3] ^ one_hot_board[4]).all().all() and
            not (one_hot_board[2] & one_hot_board[3] & one_hot_board[4]).all().all())
    if player_color == 'White':
        # white positions == player and black positions == opponent;
        assert (one_hot_board[0].equals(one_hot_board[2]) and one_hot_board[1].equals(one_hot_board[3]))
    else:
        #  white positions == opponent and player == black positions;
        assert (one_hot_board[0].equals(one_hot_board[3]) and one_hot_board[1].equals(one_hot_board[2]))
    new_board_state = [one_hot_board, board_state[1], board_state[2]]  # [x vector, win, y transition vector]
    return new_board_state


def convert_board_to_2d_matrix_policy_net(board_state, player_color):
    state = board_state[0]
    one_hot_board = [generate_binary_plane(state, player_color=player_color,
                                            who_to_filter='Player'),  # [0] player
                     generate_binary_plane(state, player_color=player_color,
                                            who_to_filter='Opponent'),  # [1] opponent
                     generate_binary_plane(state, player_color=player_color,
                                            who_to_filter='Empty')]  # [2] empty

    # ensure at most 1 bit is on at each board position for player/opponent/empty
    assert ((one_hot_board[0] ^ one_hot_board[1] ^ one_hot_board[2]).all().all() and
                not(one_hot_board[0] & one_hot_board[1] & one_hot_board[2]).all().all())
    new_board_state = [one_hot_board, board_state[1], board_state[2]]  # ex. [x vector, win, y transition vector]
    return new_board_state

def generate_binary_plane(state, player_color, who_to_filter):
    white_move = state[10]
    panda_board = pd.DataFrame(state).transpose().sort_index(ascending=False).iloc[2:]  # human-readable board
    if who_to_filter == 'White':
        who_dict = {
            'e': 0,
            'w': 1,
            'b': 0}
    elif who_to_filter == 'Black':
        who_dict = {
            'e': 0,
            'w': 0,
            'b': 1}
    elif who_to_filter == 'Player':
        if player_color == 'White':
            who_dict = {
                'e': 0,
                'w': 1,
                'b': 0}
        elif player_color == 'Black':
            who_dict = {
                'e': 0,
                'w': 0,
                'b': 1}
        else:
            print("error in convertBoard")
            exit(-190)
    elif who_to_filter == 'Opponent':
        if player_color == 'White':
            who_dict = {
                'e': 0,
                'w': 0,
                'b': 1}
        elif player_color == 'Black':
            who_dict = {
                'e': 0,
                'w': 1,
                'b': 0}
        else:
            print("error in convertBoard")
            exit(-190)
    elif who_to_filter == 'Empty':
        who_dict = {
            'e': 1,
            'w': 0,
            'b': 0}
    elif who_to_filter == 'MoveFlag':  # duplicate across 64 positions since CNN needs same dimensions
        who_dict = {
            'e': white_move,
            'w': white_move,
            'b': white_move}
    else:
        print("Error, generate_binary_plane needs a valid argument to filter")
    return panda_board.apply(lambda x: x.apply(lambda y: who_dict[y]))#filter board positions to binary


def convert_board_states_to_arrays(board_states, player_color):
    #TODO: more features: plane of 1s if next move == capture, plane of 1's on locations of possible captures? plane of 1s on locations of possible moves?
    new_board_states = board_states
    POV_states = board_states['PlayerPOV']
    states = board_states['States']
    new_board_states['States'] = []
    new_board_states['PlayerPOV'] = []
    for state in states:
        new_board_states['States'].append(
            convert_board_to_1d_array_POE(state, player_color))  # These can be inputs to value net
    for POV_state in POV_states:
        new_board_states['PlayerPOV'].append(
            convert_board_to_1d_array_POE(POV_state, player_color))  # These can be inputs to policy net
    return new_board_states


def convert_board_to_1d_array_WBPOEWmBm(board_state, player_color):
    state = board_state[0]
    white_move_index = 10
    from_white_move = state[white_move_index]

    # Do we need White/Black if black is a flipped representation? we are only asking the net "from my POV, these are
    # my pieces and my opponent's pieces what move should I take? If not, the net sees homogeneous data.
    # My fear is seeing black in a flipped representation may mess things up as, if you are white, black being in a 
    # lower row is not good, but would the Player/Opponent features cancel this out?

    # if player color == white, player and white states are mirrors; else, player and black states are mirrors
    white_move_flag = [from_white_move]  * 64  # duplicate across 64 features since CNN needs same dimensions
    black_move_flag = [from_white_move^1] *64
    one_hot_board = [generate_binary_vector(state, array_to_append=[], player_color=player_color,
                                            who_to_filter='White'),  # [0] white
                     generate_binary_vector(state, array_to_append=[], player_color=player_color,
                                            who_to_filter='Black'),  # [1] black
                     generate_binary_vector(state, array_to_append=[], player_color=player_color,
                                            who_to_filter='Player'),  # [2] player
                     generate_binary_vector(state, array_to_append=[], player_color=player_color,
                                            who_to_filter='Opponent'),  # [3] opponent
                     generate_binary_vector(state, array_to_append=[], player_color=player_color,
                                            who_to_filter='Empty'),  # [4] empty
                     white_move_flag,
                     black_move_flag]  # [5] flag indicating if the transition came from a white move
    for i in range(0, 64):  # error checking block
        # ensure at most 1 bit is on at each board position for white/black/empty
        assert ((one_hot_board[0][i] ^ one_hot_board[1][i] ^ one_hot_board[4][i]) and
                not (one_hot_board[0][i] & one_hot_board[1][i] & one_hot_board[4][i]))
        # ensure at most 1 bit is on at each board position for player/opponent/empty
        assert ((one_hot_board[2][i] ^ one_hot_board[3][i] ^ one_hot_board[4][i]) and
                not (one_hot_board[2][i] & one_hot_board[3][i] & one_hot_board[4][i]))
        if player_color == 'White':
            # white positions == player and black positions == opponent;
            assert (one_hot_board[0][i] == one_hot_board[2][i] and one_hot_board[1][i] == one_hot_board[3][i])
        else:
            #  white positions == opponent and player == black positions;
            assert (one_hot_board[0][i] == one_hot_board[3][i] and one_hot_board[1][i] == one_hot_board[2][i])
    new_board_state = [one_hot_board, board_state[1], board_state[2]]  # [x vector, win, y transition vector]
    return new_board_state

def convert_board_to_1d_array_POEPmOm(board_state, player_color):
    state = board_state[0]
    white_move_index = 10
    from_white_move = state[white_move_index]

    if player_color == 'White':
        player_to_move = from_white_move ^ 1  # if from_white_move == 1, it's not white's move this turn
    else: # player_color == 'Black'
        player_to_move = from_white_move  #  black's move = 1 iff from_white_move = 1
    player_move_flag = [player_to_move] * 64
    opponent_move_flag = [player_to_move ^ 1] * 64
    one_hot_board = [generate_binary_vector(state, array_to_append=[], player_color=player_color,
                                            who_to_filter='Player'),  # [0] player
                     generate_binary_vector(state, array_to_append=[], player_color=player_color,
                                            who_to_filter='Opponent'),  # [1] opponent
                     generate_binary_vector(state, array_to_append=[], player_color=player_color,
                                            who_to_filter='Empty'),  # [3] empty
                     player_move_flag,  # [4] flag indicating if players's turn to move
                     opponent_move_flag]  # [5] flag indicating if opponent's turn to move
    for i in range(0, 64):  # error checking block
        # ensure at most 1 bit is on at each board position for player/opponent/empty
        assert ((one_hot_board[0][i] ^ one_hot_board[1][i] ^ one_hot_board[2][i]) and
                not (one_hot_board[0][i] & one_hot_board[1][i] & one_hot_board[2][i]))
    new_board_state = [one_hot_board, board_state[1], board_state[2]]  # [x vector, win, y transition vector]
    return new_board_state

def convert_board_to_1d_array_POE(board_state, player_color):
    # 12/22 removed White/Black to avoid Curse of Dimensionality.
    # TODO: use dict for board representations so later analysis can identify without looking at this code?
    state = board_state[0]
    one_hot_board = [generate_binary_vector(state, array_to_append=[], player_color=player_color,
                                            who_to_filter='Player'),  # [0] player
                     generate_binary_vector(state, array_to_append=[], player_color=player_color,
                                            who_to_filter='Opponent'),  # [1] opponent
                     generate_binary_vector(state, array_to_append=[], player_color=player_color,
                                            who_to_filter='Empty')]  # [2] empty
    for i in range(0, 64):  # error checking block
        # ensure at most 1 bit is on at each board position for player/opponent/empty
        assert ((one_hot_board[0][i] ^ one_hot_board[1][i] ^ one_hot_board[2][i]) and
                not (one_hot_board[0][i] & one_hot_board[1][i] & one_hot_board[2][i]))
    new_board_state = [one_hot_board, board_state[1], board_state[2]]  # ex. [x vector, win, y transition vector]
    return new_board_state


def generate_binary_vector(state, array_to_append, player_color, who_to_filter):
    is_white_index = 9
    white_move_index = 10
    if who_to_filter == 'White':
        white_dict = {
            'e': 0,
            'w': 1,
            'b': 0}
        for row in sorted(state):
            if row != is_white_index and row != white_move_index:  # don't touch these indexes
                for column in sorted(state[row]):  # needs to be sorted to traverse dictionary in lexicographical order
                    array_to_append.append(white_dict[state[row][column]])
    elif who_to_filter == 'Black':
        black_dict = {
            'e': 0,
            'w': 0,
            'b': 1}
        for row in sorted(state):
            if row != is_white_index and row != white_move_index:  # don't touch these indexes
                for column in sorted(state[row]):  # needs to be sorted to traverse dictionary in lexicographical order
                    array_to_append.append(black_dict[state[row][column]])
    elif who_to_filter == 'Player':
        if player_color == 'White':
            player_dict = {
                'e': 0,
                'w': 1,
                'b': 0}
        elif player_color == 'Black':
            player_dict = {
                'e': 0,
                'w': 0,
                'b': 1}
        else:
            print("error in convertBoard")
            exit(-190)
        for row in sorted(state):
            if row != is_white_index and row != white_move_index:  # don't touch these indexes
                for column in sorted(state[row]):  # needs to be sorted to traverse dictionary in lexicographical order
                    array_to_append.append(player_dict[state[row][column]])
    elif who_to_filter == 'Opponent':
        if player_color == 'White':
            opponent_dict = {
                'e': 0,
                'w': 0,
                'b': 1}
        elif player_color == 'Black':
            opponent_dict = {
                'e': 0,
                'w': 1,
                'b': 0}
        else:
            print("error in convertBoard")
            exit(-190)
        for row in sorted(state):
            if row != is_white_index and row != white_move_index:  # don't touch these indexes
                for column in sorted(state[row]):  # needs to be sorted to traverse dictionary in lexicographical order
                    array_to_append.append(opponent_dict[state[row][column]])
    elif who_to_filter == 'Empty':
        empty_dict = {
            'e': 1,
            'w': 0,
            'b': 0}
        for row in sorted(state):
            if row != is_white_index and row != white_move_index:  # don't touch these indexes
                for column in sorted(state[row]):  # needs to be sorted to traverse dictionary in lexicographical order
                    array_to_append.append(empty_dict[state[row][column]])
    else:
        print("Error, generate_binary_vector needs a valid argument to filter")
    return array_to_append


def move_piece(board_state, to, _from, whose_move):
    empty = 'e'
    white_move_index = 10
    next_board_state = copy.deepcopy(board_state)  # edit copy of board_state
    next_board_state[int(to[1])][to[0]] = next_board_state[int(_from[1])][_from[0]]
    next_board_state[int(_from[1])][_from[0]] = empty
    if whose_move == 'White':
        next_board_state[white_move_index] = 1
    else:
        next_board_state[white_move_index] = 0
    return next_board_state


def format_move_lists_and_links(unformatted_move_list):
    move_regex = re.compile(r"[W|B]\s([a-h]\d.[a-h]\d)",
                            re.IGNORECASE)
    original_visualizer_link = mirror_visualizer_link = r'http://www.trmph.com/breakthrough/board#8,'
    move_list = list(map(lambda a: move_regex.search(a).group(1), unformatted_move_list))
    move_num = 0
    new_move_list = []
    new_mirror_move_list = []
    move = [None] * 3
    for i in range(0, len(move_list)):
        move_num += 1
        move[0] = math.ceil(move_num / 2)
        _from = str(move_list[i][0:2]).lower()
        to = str(move_list[i][3:5]).lower()
        from_column = _from[0]
        from_row = _from[1]
        to_column = to[0]
        to_row = to[1]
        mirror_from = mirror_column(from_column) + from_row
        mirror_to = mirror_column(to_column) + to_row
        if i % 2 == 0:  # white move
            assert (from_row < to_row)  # white should go forward
            move[1] = {'From': _from, 'To': to}  # set White's moves
            if i == len(move_list) - 1:  # white makes last move of game; black lost
                move[2] = "NIL"
                temp_move = {'#': move[0], 'White': move[1], 'Black': move[2]}
                new_move_list.append(temp_move)
                new_mirror_move_list.append(mirror_move(temp_move))
        else:  # black move
            assert (from_row > to_row)  # black should go backward
            move[2] = {'From': _from, 'To': to}  # set Black's moves
            temp_move = {'#': move[0], 'White': move[1], 'Black': move[2]}
            new_move_list.append(temp_move)
            new_mirror_move_list.append(mirror_move(temp_move))
        original_visualizer_link = original_visualizer_link + _from + to
        mirror_visualizer_link = mirror_visualizer_link + mirror_from + mirror_to
    return new_move_list, new_mirror_move_list, original_visualizer_link, mirror_visualizer_link


def driver(path):
    player_list = process_directory_of_breakthrough_files(path)
    write_to_disk(player_list, path)
