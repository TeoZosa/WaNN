import re as re  # regular expressions
import os
import pickle  # serialize the data structure
import mmap  # read entire files into memory (only for workstation)
import copy
import math
from multiprocessing import Pool
from tools import utils
from Breakthrough_Player.board_utils import enumerate_legal_moves

def driver(path):
    player_list = process_directory_of_breakthrough_files_multithread(path)
    write_to_disk(player_list, path)

def write_to_disk(data_to_write, path):
    write_directory = 'G:\TruncatedLogs\PythonDataSets\DataStructures'
    date = path[len(r'G:\TruncatedLogs') + 1
    :- len(r'\selfPlayLogsBreakthroughN')]
    server_name = path[- len(r'\selfPlayLogsBreakthroughN') + 1
    :]
    output_file = open(os.path.join(write_directory, (
        date  # prepend data to name of server
        + server_name
        + r'DataPython.p')), 'wb')  # append data qualifier
    pickle.dump(data_to_write, output_file, protocol=pickle.HIGHEST_PROTOCOL)

def process_directory_of_breakthrough_files_single_thread(path):
    player_list = []
    arg_lists = [[path, self_play_games] for self_play_games in utils.find_files(path, '*.txt')]
    for arg in arg_lists:
         player_list.extend(process_breakthrough_file (arg[0], arg[1]))
    return player_list

def process_directory_of_breakthrough_files_multithread(path):
    player_list = []
    arg_lists = [[path, self_play_games] for self_play_games in utils.find_files(path, '*.txt')]
    process_pool = Pool(processes=len(arg_lists))
    player_list.extend(process_pool.starmap(process_breakthrough_file, arg_lists))
    process_pool.close()
    process_pool.join()
    return player_list

def process_breakthrough_file(path, self_play_games):
    file_name = self_play_games[len(path):- len('.txt')]  # trim path & extension
    file_name = file_name.split('_SelfPlayLog')  # \BreakthroughN_SelfPlayLog00-> ['\BreakthroughN',00]
    server_name = file_name[0].strip('\\')  #'\BreakthroughN' -> 'BreakthroughN'
    self_play_log = file_name[1].strip(
                        '(').strip(
                        ')')  # if file_name was originally \BreakthroughN_SelfPlayLog(xy)
    date_range = self_play_games[len(r'G:\TruncatedLogs')
                                     + 1:len(path)
                                     - len(r'\selfPlayLogsBreakthroughN')]
    games_list, white_wins, black_wins = format_game_list(self_play_games)
    return {'ServerNode': server_name, 'Self-PlayLog': self_play_log, 'DateRange': date_range, 'Games': games_list,
            'WhiteWins': white_wins, 'BlackWins': black_wins}

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
            # self-play => same states, but win under policy for W <=> lose under policy for B
            black_board_states = generate_board_states(move_list, 'Black', black_win)

            # since a mirror image has the same strategic value
            white_mirror_board_states = generate_board_states(mirror_move_list, 'White', white_win)
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
            #equivalent to -1 or 0 if white's move, 1 if black's move
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
        utils.generate_transition_vector(move_list[0]['White']['To'], move_list[0]['White']['From'],
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
            if isinstance(move_list[i]['Black'], dict):  # if Black has a reply
                black_transition_vector = utils.generate_transition_vector(move_list[i]['Black']['To'],
                                                                           move_list[i]['Black']['From'], 'Black')
                # can't put black move block in here as it would execute before white's move
            else:  # no black move => white won 
                black_transition_vector = [0] * 154 + [1]
            state = [
                move_piece(state[0], move_list[i]['White']['To'], move_list[i]['White']['From'], whose_move='White'),
                win,
                black_transition_vector]  # Black's response to the generated state
            if player_color == 'Black':  # reflect positions tied to black transitions
                board_states['PlayerPOV'].append(reflect_board_state(state))
            board_states['States'].append(state)

        if isinstance(move_list[i]['Black'], dict):  # if not, then == resign or NIL
            if i + 1 == len(move_list):  # if no next white move => black won
                white_transition_vector = [0] * 154 + [1]  # no white move from the next generated state
            else:
                white_transition_vector = utils.generate_transition_vector(move_list[i + 1]['White']['To'],
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


def convert_board_states_to_arrays(board_states, player_color):
    #TODO: more features: 16 extra binary planes per player for # pieces left? i.e. plane 14 => 14 pieces left?
    new_board_states = board_states
    POV_states = board_states['PlayerPOV']
    states = board_states['States']
    new_board_states['States'] = []
    new_board_states['PlayerPOV'] = []
    for state in states:
        new_board_states['States'].append(
            convert_board_to_1d_array_POEB(state, player_color))  # These can be inputs to value net
    for POV_state in POV_states:
        new_board_states['PlayerPOV'].append(
            convert_board_to_1d_array_POEB(POV_state, player_color))  # These can be inputs to policy net
    return new_board_states

'''Feature Planes:
    White, Black, Player, Opponent, Empty, Came From White Move, Came From Black Move
    '''
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
    one_hot_board = [utils.generate_binary_vector(state, player_color, what_to_filter='White'),  # [0] white
                     utils.generate_binary_vector(state, player_color, what_to_filter='Black'),  # [1] black
                     utils.generate_binary_vector(state, player_color, what_to_filter='Player'),  # [2] player
                     utils.generate_binary_vector(state, player_color, what_to_filter='Opponent'),  # [3] opponent
                     utils.generate_binary_vector(state, player_color, what_to_filter='Empty'),  # [4] empty
                     white_move_flag,  # [5] flag indicating if the transition came from a white move
                     black_move_flag,
                     utils.generate_binary_vector(state, player_color, what_to_filter='Bias')]
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

'''Feature Planes:
    Player, Opponent, Empty, Came From Player Move, Came From Opponent Move
    '''
def convert_board_to_1d_array_POEPmOmB(board_state, player_color):
    state = board_state[0]
    white_move_index = 10
    from_white_move = state[white_move_index]

    if player_color == 'White':
        player_to_move = from_white_move ^ 1  # if from_white_move == 1, it's not white's move this turn
    else: # player_color == 'Black'
        player_to_move = from_white_move  #  black's move = 1 iff from_white_move = 1
    player_move_flag = [player_to_move] * 64
    opponent_move_flag = [player_to_move ^ 1] * 64
    one_hot_board = [utils.generate_binary_vector(state, player_color, what_to_filter='Player'),  # [0] player
                     utils.generate_binary_vector(state, player_color, what_to_filter='Opponent'),  # [1] opponent
                     utils.generate_binary_vector(state, player_color, what_to_filter='Empty'),  # [3] empty
                     player_move_flag,  # [4] flag indicating if players's turn to move
                     opponent_move_flag,  # [5] flag indicating if opponent's turn to move
                     utils.generate_binary_vector(state, player_color, what_to_filter='Bias')]
    for i in range(0, 64):  # error checking block
        # ensure at most 1 bit is on at each board position for player/opponent/empty
        assert ((one_hot_board[0][i] ^ one_hot_board[1][i] ^ one_hot_board[2][i]) and
                not (one_hot_board[0][i] & one_hot_board[1][i] & one_hot_board[2][i]))
    new_board_state = [one_hot_board, board_state[1], board_state[2]]  # [x vector, win, y transition vector]
    return new_board_state

'''Feature Planes:
    Player, Opponent, Empty
    '''
def convert_board_to_1d_array_POEB(board_state, player_color):
    # 12/22 removed White/Black to avoid Curse of Dimensionality.
    state = board_state[0]
    one_hot_board = [utils.generate_binary_vector(state, player_color, what_to_filter='Player'),  # [0] player
                     utils.generate_binary_vector(state, player_color, what_to_filter='Opponent'),  # [1] opponent
                     utils.generate_binary_vector(state, player_color, what_to_filter='Empty'),  # [2] empty
                     utils.generate_binary_vector(state, player_color, what_to_filter='Bias')]
    for i in range(0, 64):  # error checking block
        # ensure at most 1 bit is on at each board position for player/opponent/empty
        assert ((one_hot_board[0][i] ^ one_hot_board[1][i] ^ one_hot_board[2][i]) and
                not (one_hot_board[0][i] & one_hot_board[1][i] & one_hot_board[2][i]))
    new_board_state = [one_hot_board, board_state[1], board_state[2]]  # ex. [x vector, win, y transition vector]
    return new_board_state

"""'Feature Planes:
Player, Opponent, Empty, Moves From, Moves To, Captures From, Captures To, Player (Next State),
Opponent (Next State), Empty (Next State), If Actual Move Made Was A Capture Move
''"""
def convert_board_to_1d_array_POEMfMtCfCtPnOnEnCmB(board_state, player_color, POV_states):
    state, board_with_moves, next_state, capture_filter = generate_intermediate_boards_POEMfMtCfCtPnOnEnCmB(
        board_state, player_color, POV_states)
    one_hot_board = [utils.generate_binary_vector(state, player_color, what_to_filter='Player'),  # [0] player
                     utils.generate_binary_vector(state, player_color, what_to_filter='Opponent'),  # [1] opponent
                     utils.generate_binary_vector(state, player_color, what_to_filter='Empty'),  # [2] empty
                     utils.generate_binary_vector(board_with_moves, player_color, what_to_filter='Moves From'),  # [3]
                     utils.generate_binary_vector(board_with_moves, player_color, what_to_filter='Moves To'),  # [4]
                     utils.generate_binary_vector(board_with_moves, player_color, what_to_filter='Captures From'),  # [5]
                     utils.generate_binary_vector(board_with_moves, player_color, what_to_filter='Captures To'),  # [6]
                     # utils.generate_binary_vector(next_state, player_color, what_to_filter='Player'),  # [7] player
                     # utils.generate_binary_vector(next_state, player_color, what_to_filter='Opponent'),  # [8] opponent
                     # utils.generate_binary_vector(next_state, player_color, what_to_filter='Empty'), # [9] empty
                     # utils.generate_binary_vector(next_state, player_color, what_to_filter=capture_filter), #10
                     utils.generate_binary_vector(state, player_color, what_to_filter='Bias')]  #[11]
    for i in range(0, 64):  # error checking block
        # ensure at most 1 bit is on at each board position for player/opponent/empty
        assert ((one_hot_board[0][i] ^ one_hot_board[1][i] ^ one_hot_board[2][i]) and
                not (one_hot_board[0][i] & one_hot_board[1][i] & one_hot_board[2][i]))

    new_board_state = [one_hot_board, board_state[1], board_state[2]]  # ex. [x vector, win, y transition vector]
    return new_board_state

def generate_intermediate_boards_POEMfMtCfCtPnOnEnCmB(board_state, player_color, POV_states):
    state = board_state[0]
    next_move = utils.move_lookup_by_index(board_state[2].index(1), player_color).split('-')
    next_to = next_move[1]
    next_from = next_move[0]
    if next_from == 'no':  # 'no move' => next state is the same as the current state
        next_state = state
        capture_filter = 'Non-Capture Move'
        board_with_moves = state  # hacky but simple, generate binary vector will return a plane of all 0s since state has no move flags
    else:
        if POV_states and player_color == 'Black':  # reflect for non-POV
            original_state = reflect_board_state(board_state)[0]
            # Reflect new boards back to POV
            # Funny syntax below since reflect_board_state is expecting a list where game_board is the first element
            board_with_moves = reflect_board_state([mark_all_possible_moves(original_state, player_color)])[0]
            next_state = reflect_board_state([move_piece(original_state, next_to, next_from, player_color)])[0]
        else:
            original_state = state
            board_with_moves = mark_all_possible_moves(original_state, player_color)
            # TODO: probably should change this since player is asking which move to make, not asking whether some possible move is good.
            next_state = move_piece(original_state, next_to, next_from, player_color)
        if next_move_is_capture(original_state, next_to, player_color) == True:
            capture_filter = 'Capture Move'
        else:
            capture_filter = 'Non-Capture Move'
    return state, board_with_moves, next_state, capture_filter

def next_move_is_capture(state, to, player_color):
    if player_color == 'White':
        enemy_piece = 'b'
    else:
        enemy_piece = 'w'
    row = int(to[1])
    column = to[0]
    if state[row][column] == enemy_piece:
        capture = True
    else:
        capture = False
    return capture

def mark_all_possible_moves(state, player_color):
    #TODO: can we solve for the fact that 'to' moves don't know where they came from? 
    # different 'from' moves map to the same 'to' moves?

    #we can't do the naive solution and specify a plane of 3 distinct values, for each move (amount of moves isn't constant);
    # maybe set some upper bound on # moves (i.e. 16 x 3 = 48) and if <48, just leave them empty (plane of 0's)?
    #not feasible using binary planes since we'd need 2 boards per move (1 for move to, 1 for move from)  => 96 planes
    #not worth the information loss_init vs specifying a plane per transition

    #specifying a plane of 1's or 0's for each of the 155 transitions is also not feasible (would PROBABLY require too much data)
    legal_moves = enumerate_legal_moves(state, player_color)
    board_with_moves = copy.deepcopy(state)# edit copy of board_state
    for move in legal_moves:
        _from = move['From']
        to = move['To']
        board_with_moves = mark_possible_move(board_with_moves, to, _from, player_color)
    return board_with_moves

def mark_possible_move(state, to, _from, player_color):
    move_to_flag = 'Player Move To'
    move_from_flag = 'Player Move From'
    capture_to_flag = 'Player Capture To'
    capture_from_flag = 'Player Capture From'
    white_move_index = 10
    next_board_state = copy.deepcopy(state)  # edit copy of board_state
    if player_color == 'White':
        next_board_state[white_move_index] = 1 #next state came from white move
        enemy_piece = 'b'
    else:
        next_board_state[white_move_index] = 0 #next state did not come from white move
        enemy_piece = 'w'
    piece_at_move_location = next_board_state[int(to[1])][to[0]]
    if piece_at_move_location == enemy_piece:
        next_board_state[int(to[1])][to[0]] = capture_to_flag
        next_board_state[int(_from[1])][_from[0]] = capture_from_flag
    else:
        next_board_state[int(to[1])][to[0]] = move_to_flag
        next_board_state[int(_from[1])][_from[0]] = move_from_flag
    return next_board_state

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

def mirror_board_state(state):  # helper method for reflect_board_state
    mirror_state_with_win = copy.deepcopy(state)  # edit copy of board_state
    mirror_state = mirror_state_with_win[0]
    state = state[0]  # the board state; state[1] is the win or loss_init value, state [2] is the transition vector
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

