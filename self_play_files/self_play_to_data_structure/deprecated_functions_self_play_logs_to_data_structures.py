import warnings
import pandas as pd

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
            convert_board_to_2d_matrix_POE(state, player_color))  # These can be inputs to value net
    for POV_state in POV_states:
        new_board_states['PlayerPOV'].append(
            convert_board_to_2d_matrix_POE(POV_state, player_color))  # These can be inputs to policy net
    return new_board_states


def convert_board_to_2d_matrix_WBPOEWm(board_state, player_color):
    warnings.warn("Removed in favor of performing conversion later in the pipeline. "
                  "Else, this stage of the pipeline will be computationally expensive, "
                  "memory intensive, and require large amounts of disk space "
                  , DeprecationWarning)
    state = board_state[0]
    # Do we need White/Black if black is a flipped representation? we are only asking the net "from my POV, these are
    # my pieces and my opponent's pieces what move should I take? If not, the net sees homogeneous data.
    # My fear is seeing black in a flipped representation may mess things up as, if you are white, black being in a
    # lower row is not good, but would the Player/Opponent features cancel this out?

    # if player color == white, player and white states are mirrors; else, player and black states are mirrors
    one_hot_board = [generate_binary_plane(state, player_color,
                                           who_to_filter='White'),  # [0] white
                     generate_binary_plane(state, player_color,
                                           who_to_filter='Black'),  # [1] black
                     generate_binary_plane(state, player_color,
                                           who_to_filter='Player'),  # [2] player
                     generate_binary_plane(state, player_color,
                                           who_to_filter='Opponent'),  # [3] opponent
                     generate_binary_plane(state, player_color,
                                           who_to_filter='Empty'),  # [4] empty
                     generate_binary_plane(state, player_color,
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


def convert_board_to_2d_matrix_POE(board_state, player_color):
    warnings.warn("Removed in favor of performing conversion later in the pipeline. "
                  "Else, this stage of the pipeline will be computationally expensive, "
                  "memory intensive, and require large amounts of disk space "
                  , DeprecationWarning)
    state = board_state[0]
    one_hot_board = [generate_binary_plane(state, player_color,
                                           who_to_filter='Player'),  # [0] player
                     generate_binary_plane(state, player_color,
                                           who_to_filter='Opponent'),  # [1] opponent
                     generate_binary_plane(state, player_color,
                                           who_to_filter='Empty')]  # [2] empty

    # ensure at most 1 bit is on at each board position for player/opponent/empty
    assert ((one_hot_board[0] ^ one_hot_board[1] ^ one_hot_board[2]).all().all() and
            not (one_hot_board[0] & one_hot_board[1] & one_hot_board[2]).all().all())
    new_board_state = [one_hot_board, board_state[1], board_state[2]]  # ex. [x vector, win, y transition vector]
    return new_board_state


def generate_binary_plane(state, player_color, who_to_filter):
    warnings.warn("Removed in favor of performing conversion later in the pipeline. "
                  "Else, this stage of the pipeline will be computationally expensive, "
                  "memory intensive, and require large amounts of disk space "
                  , DeprecationWarning)
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
    return panda_board.apply(lambda x: x.apply(lambda y: who_dict[y]))  # filter board positions to binary
