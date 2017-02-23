import copy



def initial_game_board():
    empty = 'e'
    white = 'w'
    black = 'b'
    return {
        10: -1,  # (-1 for initial state, 0 if black achieved state, 1 if white achieved state)
        # equivalent to 0 if white's move, 1 if black's move
        9: 1,  # is player_color white
        8: {'a': black, 'b': black, 'c': black, 'd': black, 'e': black, 'f': black, 'g': black, 'h': black},
        7: {'a': black, 'b': black, 'c': black, 'd': black, 'e': black, 'f': black, 'g': black, 'h': black},
        6: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
        5: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
        4: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
        3: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
        2: {'a': white, 'b': white, 'c': white, 'd': white, 'e': white, 'f': white, 'g': white, 'h': white},
        1: {'a': white, 'b': white, 'c': white, 'd': white, 'e': white, 'f': white, 'g': white, 'h': white}
    }
def move_piece(board_state, move, whose_move):
    empty = 'e'
    white_move_index = 10
    is_white_index = 9
    move = move.split('-')
    _from = move[0].lower
    to = move[1].lower
    next_board_state = copy.deepcopy(board_state)  # edit copy of board_state; don't need this for breakthrough_player?
    next_board_state[int(to[1])][to[0]] = next_board_state[int(_from[1])][_from[0]]
    next_board_state[int(_from[1])][_from[0]] = empty
    if whose_move == 'White':
        next_board_state[white_move_index] = 1
        next_board_state[is_white_index] = 0 #next move isn't white's
    else:
        next_board_state[white_move_index] = 0
        next_board_state[is_white_index] = 1 #since black made this move, white makes next move
    return next_board_state

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