#cython: language_level=3, boundscheck=False

from os import walk, path
from fnmatch import filter as fnmatch_filter
from sys import stderr

# from cpython cimport array
cimport  numpy as np

# import array
import numpy as np


DTYPE = np.int8

ctypedef np.int_t DTYPE_t

def find_files(file_path, extension):
    """'
     Recursively find files at path with extension; pulled from StackOverflow
    ''"""#

    for root, dirs, files in walk(file_path):
        for file in fnmatch_filter(files, extension):
            yield path.join(root, file)

def batch_split(training_examples, labels, batch_size):#
    """'
     Lazily split examples and labels into training batches of size batch_size
    ''"""#
    X_train_batches = [training_examples[:batch_size]]
    y_train_batches = [labels[:batch_size]]
    remaining_x_train = training_examples[batch_size:]
    remaining_y_train = labels[batch_size:]
    for i in range(1, len(training_examples) // batch_size):
        X_train_batches.append(remaining_x_train[:batch_size])
        y_train_batches.append(remaining_y_train[:batch_size])
        remaining_x_train = remaining_x_train[batch_size:]
        remaining_y_train = remaining_y_train[batch_size:]
    if len(remaining_x_train) > 0:
        X_train_batches.append(remaining_x_train)  # append remaining training examples
        y_train_batches.append(remaining_y_train)
    return X_train_batches, y_train_batches

def batch_split_no_labels(inference_examples, batch_size):
    """'
     Lazily split examples into training batches of size batch_size
    ''"""#
    input_batch = [inference_examples[:batch_size]]
    remaining_inputs = inference_examples[batch_size:]
    for i in range(1, len(inference_examples) // batch_size):
        input_batch.append(remaining_inputs[:batch_size])
        remaining_inputs = remaining_inputs[batch_size:]
    if len(remaining_inputs)>0:
        input_batch.append(remaining_inputs)  # append remaining training examples
    return input_batch

cpdef str win_lookup(int index):
    """'
     For value net result lookup
    ''"""
    cdef str win
    if index == 0:
        win = 'Win'
    else:
        win = 'Lose'
    return win

cpdef int index_lookup_by_move (str move):
    cdef dict move_dict
    move_dict = { #white moves
        'a1-a2': 0, 'a1-b2': 1, 'a2-a3': 22, 'a2-b3': 23, 'a3-a4': 44, 'a3-b4': 45, 'a4-a5': 66, 'a4-b5': 67,
        'a5-a6': 88, 'a5-b6': 89, 'a6-a7': 110, 'a6-b7': 111, 'a7-a8': 132, 'a7-b8': 133, 'b1-a2': 2, 'b1-b2': 3,
        'b1-c2': 4, 'b2-a3': 24, 'b2-b3': 25, 'b2-c3': 26, 'b3-a4': 46, 'b3-b4': 47, 'b3-c4': 48, 'b4-a5': 68,
        'b4-b5': 69, 'b4-c5': 70, 'b5-a6': 90, 'b5-b6': 91, 'b5-c6': 92, 'b6-a7': 112, 'b6-b7': 113, 'b6-c7': 114,
        'b7-a8': 134, 'b7-b8': 135, 'b7-c8': 136, 'c1-b2': 5, 'c1-c2': 6, 'c1-d2': 7, 'c2-b3': 27, 'c2-c3': 28,
        'c2-d3': 29, 'c3-b4': 49, 'c3-c4': 50, 'c3-d4': 51, 'c4-b5': 71, 'c4-c5': 72, 'c4-d5': 73, 'c5-b6': 93,
        'c5-c6': 94, 'c5-d6': 95, 'c6-b7': 115, 'c6-c7': 116, 'c6-d7': 117, 'c7-b8': 137, 'c7-c8': 138, 'c7-d8': 139,
        'd1-c2': 8, 'd1-d2': 9, 'd1-e2': 10, 'd2-c3': 30, 'd2-d3': 31, 'd2-e3': 32, 'd3-c4': 52, 'd3-d4': 53,
        'd3-e4': 54, 'd4-c5': 74, 'd4-d5': 75, 'd4-e5': 76, 'd5-c6': 96, 'd5-d6': 97, 'd5-e6': 98, 'd6-c7': 118,
        'd6-d7': 119, 'd6-e7': 120, 'd7-c8': 140, 'd7-d8': 141, 'd7-e8': 142, 'e1-d2': 11, 'e1-e2': 12, 'e1-f2': 13,
        'e2-d3': 33, 'e2-e3': 34, 'e2-f3': 35, 'e3-d4': 55, 'e3-e4': 56, 'e3-f4': 57, 'e4-d5': 77, 'e4-e5': 78,
        'e4-f5': 79, 'e5-d6': 99, 'e5-e6': 100, 'e5-f6': 101, 'e6-d7': 121, 'e6-e7': 122, 'e6-f7': 123, 'e7-d8': 143,
        'e7-e8': 144, 'e7-f8': 145, 'f1-e2': 14, 'f1-f2': 15, 'f1-g2': 16, 'f2-e3': 36, 'f2-f3': 37, 'f2-g3': 38,
        'f3-e4': 58, 'f3-f4': 59, 'f3-g4': 60, 'f4-e5': 80, 'f4-f5': 81, 'f4-g5': 82, 'f5-e6': 102, 'f5-f6': 103,
        'f5-g6': 104, 'f6-e7': 124, 'f6-f7': 125, 'f6-g7': 126, 'f7-e8': 146, 'f7-f8': 147, 'f7-g8': 148, 'g1-f2': 17,
        'g1-g2': 18, 'g1-h2': 19, 'g2-f3': 39, 'g2-g3': 40, 'g2-h3': 41, 'g3-f4': 61, 'g3-g4': 62, 'g3-h4': 63,
        'g4-f5': 83, 'g4-g5': 84, 'g4-h5': 85, 'g5-f6': 105, 'g5-g6': 106, 'g5-h6': 107, 'g6-f7': 127, 'g6-g7': 128,
        'g6-h7': 129, 'g7-f8': 149, 'g7-g8': 150, 'g7-h8': 151, 'h1-g2': 20, 'h1-h2': 21, 'h2-g3': 42, 'h2-h3': 43,
        'h3-g4': 64, 'h3-h4': 65, 'h4-g5': 86, 'h4-h5': 87, 'h5-g6': 108, 'h5-h6': 109, 'h6-g7': 130, 'h6-h7': 131,
        'h7-g8': 152, 'h7-h8': 153,

        # black moves#
        'a2-a1': 153, 'a2-b1': 152, 'a3-a2': 131, 'a3-b2': 130, 'a4-a3': 109, 'a4-b3': 108, 'a5-a4': 87, 'a5-b4': 86,
        'a6-a5': 65, 'a6-b5': 64, 'a7-a6': 43, 'a7-b6': 42, 'a8-a7': 21, 'a8-b7': 20, 'b2-a1': 151, 'b2-b1': 150,
        'b2-c1': 149, 'b3-a2': 129, 'b3-b2': 128, 'b3-c2': 127, 'b4-a3': 107, 'b4-b3': 106, 'b4-c3': 105, 'b5-a4': 85,
        'b5-b4': 84, 'b5-c4': 83, 'b6-a5': 63, 'b6-b5': 62, 'b6-c5': 61, 'b7-a6': 41, 'b7-b6': 40, 'b7-c6': 39,
        'b8-a7': 19, 'b8-b7': 18, 'b8-c7': 17, 'c2-b1': 148, 'c2-c1': 147, 'c2-d1': 146, 'c3-b2': 126, 'c3-c2': 125,
        'c3-d2': 124, 'c4-b3': 104, 'c4-c3': 103, 'c4-d3': 102, 'c5-b4': 82, 'c5-c4': 81, 'c5-d4': 80, 'c6-b5': 60,
        'c6-c5': 59, 'c6-d5': 58, 'c7-b6': 38, 'c7-c6': 37, 'c7-d6': 36, 'c8-b7': 16, 'c8-c7': 15, 'c8-d7': 14,
        'd2-c1': 145, 'd2-d1': 144, 'd2-e1': 143, 'd3-c2': 123, 'd3-d2': 122, 'd3-e2': 121, 'd4-c3': 101, 'd4-d3': 100,
        'd4-e3': 99, 'd5-c4': 79, 'd5-d4': 78, 'd5-e4': 77, 'd6-c5': 57, 'd6-d5': 56, 'd6-e5': 55, 'd7-c6': 35,
        'd7-d6': 34, 'd7-e6': 33, 'd8-c7': 13, 'd8-d7': 12, 'd8-e7': 11, 'e2-d1': 142, 'e2-e1': 141, 'e2-f1': 140,
        'e3-d2': 120, 'e3-e2': 119, 'e3-f2': 118, 'e4-d3': 98, 'e4-e3': 97, 'e4-f3': 96, 'e5-d4': 76, 'e5-e4': 75,
        'e5-f4': 74, 'e6-d5': 54, 'e6-e5': 53, 'e6-f5': 52, 'e7-d6': 32, 'e7-e6': 31, 'e7-f6': 30, 'e8-d7': 10,
        'e8-e7': 9, 'e8-f7': 8, 'f2-e1': 139, 'f2-f1': 138, 'f2-g1': 137, 'f3-e2': 117, 'f3-f2': 116, 'f3-g2': 115,
        'f4-e3': 95, 'f4-f3': 94, 'f4-g3': 93, 'f5-e4': 73, 'f5-f4': 72, 'f5-g4': 71, 'f6-e5': 51, 'f6-f5': 50,
        'f6-g5': 49, 'f7-e6': 29, 'f7-f6': 28, 'f7-g6': 27, 'f8-e7': 7, 'f8-f7': 6, 'f8-g7': 5, 'g2-f1': 136,
        'g2-g1': 135, 'g2-h1': 134, 'g3-f2': 114, 'g3-g2': 113, 'g3-h2': 112, 'g4-f3': 92, 'g4-g3': 91, 'g4-h3': 90,
        'g5-f4': 70, 'g5-g4': 69, 'g5-h4': 68, 'g6-f5': 48, 'g6-g5': 47, 'g6-h5': 46, 'g7-f6': 26, 'g7-g6': 25,
        'g7-h6': 24, 'g8-f7': 4, 'g8-g7': 3, 'g8-h7': 2, 'h2-g1': 133, 'h2-h1': 132, 'h3-g2': 111, 'h3-h2': 110,
        'h4-g3': 89, 'h4-h3': 88, 'h5-g4': 67, 'h5-h4': 66, 'h6-g5': 45, 'h6-h5': 44, 'h7-g6': 23, 'h7-h6': 22,
        'h8-g7': 1, 'h8-h7': 0, 'no-move': 154}
    return move_dict[move.lower()]



cpdef str move_lookup_by_index(int index, str player_color):
    """'
    Enumerated the moves for lookup speed/visual reference (see commented out dictionary).
    Code can be prettified by calling generate_move_lookup instead
    ''"""

    cdef list transitions
    if player_color == 'White':
         transitions = ['a1-a2', 'a1-b2', 'b1-a2', 'b1-b2', 'b1-c2', 'c1-b2', 'c1-c2', 'c1-d2', 'd1-c2', 'd1-d2',
                       'd1-e2', 'e1-d2', 'e1-e2', 'e1-f2', 'f1-e2', 'f1-f2', 'f1-g2', 'g1-f2', 'g1-g2', 'g1-h2',
                       'h1-g2', 'h1-h2', 'a2-a3', 'a2-b3', 'b2-a3', 'b2-b3', 'b2-c3', 'c2-b3', 'c2-c3', 'c2-d3',
                       'd2-c3', 'd2-d3', 'd2-e3', 'e2-d3', 'e2-e3', 'e2-f3', 'f2-e3', 'f2-f3', 'f2-g3', 'g2-f3',
                       'g2-g3', 'g2-h3', 'h2-g3', 'h2-h3', 'a3-a4', 'a3-b4', 'b3-a4', 'b3-b4', 'b3-c4', 'c3-b4',
                       'c3-c4', 'c3-d4', 'd3-c4', 'd3-d4', 'd3-e4', 'e3-d4', 'e3-e4', 'e3-f4', 'f3-e4', 'f3-f4',
                       'f3-g4', 'g3-f4', 'g3-g4', 'g3-h4', 'h3-g4', 'h3-h4', 'a4-a5', 'a4-b5', 'b4-a5', 'b4-b5',
                       'b4-c5', 'c4-b5', 'c4-c5', 'c4-d5', 'd4-c5', 'd4-d5', 'd4-e5', 'e4-d5', 'e4-e5', 'e4-f5',
                       'f4-e5', 'f4-f5', 'f4-g5', 'g4-f5', 'g4-g5', 'g4-h5', 'h4-g5', 'h4-h5', 'a5-a6', 'a5-b6',
                       'b5-a6', 'b5-b6', 'b5-c6', 'c5-b6', 'c5-c6', 'c5-d6', 'd5-c6', 'd5-d6', 'd5-e6', 'e5-d6',
                       'e5-e6', 'e5-f6', 'f5-e6', 'f5-f6', 'f5-g6', 'g5-f6', 'g5-g6', 'g5-h6', 'h5-g6', 'h5-h6',
                       'a6-a7', 'a6-b7', 'b6-a7', 'b6-b7', 'b6-c7', 'c6-b7', 'c6-c7', 'c6-d7', 'd6-c7', 'd6-d7',
                       'd6-e7', 'e6-d7', 'e6-e7', 'e6-f7', 'f6-e7', 'f6-f7', 'f6-g7', 'g6-f7', 'g6-g7', 'g6-h7',
                       'h6-g7', 'h6-h7', 'a7-a8', 'a7-b8', 'b7-a8', 'b7-b8', 'b7-c8', 'c7-b8', 'c7-c8', 'c7-d8',
                       'd7-c8', 'd7-d8', 'd7-e8', 'e7-d8', 'e7-e8', 'e7-f8', 'f7-e8', 'f7-f8', 'f7-g8', 'g7-f8',
                       'g7-g8', 'g7-h8', 'h7-g8', 'h7-h8', 'no-move']
#Visual guide
        #              0: 'a1-a2',
        #              1: 'a1-b2',
        #              2: 'b1-a2',
        #              3: 'b1-b2',
        #              4: 'b1-c2',
        #              5: 'c1-b2',
        #              6: 'c1-c2',
        #              7: 'c1-d2',
        #              8: 'd1-c2',
        #              9: 'd1-d2',
        #              10: 'd1-e2',
        #              11: 'e1-d2',
        #              12: 'e1-e2',
        #              13: 'e1-f2',
        #              14: 'f1-e2',
        #              15: 'f1-f2',
        #              16: 'f1-g2',
        #              17: 'g1-f2',
        #              18: 'g1-g2',
        #              19: 'g1-h2',
        #              20: 'h1-g2',
        #              21: 'h1-h2',
        #              22: 'a2-a3',
        #              23: 'a2-b3',
        #              24: 'b2-a3',
        #              25: 'b2-b3',
        #              26: 'b2-c3',
        #              27: 'c2-b3',
        #              28: 'c2-c3',
        #              29: 'c2-d3',
        #              30: 'd2-c3',
        #              31: 'd2-d3',
        #              32: 'd2-e3',
        #              33: 'e2-d3',
        #              34: 'e2-e3',
        #              35: 'e2-f3',
        #              36: 'f2-e3',
        #              37: 'f2-f3',
        #              38: 'f2-g3',
        #              39: 'g2-f3',
        #              40: 'g2-g3',
        #              41: 'g2-h3',
        #              42: 'h2-g3',
        #              43: 'h2-h3',
        #              44: 'a3-a4',
        #              45: 'a3-b4',
        #              46: 'b3-a4',
        #              47: 'b3-b4',
        #              48: 'b3-c4',
        #              49: 'c3-b4',
        #              50: 'c3-c4',
        #              51: 'c3-d4',
        #              52: 'd3-c4',
        #              53: 'd3-d4',
        #              54: 'd3-e4',
        #              55: 'e3-d4',
        #              56: 'e3-e4',
        #              57: 'e3-f4',
        #              58: 'f3-e4',
        #              59: 'f3-f4',
        #              60: 'f3-g4',
        #              61: 'g3-f4',
        #              62: 'g3-g4',
        #              63: 'g3-h4',
        #              64: 'h3-g4',
        #              65: 'h3-h4',
        #              66: 'a4-a5',
        #              67: 'a4-b5',
        #              68: 'b4-a5',
        #              69: 'b4-b5',
        #              70: 'b4-c5',
        #              71: 'c4-b5',
        #              72: 'c4-c5',
        #              73: 'c4-d5',
        #              74: 'd4-c5',
        #              75: 'd4-d5',
        #              76: 'd4-e5',
        #              77: 'e4-d5',
        #              78: 'e4-e5',
        #              79: 'e4-f5',
        #              80: 'f4-e5',
        #              81: 'f4-f5',
        #              82: 'f4-g5',
        #              83: 'g4-f5',
        #              84: 'g4-g5',
        #              85: 'g4-h5',
        #              86: 'h4-g5',
        #              87: 'h4-h5',
        #              88: 'a5-a6',
        #              89: 'a5-b6',
        #              90: 'b5-a6',
        #              91: 'b5-b6',
        #              92: 'b5-c6',
        #              93: 'c5-b6',
        #              94: 'c5-c6',
        #              95: 'c5-d6',
        #              96: 'd5-c6',
        #              97: 'd5-d6',
        #              98: 'd5-e6',
        #              99: 'e5-d6',
        #              100: 'e5-e6',
        #              101: 'e5-f6',
        #              102: 'f5-e6',
        #              103: 'f5-f6',
        #              104: 'f5-g6',
        #              105: 'g5-f6',
        #              106: 'g5-g6',
        #              107: 'g5-h6',
        #              108: 'h5-g6',
        #              109: 'h5-h6',
        #              110: 'a6-a7',
        #              111: 'a6-b7',
        #              112: 'b6-a7',
        #              113: 'b6-b7',
        #              114: 'b6-c7',
        #              115: 'c6-b7',
        #              116: 'c6-c7',
        #              117: 'c6-d7',
        #              118: 'd6-c7',
        #              119: 'd6-d7',
        #              120: 'd6-e7',
        #              121: 'e6-d7',
        #              122: 'e6-e7',
        #              123: 'e6-f7',
        #              124: 'f6-e7',
        #              125: 'f6-f7',
        #              126: 'f6-g7',
        #              127: 'g6-f7',
        #              128: 'g6-g7',
        #              129: 'g6-h7',
        #              130: 'h6-g7',
        #              131: 'h6-h7',
        #              132: 'a7-a8',
        #              133: 'a7-b8',
        #              134: 'b7-a8',
        #              135: 'b7-b8',
        #              136: 'b7-c8',
        #              137: 'c7-b8',
        #              138: 'c7-c8',
        #              139: 'c7-d8',
        #              140: 'd7-c8',
        #              141: 'd7-d8',
        #              142: 'd7-e8',
        #              143: 'e7-d8',
        #              144: 'e7-e8',
        #              145: 'e7-f8',
        #              146: 'f7-e8',
        #              147: 'f7-f8',
        #              148: 'f7-g8',
        #              149: 'g7-f8',
        #              150: 'g7-g8',
        #              151: 'g7-h8',
        #              152: 'h7-g8',
        #              153: 'h7-h8',
        #              154: 'no-move'

    elif player_color.lower() == 'Black'.lower():
        transitions = ['h8-h7', 'h8-g7', 'g8-h7', 'g8-g7', 'g8-f7', 'f8-g7', 'f8-f7', 'f8-e7', 'e8-f7', 'e8-e7',
                       'e8-d7', 'd8-e7', 'd8-d7', 'd8-c7', 'c8-d7', 'c8-c7', 'c8-b7', 'b8-c7', 'b8-b7', 'b8-a7',
                       'a8-b7', 'a8-a7', 'h7-h6', 'h7-g6', 'g7-h6', 'g7-g6', 'g7-f6', 'f7-g6', 'f7-f6', 'f7-e6',
                       'e7-f6', 'e7-e6', 'e7-d6', 'd7-e6', 'd7-d6', 'd7-c6', 'c7-d6', 'c7-c6', 'c7-b6', 'b7-c6',
                       'b7-b6', 'b7-a6', 'a7-b6', 'a7-a6', 'h6-h5', 'h6-g5', 'g6-h5', 'g6-g5', 'g6-f5', 'f6-g5',
                       'f6-f5', 'f6-e5', 'e6-f5', 'e6-e5', 'e6-d5', 'd6-e5', 'd6-d5', 'd6-c5', 'c6-d5', 'c6-c5',
                       'c6-b5', 'b6-c5', 'b6-b5', 'b6-a5', 'a6-b5', 'a6-a5', 'h5-h4', 'h5-g4', 'g5-h4', 'g5-g4',
                       'g5-f4', 'f5-g4', 'f5-f4', 'f5-e4', 'e5-f4', 'e5-e4', 'e5-d4', 'd5-e4', 'd5-d4', 'd5-c4',
                       'c5-d4', 'c5-c4', 'c5-b4', 'b5-c4', 'b5-b4', 'b5-a4', 'a5-b4', 'a5-a4', 'h4-h3', 'h4-g3',
                       'g4-h3', 'g4-g3', 'g4-f3', 'f4-g3', 'f4-f3', 'f4-e3', 'e4-f3', 'e4-e3', 'e4-d3', 'd4-e3',
                       'd4-d3', 'd4-c3', 'c4-d3', 'c4-c3', 'c4-b3', 'b4-c3', 'b4-b3', 'b4-a3', 'a4-b3', 'a4-a3',
                       'h3-h2', 'h3-g2', 'g3-h2', 'g3-g2', 'g3-f2', 'f3-g2', 'f3-f2', 'f3-e2', 'e3-f2', 'e3-e2',
                       'e3-d2', 'd3-e2', 'd3-d2', 'd3-c2', 'c3-d2', 'c3-c2', 'c3-b2', 'b3-c2', 'b3-b2', 'b3-a2',
                       'a3-b2', 'a3-a2', 'h2-h1', 'h2-g1', 'g2-h1', 'g2-g1', 'g2-f1', 'f2-g1', 'f2-f1', 'f2-e1',
                       'e2-f1', 'e2-e1', 'e2-d1', 'd2-e1', 'd2-d1', 'd2-c1', 'c2-d1', 'c2-c1', 'c2-b1', 'b2-c1',
                       'b2-b1', 'b2-a1', 'a2-b1', 'a2-a1', 'no-move']
#Visual guide

        #               0: 'h8-h7',
        #               1: 'h8-g7',
        #               2: 'g8-h7',
        #               3: 'g8-g7',
        #               4: 'g8-f7',
        #               5: 'f8-g7',
        #               6: 'f8-f7',
        #               7: 'f8-e7',
        #               8: 'e8-f7',
        #               9: 'e8-e7',
        #               10: 'e8-d7',
        #               11: 'd8-e7',
        #               12: 'd8-d7',
        #               13: 'd8-c7',
        #               14: 'c8-d7',
        #               15: 'c8-c7',
        #               16: 'c8-b7',
        #               17: 'b8-c7',
        #               18: 'b8-b7',
        #               19: 'b8-a7',
        #               20: 'a8-b7',
        #               21: 'a8-a7',
        #               22: 'h7-h6',
        #               23: 'h7-g6',
        #               24: 'g7-h6',
        #               25: 'g7-g6',
        #               26: 'g7-f6',
        #               27: 'f7-g6',
        #               28: 'f7-f6',
        #               29: 'f7-e6',
        #               30: 'e7-f6',
        #               31: 'e7-e6',
        #               32: 'e7-d6',
        #               33: 'd7-e6',
        #               34: 'd7-d6',
        #               35: 'd7-c6',
        #               36: 'c7-d6',
        #               37: 'c7-c6',
        #               38: 'c7-b6',
        #               39: 'b7-c6',
        #               40: 'b7-b6',
        #               41: 'b7-a6',
        #               42: 'a7-b6',
        #               43: 'a7-a6',
        #               44: 'h6-h5',
        #               45: 'h6-g5',
        #               46: 'g6-h5',
        #               47: 'g6-g5',
        #               48: 'g6-f5',
        #               49: 'f6-g5',
        #               50: 'f6-f5',
        #               51: 'f6-e5',
        #               52: 'e6-f5',
        #               53: 'e6-e5',
        #               54: 'e6-d5',
        #               55: 'd6-e5',
        #               56: 'd6-d5',
        #               57: 'd6-c5',
        #               58: 'c6-d5',
        #               59: 'c6-c5',
        #               60: 'c6-b5',
        #               61: 'b6-c5',
        #               62: 'b6-b5',
        #               63: 'b6-a5',
        #               64: 'a6-b5',
        #               65: 'a6-a5',
        #               66: 'h5-h4',
        #               67: 'h5-g4',
        #               68: 'g5-h4',
        #               69: 'g5-g4',
        #               70: 'g5-f4',
        #               71: 'f5-g4',
        #               72: 'f5-f4',
        #               73: 'f5-e4',
        #               74: 'e5-f4',
        #               75: 'e5-e4',
        #               76: 'e5-d4',
        #               77: 'd5-e4',
        #               78: 'd5-d4',
        #               79: 'd5-c4',
        #               80: 'c5-d4',
        #               81: 'c5-c4',
        #               82: 'c5-b4',
        #               83: 'b5-c4',
        #               84: 'b5-b4',
        #               85: 'b5-a4',
        #               86: 'a5-b4',
        #               87: 'a5-a4',
        #               88: 'h4-h3',
        #               89: 'h4-g3',
        #               90: 'g4-h3',
        #               91: 'g4-g3',
        #               92: 'g4-f3',
        #               93: 'f4-g3',
        #               94: 'f4-f3',
        #               95: 'f4-e3',
        #               96: 'e4-f3',
        #               97: 'e4-e3',
        #               98: 'e4-d3',
        #               99: 'd4-e3',
        #               100: 'd4-d3',
        #               101: 'd4-c3',
        #               102: 'c4-d3',
        #               103: 'c4-c3',
        #               104: 'c4-b3',
        #               105: 'b4-c3',
        #               106: 'b4-b3',
        #               107: 'b4-a3',
        #               108: 'a4-b3',
        #               109: 'a4-a3',
        #               110: 'h3-h2',
        #               111: 'h3-g2',
        #               112: 'g3-h2',
        #               113: 'g3-g2',
        #               114: 'g3-f2',
        #               115: 'f3-g2',
        #               116: 'f3-f2',
        #               117: 'f3-e2',
        #               118: 'e3-f2',
        #               119: 'e3-e2',
        #               120: 'e3-d2',
        #               121: 'd3-e2',
        #               122: 'd3-d2',
        #               123: 'd3-c2',
        #               124: 'c3-d2',
        #               125: 'c3-c2',
        #               126: 'c3-b2',
        #               127: 'b3-c2',
        #               128: 'b3-b2',
        #               129: 'b3-a2',
        #               130: 'a3-b2',
        #               131: 'a3-a2',
        #               132: 'h2-h1',
        #               133: 'h2-g1',
        #               134: 'g2-h1',
        #               135: 'g2-g1',
        #               136: 'g2-f1',
        #               137: 'f2-g1',
        #               138: 'f2-f1',
        #               139: 'f2-e1',
        #               140: 'e2-f1',
        #               141: 'e2-e1',
        #               142: 'e2-d1',
        #               143: 'd2-e1',
        #               144: 'd2-d1',
        #               145: 'd2-c1',
        #               146: 'c2-d1',
        #               147: 'c2-c1',
        #               148: 'c2-b1',
        #               149: 'b2-c1',
        #               150: 'b2-b1',
        #               151: 'b2-a1',
        #               152: 'a2-b1',
        #               153: 'a2-a1',
        #               154: 'no-move'

    else:
        print("ERROR: Please specify a valid player color", file=stderr)
        exit(10)
    return transitions[index]
#

def generate_move_lookup():
    """'
    generates the lookup tables for move_look_by_index and index_lookup_by_move 
    ''"""#
    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    white_transitions = []
    for k in range (1, 9): #white's moves
        if k != 8:
            for i in range(0, len(chars)):
                if i == 0:
                    white_transitions.append(chars[i]+str(k)+'-'+chars[i]+str(k+1))
                    white_transitions.append(chars[i] + str(k) + '-' + chars[i+1] + str(k + 1))
                elif i == len(chars)-1:
                    white_transitions.append(chars[i] + str(k) + '-' + chars[i-1] + str(k + 1))
                    white_transitions.append(chars[i] + str(k) + '-' + chars[i] + str(k + 1))
                else:
                    white_transitions.append(chars[i] + str(k) + '-' + chars[i - 1] + str(k + 1))
                    white_transitions.append(chars[i] + str(k) + '-' + chars[i] + str(k + 1))
                    white_transitions.append(chars[i] + str(k) + '-' + chars[i+1] + str(k + 1))
    black_transitions = []
    for k in range (8, 0, -1): #black's moves
        if k != 1:
            for i in range(len(chars)-1, -1, -1):
                if i == 0:
                    black_transitions.append(chars[i] + str(k) + '-' + chars[i+1] + str(k - 1))
                    black_transitions.append(chars[i]+str(k)+'-'+chars[i]+str(k-1))
                elif i == len(chars)-1:
                    black_transitions.append(chars[i] + str(k) + '-' + chars[i] + str(k - 1))
                    black_transitions.append(chars[i] + str(k) + '-' + chars[i-1] + str(k - 1))
                else:
                    black_transitions.append(chars[i] + str(k) + '-' + chars[i+1] + str(k - 1))
                    black_transitions.append(chars[i] + str(k) + '-' + chars[i] + str(k - 1))
                    black_transitions.append(chars[i] + str(k) + '-' + chars[i - 1] + str(k - 1))
    white_transitions.append('no-move')
    black_transitions.append('no-move')
    # return dict([(key[1], key[0]) for key in enumerate(white_transitions)]), dict([(key[1], key[0]) for key in enumerate(black_transitions)]) # for index lookup by move
    return white_transitions, black_transitions

cpdef dict initial_game_board():
    """'
    generates an initial game board
    ''"""#
    cdef:
        int from_white = -1
        int is_white = 1
        str empty = 'e'
        str white = 'w'
        str black = 'b'
        dict gameboard= {
            10: from_white,  # (-1 for initial state, 0 if black achieved state, 1 if white achieved state)
            # equivalent to 0 if white's move, 1 if black's move
            9: is_white,  # is player_color white
            8: {'a': black, 'b': black, 'c': black, 'd': black, 'e': black, 'f': black, 'g': black, 'h': black},
            7: {'a': black, 'b': black, 'c': black, 'd': black, 'e': black, 'f': black, 'g': black, 'h': black},
            6: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
            5: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
            4: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
            3: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
            2: {'a': white, 'b': white, 'c': white, 'd': white, 'e': white, 'f': white, 'g': white, 'h': white},
            1: {'a': white, 'b': white, 'c': white, 'd': white, 'e': white, 'f': white, 'g': white, 'h': white}
        }
    return gameboard




cpdef np.ndarray convert_board_to_2d_matrix_POEB(dict game_board, str player_color):
    """'
    converts a game board to player POV representations and converts that into the corresponding neural network representation
    returns the neural network input representation of the game board
    ''"""#
    # Note: not the same as reflect_board_state in self_play_logs_to_datastructures;
    cdef:
        # the board state; state[1] is the win or loss_init value, state [2] is the transition vector
        int is_white_index = 9
        int white_move_index = 10
        int row
        str column
        dict mirror_state
        dict row1
        dict row2
        dict row3
        dict row4
    if player_color == 'Black':
        mirror_state = initial_game_board() # edit copy of board_state

        for row in sorted(game_board):
            if row != is_white_index and row != white_move_index:  # these indexes don't change
                for column in sorted(game_board[row]):
                    if column == 'a':
                        mirror_state[row]['h'] = game_board[row][column]
                    elif column == 'b':
                        mirror_state[row]['g'] = game_board[row][column]
                    elif column == 'c':
                        mirror_state[row]['f'] = game_board[row][column]
                    elif column == 'd':
                        mirror_state[row]['e'] = game_board[row][column]
                    elif column == 'e':
                        mirror_state[row]['d'] = game_board[row][column]
                    elif column == 'f':
                        mirror_state[row]['c'] = game_board[row][column]
                    elif column == 'g':
                        mirror_state[row]['b'] = game_board[row][column]
                    elif column == 'h':
                        mirror_state[row]['a'] = game_board[row][column]

        row1 = {'a': mirror_state[1]['a'], 'b': mirror_state[1]['b'], 'c': mirror_state[1]['c'], 'd':mirror_state[1]['d'], 'e': mirror_state[1]['e'], 'f':mirror_state[1]['f'], 'g': mirror_state[1]['g'], 'h': mirror_state[1]['h']}# mirror_state[1].copy()
        row2 = {'a': mirror_state[2]['a'], 'b': mirror_state[2]['b'], 'c': mirror_state[2]['c'], 'd':mirror_state[2]['d'], 'e': mirror_state[2]['e'], 'f':mirror_state[2]['f'], 'g': mirror_state[2]['g'], 'h': mirror_state[2]['h']}# mirror_state[2].copy()
        row3 = {'a': mirror_state[3]['a'], 'b': mirror_state[3]['b'], 'c': mirror_state[3]['c'], 'd':mirror_state[3]['d'], 'e': mirror_state[3]['e'], 'f':mirror_state[3]['f'], 'g': mirror_state[3]['g'], 'h': mirror_state[3]['h']}#mirror_state[3].copy()
        row4 = {'a': mirror_state[4]['a'], 'b': mirror_state[4]['b'], 'c': mirror_state[4]['c'], 'd':mirror_state[4]['d'], 'e': mirror_state[4]['e'], 'f':mirror_state[4]['f'], 'g': mirror_state[4]['g'], 'h': mirror_state[4]['h']}#mirror_state[4].copy()


        mirror_state[1] = mirror_state[8]
        mirror_state[2] = mirror_state[7]
        mirror_state[3] = mirror_state[6]
        mirror_state[4] = mirror_state[5]
        mirror_state[5] = row4
        mirror_state[6] = row3
        mirror_state[7] = row2
        mirror_state[8] = row1
        game_board =  mirror_state#reflect_board_state(game_board)

    return generate_binary_plane_POEB(game_board, player_color)


def generate_transition_vector(to, _from, player_color):
    """'
    returns a one hot vector with the index corresponding to the transition (from-to) having a value of one
    ''"""#
    # probability distribution over the 155 possible (vs legal) moves from the POV of the player.
    # Reasoning: six center columns where, if a piece was present, it could move one of three ways.
    # A piece in one of the two side columns can move one of two ways.
    # Since nothing can move in the farthest row, there are only seven rows of possible movement.
    # => (2*2*7) + (6*3*7) = 154; 154 +1 for no move
    # ==> 155 element vector of all 0s sans the 1 for the transition that was actually made.
    # i.e. a1-a2 (if White) == h8-h7 (if Black) =>
    # row 0 (closest row), column 0(farthest left)
    # moves to
    # row +1, column 0
    # <=> transition[0] = 1, transition[1:len(transition)] = 0

    # Notes: when calling NN, just reverse board state if black and decode output with black's table


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

def generate_binary_vector(state, player_color, what_to_filter):
    """'
    converts a game board to a one-hot vector representation
    returns the one-hot vector representation of the game board
    ''"""#
    bias = 1
    binary_vector = []
    is_white_index = 9
    white_move_index = 10
    if what_to_filter == 'White' or what_to_filter == 'Black':
        what_to_filter_dict = {
        'White': {
            'e': 0,
            'w': 1,
            'b': 0},
        'Black': {
            'e': 0,
            'w': 0,
            'b': 1}}
    elif what_to_filter == 'Player':
        what_to_filter_dict = {
            'White':{
                'e': 0,
                'w': 1,
                'b': 0},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 1}}
    elif what_to_filter == 'Opponent':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 1},
            'Black': {
                'e': 0,
                'w': 1,
                'b': 0}}
    elif what_to_filter == 'Empty':
        what_to_filter_dict = {
            'White':{
                'e': 1,
                'w': 0,
                'b': 0},
            'Black': {
                'e': 1,
                'w': 0,
                'b': 0}}
    elif what_to_filter == 'Capture Move':
        what_to_filter_dict = {
            'White': {
                'e': 1,
                'w': 1,
                'b': 1},
            'Black': {
                'e': 1,
                'w': 1,
                'b': 1}}
    elif what_to_filter == 'Non-Capture Move':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 0},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 0}}
    elif what_to_filter == 'Moves From':
        what_to_filter_dict = {
            'White':{
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 1,
                'Player Capture To': 0,
                'Player Capture From': 0},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 1,
                'Player Capture To': 0,
                'Player Capture From': 0}}
    elif what_to_filter == 'Moves To':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 1,
                'Player Move From': 0,
                'Player Capture To': 0,
                'Player Capture From': 0}  ,
            'Black': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 1,
                'Player Move From': 0,
                'Player Capture To': 0,
                'Player Capture From': 0}}
    elif what_to_filter == 'Captures From':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 0,
                'Player Capture To': 0,
                'Player Capture From': 1},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 0,
                'Player Capture To': 0,
                'Player Capture From': 1}}
    elif what_to_filter == 'Captures To':
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 0,
                'Player Capture To': 1,
                'Player Capture From': 0},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 0,
                'Player Capture To': 1,
                'Player Capture From': 0}}
    elif what_to_filter == 'Bias':  # duplicate across 64 positions since CNN needs same dimensions
        what_to_filter_dict = {
            'White': {
                'e': bias,
                'w': bias,
                'b': bias},
            'Black': {
                'e': bias,
                'w': bias,
                'b': bias}}
    else:
        print("Error, generate_binary_vector needs a valid argument to filter")
        what_to_filter_dict = {
            'White': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 0,
                'Player Capture To': 0,
                'Player Capture From': 0},
            'Black': {
                'e': 0,
                'w': 0,
                'b': 0,
                'Player Move To': 0,
                'Player Move From': 0,
                'Player Capture To': 0,
                'Player Capture From': 0}}
    for row in sorted(state):
        if row != is_white_index and row != white_move_index:  # don't touch these indexes
            for column in sorted(state[row]):  # needs to be sorted to traverse dictionary in lexicographical order
                binary_vector.append(what_to_filter_dict[player_color][state[row][column]])
    return binary_vector

cpdef np.ndarray generate_binary_plane_POEB(dict state, str player_color):
    """'
    converts a game board to player POV representations and converts that into the corresponding neural network representation
    returns the neural network input representation of the game board
    ''"""#
    #TODO consider reshaping for C++ input; could also put it into 3d matrix on the fly,
    # ex. if player == board[i][j], X[n][i][j] = [1, 0, 0, 1]
    cdef:
        int bias = 1
        np.ndarray binary_plane = np.array([[[0,0,0,bias]]*8]*8, dtype=DTYPE)

        int is_white_index = 9
        int white_move_index = 10
        #feature plane index
        int player_index = 0
        int opponent_index = 1
        int empty_index = 2

        str empty = 'e'
        int plane_row = 0
        int plane_column
        str player
        str opponent
        str board_square


    if player_color == 'White':
        player = 'w'
        opponent = 'b'
    else:
        player = 'b'
        opponent = 'w'

    for row in sorted(state):
        if row != is_white_index and row != white_move_index:  # don't touch these indexes
            plane_column = 0
            for column in sorted(state[row]):  # needs to be sorted to traverse dictionary in lexicographical order

                board_square = state[row][column]
                if board_square ==player:
                    binary_plane[plane_row][plane_column][player_index] = 1
                elif board_square == opponent:
                    binary_plane[plane_row][plane_column][opponent_index] = 1
                elif board_square == empty:
                    binary_plane[plane_row][plane_column][empty_index] = 1
                plane_column +=1
            plane_row +=1
    return binary_plane

