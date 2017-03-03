#NOTE: deprecated. way easier to use a pandas dataframe

import pprint
def PrintGame(game):
    board = game['BoardStates']['States']
    for stateNumber in range(0, len(board)):
        PrintTypicalBoardConfiguration(board[stateNumber][0])
        PrintTransitions(game, stateNumber)
def PrintTypicalBoardConfiguration(boardState):
    index = 8
    while index > 0:  # print row in descending order
        PrintTypicalRow(boardState, index)
        index -= 1
    print('   a b c d e f g h')

def PrintTypicalRow(boardState,index):
    if index == 8:
        print("\n")  # print extra newline for readability
    print(str(index) + ": ", end="")
    for i in sorted(boardState[index]):
        if boardState[index][i] == 'e':
            print('- ', end="")
        else:
            print(boardState[index][i]+' ', end="")
    print('\n')

def PrintHumanReadableBoardConfiguration(boardState):
    index = 8
    while index > 0:  # print row in descending order
        PrintDataStructurestyleRow(boardState, index)
        index -= 1

def PrintAnalysisFormat(boardState):
    print("\n")  # print extra newline for readability
    for index in range (0, 64):
       if index%8 ==0:
         print("\n")  # print extra newline for readability
         print(str(index) + ": ", end="")
       print(boardState[index], end="")

def PrintDataStructurestyleRow(boardState, index):
    if index == 8:
        print("\n")  # print extra newline for readability
    print(str(index) + ": ", end="")
    pprint.pprint(boardState[index])


def PrintTransitions(game, stateNumber):
    moves = game['Moves']
    nextMove = int(stateNumber/2)
    if stateNumber % 2 == 0:  # white move
        print("White Moves: ", end="")
        pprint.pprint(moves[nextMove]['White'])
    else:  # black move
        print("Black Moves: ", end="")
        pprint.pprint(moves[nextMove]['Black'])
def PrintHumanReadableBoardStatesWithTransitions(game):
    board = game['BoardStates']['States']
    for stateNumber in range (0, len(board)):
        PrintHumanReadableBoardConfiguration(board[stateNumber][0])
        PrintTransitions(game, stateNumber)
