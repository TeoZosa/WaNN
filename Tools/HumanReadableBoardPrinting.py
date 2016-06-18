import pprint
def PrintHumanReadableBoardConfiguration(boardState):
    index = 8
    while index > 0:  # print row in descending order
        PrintRow(boardState, index)
        index -= 1
def PrintAnalysisFormat(boardState):
    print("\n")  # print extra newline for readability
    for index in range (0, 64):
       if index%8 ==0:
         print("\n")  # print extra newline for readability
         #print(str(index) + ": ", end="")
       print(boardState[index], end="")
def PrintRow(boardState, index):
    if index == 8:
        print("\n")  # print extra newline for readability
    print(str(index) + ": ", end="")
    pprint.pprint(boardState[index])
def PrintTransitions(game, stateNumber):
    if stateNumber % 2 == 0:  # white move
        print("White Moves: ", end="")
        pprint.pprint(game['Moves'][int(stateNumber / 2)]['White'])
    else:  # black move
        print("Black Moves: ", end="")
        pprint.pprint(game['Moves'][int(stateNumber / 2)]['Black'])
def PrintHumanReadableBoardStatesWithTransitions(game):
    board = game['BoardStates']['States']
    for stateNumber in range (0, len(board)):
        PrintHumanReadableBoardConfiguration(board[stateNumber])
        PrintTransitions(game, stateNumber)
