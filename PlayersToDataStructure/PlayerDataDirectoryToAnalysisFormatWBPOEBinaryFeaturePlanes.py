import re as re  # regular expressions
import pprint  # pretty printing
import os, fnmatch  # to retrieve file information from path
import pickle  # serialize the data structure
import copy

def ProcessDirectoryOfBreakthroughFiles(path, playerList):
    for playerGameHistoryData in FindFiles(path, '*.txt'):
        playerList.append(ProcessBreakthroughFile(path, playerGameHistoryData))

def ProcessBreakthroughFile(path, playerGameHistoryData):
    fileName = playerGameHistoryData[
               len(path):len(playerGameHistoryData) - len('.txt')]  # trim path & extension
    fileName = fileName.split('間')  # user間2000間687687 -> ['user',2000, 687687]
    playerName = str(fileName[0])
    playerID = int(fileName[2])
    rank = int(fileName[1])
    gamesList, numWins, numLosses = FormatGameList(playerGameHistoryData, playerName)
    return {'Player': playerName, 'PlayerID': playerID, 'Rank': rank, 'Games': gamesList, 'Wins': numWins, 'Losses': numLosses}

def WriteToDisk(input, path):
    outputFile = open(path + r'PlayerDataBinaryFeaturesWBPOEDatasetTruncatedMF.p', 'wb')
    pickle.dump(input, outputFile)

def FindFiles(path, filter):  # recursively find files at path with filter extension; pulled from StackOverflow
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            yield os.path.join(root, file)

def PreprocessGamesList(playerGameHistoryData):  #normalized regex/iterable friendly list
    gamesList = [y[1] for y in list(
        enumerate([x.strip() for x in open(playerGameHistoryData, "r")]))]  # read in file and convert to list
    gamesList = filter(None, gamesList)  # remove empty strings from list
    gamesList = list(filter(lambda a: a != "[Site \"www.littlegolem.net\"]", gamesList))  # remove site from list
    return gamesList

def FormatGameList(playerGameHistoryData, playerName):
    quotesRegex = re.compile(r'"(.*)"')
    eventEntry = 0
    whiteEntry = 1
    blackEntry = 2
    resultEntry = 3
    moveEntry = 4
    games = []
    gamesList = PreprocessGamesList(playerGameHistoryData)
    numWins = 0
    numLosses = 0
    # flags to indicate if something wasn't set properly
    opponentName = None
    event = None
    playerColor = None
    opponentColor = None
    win = None
                     # format game list
    for j in range(0, len(gamesList)):
        thisRow = j % 5
        if thisRow != moveEntry:
            rowData = quotesRegex.search(gamesList[j]).group(1)
        if thisRow == eventEntry:
            # [Event "Tournament null"] -> Event: 'Tournament null'
            event = rowData
        elif thisRow == whiteEntry:
            if playerName.lower() == rowData.lower():  # ignore case just in case (no pun intended)
                playerColor = 'White'
                opponentColor = 'Black'
            else:
                opponentName = rowData
                playerColor = 'Black'
                opponentColor = 'White'
        elif thisRow == blackEntry:
            # assignment case handled above
            if playerName.lower() != rowData.lower():
                opponentName = rowData
        elif thisRow == resultEntry:
            #
            if playerColor == 'White':
                if rowData[0] == '1':
                    win = True
                elif rowData[0] == '0':
                    win = False
                elif rowData[0] == '*':
                    win = "Game In Progress"
            elif playerColor == 'Black':
                if rowData[0] == '0':
                    win = True
                elif rowData[0] == '1':
                    win = False
                elif rowData[0] == '*':
                    win = "Game In Progress"
            else:
                print("UNEXPECTED DATA FORMAT")
                win = "Undefined at line " + str(j)
        elif thisRow == moveEntry:
            # format move list
            moveList = FormatMoveList(gamesList[j])
            boardStates = GenerateBoardStates(moveList, playerColor, win)  # generate board states from moveList
            assert (playerColor != opponentColor and opponentName != playerName)
            if len(moveList) > 3 and boardStates['Win'] != "Game In Progress":
                #non-spurrious games, remove if statement for all games.
                if win == True:
                    numWins += 1
                elif win == False:
                    numLosses += 1

                games.append({'Event': event, 'PlayerColor': playerColor, 'OpponentColor': opponentColor,
                          'OpponentName': opponentName, 'Win': win,
                          'Moves': moveList, 'BoardStates': boardStates})  # append new game after formatting move list
    return games, numWins, numLosses


def GenerateBoardStates(moveList, playerColor, win):
    if win == "Game In Progress":
        return {'Win': win, 'States': []}
        #for human readability version
    empty = 'e'
    white = 'w'
    black = 'b'
    if playerColor == 'White':
        isWhite = 1
    else:
        isWhite = 0
        # win/loss 'value' symmetrical
    if win == True:
        win = 1
    elif win == False:
        win = -1
    state = [
        {
        10: -1,
         9: isWhite,
         8: {'a': black, 'b': black, 'c': black, 'd': black, 'e': black, 'f': black, 'g': black, 'h': black},
         7: {'a': black, 'b': black, 'c': black, 'd': black, 'e': black, 'f': black, 'g': black, 'h': black},
         6: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
         5: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
         4: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
         3: {'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
         2: {'a': white, 'b': white, 'c': white, 'd': white, 'e': white, 'f': white, 'g': white, 'h': white},
         1: {'a': white, 'b': white, 'c': white, 'd': white, 'e': white, 'f': white, 'g': white, 'h': white}
         }, win]# 9: is playerColor white, 10: did White's move achieve this state (-1 for a for initial state, 0 for if black achieved this state)
    mirrorState = MirrorBoardState(state)
    # boardStates = {'Win': win, 'States': [], 'MirrorStates': []}#exclude start state

    #Original start state should not be useful for tree search since it is the root of every game and has no parent.
    #however, may provide useful bias if starting player matters (i.e. 2nd player wins in solved versions (Saffidine, Jouandeau, & Cazenave, 2011)
    boardStates = {'Win': win, 'States': [state], 'MirrorStates': [mirrorState]}# including original start state

    for i in range(0, len(moveList)):
        assert (moveList[i]['#'] == i + 1)
        if isinstance(moveList[i]['White'], dict):  # if string, then == resign or NIL
            state = [MovePiece(state[0], moveList[i]['White']['To'], moveList[i]['White']['From'], whoseMove='White'), win]
            boardStates['States'].append(state)
            mirrorState = MirrorBoardState(state)
            boardStates['MirrorStates'].append(mirrorState)
        if isinstance(moveList[i]['Black'], dict):  # if string, then == resign or NIL
            state= [MovePiece(state[0], moveList[i]['Black']['To'], moveList[i]['Black']['From'], whoseMove='Black'), win]
            boardStates['States'].append(state)
            mirrorState = MirrorBoardState(state)
            boardStates['MirrorStates'].append(mirrorState)
            # for data transformation; inefficient to essentially compute board states twice, but more error-proof
    boardStates = ConvertBoardStatesToArrays(boardStates, playerColor)
    return boardStates

def MirrorBoardState(state):#since a mirror image has the same strategic value
    mirrorStateWithWin = copy.deepcopy(state)  # edit copy of boardState
    mirrorState = mirrorStateWithWin[0]
    state = state[0] #the board state; state[1] is the win or loss value
    isWhiteIndex = 9
    whiteMoveIndex = 10
    for row in sorted(state):
        if row != isWhiteIndex and row != whiteMoveIndex:  #don't touch the index that shows whose move generated this state
            for column in sorted(state[row]):
                if column == 'a':
                    mirrorState[row]['h'] = state[row][column]
                elif column == 'b':
                    mirrorState[row]['g'] = state[row][column]
                elif column == 'c':
                    mirrorState[row]['f'] = state[row][column]
                elif column == 'd':
                    mirrorState[row]['e'] = state[row][column]
                elif column == 'e':
                    mirrorState[row]['d'] = state[row][column]
                elif column == 'f':
                    mirrorState[row]['c'] = state[row][column]
                elif column == 'g':
                    mirrorState[row]['b'] = state[row][column]
                elif column == 'h':
                    mirrorState[row]['a'] = state[row][column]
    return mirrorStateWithWin

def ConvertBoardStatesToArrays(boardStates, playerColor):
    newBoardStates = boardStates
    states = boardStates['States']
    mirrorStates = boardStates['MirrorStates']
    assert len(states) == len(mirrorStates)#vacuous assertion
    newBoardStates['States'] = []
    newBoardStates['MirrorStates'] = []
    for i in range (0, len (states)):
        newBoardStates['States'].append(ConvertBoardTo1DArray(states[i], playerColor))
        newBoardStates['MirrorStates'].append(ConvertBoardTo1DArray(mirrorStates[i], playerColor))
    return newBoardStates

def ConvertBoardTo1DArray(boardState, playerColor):
    state = boardState[0]
    isWhiteIndex = 9
    whiteMoveIndex = 10
    oneDArray = []
    #if player color == white, player and white states are mirrors; else, player and black states are mirrors
    GenerateBinaryPlane(state, arrayToAppend=oneDArray, playerColor=playerColor, whoToFilter='White')#0-63 white
    GenerateBinaryPlane(state, arrayToAppend=oneDArray, playerColor=playerColor, whoToFilter='Black')#64-127 black
    GenerateBinaryPlane(state, arrayToAppend=oneDArray, playerColor=playerColor, whoToFilter='Player')# 128-191 black
    GenerateBinaryPlane(state, arrayToAppend=oneDArray, playerColor=playerColor, whoToFilter='Opponent')# 192-255 black
    GenerateBinaryPlane(state, arrayToAppend=oneDArray, playerColor=playerColor, whoToFilter='Empty')#256-319 empty
    # TODO: remove since this is now redundant? i.e. if position ==white and Player
    # whiteFlag = [state[isWhiteIndex]]*64
    # oneDArray += whiteFlag#320-383 is a flag indicating if the player is white

    moveFlag = [state[whiteMoveIndex]]#*64
    oneDArray+= moveFlag #320-383 is a flag indicating if the move came from a white move
    for i in range (0, 64):
        assert (oneDArray[i]^oneDArray[i+64]^oneDArray[i+256])#ensure at most 1 bit is on at each board position for white/black/empty
        assert (oneDArray[i+128] ^ oneDArray[i + 192] ^ oneDArray[i + 256])  # ensure at most 1 bit is on at each board position for player/opponent/empty
        if playerColor == 'White':
            #player == white positions and opponent = black positions;
            assert (oneDArray[i] == oneDArray[i+128] and oneDArray[i+64] == oneDArray[i+192])
        else:
            #player == black positions and opponent = white positions;
            assert (oneDArray[i] == oneDArray[i+192] and oneDArray[i+64] == oneDArray[i+128])

    newBoardState = [oneDArray, boardState[1]]  # [x vector, y scalar]
    return newBoardState

def GenerateBinaryPlane(state, arrayToAppend, playerColor, whoToFilter):
    isWhiteIndex = 9
    whiteMoveIndex = 10
    if whoToFilter == 'White':
        for row in sorted(state):
            if row != isWhiteIndex and row != whiteMoveIndex:  # don't touch the index that shows whose move generated this state
                for column in sorted(state[row]):
                    # needs to be sorted to traverse dictionary in lexicographical order
                    value = -5
                    if state[row][column] == 'e':
                        value = 0
                    elif state[row][column] == 'w':
                        value = 1
                    elif state[row][column] == 'b':
                        value = 0
                    else:
                        print("error in convertBoard")
                        exit(-190)
                    arrayToAppend.append(value)
    elif whoToFilter == 'Black':
        for row in sorted(state):
            if row != isWhiteIndex and row!= whiteMoveIndex:  # don't touch the index that shows whose move generated this state
                for column in sorted(state[row]):
                    # needs to be sorted to traverse dictionary in lexicographical order
                    value = -5
                    if state[row][column] == 'e':
                        value = 0
                    elif state[row][column] == 'w':
                        value = 0
                    elif state[row][column] == 'b':
                        value = 1
                    else:
                        print("error in convertBoard")
                        exit(-190)
                    arrayToAppend.append(value)
    elif whoToFilter == 'Player':
        for row in sorted(state):
            if row != isWhiteIndex and row != whiteMoveIndex:  # don't touch the index that shows whose move generated this state
                for column in sorted(state[row]):
                    # needs to be sorted to traverse dictionary in lexicographical order
                    value = -5
                    if state[row][column] == 'e':
                        value = 0
                    elif state[row][column] == 'w':
                        if playerColor == 'White':
                            value = 1
                        else:
                            value = 0
                    elif state[row][column] == 'b':
                        if playerColor == 'Black':
                            value = 1
                        else:
                            value = 0
                    else:
                        print("error in convertBoard")
                        exit(-190)
                    arrayToAppend.append(value)
    elif whoToFilter == 'Opponent':
        for row in sorted(state):
            if row != isWhiteIndex and row != whiteMoveIndex:  # don't touch the index that shows whose move generated this state
                for column in sorted(state[row]):
                    # needs to be sorted to traverse dictionary in lexicographical order
                    value = -5
                    if state[row][column] == 'e':
                        value = 0
                    elif state[row][column] == 'w':
                        if playerColor == 'White':
                            value = 0
                        else:
                            value = 1
                    elif state[row][column] == 'b':
                        if playerColor == 'Black':
                            value = 0
                        else:
                            value = 1
                    else:
                        print("error in convertBoard")
                        exit(-190)
                    arrayToAppend.append(value)
    elif whoToFilter == 'Empty':
        for row in sorted(state):
            if row != isWhiteIndex and row != whiteMoveIndex:  # don't touch the index that shows whose move generated this state
                for column in sorted(state[row]):
                    # needs to be sorted to traverse dictionary in lexicographical order
                    value = -5
                    if state[row][column] == 'e':
                        value = 1
                    elif state[row][column] == 'w' or state[row][column] == 'b':
                        value = 0
                    else:
                        print("error in convertBoard")
                        exit(-190)
                    arrayToAppend.append(value)
    else:
        print("Error, GenerateBinaryPlane needs a valid argument to filter")

def MovePiece(boardState, To, From, whoseMove):
    empty = 'e'
    whiteMoveIndex = 10
    nextBoardState = copy.deepcopy(boardState)  # edit copy of boardState
    nextBoardState[int(To[1])][To[0]] = nextBoardState[int(From[1])][From[0]]
    nextBoardState[int(From[1])][From[0]] = empty
    if whoseMove == 'White':
        nextBoardState[whiteMoveIndex] = 1
    else:
        nextBoardState[whiteMoveIndex] = 0
    return nextBoardState


def FormatMoveList(moveListString):
    moveRegex = re.compile(r'(\d+)\.\s(resign|[a-h]\d.[a-h]\d)\s(resign|[a-h]\d.[a-h]\d|\d-\d)',
                           re.IGNORECASE)
    moveList = moveRegex.findall(moveListString)
    for i in range(0, len(moveList)):
        move = list(moveList[i])
        move[0] = int(move[0])
        assert (move[0] == i + 1)
        if move[1] == "resign":
            move[2] = "NIL"
        else:
            move[1] = {'From': move[1][0:2], 'To': move[1][3:len(move[1])]}  # set White's moves
        if move[2] != "resign" and move[2] != "NIL":  # set Black's moves
            if len(move[2]) > 3:
                move[2] = {'From': move[2][0:2], 'To': move[2][3:len(move[2])]}
            else:
                move[2] = "NIL"
        moveList[i] = {'#': move[0], 'White': move[1], 'Black': move[2]}
    return moveList


                #main script
playerList = []
pathToCheck = r'/Users/TeofiloZosa/BreakthroughData/AutomatedData/'
ProcessDirectoryOfBreakthroughFiles(pathToCheck, playerList)
WriteToDisk(playerList, pathToCheck)

# Verified Working.
# #double check
#pathToCheck2 = r'/Users/TeofiloZosa/BreakthroughData/'
# newList = pickle.load(open(pathToCheck+r'PlayerDataPython.p', 'rb'))
# oldList = pickle.load(open(pathToCheck2+r'PlayerDataPython.p', 'rb'))
# assert (playerList == newList == oldList)


