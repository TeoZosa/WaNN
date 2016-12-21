import re as re  # regular expressions
import pprint  # pretty printing
import os, fnmatch  # to retrieve file information from path
import pickle  # serialize the data structure
import mmap #read entire files into memory for (only for workstation)
import copy
import math
from psutil import virtual_memory

def ProcessDirectoryOfBreakthroughFiles(path, playerList):
    for selfPlayGames in FindFiles(path, '*.txt'):
        playerList.append(ProcessBreakthroughFile(path, selfPlayGames))

def ProcessBreakthroughFile(path, selfPlayGames):
    fileName = selfPlayGames[
               len(path):len(selfPlayGames) - len('.txt')]  # trim path & extension
    fileName = fileName.split('_SelfPlayLog')  # BreakthroughN_SelfPlayLog00-> ['BreakthroughN',00]
    serverName = str(fileName[0].strip('\\'))
    selfPlayLog = str(fileName[1]).strip('(').strip(')')
    dateRange = str(selfPlayGames[
                    len(r'G:\TruncatedLogs') + 1:len(path) - len(r'\selfPlayLogsBreakthroughN')])
    gamesList, whiteWins, blackWins = FormatGameList(selfPlayGames, serverName)
    return {'ServerNode': serverName, 'selfPlayLog': selfPlayLog, 'dateRange': dateRange, 'Games': gamesList, 'WhiteWins': whiteWins, 'BlackWins': blackWins}

def WriteToDisk(input, path):
    date = str(path [
                    len(r'G:\TruncatedLogs') + 1:len(path) - len(r'\selfPlayLogsBreakthroughN')])
    outputFile = open(path + r'DataPython.p', 'wb')
    pickle.dump(input, outputFile)

def FindFiles(path, filter):  # recursively find files at path with filter extension; pulled from StackOverflow
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            yield os.path.join(root, file) #TODO: change to chunk based read for speed

def FormatGameList(selfPlayGames, serverName):
    games = []
    blackWin = None
    whiteWin = None
    endRegex = re.compile(r'.* End')
    startRegex = re.compile(r'.* Start')
    moveRegex = re.compile(r'\*play (.*)')
    blackWinRegex = re.compile(r'Black Wins:.*')
    whiteWinRegex = re.compile(r'White Wins:.*')
    numWhiteWins = 0
    numBlackWins = 0
    file = open(selfPlayGames, "r+b")# read in file
    file = mmap.mmap(file.fileno(),length=0, access= mmap.ACCESS_READ)#prot=PROT_READ only in Unix
    #iterate over list of the form:
    #Game N Start
    #...
    #(Black|White) Wins: \d
    #[Game N End]
    moveList = []
    while True:
        #line = line.decode('utf-8')
        line = file.readline().decode('utf-8')#convert to string
        if line=='':break#EOF
        if moveRegex.match(line):#put plays into move list
            moveList.append(moveRegex.search(line).group(1))
        elif blackWinRegex.match(line):
            blackWin = True
            whiteWin = False
        elif whiteWinRegex.match(line):
            whiteWin = True
            blackWin = False
        elif endRegex.match(line):
            #move list through function
            moveList, webVisualizerLink = FormatMoveList(moveList)
            whiteBoardStates = GenerateBoardStates(moveList, "White", whiteWin)  # generate board states from moveList
            blackBoardStates = GenerateBoardStates(moveList, "Black", blackWin)
            if whiteWin:
                numWhiteWins += 1
            elif blackWin:
                numBlackWins += 1
            games.append({'Win': whiteWin,
                          'Moves': moveList,
                          'BoardStates': whiteBoardStates,
                          'VisualizationURL': webVisualizerLink})  # append new white game
            games.append({'Win': blackWin,
                          'Moves': moveList,
                          'BoardStates': blackBoardStates,
                          'VisualizationURL': webVisualizerLink})  # append new black game
            moveList = []
            whiteWin = None #not necessary;redundant, but good practice
            blackWin = None
    file.close()
    return games, numWhiteWins, numBlackWins

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
    moveRegex = re.compile(r"[W|B]\s([a-h]\d.[a-h]\d)",
                           re.IGNORECASE)
    webVisualizerLink = r'http://www.trmph.com/breakthrough/board#8,'
    moveList = list(map(lambda a: moveRegex.search(a).group(1), moveListString))
    moveNum = 0
    newMoveList=[]
    move = [None] * 3
    for i in range(0, len(moveList)):
        moveNum += 1
        move[0] = math.ceil(moveNum/2)
        From = str(moveList[i][0:2]).lower()
        to = str(moveList[i][3:5]).lower()
        if i % 2 == 0:#white move
            assert(moveList[i][1] < moveList[i][4])#white should go forward
            move[1] = {'From': From, 'To': to}  # set White's moves
            webVisualizerLink = webVisualizerLink + From + to
            if i==len(moveList)-1:#white makes last move of game; black lost
                move[2] = "NIL"
                newMoveList.append({'#': move[0], 'White': move[1], 'Black': move[2]})
        else:#black move
            assert (moveList[i][1] > moveList[i][4])#black should go backward
            move[2] = {'From': From, 'To': to}# set Black's moves
            webVisualizerLink = webVisualizerLink + From + to
            newMoveList.append({'#': move[0], 'White': move[1], 'Black': move[2]})
    return newMoveList, webVisualizerLink

def Driver(path):
    playerList = []
    ProcessDirectoryOfBreakthroughFiles(path, playerList)
    WriteToDisk(playerList, path)



