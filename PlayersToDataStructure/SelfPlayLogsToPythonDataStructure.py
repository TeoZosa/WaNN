import re as re  # regular expressions
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
    rootDir = 'G:\TruncatedLogs\PythonDatasets\Datastructures\\'
    date = str(path [
                    len(r'G:\TruncatedLogs')+1:len(path) - len(r'\selfPlayLogsBreakthroughN')])
    outputFile = open(rootDir +
                      date + #prepend data to name of server
                      path[len(path) - len(r'\selfPlayLogsBreakthroughN')+1:len(path)] #name of the server
                      + r'DataPython.p', 'wb')#append data qualifier
    pickle.dump(input, outputFile)

def FindFiles(path, filter):  # recursively find files at path with filter extension; pulled from StackOverflow
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            yield os.path.join(root, file)

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
    unformattedMoveList = []
    while True:
        line = file.readline().decode('utf-8')#convert to string
        if line=='':break#EOF
        if moveRegex.match(line):#put plays into move list
            unformattedMoveList.append(moveRegex.search(line).group(1))
        elif blackWinRegex.match(line):
            blackWin = True
            whiteWin = False
        elif whiteWinRegex.match(line):
            whiteWin = True
            blackWin = False
        elif endRegex.match(line):
            #Format move list
            moveList, mirrorMoveList, originalWebVisualizerLink, mirrorWebVisualizerLink = FormatMoveListsAndURLs(unformattedMoveList)
            whiteBoardStates = GenerateBoardStates(moveList, 'White', whiteWin)  # generate board states from moveList
            whiteMirrorBoardStates = GenerateBoardStates(mirrorMoveList, 'White', whiteWin)
            blackBoardStates = GenerateBoardStates(moveList,'Black', blackWin)  # self-play => same states, but win under policy for A=> lose under policy for B
            blackMirrorBoardStates = GenerateBoardStates(mirrorMoveList, 'Black', blackWin)
            if whiteWin:
                numWhiteWins += 1
            elif blackWin:
                numBlackWins += 1
            games.append({'Win': whiteWin,
                          'Moves': moveList,
                          'MirrorMoves': mirrorMoveList,
                          'BoardStates': whiteBoardStates,
                          'MirrorBoardStates': whiteMirrorBoardStates,
                          'OriginalVisualizationURL': originalWebVisualizerLink,
                          'MirrorVisualizationURL': mirrorWebVisualizerLink}
                         )  # append new white game
            games.append({'Win': blackWin,
                          'Moves': moveList,
                          'MirrorMoves': mirrorMoveList,
                          'BoardStates': blackBoardStates,
                          'MirrorBoardStates': blackMirrorBoardStates,
                          'OriginalVisualizationURL': originalWebVisualizerLink,
                          'MirrorVisualizationURL': mirrorWebVisualizerLink}
                         )  # append new black game
            unformattedMoveList = []#reset moveList for next game
            whiteWin = None #not necessary;redundant, but good practice
            blackWin = None
    file.close()
    return games, numWhiteWins, numBlackWins

def InitialState(moveList, playerColor, win):
    empty = 'e'
    white = 'w'
    black = 'b'
    if playerColor == 'White':
        isWhite = 1
    else:
        isWhite = 0
    return [
        {
            10: -1,  # did White's move achieve this state (-1 for a for initial state, 0 for if black achieved this state)
            9: isWhite,  # is playerColor white
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
        generateTransitionVector(moveList[0]['White']['To'], moveList[0]['White']['From'], 'White')]#White's opening move

def GenerateBoardStates(moveList, playerColor, win):
    state = InitialState(moveList,playerColor,win)
    if playerColor == 'White':
       playerPOV = [state]
    else:
       playerPOV = []
    boardStates = {'Win': win, 'States': [state], 'PlayerPOV': playerPOV}
    for i in range(0, len(moveList)):
        assert (moveList[i]['#'] == i + 1)
        #structure kept for symmetry
        if isinstance(moveList[i]['White'], dict):  #for self-play, this should always happen.
            if isinstance(moveList[i]['Black'], dict):  # if no black move => white won
                blackTransitionVector = generateTransitionVector(moveList[i]['Black']['To'], moveList[i]['Black']['From'], 'Black')
                #can't put black move block in here as it would execute before white's move
            else:
                blackTransitionVector = [0]*154
            state = [MovePiece(state[0], moveList[i]['White']['To'], moveList[i]['White']['From'], whoseMove='White'),
                     win,
                     blackTransitionVector] #Black's response to the generated state
            if playerColor == 'Black': #reflect positions tied to black transitions
               boardStates['PlayerPOV'].append(ReflectBoardState(state))
            boardStates['States'].append(state)
        if isinstance(moveList[i]['Black'], dict):  # if string, then == resign or NIL
            if i+1 == len(moveList):# if no next white move => black won
                whiteTransitionVector = [0]*154 # no white move from the next generated state
            else:
                whiteTransitionVector = generateTransitionVector(moveList[i+1]['White']['To'], moveList[i+1]['White']['From'], 'White')
            state = [MovePiece(state[0], moveList[i]['Black']['To'], moveList[i]['Black']['From'], whoseMove='Black'),
                     win,
                     whiteTransitionVector]  # White's response to the generated state
            boardStates['States'].append(state)
            if playerColor == 'White':
                boardStates['PlayerPOV'].append(state)
    # for data transformation; inefficient to essentially compute board states twice, but more error-proof
    boardStates = ConvertBoardStatesToArrays(boardStates, playerColor)
    return boardStates

def generateTransitionVector(to, From, playerColor):
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
    fromColumn = From[0]
    toColumn = to[0]
    fromRow = int(From[1])
    toRow = int(to[1])
    columnOffset = (ord(fromColumn) - ord('a')) * 3 #ex if white and fromColumn = b=> 1*3; moves starting from b are [2] or [3] or [4];makes sense in context of formula
    if playerColor == 'Black':
        rowOffset = (toRow - 1) * 22 #22 possible moves per row
        assert (rowOffset == (fromRow - 2) * 22)  # double check
        index = 153 - (ord(toColumn) - ord(fromColumn) + columnOffset + rowOffset)#153 reverses the board for black
    else:
        rowOffset = (fromRow - 1) * 22 #22 possible moves per row
        assert (rowOffset == (toRow - 2) * 22)  # double check
        index = ord(toColumn) - ord(fromColumn) + columnOffset + rowOffset
    transitionVector = [0] * 154
    transitionVector[index] = 1
    return transitionVector

def MirrorMove(move):
    mirrorMove = copy.deepcopy(move)
    whiteTo = move['White']['To']
    whiteFrom = move['White']['From']
    whiteFromColumn = whiteFrom[0]
    whiteToColumn = whiteTo[0]
    whiteFromRow = int(whiteFrom[1])
    whiteToRow = int(whiteTo[1])
    mirrorMove['White']['To'] = MirrorColumn(whiteToColumn) + str(whiteToRow)
    mirrorMove['White']['From'] = MirrorColumn(whiteFromColumn) + str(whiteFromRow)

    if isinstance(move['Black'], dict):
        blackTo = move['Black']['To']
        blackFrom = move['Black']['From']
        blackFromColumn = blackFrom[0]
        blackToColumn = blackTo[0]
        blackFromRow = int(blackFrom[1])
        blackToRow = int(blackTo[1])
        mirrorMove['Black']['To'] = MirrorColumn(blackToColumn) + str(blackToRow)
        mirrorMove['Black']['From'] = MirrorColumn(blackFromColumn) + str(blackFromRow)
    #else 'Black' == NIL, don't change it
    return mirrorMove

def MirrorColumn(columnChar):
    mirrorDict ={'a': 'h',
                 'b': 'g',
                 'c': 'f',
                 'd': 'e',
                 'e': 'd',
                 'f': 'c',
                 'g': 'b',
                 'h': 'a'
                 }
    return mirrorDict[columnChar]

def ReflectBoardState(state):#since black needs to have a POV representation
  semiReflectedState = MirrorBoardState(state)
  reflectedState = copy.deepcopy(semiReflectedState)
  reflectedState[0][1] = semiReflectedState[0][8]
  reflectedState[0][2] = semiReflectedState[0][7]
  reflectedState[0][3] = semiReflectedState[0][6]
  reflectedState[0][4] = semiReflectedState[0][5]
  reflectedState[0][5] = semiReflectedState[0][4]
  reflectedState[0][6] = semiReflectedState[0][3]
  reflectedState[0][7] = semiReflectedState[0][2]
  reflectedState[0][8] = semiReflectedState[0][1]
  return reflectedState

def MirrorBoardState(state):#since a mirror image has the same strategic value
    mirrorStateWithWin = copy.deepcopy(state)  # edit copy of boardState
    mirrorState = mirrorStateWithWin[0]
    state = state[0] #the board state; state[1] is the win or loss value, state [2] is the transition vector
    isWhiteIndex = 9
    whiteMoveIndex = 10
    for row in sorted(state):
        if row != isWhiteIndex and row != whiteMoveIndex:  #these indexes don't change
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
    POVStates = boardStates['PlayerPOV']
    states = boardStates['States']
    newBoardStates['States'] = []
    newBoardStates['PlayerPOV'] = []
    for state in states:
        newBoardStates['States'].append(ConvertBoardTo1DArray(state, playerColor)) #These can be inputs to value net
    for POVState in POVStates:
        newBoardStates['PlayerPOV'].append(ConvertBoardTo1DArrayPolicyNet(POVState, playerColor)) #these can be inputs to policy net
    return newBoardStates

def ConvertBoardTo1DArray(boardState, playerColor):
    state = boardState[0]
    isWhiteIndex = 9
    whiteMoveIndex = 10
    oneHotBoard = []

    # do we need White/Black if black is a flipped representation? we are only asking the net "from my POV, these are my pieces and my opponent's pieces what move should I take?
    #if not, the net sees homogenous data. My fear is seeing black in a flipped representation may mess things up as, if you are white, black being in a lower row is not good
    #but would the Player/Opponent features cancel this out?

    #if player color == white, player and white states are mirrors; else, player and black states are mirrors
    oneHotBoard.append(GenerateBinaryPlane(state, arrayToAppend=[], playerColor=playerColor, whoToFilter='White'))#[0] white
    oneHotBoard.append(GenerateBinaryPlane(state, arrayToAppend=[], playerColor=playerColor, whoToFilter='Black'))#[1] black
    oneHotBoard.append(GenerateBinaryPlane(state, arrayToAppend=[], playerColor=playerColor, whoToFilter='Player'))#[2]player
    oneHotBoard.append(GenerateBinaryPlane(state, arrayToAppend=[], playerColor=playerColor, whoToFilter='Opponent'))# [3] opponent
    oneHotBoard.append(GenerateBinaryPlane(state, arrayToAppend=[], playerColor=playerColor, whoToFilter='Empty'))#[4] empty
    moveFlag = [state[whiteMoveIndex]]*64 #duplicate across 64 features since CNN needs same dimensions
    oneHotBoard.append(moveFlag) #320-383 is a flag indicating if the transition came from a white move
    for i in range (0, 64):  #error checking block
        assert (oneHotBoard[0][i]^oneHotBoard[1][i]^oneHotBoard[4][i])#ensure at most 1 bit is on at each board position for white/black/empty
        assert (oneHotBoard[2][i] ^ oneHotBoard[3][i] ^ oneHotBoard[4][i])  # ensure at most 1 bit is on at each board position for player/opponent/empty
        if playerColor == 'White':
            #white positions == player and black positions == opponent;
            assert (oneHotBoard[0][i] == oneHotBoard[2][i] and oneHotBoard[1][i] == oneHotBoard[3][i])
        else:
            #  white positions == opponent and player == black positions;
            assert (oneHotBoard[0][i] == oneHotBoard[3][i] and oneHotBoard[1][i] == oneHotBoard[2][i])
    newBoardState = [oneHotBoard, boardState[1], boardState[2]]  # [x vector, win, y transition vector]
    return newBoardState

def ConvertBoardTo1DArrayPolicyNet(boardState, playerColor):# 12/22 removed White/Black to avoid Curse of Dimensionality.
  state = boardState[0]
  oneHotBoard = []
  oneHotBoard.append(GenerateBinaryPlane(state, arrayToAppend=[], playerColor=playerColor, whoToFilter='Player'))#[0] player
  oneHotBoard.append(GenerateBinaryPlane(state, arrayToAppend=[], playerColor=playerColor, whoToFilter='Opponent'))#[1] opponent
  oneHotBoard.append(GenerateBinaryPlane(state, arrayToAppend=[], playerColor=playerColor, whoToFilter='Empty')) # [2] empty
  for i in range(0, 64):  # error checking block
    assert (oneHotBoard[0][i] ^ oneHotBoard[1][i] ^ oneHotBoard[2][i])  # ensure at most 1 bit is on at each board position for player/opponent/empty
  newBoardState = [oneHotBoard, boardState[1], boardState[2]]  # [x vector, win, y transition vector]
  return newBoardState


def GenerateBinaryPlane(state, arrayToAppend, playerColor, whoToFilter):
    isWhiteIndex = 9
    whiteMoveIndex = 10
    oneHotIndexes = []
    if whoToFilter == 'White':
        whiteDict = {
            'e': 0,
            'w': 1,
            'b': 0}
        for row in sorted(state):
            if row != isWhiteIndex and row != whiteMoveIndex:  # don't touch the index that shows whose move generated this state
                for column in sorted(state[row]): # needs to be sorted to traverse dictionary in lexicographical order
                    arrayToAppend.append(whiteDict[state[row][column]])
    elif whoToFilter == 'Black':
        blackDict = {
            'e': 0,
            'w': 0,
            'b': 1}
        for row in sorted(state):
            if row != isWhiteIndex and row != whiteMoveIndex:  # don't touch the index that shows whose move generated this state
                for column in sorted(state[row]): # needs to be sorted to traverse dictionary in lexicographical order
                    arrayToAppend.append(blackDict[state[row][column]])
    elif whoToFilter == 'Player':
        if playerColor == 'White':
            playerDict = {
            'e': 0,
            'w': 1,
            'b': 0}
        elif playerColor == 'Black':
            playerDict = {
                'e': 0,
                'w': 0,
                'b': 1}
        else:
            print("error in convertBoard")
            exit(-190)
        for row in sorted(state):
            if row != isWhiteIndex and row != whiteMoveIndex:  # don't touch the index that shows whose move generated this state
                for column in sorted(state[row]): # needs to be sorted to traverse dictionary in lexicographical order
                    arrayToAppend.append(playerDict[state[row][column]])
    elif whoToFilter == 'Opponent':
        if playerColor == 'White':
            opponentDict = {
            'e': 0,
            'w': 0,
            'b': 1}
        elif playerColor == 'Black':
            opponentDict = {
                'e': 0,
                'w': 1,
                'b': 0}
        else:
            print("error in convertBoard")
            exit(-190)
        for row in sorted(state):
            if row != isWhiteIndex and row != whiteMoveIndex:  # don't touch the index that shows whose move generated this state
                for column in sorted(state[row]): # needs to be sorted to traverse dictionary in lexicographical order
                    arrayToAppend.append(opponentDict[state[row][column]])
    elif whoToFilter == 'Empty':
        emptyDict = {
            'e': 1,
            'w': 0,
            'b': 0}
        for row in sorted(state):
            if row != isWhiteIndex and row != whiteMoveIndex:  # don't touch the index that shows whose move generated this state
                for column in sorted(state[row]): # needs to be sorted to traverse dictionary in lexicographical order
                    arrayToAppend.append(emptyDict[state[row][column]])
    else:
        print("Error, GenerateBinaryPlane needs a valid argument to filter")
    return arrayToAppend

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

def FormatMoveListsAndURLs(unformattedMoveList):
    moveRegex = re.compile(r"[W|B]\s([a-h]\d.[a-h]\d)",
                           re.IGNORECASE)
    originalWebVisualizerLink = mirrorWebVisualizerLink = r'http://www.trmph.com/breakthrough/board#8,'
    moveList = list(map(lambda a: moveRegex.search(a).group(1), unformattedMoveList))
    moveNum = 0
    newMoveList=[]
    newMirrorMoveList=[]
    move = [None] * 3
    for i in range(0, len(moveList)):
        moveNum += 1
        move[0] = math.ceil(moveNum/2)
        From = str(moveList[i][0:2]).lower()
        to = str(moveList[i][3:5]).lower()
        fromColumn = From[0]
        fromRow = From[1]
        toColumn = to[0]
        toRow = to[1]
        mirrorFrom = MirrorColumn(fromColumn)+fromRow
        mirrorTo = MirrorColumn(toColumn)+toRow
        if i % 2 == 0:#white move
            assert(fromRow < toRow)#white should go forward
            move[1] = {'From': From, 'To': to}  # set White's moves
            if i==len(moveList)-1:#white makes last move of game; black lost
                move[2] = "NIL"
                tempMove = {'#': move[0], 'White': move[1], 'Black': move[2]}
                newMoveList.append(tempMove)
                newMirrorMoveList.append(MirrorMove(tempMove))
        else:#black move
            assert (fromRow > toRow)#black should go backward
            move[2] = {'From': From, 'To': to}# set Black's moves
            tempMove = {'#': move[0], 'White': move[1], 'Black': move[2]}
            newMoveList.append(tempMove)
            newMirrorMoveList.append(MirrorMove(tempMove))
        originalWebVisualizerLink = originalWebVisualizerLink + From + to
        mirrorWebVisualizerLink = mirrorWebVisualizerLink + mirrorFrom + mirrorTo
    return newMoveList, newMirrorMoveList, originalWebVisualizerLink, mirrorWebVisualizerLink

def Driver(path):
    playerList = []
    ProcessDirectoryOfBreakthroughFiles(path, playerList)
    WriteToDisk(playerList, path)
