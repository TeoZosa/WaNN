import re as re #regular expressions
import pprint #pretty printing
import os, fnmatch #to retrieve file information from path
import pickle #serialize the data structure
import copy
def processDirectoryOfBreakthroughFiles(path, playerList):
    for playerGameHistoryData in findFiles(path, '*.txt'):
        playerList.append(processBreakthroughFile(path, playerGameHistoryData))
def processBreakthroughFile(path, playerGameHistoryData):
    fileName = playerGameHistoryData[
               len(path):len(playerGameHistoryData) - len('.txt')]  # trim path & extension
    fileName = fileName.split('間')  # arbitrary delimiter; user間2000間687687 -> ['user',2000, 687687]
    playerName = str(fileName[0])
    playerID = int(fileName[2])
    rank = int(fileName[1])
    gamesList = formatGameList(playerGameHistoryData, playerName)
    return {'Player': playerName, 'PlayerID': playerID, 'Rank': rank, 'Games': gamesList}
def writeToDisk(input, path):
    outputFile = open(path + r'PlayerDataPython.p', 'wb')
    pickle.dump(input, outputFile)
def findFiles (path, filter):#recursively find files at path with filter extension
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            yield os.path.join(root, file)
def preprocessGamesList(playerGameHistoryData):#normalized regex/iterable friendly list
    gamesList = [y[1] for y in list(
        enumerate([x.strip() for x in open(playerGameHistoryData, "r")]))]  # read in file and convert to list
    gamesList = filter(None, gamesList)  # remove empty strings from list
    gamesList = list(filter(lambda a: a != "[Site \"www.littlegolem.net\"]", gamesList))  # remove site from list
    return gamesList
def formatGameList(playerGameHistoryData, playerName):
    quotesRegex = re.compile(r'"(.*)"')
    eventEntry = 0
    whiteEntry = 1
    blackEntry = 2
    resultEntry = 3
    moveEntry = 4
    games = []
    gamesList = preprocessGamesList(playerGameHistoryData)
    #flags to indicate if something wasn't set properly
    opponentName = None
    event = None
    playerColor = None
    opponentColor = None
    win = None
                                   #format game list
    for j in range(0, len(gamesList)):
        thisRow = j % 5
        if thisRow !=moveEntry:
            rowData = quotesRegex.search(gamesList[j]).group(1)
        if thisRow == eventEntry:
            # [Event "Tournament null"] -> Event: "Tournament null"
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
                print ("UNEXPECTED DATA FORMAT")
                win = "Undefined at line " + str(j)
        elif thisRow == moveEntry:
                                         #format move list
            moveList = formatMoveList(gamesList[j])
            boardStates = generateBoardStates(moveList, playerColor, win)#generate board states from moveList
            assert (playerColor != opponentColor and opponentName != playerName)
            if len(moveList) > 3 and boardStates['Win'] != "Game In Progress":
                # non-spurrious games, remove if statement for all games.
                games.append({'Event': event, 'PlayerColor': playerColor, 'OpponentColor': opponentColor,
                          'OpponentName': opponentName, 'Win': win,
                          'Moves': moveList, 'BoardStates': boardStates})  # append new game after formatting move list
    return games

def generateBoardStates(moveList, playerColor, win):
    if win == "Game In Progress":
        return {'Win': win, 'States': []}
    empty = 'e'
    white = 'w'
    black = 'b'
    boardState = {8:{'a': black, 'b': black, 'c': black, 'd': black, 'e': black, 'f': black, 'g': black, 'h': black},
                  7:{'a': black, 'b': black, 'c': black, 'd': black, 'e': black, 'f': black, 'g': black, 'h': black},
                  6:{'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
                  5:{'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
                  4:{'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
                  3:{'a': empty, 'b': empty, 'c': empty, 'd': empty, 'e': empty, 'f': empty, 'g': empty, 'h': empty},
                  2:{'a': white, 'b': white, 'c': white, 'd': white, 'e': white, 'f': white, 'g': white, 'h': white},
                  1:{'a': white, 'b': white, 'c': white, 'd': white, 'e': white, 'f': white, 'g': white, 'h': white}
                }
    boardStatistics = {'Win': win, 'States':[boardState]}
    for i in range (0, len(moveList)):
        assert (moveList[i]['#'] == i+1)
        if isinstance(moveList[i]['White'], dict):#if string, then == resign or NIL
            boardState = movePiece(boardState, moveList[i]['White']['To'], moveList[i]['White']['From'])
            boardStatistics['States'].append(boardState)
        if isinstance(moveList[i]['Black'], dict):#if string, then == resign or NIL
            boardState = movePiece(boardState, moveList[i]['Black']['To'], moveList[i]['Black']['From'])
            boardStatistics['States'].append(boardState)
    return boardStatistics

def movePiece(boardState, To, From):
    empty = 'e'
    nextBoardState = copy.deepcopy(boardState)#edit copy of boardState
    nextBoardState[int(To[1])][To[0]] = nextBoardState[int(From[1])][From[0]]
    nextBoardState[int(From[1])][From[0]] = empty
    return nextBoardState
def formatMoveList(moveListString):
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
            move[1] = {'From': move[1][0:2], 'To':move[1][3:len(move[1])]} #set White's moves
        if move[2] != "resign" and move[2] != "NIL": #set Black's moves
            if len(move[2]) > 3:
                move[2] = {'From': move[2][0:2], 'To':move[2][3:len(move[2])]}
            else:
                move[2] = "NIL"
        moveList[i] = {'#': move[0], 'White': move[1], 'Black':move[2]}
    return moveList

playerList = []
pathToCheck2 = r'/Users/TeofiloZosa/BreakthroughData/'
pathToCheck = r'/Users/TeofiloZosa/BreakthroughData/AutomatedData/'
processDirectoryOfBreakthroughFiles(pathToCheck, playerList)
for i in range(0, len(playerList)):
    pprint.pprint("Player # " + str(i+1) +": " + playerList[i]['Player'])
writeToDisk(playerList, pathToCheck)

#Verified Working.
# #double check
# newList = pickle.load(open(pathToCheck+r'PlayerDataPython.p', 'rb'))
# oldList = pickle.load(open(pathToCheck2+r'PlayerDataPython.p', 'rb'))
# assert (playerList == newList == oldList)


