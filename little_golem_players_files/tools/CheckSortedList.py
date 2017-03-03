'''Various unit-tests to ensure coherency of the various versions of the player list data structure. Not exhaustive.'''
def PrintStatistics(playerList):
    games = 0
    states = 0
    moves = 0
    for player in playerList:
      #  print ("Player " + player['Player'] +" has ID# "+ str(player['PlayerID'])+" and is Rank: "+ str(player['Rank']))
        games += len(player['Games'])
        gameList = player['Games']
        for game in gameList:
            if isinstance(game['BoardStates'], dict):#else game In Progress
                boardStates = game['BoardStates']['States']
                mirrorBoardStates = game['BoardStates']['MirrorStates']
                assert len(boardStates) == len(mirrorBoardStates)
                states += len(boardStates)
                states += len(mirrorBoardStates)
            moves += len(game['Moves'])
    print ('Total Number of Games: {games}'.format(games = games))
    print ('Total Number of States: {states}'.format(states = states))
    print ('Total Number of Moves: {moves}'.format(moves = moves))

#test to make sure analysis format matches original format
def TestAnalysisVsOriginal(originalPlayerList, analysisPlayerList):
    for i in range (0, len(originalPlayerList[1]['Games'])):
        playerColor = originalPlayerList[1]['Games'][i]['PlayerColor']
        gameStates = originalPlayerList[1]['Games'][i]['BoardStates']['States']
        assert (playerColor == analysisPlayerList[1]['Games'][i]['PlayerColor'])
        assert (originalPlayerList[1]['Games'][i]['OpponentName'] == analysisPlayerList[1]['Games'][i]['OpponentName'])
        assert (originalPlayerList[1]['Games'][i]['OpponentColor'] == analysisPlayerList[1]['Games'][i]['OpponentColor'])
        assert (originalPlayerList[1]['Games'][i]['Moves'] == analysisPlayerList[1]['Games'][i]['Moves'])
        assert (originalPlayerList[1]['Games'][i]['Event'] == analysisPlayerList[1]['Games'][i]['Event'])
        win = -10
        if originalPlayerList[1]['Games'][i]['Win'] == True:
            win = 1
        elif originalPlayerList[1]['Games'][i]['Win'] == False:
            win = -1
        for stateNum in range(0, len(gameStates)):
            assert(analysisPlayerList[1]['Games'][i]['BoardStates']['States'][stateNum][1] == win)
            for row in gameStates[stateNum]:
                for column in sorted(gameStates[stateNum][row]):
                    column_offset = int(ord(column) - ord('a'))
                    offset = (row - 1) * 8 + column_offset
                    value = analysisPlayerList[1]['Games'][i]['BoardStates']['States'][stateNum][0][offset]
                    if gameStates[stateNum][row][column] == 'e':
                        assert (value == 0)
                    if playerColor =='White':
                        if gameStates[stateNum][row][column] == 'w':
                            assert (value == 1)
                        if gameStates[stateNum][row][column] == 'b':
                            assert (value == -1)
                    if playerColor =='Black':
                        if gameStates[stateNum][row][column] == 'b':
                            assert (value == 1)
                        if gameStates[stateNum][row][column] == 'w':
                            assert (value == -1)

def TestAnalysisVsMirrors(analysisPlayerList):
    for player in analysisPlayerList:
        for game in player['Games']:
            boardStates = game['BoardStates']
            originalStates = boardStates['States']
            mirrorStates = boardStates['MirrorStates']
            for stateNum in range(0, len(boardStates['States'])):
                row = 0
                while row <8:
                    column = 0
                    while column <8:
                        offset = (row) * 8
                        originalValueIndex = offset + column
                        mirrorValueIndex = offset + 7 - column
                        originalValue = originalStates[stateNum][0][originalValueIndex]
                        mirrorValue = mirrorStates[stateNum][0][mirrorValueIndex]
                        assert originalValue == mirrorValue
                        column+=1
                    row += 1
                assert (originalStates[stateNum][0][64] == mirrorStates[stateNum][0][64])

def TestOriginalVsMirrorStates(analysisPlayerList):
    for player in analysisPlayerList:
        for game in player['Games']:
            boardStates = game['BoardStates']
            originalStates = boardStates['States']
            mirrorStates = boardStates['MirrorStates']
            isWhiteIndex = 192
            isWhite = originalStates[0][0][isWhiteIndex]
            mirrorIsWhite = mirrorStates[0][0][isWhiteIndex]
            for stateNum in range(0, len(boardStates['States'])):
                plane = 0
                while plane <3:
                    row = 0
                    while row <8:
                        column = 0
                        while column <8:
                            offset = (row * 8) + plane*64
                            originalValueIndex = offset + column
                            mirrorValueIndex = offset + 7 - column
                            originalValue = originalStates[stateNum][0][originalValueIndex]
                            mirrorValue = mirrorStates[stateNum][0][mirrorValueIndex]
                            assert originalValue == mirrorValue
                            column+=1
                        row += 1
                    plane+=1
                assert (originalStates[stateNum][0][isWhiteIndex] == mirrorStates[stateNum][0][isWhiteIndex] == isWhite == mirrorIsWhite)
                assert (originalStates[stateNum][0][193] == mirrorStates[stateNum][0][193])

def TestBinaryVsTernary(ternaryPlayerList, binaryPlayerList):
    for player in binaryPlayerList:
        for game in player['Games']:
            boardStates = game['BoardStates']
            originalStates = boardStates['States']
            mirrorStates = boardStates['MirrorStates']
            for stateNum in range(0, len(boardStates['States'])):
                plane = 0
                while plane <3:
                    row = 0
                    while row <8:
                        column = 0
                        while column <8:
                            offset = (row * 8) + plane*64
                            originalValueIndex = offset + column
                            mirrorValueIndex = offset + 7 - column
                            originalValue = originalStates[stateNum][0][originalValueIndex]
                            mirrorValue = mirrorStates[stateNum][0][mirrorValueIndex]
                            assert originalValue == mirrorValue
                            column+=1
                        row += 1
                    plane+=1
                assert (originalStates[stateNum][0][192] == mirrorStates[stateNum][0][192])

def TestAnalysisVsAnalysisWithTransition(originalAnalysisList, transitionAnalysisList):
    for j in range (0, len(originalAnalysisList)):
        for i in range(0, len(originalAnalysisList[j]['Games'])):
            playerColor = originalAnalysisList[j]['Games'][i]['PlayerColor']
            originalgameStates = originalAnalysisList[j]['Games'][i]['BoardStates']['States']
            newGameStates = transitionAnalysisList[j]['Games'][i]['BoardStates']['States']
            assert (playerColor == transitionAnalysisList[j]['Games'][i]['PlayerColor'])
            assert (originalAnalysisList[j]['Games'][i]['OpponentName'] == transitionAnalysisList[j]['Games'][i]['OpponentName'])
            assert (
            originalAnalysisList[j]['Games'][i]['OpponentColor'] == transitionAnalysisList[j]['Games'][i]['OpponentColor'])
            assert (originalAnalysisList[j]['Games'][i]['Moves'] == transitionAnalysisList[j]['Games'][i]['Moves'])
            assert (originalAnalysisList[j]['Games'][i]['Event'] == transitionAnalysisList[j]['Games'][i]['Event'])
            win = -10
            if originalAnalysisList[j]['Games'][i]['Win'] == True:
                win = 1
            elif originalAnalysisList[j]['Games'][i]['Win'] == False:
                win = -1
            for stateNum in range(0, len(originalgameStates)):
                assert (transitionAnalysisList[j]['Games'][i]['BoardStates']['States'][stateNum][1] == win)
                originalStateToCheck = originalgameStates[stateNum][0]
                newStateToCheck = newGameStates[stateNum][0]
                for p in range (0, len(originalStateToCheck)):
                    assert (originalStateToCheck[p] == newStateToCheck[p])


#
# path = r'/Users/TeofiloZosa/BreakthroughData/AutomatedData/'
# #Original format
# playerList = pickle.load(open(path + r'PlayerDataPythonNonSpurriousGamesSorted.p', 'rb'))
# #Analysis format
# playerList2 = pickle.load(open(path + r'PlayerDataPythonDataSetsorted.p', 'rb'))
# #PlayerDataBinaryFeaturesDataset
# playerList3 = pickle.load(open(path + r'PlayerDataBinaryFeaturesWBEDataSetsorted.p', 'rb'))
# TestOriginalVsMirrorStates(playerList3)
#TestAnalysisVsMirrors(playerList2)
#TestAnalysisVsOriginal(playerList, playerList2)
#printBoard.PrintHumanReadableBoardStatesWithTransitions(gameToPrint)
#
