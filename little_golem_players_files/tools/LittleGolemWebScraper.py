import urllib.request
import re
from bs4 import BeautifulSoup
import pprint

# Note: Implementation has not been optimized nor does it conform to programming best practices
# Used text parsing beforehand to get player names & rank in user間rank format.
# List is assumed to match order of plids retrieved
# Note: Be sure to label output folder by date/time since the games list gets updated frequently.

def GetPLIDs():
    breakthroughRankPath = r'http://www.littlegolem.net/jsp/info/player_list.jsp?gtvar=brkthr_DEFAULT&filter=&countryid=&page='
    plids = []
    plidRegExp = re.compile(r'.*plid=(\d+)')
    for i in range(1, 15):
        # get plids from each page; as of 05-28-2016, 14 pages
        req = urllib.request.Request(breakthroughRankPath + str(i))
        with urllib.request.urlopen(req) as response:
            the_page = response.read()
        readablePage = BeautifulSoup(the_page, "html.parser")
        plids += plidRegExp.findall(str(readablePage.prettify()))
    return plids
def GetPlayerNameList(path):
    playerNameFile = [y[1] for y in list(
        enumerate([x.strip() for x in open(path, 'r')]))]  # read in file and convert to list
    playerNameFile = list(filter(None, playerNameFile))  # remove empty strings from list
    pprint.pprint(playerNameFile)
    return playerNameFile
def WriteToDisk(playerName, plid, page, path):
    delimiter = '間'  # 'ma' in romaji; meaning: empty, space, nothingness
    gameText = str(page.prettify())
    outputFile = open(path + playerName + delimiter + str(plid) + ".txt", 'w')
    outputFile.write(gameText)
    outputFile.close()
def GetPlayerGameData(plid):
    playerGameListPath = 'http://www.littlegolem.net/jsp/info/player_game_list_txt.jsp?plid={playerID}&gtid=brkthr' \
        .format(playerID=plid)
    req = urllib.request.Request(playerGameListPath)
    with urllib.request.urlopen(req) as response:
        the_page = response.read()
    readablePage = BeautifulSoup(the_page, "html.parser")
    return readablePage
def WriteAllPlayersGamesDataToDisk(plidList, playerNamesList, writePath):
    for i in range(0, len(plidList)):
        gameData = GetPlayerGameData(plidList[i])
        WriteToDisk(playerNamesList[i], plidList[i], gameData, writePath)
def driver():
    writePath = r'/Users/TeofiloZosa/BreakthroughData/AutomatedData/'
    playerNamesFilePath = r'/Users/TeofiloZosa/BreakthroughData/BreakthroughPlayerNamesFormatted.txt'
    playerNameList = GetPlayerNameList(playerNamesFilePath)
    plids = GetPLIDs()
    WriteAllPlayersGamesDataToDisk(plids, playerNameList, writePath)
