import urllib.request
import re
from bs4 import BeautifulSoup
import pprint

# Note: Implementation has not been optimized nor does it conform to programming best practices
# Used text parsing beforehand to get player names & rank in user間rank format.
# List is assumed to match order of plids retrieved
# Note: Be sure to label output folder by date/time since the games list gets updated frequently.

def get_plids():
    breakthrough_rank_path = r'http://www.littlegolem.net/jsp/info/player_list.jsp?gtvar=brkthr_DEFAULT&filter=&countryid=&page='
    plids = []
    plid_reg_exp = re.compile(r'.*plid=(\d+)')
    for i in range(1, 15):
        # get plids from each page; as of 05-28-2016, 14 pages
        req = urllib.request.Request(breakthrough_rank_path + str(i))
        with urllib.request.urlopen(req) as response:
            the_page = response.read()
        readablePage = BeautifulSoup(the_page, "html.parser")
        plids += plid_reg_exp.findall(str(readablePage.prettify()))
    return plids

def get_player_names_list(path):
    player_name_file = [y[1] for y in list(
        enumerate([x.strip() for x in open(path, 'r')]))]  # read in file and convert to list
    player_name_file = list(filter(None, player_name_file))  # remove empty strings from list
    pprint.pprint(player_name_file)
    return player_name_file

def write_to_disk(playerName, plid, page, path):
    delimiter = '間'  # 'ma' in romaji; meaning: empty, space, nothingness
    game_text = str(page.prettify())
    output_file = open(path + playerName + delimiter + str(plid) + ".txt", 'w')
    output_file.write(game_text)
    output_file.close()

def get_player_game_data(plid):
    playerGameListPath = 'http://www.littlegolem.net/jsp/info/player_game_list_txt.jsp?plid={playerID}&gtid=brkthr' \
        .format(playerID=plid)
    req = urllib.request.Request(playerGameListPath)
    with urllib.request.urlopen(req) as response:
        the_page = response.read()
    readable_page = BeautifulSoup(the_page, "html.parser")
    return readable_page

def write_all_player_data_to_disk(plidList, player_names, path):
    for i in range(0, len(plidList)):
        game_data = get_player_game_data(plidList[i])
        write_to_disk(player_names[i], plidList[i], game_data, path)

def driver():
    write_path = r'/Users/TeofiloZosa/BreakthroughData/AutomatedData/'
    player_names_file_path = r'/Users/TeofiloZosa/BreakthroughData/BreakthroughPlayerNamesFormatted.txt'
    player_name_list = get_player_names_list(player_names_file_path)
    plids = get_plids()
    write_all_player_data_to_disk(plids, player_name_list, write_path)
