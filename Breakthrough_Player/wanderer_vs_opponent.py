from Breakthrough_Player import breakthrough_player
from multiprocessing import freeze_support
import os
if __name__ == '__main__':#for Windows since it lacks os.fork
  freeze_support()

  white_wins = 0
  black_wins = 0

  num_games_to_play = 20
  time_to_think = 30
  depth_limit = 5
  date = r'03142017'
  file_designator = 'z'
  expansion_MCTS = 'Expansion MCTS'
  expansion_MCTS_pruning = 'Expansion MCTS Pruning'
  expansion_MCTS_post_pruning = 'Expansion MCTS Post-Pruning'

  human = 'Human'
  wanderer = 'Wanderer'

  random_moves = 'Random'
  BFS_MCTS = 'BFS MCTS'
  EBFS_MCTS = 'EBFS MCTS'
  policy = "Policy"
  # player_for_path = 'EMCTSPruning'
  player_for_path = EBFS_MCTS
  Windows_path = r'G:\TruncatedLogs\PythonDatasets'
  OSX_path = r'/Users/TeofiloZosa/BreakthroughData/03122017SelfPlay'
  path = Windows_path
  white_player = EBFS_MCTS
  black_opponent = wanderer

  for i in range(0, num_games_to_play):
    gameplay_file = open(os.path.join(path,
                                    r'{date}'
                                    # r'_2RandStartMoves_randBestMoves_'
                                    # r'normalizedNNupdate_rankingOffset_'
                                    r'White{opponent}vsWanderer{designator}.txt'.format(date=date, opponent=player_for_path,
                                                                                        designator=file_designator)),
                       'a')
    MCTS_logging_file = open(os.path.join(path,
                                            r'{date}{opponent}vsWanderer'
                                            r'depth{depth}_'
                                            r'ttt{time_to_think}{designator}.txt'.format(date=date, opponent=player_for_path,
                                                                                         designator=file_designator,
                                                                                         depth=depth_limit,
                                                                                         time_to_think=time_to_think)),
                               'a')
    winner_color =   breakthrough_player.play_game_vs_wanderer(white_player, black_opponent, depth_limit, time_to_think, gameplay_file, MCTS_logging_file)

    if winner_color == 'White':
      white_wins += 1
    else:
      black_wins += 1
    print("Game {game}  White Wins: {white_wins}    Black Wins: {black_wins}".format(
        game=i+1, white_wins=white_wins, black_wins=black_wins ), file=gameplay_file)
    print("game ended")

    gameplay_file.close()
    MCTS_logging_file.close()