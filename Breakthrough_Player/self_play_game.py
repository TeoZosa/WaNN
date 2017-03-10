from Breakthrough_Player.breakthrough_player import self_play_game
from multiprocessing import freeze_support
import os.path

if __name__ == '__main__':#for Windows since it lacks os.fork
  freeze_support()
  num_games_to_play = 20
  black_wins = 0
  white_wins = 0
  time_to_think = 60
  depth_limit = 1 #for EBFS MCTS, will get to depth 7 by move 60
  date = r'03092017'
  file_designator = ''
  expansion_MCTS = 'Expansion MCTS'
  expansion_MCTS_pruning = 'Expansion MCTS Pruning'
  expansion_MCTS_post_pruning = 'Expansion MCTS Post-Pruning'

  random_moves = 'Random'
  BFS_MCTS = 'BFS MCTS'

  EBFS_MCTS = 'EBFS MCTS'

  policy = "Policy"

  opponent = 'EMCTSPruning'
  path = r'G:\TruncatedLogs\PythonDatasets'

  #possible policy net opponents


  for i in range(0, num_games_to_play):
    gameplay_file = open(os.path.join(path,
                                    r'{date}'
                                    # r'_2RandStartMoves_randBestMoves_'
                                    # r'normalizedNNupdate_rankingOffset_'
                                    r'White{opponent}vsPolicy{designator}.txt'.format(date=date, opponent=opponent,
                                                                                      designator=file_designator)),
                       'a')
    MCTS_logging_file = open(os.path.join(path,
                                            r'{date}{opponent}_'
                                            r'depth{depth}_'
                                            r'ttt{time_to_think}{designator}.txt'.format(date=date, opponent=opponent,
                                                                                         designator=file_designator,
                                                                                         depth=depth_limit,
                                                                                         time_to_think=time_to_think)),
                               'a')
    winner_color = self_play_game(white_player=expansion_MCTS_pruning, black_opponent=policy, depth_limit=depth_limit,
                                  time_to_think=time_to_think, file_to_write=gameplay_file, MCTS_log_file=MCTS_logging_file)
    if winner_color == 'White':
      white_wins += 1
    else:
      black_wins += 1
    print("Game {game}  White Wins: {white_wins}    Black Wins: {black_wins}".format(
        game=i+1, white_wins=white_wins, black_wins=black_wins ), file=gameplay_file)
    print("game ended")

    gameplay_file.close()
    MCTS_logging_file.close()