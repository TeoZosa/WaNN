from Breakthrough_Player.breakthrough_player import self_play_game
from multiprocessing import freeze_support
import os.path

if __name__ == '__main__':#for Windows since it lacks os.fork
  freeze_support()
  num_games_to_play = 20
  black_wins = 0
  white_wins = 0
  time_to_think = 60
  depth_limit = 5
  date = r'03072017'
  file_designator = ''
  expansion_MCTS = 'Expansion MCTS'
  random_moves = 'Random'
  BFS_MCTS = 'BFS MCTS'

  EBFS_MCTS = 'EBFS MCTS'

  policy = "Policy"

  opponent = 'EMCTS'
  path = r'G:\TruncatedLogs\PythonDatasets'
  gameplay_file = open(os.path.join(path,
                      r'{date}_2RandStartMoves_randBestMoves_'
                       r'normalizedNNupdate_rankingOffset_'
                       r'White{opponent}vsPolicy{designator}.txt'.format(date=date, opponent=opponent, designator=file_designator)), 'a')
  MCTS_logging_file = open(os.path.join(path,
                           r'{date}{opponent}_'
                           r'depth{depth}_'
                           r'ttt{time_to_think}{designator}.txt'.format(date=date, opponent=opponent, designator=file_designator,
        depth=depth_limit, time_to_think=time_to_think)), 'a')
  #possible policy net opponents


  for i in range(0, num_games_to_play):
    winner_color = self_play_game(white_player=expansion_MCTS, black_opponent=policy, depth_limit=depth_limit,
                                  time_to_think=time_to_think, file_to_write=gameplay_file, MCTS_log_file=MCTS_logging_file)
    if winner_color == 'White':
      white_wins += 1
    else:
      black_wins += 1
    print("Game {game}  White Wins: {white_wins}    Black Wins: {black_wins}".format(
        game=i+1, white_wins=white_wins, black_wins=black_wins ), file=gameplay_file)

  gameplay_file.close()
  MCTS_logging_file.close()