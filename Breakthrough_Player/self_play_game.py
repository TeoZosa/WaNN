from Breakthrough_Player.breakthrough_player import self_play_game
from multiprocessing import freeze_support

if __name__ == '__main__':#for Windows since it lacks os.fork
  freeze_support()
  num_games_to_play = 1
  black_wins = 0
  white_wins = 0
  file_to_write = open(r'G:\TruncatedLogs\PythonDatasets\03062017_2RandStartMoves_randBestMoves_normalizedNNupdate_rankingOffset_BlackEMCTSvsPolicy.txt','a')
  #possible policy net opponents
  expansion_MCTS = 'Expansion MCTS'
  random_moves = 'Random'
  BFS_MCTS = 'BFS MCTS'
  policy = "Policy"

  for i in range(0, num_games_to_play):
    winner_color = self_play_game(True, policy_opponent=expansion_MCTS, )
    if winner_color == 'White':
      white_wins += 1
    else:
      black_wins += 1
    print("Game {game}  White Wins: {white_wins}    Black Wins: {black_wins}".format(
        game=i+1, white_wins=white_wins, black_wins=black_wins ), file=file_to_write)
  file_to_write.close()