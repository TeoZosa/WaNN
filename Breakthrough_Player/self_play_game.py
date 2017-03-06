from Breakthrough_Player.breakthrough_player import self_play_game
from multiprocessing import freeze_support

if __name__ == '__main__':#for Windows since it lacks os.fork
  freeze_support()
  num_games_to_play = 25
  black_wins = 0
  white_wins = 0
  file_to_write = open(r'G:\TruncatedLogs\PythonDatasets\03052017_2randStartMoves_randBestMoves_normalizedNNupdate_rankingOffset_ExpansionMCTSvsPolicy.txt','a')
  #possible policy net opponents
  expansion_MCTS = 'Expansion MCTS'
  random_moves = 'Random'
  BFS_MCTS = 'BFS MCTS'

  for i in range(1, num_games_to_play):
    winner_color = self_play_game(False, policy_opponent=expansion_MCTS, file_to_write=file_to_write)
    if winner_color == 'White':
      white_wins += 1
    else:
      black_wins += 1
    print("Game {game}  White Wins: {white_wins}    Black Wins: {black_wins}".format(
        game=i, white_wins=white_wins, black_wins=black_wins ), file=file_to_write)
  file_to_write.close()