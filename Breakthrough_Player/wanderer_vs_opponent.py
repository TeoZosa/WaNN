from Breakthrough_Player import breakthrough_player
from multiprocessing import freeze_support

if __name__ == '__main__':#for Windows since it lacks os.fork
  freeze_support()


  num_games_to_play = 10
  time_to_think = 10
  depth_limit = 1
  date = r'03122017'
  file_designator = ''
  expansion_MCTS = 'Expansion MCTS'
  expansion_MCTS_pruning = 'Expansion MCTS Pruning'
  expansion_MCTS_post_pruning = 'Expansion MCTS Post-Pruning'

  human = 'Human'
  wanderer = 'Wanderer'

  random_moves = 'Random'
  BFS_MCTS = 'BFS MCTS'
  EBFS_MCTS = 'EBFS MCTS'
  policy = "Policy"
  opponent_for_path = 'EMCTSPruning'
  # path = r'G:\TruncatedLogs\PythonDatasets'

  breakthrough_player.play_game_vs_wanderer(expansion_MCTS_pruning, human, depth_limit, time_to_think)
