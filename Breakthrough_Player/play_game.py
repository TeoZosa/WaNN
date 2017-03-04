from Breakthrough_Player import breakthrough_player
from multiprocessing import freeze_support

if __name__ == '__main__':#for Windows since it lacks os.fork
  freeze_support()
  breakthrough_player.play_game(False)