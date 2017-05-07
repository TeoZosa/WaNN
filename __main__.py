#whatever scripts I want to run here.
import time
from multiprocessing import freeze_support

from self_play_files.self_play_to_data_structure import self_play_logs_to_training_data

##little golem player data to numpy array
# readPath = r'/Users/teofilozosa/BreakthroughData/AutomatedData/'
# #readPath = ''#default directory for 2011 MBP 2
# fileName = r'PlayerDataBinaryFeaturesWBPOEDatasetSorted.p'
# filter = assign_filter(fileName)
# playerListDataFriendly = pickle.load(open(readPath + fileName, 'rb'))
# X, y = filter_training_examples_and_labels(playerListDataFriendly, filter= filter)
# writePath = assign_path("MBP2014")
# write_np_array_to_disk(writePath, X, y, filter)

#human readable testing
#SelfPlayLogsToPythonDataStructure.Driver(r'G:\TruncatedLogs\07xx-07yy\selfPlayLogsMBP2011xxxxxx')
#
# testData = pickle.load(open(r'G:\TruncatedLogs\07xx-07yy\selfPlayLogsMBP2011xxxxxxDataPython.p', "r+b"))
# HumanReadableBoardPrinting.PrintGame(testData[0]['Games'][0])

if __name__ == '__main__':#for Windows since it lacks os.fork
  freeze_support()
  #
  # startTime = time.time()
  # self_play_logs_to_training_data.SelfPlayLogsToDataStructures()
  # print("Minutes to convert to data structures: {time}".format(time=(time.time() - startTime) / (60)))

  startTime = time.time()
  self_play_logs_to_training_data.self_play_data_structures_to_numpy()
  print("Minutes to numpy array: {time}".format(time=(time.time() - startTime) / (60)))
