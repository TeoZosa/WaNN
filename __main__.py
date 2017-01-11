#whatever scripts I want to run here.
import pickle
from multiprocessing import freeze_support
from PlayersToDataStructure import  SelfPlayLogsToPythonDataStructure, SelfPlayLogsToTrainingData
from Tools import HumanReadableBoardPrinting
import time
##little golem player data to numpy array
# readPath = r'/Users/teofilozosa/BreakthroughData/AutomatedData/'
# #readPath = ''#default directory for 2011 MBP 2
# fileName = r'PlayerDataBinaryFeaturesWBPOEDatasetSorted.p'
# filter = AssignFilter(fileName)
# playerListDataFriendly = pickle.load(open(readPath + fileName, 'rb'))
# X, y = generateArray(playerListDataFriendly, filter= filter)
# writePath = AssignPath("MBP2014")
# WriteNPArrayToDisk(writePath, X, y, filter)

#human readable testing
#SelfPlayLogsToPythonDataStructure.Driver(r'G:\TruncatedLogs\07xx-07yy\selfPlayLogsMBP2011xxxxxx')
#
# testData = pickle.load(open(r'G:\TruncatedLogs\07xx-07yy\selfPlayLogsMBP2011xxxxxxDataPython.p', "r+b"))
# HumanReadableBoardPrinting.PrintGame(testData[0]['Games'][0])

if __name__ == '__main__':#for Windows since it lacks os.fork
  freeze_support()

  # startTime = time.time()
  # SelfPlayLogsToTrainingData.SelfPlayLogsToDataStructures()
  # print("Minutes to convert to data structures: {time}".format(time=(time.time() - startTime) / (60)))

  startTime = time.time()
  SelfPlayLogsToTrainingData.SelfPlayDataStructuresToNumpyArrays()
  print("Minutes to numpy array: {time}".format(time=(time.time() - startTime) / (60)))
