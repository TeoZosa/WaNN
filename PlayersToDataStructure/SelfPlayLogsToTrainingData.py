
import  pickle
import threading
import time
from PlayersToDataStructure import SelfPlayLogsToPythonDataStructure as convertLog
from Tools import NumpyArray



#15 threads
def SelfPlayLogsToDataStructures():
  ###Super I/O Bound, so multithreading doesn't help too much, even with mmap
  class convertLogThread(threading.Thread):
    def __init__(self, path):
      threading.Thread.__init__(self)
      self.path = path

    def run(self):
      convertLog.Driver(self.path)
  threads = [None]*15
  paths = [r'G:\TruncatedLogs\07xx-07yy\selfPlayLogsMBP2011xxxxxx',
    r'G:\TruncatedLogs\0802-0805\selfPlayLogsWorkstationxx',
    r'G:\TruncatedLogs\0806-0824\selfPlayLogsBreakthrough4',
    r'G:\TruncatedLogs\0824-1006\selfPlayLogsBreakthrough1',
    r'G:\TruncatedLogs\0824-1006\selfPlayLogsBreakthrough2',
    r'G:\TruncatedLogs\0824-1006\selfPlayLogsBreakthrough3',
    r'G:\TruncatedLogs\0824-1006\selfPlayLogsBreakthrough4',
    r'G:\TruncatedLogs\1018-1024\selfPlayLogsBreakthrough1',
    r'G:\TruncatedLogs\1018-1024\selfPlayLogsBreakthrough2',
    r'G:\TruncatedLogs\1018-1024\selfPlayLogsBreakthrough3',
    r'G:\TruncatedLogs\1018-1024\selfPlayLogsBreakthrough4',
    r'G:\TruncatedLogs\1024-1129\selfPlayLogsBreakthrough1',
    r'G:\TruncatedLogs\1024-1129\selfPlayLogsBreakthrough2',
    r'G:\TruncatedLogs\1024-1129\selfPlayLogsBreakthrough3',
    r'G:\TruncatedLogs\1024-1129\selfPlayLogsBreakthrough4']
  for i in range(0,len(paths)):
    threads[i]= convertLogThread(paths[i])
  startTime = time.time()
  for i in range (0,len(paths)):
    threads[i].start()
    print("thread {thread} starting".format(thread = i))
  for i in range(0,len(paths)):
    threads[i].join()
  print("Time elapsed: {time}".format(time = time.time() - startTime))

def AggregateSelfPlayDataStructures():
  path = r'G:\TruncatedLogs\PythonDatasets\Datastructures\\'
  files = [r'07xx-07yyselfPlayLogsMBP2011xxxxxxSelfPlayDataPython.p',
    r'0802-0805selfPlayLogsWorkstationxxSelfPlayDataPython.p',
    r'0806-0824selfPlayLogsBreakthrough4SelfPlayDataPython.p',
    r'0824-1006selfPlayLogsBreakthrough1SelfPlayDataPython.p',
    r'0824-1006selfPlayLogsBreakthrough2SelfPlayDataPython.p',
    r'0824-1006selfPlayLogsBreakthrough3SelfPlayDataPython.p',
    r'0824-1006selfPlayLogsBreakthrough4SelfPlayDataPython.p',
    r'1018-1024selfPlayLogsBreakthrough1SelfPlayDataPython.p',
    r'1018-1024selfPlayLogsBreakthrough2SelfPlayDataPython.p',
    r'1018-1024selfPlayLogsBreakthrough3SelfPlayDataPython.p',
    r'1018-1024selfPlayLogsBreakthrough4SelfPlayDataPython.p',
    r'1024-1129selfPlayLogsBreakthrough1SelfPlayDataPython.p',
    r'1024-1129selfPlayLogsBreakthrough2SelfPlayDataPython.p',
    r'1024-1129selfPlayLogsBreakthrough3SelfPlayDataPython.p',
    r'1024-1129selfPlayLogsBreakthrough4SelfPlayDataPython.p']
  combinedList =[]
  for fileName in files:
    file = open(path+fileName,'r+b')
    combinedList.append(pickle.load(file))
    file.close()
  outputList = open(path + r'07xx-1129SelfPlayGames.p', 'wb')
  pickle.dump(combinedList, outputList)
  outputList.close()

def SelfPlayDataStructuresToNumpyArrays():
  NumpyArray.SelfPlayDriver("Self-Play",r'G:\TruncatedLogs\PythonDatasets\Datastructures\\', "07xx-1129SelfPlayGames.p")
