import  pickle
from multiprocessing import Process, Pool, TimeoutError, freeze_support
from PlayersToDataStructure import SelfPlayLogsToPythonDataStructure as convertLog
from Tools import NumpyArray



#15 processes
def SelfPlayLogsToDataStructures():
  processes = Pool(processes=15)
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
  processes.map(convertLog.Driver, paths)#map processes to arg lists
   
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
