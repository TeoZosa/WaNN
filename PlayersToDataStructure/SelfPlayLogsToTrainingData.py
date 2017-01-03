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
  path = r'G:\TruncatedLogs\PythonDataSets\DataStructures\\'
  files = [r'07xx-07yyselfPlayLogsMBP2011xxxxxxDataPython.p',
    r'0802-0805selfPlayLogsWorkstationxxDataPython.p',
    r'0806-0824selfPlayLogsBreakthrough4DataPython.p',
    r'0824-1006selfPlayLogsBreakthrough1DataPython.p',
    r'0824-1006selfPlayLogsBreakthrough2DataPython.p',
    r'0824-1006selfPlayLogsBreakthrough3DataPython.p',
    r'0824-1006selfPlayLogsBreakthrough4DataPython.p',
    r'1018-1024selfPlayLogsBreakthrough1DataPython.p',
    r'1018-1024selfPlayLogsBreakthrough2DataPython.p',
    r'1018-1024selfPlayLogsBreakthrough3DataPython.p',
    r'1018-1024selfPlayLogsBreakthrough4DataPython.p',
    r'1024-1129selfPlayLogsBreakthrough1DataPython.p',
    r'1024-1129selfPlayLogsBreakthrough2DataPython.p',
    r'1024-1129selfPlayLogsBreakthrough3DataPython.p',
    r'1024-1129selfPlayLogsBreakthrough4DataPython.p']
  combinedList =[]
  for fileName in files:
    file = open(path+fileName,'r+b')
    combinedList.append(pickle.load(file))
    file.close()
  outputList = open(path + r'07xx-1129SelfPlayGames.p', 'wb')
  pickle.dump(combinedList, outputList)
  outputList.close()

def SelfPlayDataStructuresToNumpyArrays():
  NumpyArray.SelfPlayDriver("Self-Play", 'Policy', r'G:\TruncatedLogs\PythonDataSets\DataStructures\\', "07xx-1129SelfPlayGames.p")
  # NumpyArray.SelfPlayDriver("Self-Play", 'Policy', r'G:\TruncatedLogs\PythonDataSets\DataStructures\\', r'1018-1024selfPlayLogsBreakthrough1DataPython.p')
