import os
import  pickle
import warnings
from multiprocessing import Process, pool, Pool, TimeoutError, freeze_support
from PlayersToDataStructure import SelfPlayLogsToPythonDataStructure as convertLog
from Tools import NumpyArray

class NoDaemonProcess(Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(pool.Pool):  # Had to make a special class to allow for an inner process pool
    Process = NoDaemonProcess

def SelfPlayLogsToDataStructures():
  paths = [
    r'G:\TruncatedLogs\07xx-07yy\selfPlayLogsMBP2011xxxxxx',
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
    r'G:\TruncatedLogs\1024-1129\selfPlayLogsBreakthrough4'
  ]
  processes = MyPool(processes=len(paths))
  processes.map_async(convertLog.driver, paths)#map processes to arg lists
  processes.close()
  processes.join()

   
def AggregateSelfPlayDataStructures():
    warnings.warn("Removed in favor of aggregating/queuing later in the pipeline. "
                  "Else, this is time consuming, creates a large, redundant file, "
                  "and precludes the possibility of embarrassingly parallel numpy array conversion."
                  , DeprecationWarning)
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
        file = open(os.path.join(path, fileName),'r+b')
        combinedList.extend(pickle.load(file))
        file.close()
    outputList = open(path + r'07xx-1129SelfPlayGames.p', 'wb')
    pickle.dump(combinedList, outputList, protocol=4)
    outputList.close()

def SelfPlayDataStructuresToNumpyArrays():
  path = r'G:\TruncatedLogs\PythonDataSets\DataStructures'
  files = [
           r'07xx-07yyselfPlayLogsMBP2011xxxxxxDataPython.p',
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
           r'1024-1129selfPlayLogsBreakthrough4DataPython.p'
           ]

  arg_lists = [[r'Self-Play', r'Value', path, file] for file in files]
  processes = Pool(processes=len(arg_lists))
  processes.starmap_async(NumpyArray.self_player_driver, arg_lists)#map processes to arg lists
  processes.close()
  processes.join()
  # NumpyArray.self_player_driver(r'Self-Play', r'Policy', path, files[0])
