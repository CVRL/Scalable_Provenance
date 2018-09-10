import sys
import os

numberOfServers = int(sys.argv[1])
startingPort = int(sys.argv[2])
preload=True
indexFolder = None
if len(sys.argv) > 3:
    indexFolder = sys.argv[3]
i=0
preloadStr = 'LD_PRELOAD=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_core.so:/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_sequential.so '
if not preload:
    preloadStr = ''
if indexFolder is not None:
    for index in os.listdir(indexFolder):
        os.system(preloadStr+"python3.5 queryServer.py "+str(startingPort+i)+" " + 'localhost '+ os.path.join(indexFolder,index) + ' '+ str(i) + ' &')
        i+=1
else:
    for i in range(numberOfServers):
        os.system(preloadStr+"python3.5 queryServer.py " + str(startingPort+i) + '&')
