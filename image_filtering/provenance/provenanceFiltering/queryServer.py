from threading import Thread
import os
import sys
sys.path.append('../featureExtraction')
sys.path.append('../helperLibraries')
from featureExtraction import featureExtraction
import socket
from queryIndex import queryIndex
from resources import Resource
import collections
import pickle
import progressbar

class queryIndexServer:

    def __init__(self, port,address=socket.gethostname(),sock=None,indexFile=None,index=None, preproc=None, map=None,id=None):
        # Thread.__init__(self)
        if sock is None:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((address, port))
        else:
            self.server_socket = sock
        self.ex = featureExtraction()
        self.indexIsLoaded = False
        self.queryIndexObject = None
        self.port = port
        print('bound socket to ', address, ' on port ', port)
        if indexFile is not None:
            findex = open(indexFile,'rb')
            indexResource = Resource('index', findex.read(), 'application/octet-stream')
            findex.close()
            self.queryIndexObject = queryIndex(indexResource,id=id)
            self.indexIsLoaded = True
            print('index loaded!')
    def mainDispatcher(self,port,bufsize,numConnections):
        self.server_socket.listen(numConnections)
        while True:
            print('server is listening...')
            (client_socket,
             client_address) = self.server_socket.accept()  # connect to the client which will send the pickled resource
            d = Thread(target=self.runServerThreadLoop,args=(port,bufsize,client_socket,client_address))
            d.start()
    def runServerThreadLoop(self,port,bufsize,client_socket,client_address):
        # port = arg[0]
        # bufsize = arg[1]
        self.server_socket.listen(5)
        # while True:


        # (client_socket,client_address) = self.server_socket.accept()  # connect to the client which will send the pickled resource
        print('connected to ', client_address)
        stringCollect = []
        c = 0
        dataType = client_socket.recv(bufsize).decode()
        print(dataType)
        numberOfResultsToRetrieve = 0
        if dataType == "isIndexLoaded":
            print('sending ', str(self.indexIsLoaded))
            client_socket.send(str(self.indexIsLoaded).encode())
        else:
            client_socket.send("typeCallback".encode())


            if dataType == "query" or dataType == "queryf":
                numberOfResultsToRetrieve = int(client_socket.recv(bufsize).decode())
                client_socket.send("kCallback".encode())
            sizeData = int(client_socket.recv(bufsize).decode())
            client_socket.send("sizeCallback".encode())
            dataLength = 0
            bar = progressbar.ProgressBar(max_value=sizeData)
            while dataLength < sizeData:
                data = client_socket.recv(bufsize)
                dataLength+=len(data)
                if not data:
                    print('broke early')
                    break
                else:
                    stringCollect.append(data)
                c+=1
                bar.update(dataLength)
            allData = b''.join(stringCollect)
            if dataType == "index":
                self.loadIndex(allData)
                if self.indexIsLoaded:
                    client_socket.send("indexLoadedCallback_1".encode())
                else:
                    client_socket.send("indexLoadedCallback_0".encode())
            elif dataType == "query" or dataType == "queryf":
                try:
                    if dataType == "query":
                        rDict = self.queryImage(allData,numberOfResultsToRetrieve)
                    elif dataType == "queryf":
                        rDict = self.queryFeatures(allData, numberOfResultsToRetrieve)
                    rDict_data = pickle.dumps(rDict)
                    dataLength = len(rDict_data)
                    client_socket.send(str(dataLength).encode())
                    callback = client_socket.recv(bufsize).decode()
                    if callback == "sizeCallback":
                        client_socket.send(rDict_data)
                except:
                    print('couldnt query')


            client_socket.close()

    def loadIndex(self,data):
        indexResource = Resource('index', data, 'application/octet-stream')
        print('starting up index...')
        # try:
        self.queryIndexObject = queryIndex(indexResource,id=self.port)
        print('index loaded!')
        self.indexIsLoaded = True
        # except:
        #     print('error loading index!')
    def queryImage(self,data,numberOfResultsToRetrieve):
        if self.indexIsLoaded:
            imageResource = Resource("",data,'application/octet-stream')
            print('running query!')
            result = self.queryIndexObject.queryImage(imageResource, numberOfResultsToRetrieve)
            if result is None:
                return 'bad'
            resultDict = {'probeImage':result.probeImage,'I':result.I,'D':result.D,'map':result.map,'scores':result.scores}
            print('query done')
            return resultDict
        else:
            print('Error: no index loaded!')
            return None
    def queryFeatures(self,data,numberOfResultsToRetrieve):
        if self.indexIsLoaded:
            print('running query!')
            featureResource = Resource('', pickle.loads(data),'application/octet-stream')
            result = self.queryIndexObject.queryFeatures(featureResource, numberOfResultsToRetrieve)
            resultDict = {'probeImage':result.probeImage,'I':result.I,'D':result.D,'map':result.map,'scores':result.scores}
            return resultDict
        else:
            print('Error: no index loaded!')
            return None
    def startServer(self,port,bufsize):
        t = Thread(target=self.serverThread,args=((port,bufsize),))
        t.start()

port  = int(sys.argv[1])
address = socket.gethostname()
indexFile = None
id = None
if len(sys.argv) > 2:
    if not sys.argv[2] == 'local':
        address = sys.argv[2]
if len(sys.argv) > 3:
    indexFile = sys.argv[3]
if len(sys.argv) > 4:
    id = sys.argv[4]
bufsize = 2048
print('starting index with ' , indexFile)
server = queryIndexServer(port,address=address,indexFile=indexFile,id=id)
server.mainDispatcher(port,bufsize,6)
# server.runServerThreadLoop((port,bufsize))

