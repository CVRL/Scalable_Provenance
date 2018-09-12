from featureExtraction import featureExtraction

import numpy as np
import pickle
import traceback
import cv2
import sys
from resources import Resource
import faiss
import indexfunctions
import fileutil
import os
import json
import collections
import socket
import time
from scoreMerge import mergeResultsMatrix

from threading import Thread
#this class will load a single index and allow queries using either an image or features
#This allows for querying on a distributed index

class struct(object):
    pass


class queryIndex:
      index=None
      id = None
      preproc = None
      ngpu = 0
      nprobe = 16
      tempmem = 1536 * 1024 * 1024  # if -1, use system default
      IDToImage = {}
      isDeserialized = False
      algorithmName = "ND_dsurf_5000_filtering"
      currentQueryFeatureResource = None
      #Load Index at initialization
      #indexFileResource is a resource object 
      def __init__(self, indexFileResource,index=None,preproc=None,map=None,id=None):
          if index is not None and preproc is not None and map is not None:
              self.index = index
              self.preproc = preproc
              self.IDToImage = map
          else:
              self.deserializeIndex(indexFileResource,id)
          cacheroot = fileutil.getResourcePath(self.algorithmName)
          self.currentQueryFeatureResource = None
          self.gpu_resources = indexfunctions.wake_up_gpus(self.ngpu, self.tempmem)
          print("gpus found: ", len(self.gpu_resources))
          # preproc = faiss.read_VectorTransform(self.preproc_cachefile)
          # with open(self.IDMap_cachefile,'r') as fp:
          #   self.IDToImage = json.load(fp)\

      #queryImage conatins resourse object containing image
      def queryImage (self, imageResource, numberOfResultsToRetrieve):
          #create score object
          resultScores =filteringResults()
          #get probe features
          featureExtractor = featureExtraction()
          feature  = featureExtractor.processImage(imageResource)
          print('processed image to featres...')
          if feature is not None:
              feature_r = featureExtractor.processImage(imageResource,flip=True)
              self.currentQueryFeatureResource = (feature,feature_r)
              features_all = self.concatFeatures(feature,feature_r)
              results = self.queryFeatures(features_all['supplemental_information']['value'], numberOfResultsToRetrieve)
              return results
          return None

      #queryFeature contains resource object containing feature(s)
      #this allows for non-image queries
      def queryFeatures (self, featureResource, numberOfResultsToRetrieve):
          numberOfResultsToRetrieve = int(numberOfResultsToRetrieve)
          ps = None
          if self.ngpu > 0:
              ps = faiss.GpuParameterSpace()
              ps.initialize(self.index)
              #ps.set_index_parameter(self.index, 'nprobe', self.nprobe)
          features = self.deserializeFeatures(featureResource)
          pfeatures = self.preproc.apply_py(indexfunctions.sanitize(features))
          D, I = self.index.search(pfeatures, numberOfResultsToRetrieve)
          sortedIDs, sortedVotes,maxvoteval = indexfunctions.tallyVotes(D,I,numcores=1)
          #print('number of ids: ',len(self.IDToImage))
          # voteScores = 1.0 * sortedVotes / (1.0 * np.max(sortedVotes))
          voteScores = 1.0 * sortedVotes / (maxvoteval)
          resultScores = filteringResults()
          resultScores.D = D
          resultScores.I = I
          #print(list(self.IDToImage.keys())[0]+'\n')
          for i in range(0, min(len(sortedIDs), numberOfResultsToRetrieve)):
              id = sortedIDs[i]
              id_str = str(id)
              #print(id_str)
              if id_str in self.IDToImage:
                  imname = self.IDToImage[id_str]
                  score = voteScores[i]
                  resultScores.addScore(imname,score,ID=id)
          resultScores.pairDownResults(numberOfResultsToRetrieve)
          return resultScores
      # @profile(precision=5)
      def deserializeIndex (self, indexFileResource,id=None):
          # the joined index file contains populated_index,empty_trained_index,preproc,IDMap, (possible) IVF file, in that order
          bytearrays = fileutil.splitMultiFileByteArray(indexFileResource._data,12,6)
          tmppath = 'tmp'
          mv = memoryview(indexFileResource._data)
          if id is not None:
              tmppath += '_'+str(id)
          all_tmp_paths = []
          mmap_path = None
          count = 0
          for bytearray in bytearrays:
              if count == 4:
                  # This is the path to where the ivf mmap file must be stored
                  mmap_path = mv[bytearray[0]:bytearray[1]].tobytes().decode('ascii')
                  print('making directory ', os.path.dirname(mmap_path), ' to store ivf data')
                  if not os.path.exists(os.path.dirname(mmap_path)):
                      os.makedirs(os.path.dirname(mmap_path))
              p = tmppath+str(count)+'.dat'
              if count == 5:
                  p = mmap_path
              with open(p,'wb') as fp:
                  fp.write(mv[bytearray[0]:bytearray[1]])
              count+=1
              all_tmp_paths.append(p)

          self.preproc = faiss.read_VectorTransform(all_tmp_paths[2])
          self.IDToImage = json.loads(mv[bytearrays[3][0]:bytearrays[3][1]].tobytes().decode('ascii'))
          #print(mv[bytearrays[3][0]:bytearrays[3][1]].tobytes())
          del bytearrays
          del indexFileResource
          print('initializing index...')
          self.index = faiss.read_index(all_tmp_paths[0],faiss.IO_FLAG_MMAP)


          print('index size: ',self.index.ntotal)
          print('map size:',len(self.IDToImage))
          self.isDeserialized = True
          return(self.index,self.preproc,self.IDToImage)

      def deserializeFeatures(self, featureResource):
          data = featureResource._data
          return np.reshape(data[:-2], (int(data[-2]), int(data[-1])))

      def concatFeatures(self,r1,r2):
          featureExtractor = featureExtraction()
          cat = np.vstack((self.deserializeFeatures(r1['supplemental_information']['value']),self.deserializeFeatures(r2['supplemental_information']['value'])))
          filename = r1['supplemental_information']['value'].key
          featureResource = Resource(filename, featureExtractor.serializeFeature(cat), 'application/octet-stream')
          return featureExtractor.createOutput(featureResource)


#Thiss class produces the data needed for the Provenance Filtering JSON
#the function merge will be used to merge results when indexing is parallelized
# you can modify the class implementations to meet your needs, but function calls 
# should be kept the same
class filteringResults:
      map = {}
      scores = collections.OrderedDict()
      def __init__(self):
          self.probeImage = ""
          self.I = None
          self.D = None
          self.map = {}
          self.scores = collections.OrderedDict()
      def addScore(self,filename, score,ID=None):
          self.scores[filename]=score
          if ID is not None:
              self.map[ID] = filename
      #this function merges two results
      def mergeScores(self,additionalScores,ignoreIDs = []):
          if self.I is not None and self.D is not None and additionalScores is not None and additionalScores.I is not None and additionalScores.D is not None:
              # Merge results based on I and D matrixes (not heuristic!)
              mergedresults = mergeResultsMatrix(self.D,additionalScores.D,self.I,additionalScores.I,self.map,additionalScores.map,k=min(len(self.scores),self.I.shape[1]),numcores=12)
              self.I = mergedresults[0]
              self.D = mergedresults[1]
              self.map = mergedresults[2]
              sortedIDs, sortedVotes,maxvoteval = indexfunctions.tallyVotes(self.D, self.I)
              # voteScores = 1.0 * sortedVotes / (1.0 * np.max(sortedVotes))
              voteScores = 1.0 * sortedVotes / (maxvoteval)
              self.scores = collections.OrderedDict()
              for i in range(0, len(sortedIDs)):
                  id = sortedIDs[i]
                  if id not in ignoreIDs:
                      id_str = str(id)
                      if id in self.map:
                          imname = self.map[id]
                          score = voteScores[i]
                          self.addScore(imname, score, ID=id)

          elif additionalScores is None:
              #if additional scores contains nothing don't add anything!
              pass
          elif self.I is None and self.D is None and additionalScores.I is not None and additionalScores.D is not None:
              # Pushing into empty results, just populate the object with the additionalScores
              self.I = additionalScores.I
              self.D = additionalScores.D
              self.map = additionalScores.map
              self.scores = additionalScores.scores

          else:
              # Merge in a heuristic way
              self.scores.update(additionalScores.scores)
              sortedscores = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
              self.scores = sortedscores
          for id in ignoreIDs:
              if id in self.scores:
                  del self.scores[id]
      # this function merges two results
      def dictSort(self, additionalScores):
          od = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
          self.scores.update(additionalScores.scores)
          sortedscores = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
          self.scores = sortedscores

      #Once scores are merged together, at most "numberOfResultsToRetrieve" will be retained
      def pairDownResults(self,numberOfResultsToRetrieve):
          numberOfResultsToRetrieve = int(numberOfResultsToRetrieve)
          if len(self.scores) > numberOfResultsToRetrieve:
              newscores = collections.OrderedDict(
                  sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:numberOfResultsToRetrieve])
              self.scores = newscores
      def normalizeResults(self):
          maxVal = self.scores[list(self.scores.keys())[0]]
          for s in self.scores:
              self.scores[s] = self.scores[s]/maxVal
class queryIndex_Client:
    def __init__(self, indexFileResource=None, address='ndm2',port=8000,bufsize=2048):
        self.TCP_IP = address
        self.TCP_PORT = port
        self.BUFFER_SIZE = bufsize
        #self.s = socket.socket()
        self.indexIsLoaded = self.isIndexLoadedOnServer()
        if not  self.indexIsLoaded:
            print('Server does not have index. Loading index to server')
            self.sendIndexToServer(indexFileResource)
    def isIndexLoadedOnServer(self):
        # print('connecting...')
        s=socket.socket()
        s.connect((self.TCP_IP, self.TCP_PORT))
        s.send("isIndexLoaded".encode())
        callback = s.recv(self.BUFFER_SIZE).decode().rstrip()
        if callback == "True":
            return True
        elif callback == "False":
            return False
        return False
        s.close()

    def queryImage(self,imageResource,numberOfResultsToRetrieve):
        imageData = imageResource._data
        # print('connecting...')
        s = socket.socket()
        s.connect((self.TCP_IP, self.TCP_PORT))
        dataSize = len(imageData)
        s.send("query".encode())
        callback = s.recv(self.BUFFER_SIZE).decode()
        if callback == "typeCallback":
            s.send(str(numberOfResultsToRetrieve).encode())
            callback = s.recv(self.BUFFER_SIZE).decode()
            if callback == "kCallback":

                s.send(str(dataSize).encode())
                callback = s.recv(self.BUFFER_SIZE).decode()
                if callback == "sizeCallback":
                    s.sendall(imageData)
                    callback = s.recv(self.BUFFER_SIZE).decode()
                    if not callback == '':
                        resultSize = int(callback)
                        # print("size of results: ",resultSize)
                        s.send("sizeCallback".encode())
                        dataLength = 0
                        stringCollect = []
                        while dataLength < resultSize:
                            data = s.recv(self.BUFFER_SIZE)
                            dataLength += len(data)
                            if not data:
                                break
                            else:
                                stringCollect.append(data)
                        allData = b''.join(stringCollect)
                        resultDict = pickle.loads(allData)
                        results = filteringResults()
                        if resultDict == 'bad':
                            print('didnt get results from server', self.TCP_IP)
                            return None
                        else:
                            results.probeImage = resultDict['probeImage']
                            results.I = resultDict['I']
                            results.D = resultDict['D']
                            results.map = resultDict['map']
                            results.scores = resultDict['scores']
                        # print('we got it all!')
                            return results
                    return None
        s.close()

    def queryFeatures(self, featureResource, numberOfResultsToRetrieve):
        # print('connecting...')
        s = socket.socket()
        s.connect((self.TCP_IP, self.TCP_PORT))
        data = pickle.dumps(featureResource)
        dataSize = len(data)
        s.send("queryf".encode())
        callback = s.recv(self.BUFFER_SIZE).decode()
        if callback == "typeCallback":
            s.send(str(numberOfResultsToRetrieve).encode())
            callback = s.recv(self.BUFFER_SIZE).decode()
            if callback == "kCallback":

                s.send(str(dataSize).encode())
                callback = s.recv(self.BUFFER_SIZE).decode()
                if callback == "sizeCallback":

                    s.send(data)
                    callback = s.recv(self.BUFFER_SIZE).decode()
                    resultSize = int(callback)
                    s.send("sizeCallback".encode())
                    dataLength = 0
                    stringCollect = []
                    while dataLength < resultSize:
                        data = s.recv(self.BUFFER_SIZE)
                        dataLength += len(data)
                        if not data:
                            break
                        else:
                            stringCollect.append(data)
                    allData = b''.join(stringCollect)
                    resultDict = pickle.loads(allData)
                    results = filteringResults()
                    results.probeImage = resultDict['probeImage']
                    results.I = resultDict['I']
                    results.D = resultDict['D']
                    results.map = resultDict['map']
                    results.scores = resultDict['scores']
                    return results
        s.close()
    def sendIndexToServer(self,indexResource):
        s = socket.socket()
        indexData = indexResource._data
        # print('connecting...')
        s.connect((self.TCP_IP, self.TCP_PORT))
        s.send("index".encode())
        callback = s.recv(self.BUFFER_SIZE).decode()
        print(callback)
        if callback == "typeCallback":
            dataSize = len(indexData)
            print('sending size ', dataSize)
            s.send(str(dataSize).encode())
            callback = s.recv(self.BUFFER_SIZE).decode()
            if callback == "sizeCallback":
                s.send(indexData)
                callback = s.recv(self.BUFFER_SIZE).decode()
                callback = ""
                if callback.startswith('indexLoadedCallback') and callback.endswith('1'):
                    print('index loaded succesfully')
        s.close()


if __name__ == "__main__":
    im1 ='/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/probe/ff7c1f46c84e6efce1cb563bf9d1b65d.JPG'
    im2 = '/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/probe/797a0b5dd9c47a4b391f4cee60ba9354.jpg'
    indexFile = '/home/jbrogan4/Documents/Projects/Medifor/GPU_Prov_Filtering/provenance/tutorial/index/index'
    r = Resource.from_file(os.path.basename(im1),im1)
    indexfile = open(indexFile, 'rb')
    indexResource = Resource('index', indexfile.read(), 'application/octet-stream')
    queryClient = queryIndex_Client(indexResource)
    results = queryClient.queryImage(r,10)
    queryClient2 = queryIndex_Client(indexResource)
    #sendIndexToServer(indexFile)
    #
    print('here')
