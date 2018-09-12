#this class will be modified by the MediFor Compute team (It should not be touched by TA1 provenance performers)
#It provides client stubs that can be called for provenance filtering that are guarenteed to 
#run against all indices

#eventually the hope is to load the entire index into RAM to minimize disk read / write by wrapping these using MESOS / Docker

import numpy as np
import pickle
import traceback
import sys
from resources import Resource
from queryIndex import  queryIndex,filteringResults,queryIndex_Client
import multiTierFeatureBuilder
import os
from joblib import Parallel,delayed
import progressbar
import scoreMerge
from featureExtraction import featureExtraction

class distributedQuery:
      indexFiles=[]
      imageDirectory=""
      currentQueryFeatureResource = None
      #Set this to true if you are testing this locally (loads the query with a single index once for faster querying)
      isTest = False
      # Set this to true if you are using socket servers to run your queries
      useServers = False
      serversAndPorts = []
      defaultServerName = '127.0.0.1'
      #Directories for indexes and images 
      def __init__(self, indexDir,outputImageDir):
          self.indexFiles = []
          for filename in os.listdir(indexDir):
             self.indexFiles.append(os.path.join(indexDir,filename))
          self.imageDirectory=outputImageDir
          if self.isTest:
              indexfile = open(self.indexFiles[0], 'rb')
              indexResource = Resource('index', indexfile.read(), 'application/octet-stream')
              self.curQuery = queryIndex(indexResource)
              self.currentQueryFeatureResource = None
      # currently very ineffcient due to disk read. Will parallelize
      # queryImages is an array of image resourcese

      def setServerList(self,serverListFile):

          self.serversAndPorts = []
          with open(serverListFile,'r') as fp:
              serverList = fp.readlines()
          for s in serverList:
              parts = s.split(',')
              # print('setting server ', parts[0],' on port ',parts[1])
              self.serversAndPorts.append((parts[0],int(parts[1])))
      def queryImages (self, queryImages, numberOfResultsToRetrieve,):
          allresults = []
          if self.useServers:
              port = 8000
              # NOT FOR USE WITH SCALABLE API, FOR USE ONLY ON LOCAL NOTRE DAME SERVERS. PLEASE SEE ELSE STATEMENT.
              for image in queryImages:
                  # print('running image ')
                  if self.serversAndPorts is not None and not self.serversAndPorts == []:
                      allIndexResults = Parallel(n_jobs=len(self.serversAndPorts))(
                          delayed(runQuery)(image, numberOfResultsToRetrieve, port, 'Image',address) for address,port in self.serversAndPorts)
                  else:
                      allIndexResults = Parallel(n_jobs=len(self.indexFiles))(
                          delayed(runQuery)(image, numberOfResultsToRetrieve, port,'Image',self.defaultServerName) for port in range(port,port+len(self.indexFiles)))
                  result = filteringResults()

                  result = scoreMerge.mergeScoreSet(allIndexResults)
                  # for r in bar(allIndexResults):
                  #     result.mergeScores(r)
                      # print(result.scores)
                  allresults.append(result)
          else:
              for i in queryImages:
                 allresults.append(filteringResults())
              for index in self.indexFiles:
                 # This code is put here only for testing: Prevents the distributed query from reloading the index file every damn time you query an image
                 if not self.isTest:
                     indexfile = open(index, 'rb')
                     indexResource = Resource('index', indexfile.read(), 'application/octet-stream')
                     print('initializing a new query index....')
                     self.curQuery = queryIndex(indexResource)
                     #print('index size: ', self.curQuery.index.ntotal)
                 c=0
                 for image in queryImages:
                     result = self.curQuery.queryImage(image,numberOfResultsToRetrieve)
                     print(result.scores)
                     allresults[c].mergeScores(result)
                     c=c+1
              c = 0
              print(allresults)
          return allresults
       
      def getWorldImage(self,fileKey):
          filePath = os.path.join(self.imageDirectory,fileKey)
          worldImageResource= Resource.from_file(fileKey, filePath)
          return worldImageResource

      # currently very ineffcient due to disk read. Wll parallelize
      # queryImages is an array of image resourceseeryFeatures is an array of Images
      def queryFeatures (self, queryFeatures, numberOfResultsToRetrieve,ignoreIDs = []):
          allresults = []
          # for i in queryImages:
          #    allresults.append(filteringResults())
          # TODO:Feature concatination for faster query batches
          # concatinate features
          # allFeats = []
          # featureExtractor = featureExtraction()
          # for feature in queryFeatures:
          #     allFeats.append(self.deserializeFeatures(feature))
          # allFeats = np.concatenate(allFeats,axis=0)
          # allFeatsResource = featureExtractor.createOutput(Resource("", featureExtractor.serializeFeature(allFeats), 'application/octet-stream'))
          # allFeatsResource = ['supplemental_information']['value']
          if self.useServers:
              # NOT FOR USE WITH SCALABLE API, FOR USE ONLY ON LOCAL NOTRE DAME SERVERS. PLEASE SEE ELSE STATEMENT.
              port = 8000
              for feature in queryFeatures:
                  # print('running image ')
                  if self.serversAndPorts is not None and not self.serversAndPorts == []:
                      allIndexResults = Parallel(n_jobs=len(self.serversAndPorts))(
                          delayed(runQuery)(feature._data, numberOfResultsToRetrieve, port, 'Features', address) for port, address in
                          self.serversAndPorts)
                  else:
                      allIndexResults = Parallel(n_jobs=len(self.indexFiles))(delayed(runQuery)(feature._data, numberOfResultsToRetrieve, port,'Features',self.defaultServerName) for port in
                      range(port, port + len(self.indexFiles)))
                  result = filteringResults()
                  bar = progressbar.ProgressBar()
                  for r in bar(allIndexResults):
                      result.mergeScores(r,ignoreIDs=ignoreIDs)
                      # print(result.scores)
                  allresults.append(result)
          else:
              for i in queryFeatures:
                 allresults.append(filteringResults())
              for index in self.indexFiles:
                 indexfile = open(index,'rb')
                 indexResource = Resource('index', indexfile.read(),'application/octet-stream')
                 # curQuery = queryIndex(indexResource)
                 if not self.isTest and self.curQuery is None:
                     print('initializing a new query index....')
                     self.curQuery = queryIndex(indexResource)
                 c=0
                 for feature in queryFeatures:
                     result = self.curQuery.queryFeatures(feature,numberOfResultsToRetrieve)
                     allresults[c].mergeScores(result,ignoreIDs=ignoreIDs)
                     c=c+1
          return allresults

def runQuery(image,numberOfResultsToRetrieve,port,type,address):
    indexResource = Resource('index', "none".encode(), 'application/octet-stream')
    curQuery = queryIndex_Client(indexResource, port=port,address=address)
    if type == "Image":
        q = curQuery.queryImage(image, numberOfResultsToRetrieve)
    elif type == "Features":
        q = curQuery.queryFeatures(image,numberOfResultsToRetrieve)
    if q is None:
        print('got none from ', address)
    else:
        # print('got result from ', address)
        pass
    # print('done on ', port)
    return q

def loadServer(indexpath,port):
  indexfile = open(indexpath, 'rb')
  indexResource = Resource('index', indexfile.read(), 'application/octet-stream')
  client = queryIndex_Client(indexResource,port=port)
def startServers(indexDir,startingPort):
  from threading import Thread
  c = 0
  for index in os.listdir(indexDir):
      print('sending index ', index)
      indexpath = os.path.join(indexDir,index)
      t = Thread(target=loadServer,args=(indexpath,startingPort+c))
      t.start()
      c+=1
def concatFeatures(self,r1,r2):
  featureExtractor = featureExtraction()
  cat = np.vstack((self.deserializeFeatures(r1['supplemental_information']['value']),self.deserializeFeatures(r2['supplemental_information']['value'])))
  filename = r1['supplemental_information']['value'].key
  featureResource = Resource(filename, featureExtractor.serializeFeature(cat), 'application/octet-stream')
  return featureExtractor.createOutput(featureResource)

if __name__ == "__main__":
    indexfolder = sys.argv[1]
    startingPort = int(sys.argv[2])
    serverNames = 'localhost'
    if len(sys.argv) > 2:
        startServers = sys.argv[3]
    startServers(serverNames,startingPort)
