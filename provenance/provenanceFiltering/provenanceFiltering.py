import distributedQuery
from queryIndex import  queryIndex,filteringResults
import multiTierFeatureBuilder
import numpy as np
import pickle
import traceback
import cv2
import sys
import os
from resources import Resource
import scoreMerge

class provenanceFiltering:
   
      #modify these variables
      algorithmName="SystemName"
      algorithmVersion="1.0"

      scalableQuery=None
      useSocketServers = False
      useSocketServerFile = False
      MultiTier = False
      #image results will go here
      imageOutputDir=""

      #Load Index at initialization
      #indexFileResource is a distributedQuery object 
      def __init__(self, distributedQueryObject):
          self.scalableQuery=distributedQueryObject
          if os.path.exists('./serverList.txt') and self.useSocketServers and self.useSocketServerFile:
              print('setting server list!')
              self.scalableQuery.setServerList('./serverList.txt')
          #image results from the distributed query will go here
          self.imageOutputDir= self.scalableQuery.imageDirectory

      def showTopResults(self,results,k):
          import featureExtraction
          import math
          import matplotlib.pyplot as plt
          fe = featureExtraction.featureExtraction()
          images = list(results.scores.keys())
          numImages = min(k,len(images))
          dim = math.ceil(np.sqrt(numImages))
          fig = plt.figure()
          i = 0
          for im in images[:numImages]:
              image = fe.deserialize_image(self.scalableQuery.getWorldImage(im)._data)
              sub = fig.add_subplot(dim,dim,i+1)
              sub.imshow(image)
              i+=1

          pass
      #probeImage conatins Image Data
      def processImage (self, probeImage, numberOfResultsToRetrieve):
          #get filename
          probeFilename  = probeImage.key
          #create score object
          resultScores =filteringResults()

          allQueries = []
          allQueries.append(probeImage)

          #this can be called as many times as needed
          #image files will be put in 
          allResults = self.scalableQuery.queryImages(allQueries,numberOfResultsToRetrieve)
          #Tier2
          maxScore = allResults[0].scores[list(allResults[0].scores.keys())[0]]
          if self.MultiTier and maxScore > .03: #only do multitier search if the first query gets enough votes (3% of all features match)
              try:
                  mainResult = allResults[0]
                  tier2ImageResources = []
                  for r in list(mainResult.scores):
                      tier2ImageResources.append(self.scalableQuery.getWorldImage(r))
                  fullTier2FeatureResource,tier2FeatureSets, featureIDList, featureObjectIDList, featureDictionary,queryOrResultList,featureSetMap,visDict = multiTierFeatureBuilder.getTier2Features(probeImage,tier2ImageResources,30)
                  if fullTier2FeatureResource is not None:
                    # allTier2Results = self.scalableQuery.queryFeatures([fullTier2FeatureResource['supplemental_information']['value']], 100,ignoreIDs=list(allResults[0].map))
                    allTier2Results = self.scalableQuery.queryFeatures(tier2FeatureSets,75,ignoreIDs=list(allResults[0].map))
                    print('found results for ',len(allTier2Results),' tier 2 objects')
                    # allTier2Scores = multiTierFeatureBuilder.getObjectScores(allTier2Results[0],featureIDList,featureObjectIDList,featureDictionary,queryOrResultList,objectWise=True,ignoreIDs=list(allResults[0].map))
                    allTier2Scores = allTier2Results
                    finalTier2Ranks = filteringResults()
                    for r in allTier2Scores:
                        r.I = None
                        r.D = None
                        r.pairDownResults(2)
                    print('merging tier 2 scores')
                    for r in allTier2Scores:
                        finalTier2Ranks.mergeScores(r,ignoreIDs=allResults[0].map)
                    # scoreMerge.mergeScoreSet(allTier2Scores)
                    allResults[0].mergeScores(finalTier2Ranks)
                  else:
                      allTier2Results = None
              except:
                  print('failed tier 2 search')
                  allTier2Results = None
          allResults[0].normalizeResults()
          outputJson = self.createOutput(probeFilename,allResults[0])

          return outputJson


      def createOutput(self,probeFilename, resultScores):
          return {'algorithm': self.createAlgOutput(), 'provenance_filtering': self.createFilteringOutput(probeFilename,resultScores)}

      def createAlgOutput(self,):
          return  {'name': self.algorithmName.replace(" ", ""), 'version': self.algorithmVersion.replace(" ", "")}

      def createFilteringOutput(self, probeFilename,resultScores):
          return {'probe': probeFilename, 'matches':resultScores.scores}    
