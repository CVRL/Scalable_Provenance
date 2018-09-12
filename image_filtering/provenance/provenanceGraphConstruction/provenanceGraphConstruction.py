import sys
sys.path.insert(2, '/root/opencv-2.4.13.5/release/lib/')
sys.path.insert(2, '/root/opencv-2.4.13.5/modules/python/src2/')

from featureExtraction import featureExtraction
import numpy as np
import pickle
import traceback
import cv2
import sys
import io
import rawpy
from scipy.misc import imread, imsave
from resources import Resource

#for tutorial example
import random

# Notre Dame libraries and dependencies
import time, AdjacencyMatrixBuilder, ClusterGraphBuilder

#Thiss class produces the API output needed for the Provenance Graph JSON
class associationResult:
      association={}
      def __init__(self):
          self.association['link_type']= 'provenance'
          self.association['link']= {}

      #probeID = probe filename
      #donorID = donor filename
      #score = confidence of the link
      #probeMask, donorMask = localization objects. Not used in evaluation quite yet. Can be created with localization_output given a mask)
      def addLink(self,probeID, donorID,score, probeMask=None, donorMask=None, confidence = None, explanation=None):
          self.association['link']['from']=probeID
          self.association['link']['to']=donorID
          self.association['link']['score']=score
          if probeMask:
              self.association['link']['from_mask']=probeMask
          if donorMask:
              self.association['link']['to_mask']=donorMask
          if confidence:
              self.association['link']['confidence']=confidence
          if explanation:
              self.association['link']['explanation']=explanation

      def getAssociation(self):
          return self.association.copy()

class provenanceGraphConstruction:
      #Put variables for your algorithm here
      rankSize = 100
      npzFolder = './npz/'
   
      #modify these variables
      #algorithmName="YourSystemName"
      #algorithmVersion="1.0"
      algorithmName="Purdue-NotreDame"
      algorithmVersion="1.0"

      #Initialization if needed 
      #def __init__(self):
          #Initialization code
          

      #probeImage = conatains Image Resources for the probe that were used
      #filteringImages = (dict) conatains Image Resources for matching images found during filtering
      #filteringResults = (dict) conatains match score for matching images found during filtering
      #return Provenance Grap API Json (dict)
      def processGraph (self, probeImage, filteringImages, filteringResults):
          #get prove filename and image
          probeFilename  = probeImage.key
          #probeImage = self.deserialize_image(probeImage.data)
          npzFilename = self.npzFolder + probeFilename + '.npz'

          print "Processing probe", probeFilename, "and saving file", npzFilename
          print "AdjacencyMatrixBuilder.buildAdjMat", probeImage.key, self.rankSize, npzFilename

          i_time = time.time()
          AdjacencyMatrixBuilder.buildAdjMat(probeImage, filteringImages, self.rankSize, npzFilename)
          print("--- %s seconds ---" % (time.time() - i_time))
          print ""

          print "Processing NPZ file", npzFilename
          print "ClusterGraphBuilder.buildGraph", npzFilename
          i_time = time.time()
          dag = ClusterGraphBuilder.buildGraph(npzFilename)
          print("--- %s seconds ---" % (time.time() - i_time))
          print ""
     
          associations = []
          
          ########################################  
          #put your proveance graph code here, creating new associations you find for provenance nodes related to the probe image
          #Replace this with your code, hopefully you are doing somethign better then randomly assigning assciations
          r = 0
          c = 0

          i = 0
          for fromKey in filteringImages:
              i = i + 1

              #This commented out line is an example of how to open the image. Commented out due ot speed reasons for the tuorial
              #fromImage = self.deserialize_image(filteringImages[fromFileName].data)

              #score may not be reliable if coming from oracle provenance results
              #fromScore = filteringResults[fromKey]
              fromFileName = filteringImages[fromKey].key

              if fromFileName == probeFilename:
                  r = 0
              else:
                  r = i

              j = 0
              for toKey in filteringImages:
                  j = j + 1

                  #This commented out line is an example of how to open the image. Commented out due ot speed reasons for the tuorial
                  #toImage = self.deserialize_image(filteringImages[toFileName].data)

                  #score may not be reliable if coming from oracle provenance results
                  #toScore = filteringResults[toKey]
                  toFileName = filteringImages[toKey].key
                  if toFileName == probeFilename:
                      c = 0
                  else:
                      c = j

                  #if fromFileName==toFileName:
                  #   continue
                   
                  #random link score
                  #linkConfidence = random.uniform(0, 1)
                  #if(linkConfidence>.9):
                  #    #add an association result
                  #    newAssociation = associationResult()
                  #    newAssociation.addLink(fromFileName,toFileName,linkConfidence)
                  #    associations.append(newAssociation.getAssociation())

                  linkConfidence = 0.0
                  if r < dag.shape[0] and c < dag.shape[1]:
                      linkConfidence = dag[r][c]
                  if (linkConfidence > 0.0):
                      # add an association result
                      newAssociation = associationResult()
                      newAssociation.addLink(fromFileName, toFileName, linkConfidence)
                      associations.append(newAssociation.getAssociation())

          #########################################

          #Valid optout values are Processed | NonProcessed | OptOut
          optout="Processed"
 
          outputJson = self.createOutput(optout,associations)
          return outputJson
          

      def createOutput(self,outout, associationArray):
          return {'algorithm': self.createAlgOutput(), 'provenance_graph': self.createAssociationOutput(outout,associationArray)}

      def createAlgOutput(self):
          return  {'name': self.algorithmName.replace(" ", ""), 'version': self.algorithmVersion.replace(" ", "")}

      #optout:  ( Processed | NonProcessed | OptOut)
      def createAssociationOutput(self, optout, associations):
          return {'optout': optout, 'association':associations}

      #def deserialize_image(self,data, flatten=False):
      #    with io.BytesIO(data) as stream:
      #       return imread(stream, flatten=flatten) 
      def deserialize_image(self, data, flatten=False):
          fail = False
          try:
              imageStream = io.BytesIO()
              imageStream.write(data)
              imageStream.seek(0)
              imageBytes = numpy.asarray(bytearray(imageStream.read()), dtype=numpy.uint8)
              img = cv2.imdecode(imageBytes, cv2.IMREAD_COLOR)
              img = img.astype(numpy.uint8)
          except Exception as e:
              fail = True
          
          if not fail and img is not None:
              return img
          
          with io.BytesIO(data) as stream:
              fail = False
              try:
                  img = imread(stream, flatten=False)
                  img = img.astype(numpy.uint8)
              except Exception as e:
                  fail = True
              
              if not fail and img is not None:
                  print('Read image with scipy.')
                  return img
              
              fail = False
              try:
                  img = rawpy.imread(stream).postprocess()
                  img = img.astype(numpy.uint8)
              except Exception as e:
                  fail = True
              
              if not fail and img is not None:
                  print('Read image with rawpy.')
                  return img
          
          print('Could not read image')
          return numpy.zeros((10, 10), dtype=numpy.uint8)

      #create localization json object
      #mask is an grey scale image
      #Not used in the example, but here in case teams wish to evetually produce a mask
      def localization_output(mask, maskThreshold=None, maskOptOut=None):
          localization ={}
          if mask is not None:
              localization['mask']= Resource('mask', serialize_image(mask), 'image/png')
          if maskThreshold:
              localization['mask_threshold']= maskThreshold
          if maskOptOut is not None:
              localization['mask_optout"']= Resource('maskOptOut', serialize_image(maskOptOut), 'image/png')
          return localization
