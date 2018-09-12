#simple example with color histograms (good for speed, bad for accuracy...)
import numpy as np
from scipy import ndimage
# from imageio import imread
from scipy.misc import imread, imsave

import cv2
import io
import pickle as pickle
import traceback
import featureExtractor
import rawpy
import math
#import pickle

#use the logging class to write logs out
import logging
from resources import Resource

class featureExtraction:
      #Put variables for your apprach here      
      detetype = 'SURF3'
      desctype = 'SURF3'
      kmax = 5000
      #modify these variables
      algorithmName="ND_dsurf_5000_filtering"
      algorithmVersion="1.0"
      featuredims = 64
      #Set any parameteres needed fo the model here
      #def __init__(self):          

      #extract features from a single image and save to an object
         #input image output object containing
         #worldImage conatains a resource object (includes the filename and data) 
      def processImage (self, worldImage,flip=False,downsize=False):
          
          #extract data
          filename  = worldImage.key
          imagedata  = worldImage._data
          image = self.deserialize_image(imagedata)
          if downsize:
              image = self.downSize(image)
          if image is not None:
              if flip:
                  image = cv2.flip(image,0)
              #extract your features, in this example color histograms
              featuresStruct = featureExtractor.local_feature_detection_and_description(filename, self.detetype, self.desctype, self.kmax, image, mask=None, dense_descriptor=False,
                                                default_params=True)
              features = featuresStruct[1]

              #serialize your features into a Resource object that can be written to a file
              #The resource key should be the input filename
              if not features == [] and features is not None:
                  totalFeatures = features.shape[1]
              else:
                  features = np.zeros((0,0),dtype='float32')
                  totalFeatures = 0
              featureResource = Resource(filename, self.serializeFeature(features), 'application/octet-stream')
              return self.createOutput(featureResource)
          return None

      #creates the API struct
      def createOutput(self,featureResource):
          return {'algorithm': self.createAlgOutput(), 'supplemental_information': self.createFeatureOutput(featureResource)}

      def createAlgOutput(self):
          return  {'name': self.algorithmName.replace(" ", ""), 'version': self.algorithmVersion.replace(" ", "")}

      def createFeatureOutput(self,featureResource):
          return  {'name': 'provenance_features', 'description': 'features extracted for provenance filtering', 'value': featureResource}

      def serializeFeature(self, features):
          dims = features.shape
          data = np.insert(features.flatten().astype('float32'),dims[0]*dims[1],dims)
          return data

      def deserialize_image(self,data, flatten=False):
          fail = False
          try:
              imageStream = io.BytesIO()
              imageStream.write(data)
              imageStream.seek(0)
              imageBytes = np.asarray(bytearray(imageStream.read()), dtype=np.uint8)
              img = cv2.imdecode(imageBytes, cv2.IMREAD_COLOR)
              img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
          except Exception as e:
              print(e)
              fail = True

          if not fail:
              return img

          with io.BytesIO(data) as stream:
              fail = False
              try:
                  img = imread(stream, flatten=False)
              except Exception as e:
                  fail = True
                  print(e)
              if not fail:
                  print('Read image with scipy.')
                  return img

              fail = False
              try:
                  img = rawpy.imread(stream).postprocess()
              except Exception as e:
                  fail = True
                  print(e)
              if not fail:
                  print('Read image with rawpy.')
                  return img

          print('Could not read image')
          return None

      def downSize(img):
          maxPixelCount = 2073600  # HDV

          newHeight = 0
          newWidth = 0

          if img.shape[0] * img.shape[1] > maxPixelCount:
              aspectRatio = img.shape[0] * pow(img.shape[1], -1)

              newWidth = int(round(math.sqrt(maxPixelCount / aspectRatio)))
              newHeight = int(round(aspectRatio * newWidth))

              return cv2.resize(img, (newWidth, newHeight))
          else:
              return img
#      def deserialize_image(self,data, flatten=False):
#          with io.BytesIO(data) as stream:
#             try:
#                img = imread(stream, flatten=flatten)
#                return img
#             except Exception as e:
#                 print('Using rawpy')
#             try:
#                 img = rawpy.imread(stream).postprocess()
#                 return img
#             except Exception as e:
#                 print('error: ',e)
#                 return None


