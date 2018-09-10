import os
import sys
from scipy.misc import imread
import rawpy
import numpy
import io
import cv2
import progressbar
def _deserialize_image_file(file, flatten=False):
  fail = False
  try:
      imageStream = io.BytesIO()
      imageStream.write(data)
      imageStream.seek(0)
      imageBytes = numpy.asarray(bytearray(imageStream.read()), dtype=numpy.uint8)
      img = cv2.imdecode(imageBytes, cv2.IMREAD_COLOR)
  except Exception as e:
      fail = True

  if not fail:
      return img

  with io.BytesIO(data) as stream:
      fail = False
      try:
          img = imread(stream, flatten=False)
      except Exception as e:
          fail = True

      if not fail:
          print('Read image with scipy.')
          return img

      fail = False
      try:
          img = rawpy.imread(stream).postprocess()
      except Exception as e:
          fail = True

      if not fail:
          print('Read image with rawpy.')
          return img

  print('Could not read image')
  return None

def _deserialize_image(data, flatten=False):
  fail = False
  try:
      imageStream = io.BytesIO()
      imageStream.write(data)
      imageStream.seek(0)
      imageBytes = numpy.asarray(bytearray(imageStream.read()), dtype=numpy.uint8)
      img = cv2.imdecode(imageBytes, cv2.IMREAD_COLOR)
  except Exception as e:
      fail = True

  if not fail:
      return img

  with io.BytesIO(data) as stream:
      fail = False
      try:
          img = imread(stream, flatten=False)
      except Exception as e:
          fail = True

      if not fail:
          print('Read image with scipy.')
          return img

      fail = False
      try:
          img = rawpy.imread(stream).postprocess()
      except Exception as e:
          fail = True

      if not fail:
          print('Read image with rawpy.')
          return img

  print('Could not read image')
  return None

imgListFile = sys.argv[1]
imgFolder = sys.argv[2]
outputFolder = sys.argv[3]
outputIndexFile = sys.argv[4]

with open(imgListFile,'r') as fp:
    imageList = fp.readlines()

outputIndexList = ['TaskID|ProvenanceProbeFileID|ProvenanceProbeFileName|ProvenanceProbeWidth|ProvenanceProbeHeight']
bar = progressbar.ProgressBar()
for f in bar(imageList):
    fileName = os.path.join(imgFolder,f)
    parts = f.split('.')
    if len(parts) > 1:
        id = parts[0]
        outputName = id+'.png'
        outputPath = os.path.join(outputFolder,outputName)
        with open(f,'rb') as data:
            img = _deserialize_image(data)
        if img is not None:
            cv2.imwrite(outputPath,img)
            line = 'provenancefiltering|' + id + '|' + os.path.join(os.path.basename(outputFolder),outputName) + '|' + str(img.shape[1]) + '|' + str(img.shape[0])
            outputIndexList.append(line)
with open(outputIndexFile,'w') as fp:
    fp.write('\n'.join(outputIndexList))


