# Traces back the rank images that might be connected to a probe.
import itertools, numpy, cv2, KeypointMatcher, ImageDescriptor  # , GraphJSONReaderWriter

import io
from scipy.misc import imread
from resources import Resource
import rawpy

import cv2
import random

#def _deserialize_image(data):
#    with io.BytesIO(data) as stream:
#        return imread(stream, flatten=False)

def _deserialize_image(data, flatten=False):
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


def _areRelatedIJ(i, j, keypoints, descriptions, imgs,
                  minMatchCountThreshold, maxMatchCountThreshold, keypointDistanceThreshold):
    print "Trying to link", i, j

    if imgs[i] is None or imgs[j] is None:
        return []

    if len(keypoints[i]) == 0 or len(keypoints[j]) == 0:
        return []

    matches = KeypointMatcher.match(keypoints[i], descriptions[i], imgs[i],
                                    keypoints[j], descriptions[j], imgs[j])

    matchCount = len(matches)
    if matchCount < maxMatchCountThreshold:
        if matchCount < minMatchCountThreshold:
            return []

        xDistances = numpy.zeros((matchCount, matchCount), float)
        yDistances = numpy.zeros((matchCount, matchCount), float)

        for a in range(0, matchCount):
            keypointA = keypoints[i][matches[a].queryIdx]

            for b in range(a + 1, matchCount):
                keypointB = keypoints[i][matches[b].queryIdx]

                xDist = abs(keypointA.pt[0] - keypointB.pt[0])
                yDist = abs(keypointA.pt[1] - keypointB.pt[1])

                xDistances[a, b] = xDist
                xDistances[b, a] = xDist
                yDistances[a, b] = yDist
                yDistances[b, a] = yDist

            if (numpy.max(xDistances[a, :]) > keypointDistanceThreshold * imgs[i].shape[0] or
                    numpy.max(yDistances[a, :]) > keypointDistanceThreshold * imgs[i].shape[1]):
                return []

    return matches


##def getRelatedImagesToProbe(probeFilePath, jsonRankFilepath, imageDirPath, rankSize,
##                            minMatchCountThreshold = 4, maxMatchCountThreshold = 10,
##                            keypointDistanceThreshold = 0.02):
def getRelatedImagesToProbe(probeImage, filteringImages, rankSize,
                            minMatchCountThreshold=4, maxMatchCountThreshold=10,
                            keypointDistanceThreshold=0.02):
    # obtains the image file paths
    ##imageFilePaths = []
    ##for image in GraphJSONReaderWriter.readRank(jsonRankFilepath, imageDirPath):
    ##    if image[0] != probeFilePath:
    ##        imageFilePaths.append(image[0])
    ##
    ##    if len(imageFilePaths) + 1 == rankSize:
    ##        break
    ##imageFilePaths.append(probeFilePath)

    # images, descriptions, and keypoints
    ##imgs = list(map(lambda p: cv2.imread(p), imageFilePaths))
    ##imageCount = len(imgs)

    imageKeys = []
    imgs = []

    for k in filteringImages:
        if len(imgs) + 1 == rankSize:
            break

        if filteringImages[k].key != probeImage.key:
            imageKeys.append(filteringImages[k].key)
            imgs.append(numpy.array(_deserialize_image(filteringImages[k].data)).astype(numpy.uint8))


    imageKeys.append(probeImage.key)
    imgs.append(numpy.array(_deserialize_image(probeImage.data)).astype(numpy.uint8))
    imageCount = len(imgs)

    keypoints = []
    descriptions = []

    for i in list(map(lambda i: ImageDescriptor.surfDescribe(i), imgs)):
        keypoints.append(i[0])
        descriptions.append(i[1])
    print "Described", imageCount, "images."

    # answers of the method
    probeRelatedImageIndexes = []

    matches = []
    for i in range(imageCount * imageCount):
        matches.append([])

    # current sources
    sourceImageFileIndexes = []
    sourceImageFileIndexes.append(-1)

    # current targets
    targetImageFileIndexes = []
    for i in range(len(imageKeys) - 1):
        targetImageFileIndexes.append(i)

    # while there are available sources
    while len(sourceImageFileIndexes) > 0:
        probeRelatedImageIndexes.append(sourceImageFileIndexes[:])

        currentSourceImageFileIndexes = sourceImageFileIndexes[:]
        del sourceImageFileIndexes[:]

        for i in currentSourceImageFileIndexes:
            currentTargetImageFileIndexes = targetImageFileIndexes[:]

            currentMatches = list(map(lambda j: _areRelatedIJ(i, j, keypoints, descriptions, imgs,
                                                              minMatchCountThreshold, maxMatchCountThreshold,
                                                              keypointDistanceThreshold),
                                      currentTargetImageFileIndexes))

            for p in range(len(currentTargetImageFileIndexes)):
                j = currentTargetImageFileIndexes[p]
                if len(currentMatches[p]) > 0:
                    sourceImageFileIndexes.append(j)
                    targetImageFileIndexes.remove(j)

                p1 = imageCount * (i + 1) + (j + 1)
                p2 = imageCount * (j + 1) + (i + 1)
                matches[p1] = matches[p2] = currentMatches[p][:]

            del currentTargetImageFileIndexes[:]

        del currentSourceImageFileIndexes[:]

    # return probeRelatedImageIndexes, keypoints, descriptions, imageFilePaths, matches
    return probeRelatedImageIndexes, keypoints, descriptions, imageKeys, matches

