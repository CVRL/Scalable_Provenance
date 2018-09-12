# Builds N provenance adjacency matrices based on:
#    i. keypoint count;
#    ii. average distance;
#    iii. MI;
#    iv. MSE.
import datetime, numpy, cv2, KeypointMatcher, ProbeLinker, LocalSimilarityAnalyser

import io
from scipy.misc import imread
from resources import Resource
import rawpy

import cv2
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


def _calculateIJValues(i, j, keypoints, descriptions, imageI, imageJ, allMatches, imageCount, matchCountForSimilarity):
    p1 = imageCount * (i + 1) + (j + 1)
    p2 = imageCount * (j + 1) + (i + 1)
    matches = allMatches[p1]

    # if matches were not calculated yet...
    if len(matches) == 0:
        matches = KeypointMatcher.match(keypoints[i], descriptions[i], imageI,
                                        keypoints[j], descriptions[j], imageJ)

    # if there are enough matches for homography (more than 3)
    if len(matches) > 3:
        wkpsI, warpI, wkpsJ, warpJ = KeypointMatcher.warpMatches(keypoints[i], imageI,
                                                                 keypoints[j], imageJ,
                                                                 matches)

        distances = []
        filteredKeypointsI = []
        filteredKeypointsJ = []
        filteredWarpKeypsI = []
        filteredWarpKeypsJ = []
        for match in matches:
            distances.append(match.distance)
            if matchCountForSimilarity > 0:
                filteredKeypointsI.append(keypoints[i][match.queryIdx])
                filteredKeypointsJ.append(keypoints[j][match.trainIdx])
                filteredWarpKeypsI.append(wkpsI[match.queryIdx])
                filteredWarpKeypsJ.append(wkpsJ[match.trainIdx])
                matchCountForSimilarity = matchCountForSimilarity - 1

        if len(distances) > 0:
            miIJ, mseIJ = LocalSimilarityAnalyser.getSimilarity(warpI, filteredWarpKeypsI, imageJ, filteredKeypointsJ)
            miJI, mseJI = LocalSimilarityAnalyser.getSimilarity(warpJ, filteredWarpKeypsJ, imageI, filteredKeypointsI)

            kpCount = len(distances)
            avgDistance = numpy.average(distances)

            return True, kpCount, kpCount, avgDistance, avgDistance, miIJ, miJI, mseIJ, mseJI

    return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


##def buildAdjMat(probeFilePath, jsonRankFilepath, imageDirPath, rankSize,
##                outputFilePath, matchCountForSimilarity = 100):
def buildAdjMat(probeImage, filteringImages, rankSize,
                outputFilePath, matchCountForSimilarity=100):
    # obtains only the images that are related to the probe
    ##relatedImages, keypoints, descriptions, imageFilePaths, allMatches = ProbeLinker.getRelatedImagesToProbe(
    ##    probeFilePath, jsonRankFilePath, imageDirPath, rankSize)
    relatedImages, keypoints, descriptions, imageKeys, allMatches = ProbeLinker.getRelatedImagesToProbe(
        probeImage, filteringImages, rankSize)
    imageCount = len(imageKeys)
    print relatedImages, imageCount, str(datetime.datetime.now())

    # adjancency matrices as lists
    kpCountMat = numpy.zeros((imageCount, imageCount))
    avgDistMat = numpy.ones((imageCount, imageCount)) * -1.0
    mutInfoMat = numpy.zeros((imageCount, imageCount))
    mSqErroMat = numpy.ones((imageCount, imageCount)) * -1.0
    # other matrices can be added here...

    # for each pair of related images...
    for l in range(0, len(relatedImages) - 1):
        sources = []
        targets = []

        for m in range(0, len(relatedImages[l])):
            sources.append(relatedImages[l][m])

        for m in range(0, len(relatedImages[l + 1])):
            sources.append(relatedImages[l + 1][m])
            targets.append(relatedImages[l + 1][m])

        for i in sources:
            currentJs = []
            for j in targets:
                if i != j and (i not in targets or i < j):
                    currentJs.append(j)

            if len(currentJs) > 0:
                ##imageI = cv2.imread(imageFilePaths[i])
                ##imagesJ = list(map(lambda p: cv2.imread(imageFilePaths[p]), currentJs))

                imageI = [[]]
                if i == -1 or i == len(imageKeys) - 1 or imageKeys[i] == probeImage.key:
                    imageI = numpy.array(_deserialize_image(probeImage.data)).astype(numpy.uint8)
                else:
                    imageI = numpy.array(_deserialize_image(filteringImages[imageKeys[i]].data)).astype(numpy.uint8)

                imagesJ = []
                for p in currentJs:
                    if p == -1 or p == len(imageKeys) - 1 or imageKeys[i] == probeImage.key:
                        imagesJ.append(numpy.array(_deserialize_image(probeImage.data)).astype(numpy.uint8))
                    else:
                        imagesJ.append(
                            numpy.array(_deserialize_image(filteringImages[imageKeys[i]].data)).astype(numpy.uint8))

                print "Processing", i, "against", currentJs

                ij = []
                for x in range(0, len(currentJs)):
                    ij.append((currentJs[x], x))

                values = list(map(lambda (j, p): _calculateIJValues(i, j, keypoints, descriptions, imageI, imagesJ[p],
                                                                    allMatches, imageCount, matchCountForSimilarity),
                                  ij))

                for j in range(len(currentJs)):
                    if values[j][0]:
                        a = i + 1
                        b = currentJs[j] + 1

                        kpCountMat[a, b] = kpCountMat[b, a] = values[j][1]
                        avgDistMat[a, b] = avgDistMat[b, a] = values[j][3]
                        mutInfoMat[a, b] = values[j][5]
                        mutInfoMat[b, a] = values[j][6]
                        mSqErroMat[a, b] = values[j][7]
                        mSqErroMat[b, a] = values[j][8]
                        # saves everything
    numpy.savez(outputFilePath, kpCountMat, avgDistMat, mutInfoMat, mSqErroMat)
