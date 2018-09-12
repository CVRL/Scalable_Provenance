import itertools, numpy, cv2, KeypointMatcher, ImageDescriptor

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

def getRelatedImagesToProbe(probeFilePath, imageListFilePath, 
                            minMatchCountThreshold = 4, maxMatchCountThreshold = 10,
                            keypointDistanceThreshold = 0.02):    
    # obtains the image file paths
    imageFilePaths = []
    file = open(imageListFilePath, 'r')
    for filePath in file:
        imageFilePaths.append(filePath.strip())
    file.close()
    imageFilePaths.append(probeFilePath)
    
    # images, descriptions, and keypoints
    imgs = list(map(lambda p: cv2.imread(p), imageFilePaths))
    imageCount = len(imgs)
    print "Read", imageCount, "images."
    
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
    for i in range(len(imageFilePaths) - 1):
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
                                                              keypointDistanceThreshold), currentTargetImageFileIndexes))
            
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
                
    return probeRelatedImageIndexes, keypoints, descriptions, imageFilePaths, matches

