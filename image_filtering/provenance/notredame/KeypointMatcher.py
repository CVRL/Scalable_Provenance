# Matches SURF keypoints between two given images with geometric consistency.
import numpy, cv2, math


def _copyKeypoints(keypoints):
    answer = []

    for keypoint in keypoints:
        answer.append(cv2.KeyPoint(keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response))

    return answer


def _filterIJMatches(matches, i, j, queryPoints, trainPoints, displMatchThreshold):
    consistentMatches = []

    # queryImgKeypointCount = queryPoints.shape[1]
    trainImgKeypointCount = trainPoints.shape[1]

    queryKeypointIndex1 = matches[i].queryIdx
    queryKeypointIndex2 = matches[j].queryIdx
    queryDistance = math.sqrt(pow(queryPoints[0, queryKeypointIndex1] - queryPoints[0, queryKeypointIndex2], 2) +
                              pow(queryPoints[1, queryKeypointIndex1] - queryPoints[1, queryKeypointIndex2], 2))

    trainKeypointIndex1 = matches[i].trainIdx
    trainKeypointIndex2 = matches[j].trainIdx
    trainDistance = math.sqrt(pow(trainPoints[0, trainKeypointIndex1] - trainPoints[0, trainKeypointIndex2], 2) +
                              pow(trainPoints[1, trainKeypointIndex1] - trainPoints[1, trainKeypointIndex2], 2))

    # puts query and train images in the same scale
    distanceRate = 1.0
    if trainDistance > 0:
        distanceRate = queryDistance / trainDistance

    if distanceRate > 0:
        if distanceRate > 1.0:
            scaleMatrix = numpy.zeros((2, 2))
            scaleMatrix[0, 0] = distanceRate
            scaleMatrix[1, 1] = distanceRate
            trainPoints = numpy.mat(scaleMatrix) * numpy.mat(trainPoints)

        elif distanceRate < 1.0:
            scaleMatrix = numpy.zeros((2, 2))
            scaleMatrix[0, 0] = 1.0 / distanceRate
            scaleMatrix[1, 1] = 1.0 / distanceRate
            queryPoints = numpy.mat(scaleMatrix) * numpy.mat(queryPoints)

        # calculates the difference of angles between query and train images
        # angle between points 1 and 2, within query image
        queryAngle = math.atan2(queryPoints[1, queryKeypointIndex2] - queryPoints[1, queryKeypointIndex1],
                                queryPoints[0, queryKeypointIndex2] - queryPoints[0, queryKeypointIndex1])

        # angle between point 1 and 2, within train image
        trainAngle = math.atan2(trainPoints[1, trainKeypointIndex2] - trainPoints[1, trainKeypointIndex1],
                                trainPoints[0, trainKeypointIndex2] - trainPoints[0, trainKeypointIndex1])

        # difference of angles
        diffAngle = queryAngle - trainAngle
        pipi = 2.0 * 3.1416;

        if abs(diffAngle) > pipi:
            diffAngle = diffAngle / pipi
            diffAngle = diffAngle - int(diffAngle)

        if diffAngle < 0.0:
            diffAngle = diffAngle + pipi

        # obtains the rotation-translation matrix
        # 1. rotation
        sinAngle = math.sin(diffAngle)
        cosAngle = math.cos(diffAngle)

        rotationTranslationMatrix = numpy.zeros((3, 3))
        rotationTranslationMatrix[0, 0] = cosAngle
        rotationTranslationMatrix[0, 1] = -sinAngle
        rotationTranslationMatrix[1, 0] = sinAngle
        rotationTranslationMatrix[1, 1] = cosAngle

        # 2. translation
        pointA1 = queryPoints[:, queryKeypointIndex1]
        pointA1.shape = (2, 1)
        pointB1 = trainPoints[:, trainKeypointIndex1]
        pointB1.shape = (2, 1)

        pointA2 = queryPoints[:, queryKeypointIndex2]
        pointA2.shape = (2, 1)
        pointB2 = trainPoints[:, trainKeypointIndex2]
        pointB2.shape = (2, 1)
        translationMatrix = (pointA1 - pointB1 + pointA2 - pointB2) / 2.0

        rotationTranslationMatrix[0, 2] = translationMatrix[0, 0]
        rotationTranslationMatrix[1, 2] = translationMatrix[1, 0]

        # translates and rotates all keypoints
        trainPoints = numpy.delete(
            numpy.mat(rotationTranslationMatrix) * numpy.mat(
                numpy.vstack((trainPoints, numpy.ones((1, trainImgKeypointCount))))),
            -1, 0)

        # mounts the answer of the method, with only
        # the geometrically consistent matches
        for k in range(len(matches)):
            xA = queryPoints[0, matches[k].queryIdx]
            yA = queryPoints[1, matches[k].queryIdx]

            xB = trainPoints[0, matches[k].trainIdx]
            yB = trainPoints[1, matches[k].trainIdx]

            if abs(xA - xB) < displMatchThreshold and abs(yA - yB) < displMatchThreshold:
                consistentMatches.append(k)

    return consistentMatches


def _filterGeomConsMatches(matches, queryImgKeypoints, trainImgKeypoints, refMatchCount, displMatchThreshold):
    matchCount = len(matches)
    if refMatchCount > matchCount:
        refMatchCount = matchCount

    queryImgKeypointCount = len(queryImgKeypoints)
    trainImgKeypointCount = len(trainImgKeypoints)

    queryPoints = numpy.zeros((2, queryImgKeypointCount))
    for k in range(queryImgKeypointCount):
        queryPoints[0, k] = queryImgKeypoints[k].pt[0]
        queryPoints[1, k] = queryImgKeypoints[k].pt[1]

    trainPoints = numpy.zeros((2, trainImgKeypointCount))
    for k in range(trainImgKeypointCount):
        trainPoints[0, k] = trainImgKeypoints[k].pt[0]
        trainPoints[1, k] = trainImgKeypoints[k].pt[1]

    filteredMatches = []
    usedMatches = []

    for i in range(refMatchCount - 1):
        currentJs = []
        for j in range(i + 1, refMatchCount):
            if j not in usedMatches:
                currentJs.append(j)

        if len(currentJs) > 0:
            consistentMatches = list(
                map(lambda j: _filterIJMatches(matches, i, j, queryPoints, trainPoints, displMatchThreshold),
                    currentJs))
            for m in consistentMatches:
                if len(m) > 0:
                    if len(m) > len(filteredMatches):
                        filteredMatches = m[:]

                    usedMatches = usedMatches + m

    answer = []
    for i in filteredMatches:
        answer.append(matches[i])
    return answer


def match(keypoints1, descriptions1, image1, keypoints2, descriptions2, image2,
          nndrThreshold=0.8, refMatchCount=50, displMatchThreshold=20):
    # if there are more points in image 1 than in 2, swaps them
    swap = 0
    if len(keypoints1) < len(keypoints2):
        swap = 1
        keypoints1, keypoints2 = keypoints2, keypoints1
        descriptions1, descriptions2 = descriptions2, descriptions1
        image1, image2 = image2, image1

    # finds the good matches between the keypoints
    firstAndSecondMatches = []
    if len(descriptions1) > 0 and len(descriptions2) > 0:
        matcher = cv2.BFMatcher()
        firstAndSecondMatches = matcher.knnMatch(descriptions1, descriptions2, k=2)

    goodMatches = []
    if len(firstAndSecondMatches) > 0:
        try:
            for i, (a, b) in enumerate(firstAndSecondMatches):
                if b.distance != 0 and a.distance / b.distance < nndrThreshold:
                    goodMatches.append(a)

            goodMatches = _filterGeomConsMatches(goodMatches, keypoints1, keypoints2, refMatchCount,
                                                 displMatchThreshold)
        except:
            print "no good matches"

    # re-swaps the matches, if it is the case
    if swap == 1:
        for match in goodMatches:
            match.queryIdx, match.trainIdx = match.trainIdx, match.queryIdx

    return goodMatches


def warpMatches(keypoints1, image1, keypoints2, image2, matches):
    img1Points = []
    img2Points = []
    for match in matches:
        img1Points.append(keypoints1[match.queryIdx].pt)
        img2Points.append(keypoints2[match.trainIdx].pt)

    warps = []
    keypoints = []
    for i in range(0, 2):
        aPoints = []
        bPoints = []
        imageA = [[]]
        imageB = [[]]
        kps = []

        if i == 0:
            aPoints = img1Points
            bPoints = img2Points
            imageA = image1
            imageB = image2
            kps = _copyKeypoints(keypoints1)

        else:
            aPoints = img2Points
            bPoints = img1Points
            imageA = image2
            imageB = image1
            kps = _copyKeypoints(keypoints2)

        homography, _ = cv2.findHomography(numpy.array(aPoints), numpy.array(bPoints), cv2.LMEDS)
        warpedAPoints = cv2.perspectiveTransform(numpy.array([numpy.array(aPoints)]), homography)
        warpImage = cv2.warpPerspective(imageA, homography, (imageB.shape[1], imageB.shape[0]))

        warpedAPoints = warpedAPoints.reshape(len(aPoints), 2)
        if i == 0:
            j = 0
            for match in matches:
                kps[match.queryIdx].pt = (warpedAPoints[j][0], warpedAPoints[j][1])
                kps[match.queryIdx].size = keypoints2[match.trainIdx].size
                j = j + 1
        else:
            j = 0
            for match in matches:
                kps[match.trainIdx].pt = (warpedAPoints[j][0], warpedAPoints[j][1])
                kps[match.trainIdx].size = keypoints1[match.queryIdx].size
                j = j + 1

        warps.append(warpImage)
        keypoints.append(kps)

    return keypoints[0], warps[0], keypoints[1], warps[1]
