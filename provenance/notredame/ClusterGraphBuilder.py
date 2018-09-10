# Generates a provenance DAG from a given set of adjacency matrices.
# Clustering (ad-hoc controled graph expansion) strategy.
import numpy  # , GraphJSONReaderWriter


def _isNearDuplicate(targetNodeIndex, nodeIndexes, kpMatrix):
    if len(nodeIndexes) > 2:
        kps = []
        for i in range(len(nodeIndexes) - 1):
            for j in range(i + 1, len(nodeIndexes)):
                kps.append(kpMatrix[nodeIndexes[i], nodeIndexes[j]])

        mean = numpy.mean(kps)
        std = numpy.std(kps)

        dCount = 0
        for i in nodeIndexes:
            if (kpMatrix[i, targetNodeIndex] < mean - std or
                    kpMatrix[i, targetNodeIndex] > mean + std):
                dCount = dCount + 1

        if dCount > len(nodeIndexes) / 2.0:
            return False

    return True


def _getStrongestLink(nodeIndex, kpMatrix, miMatrix, dsMatrix, avoidedIndexes):
    index = -1
    kp = 0.0
    mi = -1.0
    dir = -1

    imCount = kpMatrix.shape[0]
    for i in range(imCount):
        if (i != nodeIndex and i not in avoidedIndexes and
                (miMatrix[nodeIndex, i] != 0.0 or miMatrix[i, nodeIndex] != 0.0) and
                kpMatrix[nodeIndex, i] >= 4 and
                (kpMatrix[nodeIndex, i] >= 10 or dsMatrix[nodeIndex, i] < 0.1)):
            if kp < kpMatrix[nodeIndex, i]:
                index = i
                kp = kpMatrix[nodeIndex, i]
                mi = miMatrix[nodeIndex, i]

                if miMatrix[nodeIndex, i] > miMatrix[i, nodeIndex]:
                    dir = 0
                elif miMatrix[nodeIndex, i] < miMatrix[i, nodeIndex]:
                    dir = 1

            elif kp > 0.0 and kp == kpMatrix[nodeIndex, i]:
                if mi < miMatrix[nodeIndex, i]:
                    index = i
                    kp = kpMatrix[nodeIndex, i]
                    mi = miMatrix[nodeIndex, i]
                    dir = 0

                if mi < miMatrix[i, nodeIndex]:
                    index = i
                    kp = kpMatrix[i, nodeIndex]
                    mi = miMatrix[i, nodeIndex]
                    dir = 1

    return nodeIndex, index, dir, kp, mi


def _getExpansionPoint(nodeIndexes, kpMatrix, miMatrix, dsMatrix):
    candidates = list(map(lambda i: _getStrongestLink(i, kpMatrix, miMatrix, dsMatrix, nodeIndexes), nodeIndexes))
    candidates = sorted(candidates, key=lambda c: c[3] + c[4], reverse=True)

    if len(candidates) > 0 and candidates[0][1] > -1:
        return candidates[0]
    else:
        return -1, -1, -1, 0.0, -1.0


def _expandFromProbe(kpMatrix, miMatrix, dsMatrix, dag):
    usedIndexes = []
    usedIndexes.append(0)  # probe

    outCount = 0.0
    inCount = 0.0

    iFront1 = 0
    iFront2 = 0
    front = -1

    expansion = _getStrongestLink(iFront1, kpMatrix, miMatrix, dsMatrix, usedIndexes)
    front = 1
    if expansion[2] == 0:
        outCount = outCount + 0.5
    else:
        inCount = inCount + 0.5

    while expansion[1] > -1:
        if expansion[2] == 0:
            outCount = outCount + 1.0
        else:
            inCount = inCount + 1.0

        if front == 1:
            usedIndexes.append(expansion[1])
            iFront1 = expansion[1]

        else:
            usedIndexes.insert(0, expansion[1])
            iFront2 = expansion[1]

        expansion1 = _getStrongestLink(iFront1, kpMatrix, miMatrix, dsMatrix, usedIndexes)
        valid1 = expansion1[1] > -1 and _isNearDuplicate(expansion1[1], usedIndexes, kpMatrix)

        expansion2 = _getStrongestLink(iFront2, kpMatrix, miMatrix, dsMatrix, usedIndexes)
        valid2 = expansion2[1] > -1 and _isNearDuplicate(expansion2[1], usedIndexes, kpMatrix)

        if not valid1 and not valid2:
            break

        elif valid1 and (not valid2 or expansion1[3] >= expansion2[3]):
            expansion = expansion1[:]
            front = 1
        else:
            expansion = expansion2[:]
            front = 2

    if outCount > inCount:
        for i in range(len(usedIndexes) - 1):
            dag[usedIndexes[i], usedIndexes[i + 1]] = 1.0
    else:
        for i in range(len(usedIndexes) - 1):
            dag[usedIndexes[i + 1], usedIndexes[i]] = 1.0

    return dag, usedIndexes


def _regularExpand(kpMatrix, miMatrix, dsMatrix, dag, usedIndexes):
    outCount = 0.0
    inCount = 0.0

    expansion = _getExpansionPoint(usedIndexes, kpMatrix, miMatrix, dsMatrix)
    outNode = expansion[0]
    if expansion[2] == 0:
        outCount = outCount + 0.5
    else:
        inCount = inCount + 0.5

    while expansion[1] > -1:
        usedIndexes.append(expansion[1])

        if expansion[2] == 0:
            outCount = outCount + 1.0
        else:
            inCount = inCount + 1.0

        innerUsedIndexes = []
        innerUsedIndexes.append(expansion[1])
        innerExpansion = _getStrongestLink(expansion[1], kpMatrix, miMatrix, dsMatrix, usedIndexes)
        while innerExpansion[1] > -1:
            usedIndexes.append(innerExpansion[1])
            innerUsedIndexes.append(innerExpansion[1])

            if (innerExpansion[2] == 0):
                outCount = outCount + 1.0
            else:
                inCount = inCount + 1.0

            innerExpansion = _getStrongestLink(innerExpansion[1], kpMatrix, miMatrix, dsMatrix, usedIndexes)
            if innerExpansion[1] == -1 or not _isNearDuplicate(innerExpansion[1], innerUsedIndexes, kpMatrix):
                break

        if outCount > inCount:
            dag[outNode, innerUsedIndexes[0]] = 1.0
            for i in range(len(innerUsedIndexes) - 1):
                dag[innerUsedIndexes[i], innerUsedIndexes[i + 1]] = 1.0
        else:
            dag[innerUsedIndexes[0], outNode] = 1.0
            for i in range(len(innerUsedIndexes) - 1):
                dag[innerUsedIndexes[i + 1], innerUsedIndexes[i]] = 1.0

        expansion = _getExpansionPoint(usedIndexes, kpMatrix, miMatrix, dsMatrix)
        outNode = expansion[0]
        if expansion[2] == 0:
            outCount = outCount + 0.5
        else:
            inCount = inCount + 0.5

    return dag, usedIndexes


##def jsonIt(jsonRankFilepath, imageDirPath, npzFilePath, outFilePath):
def buildGraph(npzFilePath):
    npzFile = numpy.load(npzFilePath)
    kpMatrix = npzFile['arr_0']
    miMatrix = npzFile['arr_2']
    dsMatrix = npzFile['arr_1']

    nodeCount = kpMatrix.shape[0]
    dag = numpy.zeros(kpMatrix.shape)
    dag, usedIndexes = _expandFromProbe(kpMatrix, miMatrix, dsMatrix, dag)
    dag, usedIndexes = _regularExpand(kpMatrix, miMatrix, dsMatrix, dag, usedIndexes)

    ##imageFilePaths=[]
    ##imageConfidenceScores=[]
    ##for image in GraphJSONReaderWriter.readRank(jsonRankFilepath, imageDirPath):
    ##    imageFilePaths.append(image[0])
    ##    imageConfidenceScores.append(image[1])
    ##
    ##    if len(imageFilePaths) == nodeCount:
    ##        break

    ##GraphJSONReaderWriter.saveGraph(dag, kpMatrix, imageFilePaths, imageConfidenceScores, outFilePath)
    return dag
