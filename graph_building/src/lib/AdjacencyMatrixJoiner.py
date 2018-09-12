import numpy

def _minMaxNorm(adjMatrix, dissimilar):
    if dissimilar:
        adjMatrix = adjMatrix * -1.0
        
    values = []
    mSize = adjMatrix.shape[0]
    for i in range(mSize):
        for j in range(mSize):
            value = adjMatrix[i][j]            
            if not dissimilar or value != 1.0:
                values.append(value)
                
    min = numpy.min(values)
    max = numpy.max(values)
    itv = max - min

    for i in range(mSize):
        for j in range(mSize):
            value = adjMatrix[i][j]
            
            if ((dissimilar and value == 1.0) or
                (not dissimilar and value == 0.0) or
                itv == 0.0):
                adjMatrix[i][j] = -1.0
            else:
                adjMatrix[i][j] = (value - min) / itv

    return adjMatrix

def _join(matrices):
    normMatrices = []
    
    for i in range(len(matrices)):
        normMatrix = _minMaxNorm(matrices[i], (matrices[i][0, 0] == -1.0))
        normMatrices.append(normMatrix)
    
    adjMatrix = numpy.zeros((normMatrices[0].shape[0], normMatrices[0].shape[1]))
    for matrix in normMatrices:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if adjMatrix[i, j] != -1.0:
                    if matrix[i, j] == -1.0:
                        adjMatrix[i, j] = -1.0
                    else:
                        adjMatrix[i, j] = adjMatrix[i, j] + matrix[i, j]
    adjMatrix = adjMatrix / len(normMatrices)
    
    return adjMatrix

def buildJointAdjMat(npzFilePath, matrixPs, outputFilePath):
    matrices = []
    npzFile = numpy.load(npzFilePath)
    for i in matrixPs:
        label = 'arr_' + i
        matrices.append(npzFile[label])
    npzFile.close()

    newMatrix = _join(matrices)
    numpy.savez(outputFilePath, newMatrix)