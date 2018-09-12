import numpy, GraphJSONWriter

class Vertex(object):
    def __init__(self, id):
        self.id = id
        self.leader = id
        self.set = [self]
        
    def join(self, v):
        newLeader = self.leader
        newSet = self.set + v.set
        
        for _v in newSet:
            _v.leader = newLeader            
            del _v.set
            _v.set = newSet
    
def applyKruskal(adjMatrix):
    vertexCount = adjMatrix.shape[0]
    answer = numpy.zeros((vertexCount, vertexCount))
    
    vertices = []
    connectedVerticeIds = []
    for i in range(vertexCount):
        vertices.append(Vertex(i))
                
        for j in range(vertexCount):
            if adjMatrix[i, j] > 0 or adjMatrix[j, i] > 0:                
                connectedVerticeIds.append(i)
                break
                
    vertexCount = len(connectedVerticeIds)
    
    edges = []
    for i in connectedVerticeIds:
        for j in connectedVerticeIds:
            edges.append((i, j, adjMatrix[i, j]))    
    edges.sort(key=lambda e: e[2], reverse = True)
    
    currentEdge = 0
    while vertexCount > 1:
        i = edges[currentEdge][0]
        j = edges[currentEdge][1]
    
        if vertices[i].leader != vertices[j].leader:
            answer[i, j] = 1.0
            vertices[i].join(vertices[j])
            vertexCount = vertexCount - 1
        
        currentEdge = currentEdge + 1
    
    return answer

def jsonIt(rankFilePath, npzFilePath, matrixId, outFilePath):
    npzFile = numpy.load(npzFilePath)
    adjMatrix = npzFile['arr_' + matrixId]
    nodeCount = adjMatrix.shape[0]  
    
    tree = applyKruskal(adjMatrix)
    
    imageFilePaths=[]
    imageConfidenceScores=[]
    rankFile = open(rankFilePath, 'r')
    for image in rankFile:
        imageInfo = image.strip().split(',')
        
        imageFilePaths.append('world/' + imageInfo[0])
        imageConfidenceScores.append(float(imageInfo[1]))
        
        if len(nodeLabels) == nodeCount:
            break        
    rankFile.close()
    
    GraphJSONWriter.save(tree, adjMatrix, imageFilePaths, imageConfidenceScores, outFilePath)
