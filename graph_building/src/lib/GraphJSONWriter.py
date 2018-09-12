import io, json

def save(graph, adjMatrix, imageFilePaths, imageConfidenceScores, outputFilePath):
    # make it work for Python 2+3 and with Unicode
    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    # node info
    nodeIds = []
    nodeFiles = []
    nodeScores = []
    
    nodeCount = graph.shape[0]
    for i in range(nodeCount):
        nodeIds.append(i)
        nodeFiles.append(imageFilePaths[i])
        nodeScores.append(imageConfidenceScores[i])
    connectCount = 0
    
    # json graph
    nodeList = []
    edgeList = []
        
    jsGraph = {}
    jsGraph['directed'] = True
    
    for i in range(nodeCount):
        connected = False
        for j in range(nodeCount):
            if graph[i, j] == 1.0 or graph[j, i] == 1.0:
                connected = True
                break
        
        if connected or i == 0:
            nodeIds[i] = connectCount
            connectCount = connectCount + 1
            
            node = {}
            node['id'] = 'id' + str(nodeIds[i])
            node['file'] = nodeFiles[i]
            node['fileid'] = nodeFiles[i].split(".")[0]
            node['nodeConfidenceScore'] = nodeScores[i]
            nodeList.append(node)
            
    for i in range(nodeCount):               
        for j in range(nodeCount):
            if graph[i, j] == 1:
                edge = {}
                edge['source'] = nodeIds[i]
                edge['target'] = nodeIds[j]
                edge['relationshipConfidenceScore'] = adjMatrix[i, j]
                edgeList.append(edge)
                
    jsGraph['nodes'] = nodeList
    jsGraph['links'] = edgeList
    
    # saves JSON file
    with io.open(outputFilePath, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(jsGraph, indent=4, sort_keys=False, separators=(',', ':'), ensure_ascii=False)
        outfile.write(to_unicode(str_))
    