import os
import sys
import json
from random import randint
from shutil import copy
import numpy as np

journalFolderPath = sys.argv[1]
dataPath = sys.argv[2]
indexFile = sys.argv[3] #world index file
outfolder = sys.argv[4]
numOriginals = int(sys.argv[5])
offset = int(sys.argv[6])
try:
    os.makedirs(outfolder)
except:
    pass
with open(indexFile, 'r') as fp:
    dsIndex = fp.readlines()
c = 0
dsIndexDict = {}
print("generating map for world index...")
for l in dsIndex[1:]:
    l = l.rstrip()
    part = l.split('|')[2]
    dsIndexDict[os.path.basename(part)] = part
    # print(os.path.basename(part))
count = offset
for f in os.listdir(journalFolderPath):
    # print(f)
    with open(os.path.join(journalFolderPath,f)) as fp:
        journal = json.load(fp)
    clinkold = -1
    links = journal['links']
    nodes = journal['nodes']
    amforward = np.zeros((len(nodes),len(nodes)))
    ambackward = np.zeros((len(nodes), len(nodes)))
    for l in links:
        y = int(l['target'])
        x = int(l['source'])
        if not amforward[y,x]:
            amforward[y,x] = 1
        if not ambackward[y,x]:
            ambackward[y,x] = 1
    curNode = links[0]['source']
    lastNode = -1
    chain = 0
    while curNode != lastNode:
        lastNode = curNode
        curNode = np.where(amforward[curNode] == 1)[0]
        if len(curNode) > 0:
            curNode=curNode[0]
            # print(curNode)
            chain += 1
        else:
            break
    unmodifiedImg = nodes[lastNode]['file']
    modifiedImage = nodes[links[0]['target']]['file']

    # curNode = lastNode
    # children = []
    # print("going to leaf")
    # for i in range(0,10):
    #     lastNode = curNode
    #     print(curNode)
    #     curNode = np.where(ambackward[curNode] == 1)[0]
    #     if len(curNode) > 0:
    #         curNode=curNode[0]
    #         if curNode == lastNode:
    #             break
    #         children.append(curNode)
    # if len(children) > 1:
    #     childNode = None
    #     for c in children[::-1]:
    #         print(c)
    #         if nodes[c]['file'] in dsIndexDict:
    #             childNode = c
    #             break
    #     if childNode:
    #         modifiedImage = nodes[childNode]['file']
    if modifiedImage in dsIndexDict and unmodifiedImg in dsIndexDict and not modifiedImage == unmodifiedImg :
        print(count, ' ', unmodifiedImg, ' ', modifiedImage)
        outdirOrig = os.path.join(outfolder,str("%04d" % count),'original')
        outdirMod = os.path.join(outfolder,str("%04d" % count),'modified')
        try:
            os.makedirs(outdirOrig)
        except:
            pass
        try:
            os.makedirs(outdirMod)
        except:
            pass
        try:
            copy(os.path.join(dataPath, dsIndexDict[unmodifiedImg]), os.path.join(outdirOrig, unmodifiedImg))
            copy(os.path.join(dataPath, dsIndexDict[modifiedImage]), os.path.join(outdirMod, modifiedImage))
            count += 1
            if count >= numOriginals:
                break
        except:
            print("couldnt copy ",os.path.join(dataPath, dsIndexDict[unmodifiedImg]),' ', os.path.join(dataPath, dsIndexDict[modifiedImage]))
            pass
    # break




