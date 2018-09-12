import sys
import os
import json
import time
from joblib import Parallel,delayed
import fnmatch
import numpy as np
import progressbar
from threading import Thread
from time import sleep

def threaded_function(arg):
    while True:
        print("running")
        sleep(20)
def isImage(name):
    name = name.lower()
    if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.ppm') or name.endswith('.pgm') or name.endswith('.pbm') or name.endswith('.pnm') or name.endswith('.bmp') or name.endswith('.exif') or name.endswith('.tiff') or name.endswith('.tif') or name.endswith('.gif') or name.endswith('.nef') or name.endswith('.dng') or name.endswith('.cr2') or name.endswith('.raf') or name.endswith('.arw'):
        return True
    return False
def generateFileDictionary(directoryToIndex,isFeatures,basedir):
    nameToPathDictionary = {}
    nameToIDDictionary = {}
    IDToNameDictionary = {}
    imageList = []
    count = 0

    notfindFileTypes = True
    i = 0
    thread = Thread(target=threaded_function,args=[1])
    thread.start()
    for maindir in directoryToIndex:
        print("searching include " + maindir)
        #for root, dirs, files in os.walk(maindir,followlinks=True):
        for i in range(1):
            #dlength = len(files)
            count = 0
            root = maindir
            print('indexing ', root)
            if isFeatures:
                bar = progressbar.ProgressBar()
                for file in bar(files):

                    if notfindFileTypes or file.endswith('.npy'):
                        if file not in nameToPathDictionary:
                            nameToPathDictionary[file] = os.path.abspath(root)
                            nameToIDDictionary[file] = count
                            IDToNameDictionary[count] = file
                            count += 1
                        else:
                            pass
                            # print('file already exists, skip')
                        if count%1000 == 0:
                            print(str(count)+"/"+str(dlength))
            elif not root.endswith('thumb'):
                files = os.listdir(os.path.join(root))
                bar = progressbar.ProgressBar()
                for file in bar(files):
                    file = file.rstrip()
                    # print(file,', ',isImage(file))
                    if isImage(file):
                        imageList.append(os.path.join(root,file))
                        count +=1
                        i += 1
                    else:
                        print(file)
                    # if count % 1000 == 0:
                    #     print(str(i) + "/" + str(dlength))

    nameToPathDictionary["_rootDirectory_"] = os.path.abspath(basedir)
    print('all done!')
    return (nameToPathDictionary,nameToIDDictionary,IDToNameDictionary,imageList)

def calcFeatureNumber(fileDict,keys):
    totalSize = 0
    i = 0
    integratedSize = []
    names = []
    for key in keys:
        filename = os.path.join(fileDict[key],key)
        fsize = os.path.getsize(filename)
        totalSize += (fsize-80)/4/dimensions
        integratedSize.append(totalSize)
        names=[]
        i+=1
        if i%1000 == 0:
            print(str(i)+'/'+str(len(keys)))

    return (totalSize,integratedSize)
def runFeatureNumberCalc(dictionary,keys,numJobs):
    sep = len(keys) * 1.0 / numJobs
    keySet = []
    for i in range(0, numJobs):
        keySet.append(keys[int(i * sep):min(int((i + 1) * sep), len(keys))])
    if numJobs > 1:
        counts = Parallel(n_jobs=numJobs)(delayed(calcFeatureNumber)(dictionary, k) for k in keySet)
    else:
        counts = []
        counts.append(calcFeatureNumber(dictionary, keySet[0]))
    total = 0
    integratedSize = []
    for c in counts:
        sizes = np.asarray(c[1]) + total
        total += c[0]
        integratedSize.append(sizes)
    integratedSize = np.concatenate(integratedSize)
    print("Total features: " + str(total))
    return (total,integratedSize)

def usage():
    print('build a dictionary for indexing')

if __name__ == "__main__":
    args = sys.argv[1:]
    dimensions = 64
    numJobs = 1
    mapName = None
    isFeatures = True
    directoryToIndex = None
    mapName = None
    mainJSON = {}
    includeFolders = []

    while args:
        a = args.pop(0)
        if a == '-h':
            usage()
            sys.exit(1)
        elif a == '-dims':
            dimensions = int(args.pop(0))
        elif a == '-numJobs':
            numJobs = int(args.pop(0))
        elif a == '-dataset':
            isFeatures = False
        elif a == '-includeDirs':
            includeFolders = args.pop(0).split(',')
        elif not directoryToIndex:
            directoryToIndex = os.path.abspath(a)
        elif not mapName:
            mapName = a
        else:
            print("argument %s unknown" % a)
            sys.exit(1)
    if mapName is None:
        mapName = ""
    if os.path.isdir(directoryToIndex):
        t0 = time.time()
        dirsToIndex = []
        if len(includeFolders) > 0:
            for i in includeFolders:
                dirsToIndex.append(os.path.join(directoryToIndex,i))
        else:
            dirsToIndex.append(directoryToIndex)
        dicts = generateFileDictionary(dirsToIndex,isFeatures,directoryToIndex)
        nameToPathDictionary = dicts[0]
        nameToIDDictionary = dicts[1]
        IDToNameDictionary = dicts[2]
        imageList = dicts[3]
        fileSaveName = os.path.abspath(directoryToIndex).split('/')[-1]+'_'
        fileSaveName_main = mapName+ '_' + fileSaveName +"dictionary.json"
        fileSaveName_nameID = fileSaveName + "_nameIDdictionary.json"
        fileSaveName_Idname = fileSaveName + "_IDnamedictionary.json"
        if isFeatures:
            mainJSON['imageIDMap'] = nameToIDDictionary
            mainJSON['IDimageMap'] = IDToNameDictionary
            mainJSON['nameToPathMap'] = nameToPathDictionary
            featureKeys = sorted(fnmatch.filter(list(nameToPathDictionary.keys()), '*.npy'))
            mainJSON['sortedFeatureKeys'] = featureKeys
            print('building feature counts now')
            featCounts = runFeatureNumberCalc(nameToPathDictionary,featureKeys,numJobs)
            mainJSON['featureCounts'] = list(featCounts[1])
            mainJSON['totalFeatures'] = featCounts[0]
            print("Saving file to " + fileSaveName)

        else:
            mainJSON['imageList'] = imageList
            print(len(imageList))
            mainJSON['baseDir'] = directoryToIndex
        with open(fileSaveName_main, 'w') as fp:
            json.dump(mainJSON, fp)
        # with open(fileSaveName_nameID, 'w') as fp:
        #     json.dump(nameToIDDictionary, fp)
        # with open(fileSaveName_Idname, 'w') as fp:
        #     json.dump(IDToNameDictionary, fp)
        t1 = time.time()
        print("Total time: ",t1-t0,' seconds')
    else:
        print("Error: " + directoryToIndex + "is not a directory")
else:
    print("Usage: BuildFileDictionary <base/path/of/files>")
