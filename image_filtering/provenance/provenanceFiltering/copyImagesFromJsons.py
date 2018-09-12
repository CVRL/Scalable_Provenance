import os
import sys
import json
import shutil
import progressbar
from joblib import Parallel,delayed

def copyFilesFromJson(jsonFilePath,outputFolder,baseDir):
    with open(jsonFilePath,'r') as fp:
        jsonFile = json.load(fp)
    nodes = jsonFile['nodes']
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    bar = progressbar.ProgressBar()
    for n in bar(nodes):
        imgPath = os.path.join(baseDir,n['file'])
        outputPath = os.path.join(outputFolder,os.path.basename(n['file']))
        if not os.path.exists(outputPath):
            shutil.copy(imgPath,outputPath)

jsonDir = sys.argv[1]
outputDir = sys.argv[2]
baseDir = sys.argv[3]
nproc = 10
jsonFiles = os.listdir(jsonDir)
Parallel(n_jobs=nproc)(
        delayed(copyFilesFromJson)(os.path.join(jsonDir,jsonFilePath), outputDir, baseDir) for jsonFilePath in jsonFiles)

