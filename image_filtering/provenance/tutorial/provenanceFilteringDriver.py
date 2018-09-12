from provenanceFiltering import  provenanceFiltering,filteringResults
from distributedQuery import distributedQuery
import argparse
import os
import sys
import logging
import traceback
from resources import Resource
import pickle
import json
from fileutil import isRaw
from joblib import Parallel,delayed

def convertNISTJSON(results):
    jsonResults={}
    nodes=[]
    jsonResults['directed']=True

    scores = results['provenance_filtering']['matches']
    count=1
    for filename in scores:
        node={}
        node['id']= str(count)
        node['file']='world/'+filename
        node['fileid']= os.path.splitext(os.path.basename(filename))[0]
        node['nodeConfidenceScore']= scores[filename]
        nodes.append(node)
        count=count+1
    jsonResults['nodes']=nodes
    jsonResults['links']=[]
    jsonstring = json.dumps(jsonResults)
    return jsonstring


parser = argparse.ArgumentParser()
parser.add_argument('--NISTProbeFileList', help='provenance index file')
parser.add_argument('--NISTDataset', help='nist dataset directory')
parser.add_argument('--IndexOutputDir', help='directory containing Indexes')
parser.add_argument('--ProvenanceOutputFile', help='output directory for results')
parser.add_argument('--Recall', help='output directory for the Index')

args = parser.parse_args()

provenanceresults=open(args.ProvenanceOutputFile,'w')
provenanceresults.write('ProvenanceProbeFileID|ConfidenceScore|ProvenanceOutputFileName|ProvenanceProbeStatus\n')
provenanceresults.close()
def runTotalQuery(line,fileIndex):
    values = line.split('|')
    filepath = os.path.join(args.NISTDataset, values[fileIndex])
    # try:

    filename = os.path.basename(filepath)
    probeImage = Resource.from_file(filename, filepath)
    distQuery = distributedQuery(args.IndexOutputDir, os.path.join(args.NISTDataset, "world"))
    provenanceFilter = provenanceFiltering(distQuery)

    results = provenanceFilter.processImage(probeImage, int(args.Recall))

    fileID = os.path.splitext(os.path.basename(probeImage.key))[0]
    resultDir = os.path.dirname(args.ProvenanceOutputFile)
    jsonFile = 'json/' + fileID + '.json'
    provenanceresults = open(args.ProvenanceOutputFile, 'a')
    provenanceresults.write(fileID + '|1.0|' + jsonFile + '|Processed\n')
    provenanceresults.close()
    jsonPath = os.path.join(resultDir, jsonFile)
    jsonString = convertNISTJSON(results)
    print(jsonPath)
    jsonFile = open(jsonPath, 'w')
    jsonFile.write(jsonString)

with open(args.NISTProbeFileList) as f:
  lines = f.readlines()
fileIndex = -1
numcores = 1
if numcores > 1:
    fileIndex = -1
    values = lines[0].split('|')
    if fileIndex == -1:
        fileIndex = values.index('ProvenanceProbeFileName')
    Parallel(n_jobs=numcores)(
        delayed(runTotalQuery)(line,fileIndex) for line in lines[1:])
else:
  values = lines[0].split('|')
  if fileIndex == -1:
     fileIndex = values.index('ProvenanceProbeFileName')
  for line in lines[1:]:
     values = line.split('|')
     if fileIndex==-1:
         fileIndex=values.index('ProvenanceProbeFileName')
     else:
        filepath = os.path.join(args.NISTDataset,values[fileIndex])
        try:
            filename = os.path.basename(filepath)

            fileID = os.path.splitext(os.path.basename(filename))[0]

            resultDir = os.path.dirname(args.ProvenanceOutputFile)
            jsonFile = 'json/' + fileID + '.json'
            jsonPath = os.path.join(resultDir, jsonFile)
            if (not os.path.exists(jsonPath) and not isRaw(filename)) or True:
                print(filename)
                probeImage = Resource.from_file(filename, filepath)
                distQuery = distributedQuery(args.IndexOutputDir,os.path.join(args.NISTDataset,"world"))
                provenanceFilter = provenanceFiltering(distQuery)

                results = provenanceFilter.processImage(probeImage,int(args.Recall))


                provenanceresults = open(args.ProvenanceOutputFile, 'a')
                provenanceresults.write(fileID+'|1.0|'+jsonFile+'|Processed\n')
                provenanceresults.close()

                jsonString = convertNISTJSON(results)
                print(jsonPath)
                jsonFile = open(jsonPath,'w')
                jsonFile.write( jsonString)

                # except IOError as e:
                #    print('skipping')
                #    logging.info('skipping '+filepath);
                #    provenanceresults.write(os.path.splitext(os.path.basename(values[fileIndex]))[0]+'|1.0||NonProcessed\n')
                # except Exception as e:
                #    print('skipping2')
                #    logging.error(traceback.format_exc())
                #    provenanceresults.write(os.path.splitext(os.path.basename(values[fileIndex]))[0]+'|1.0||NonProcessed\n')
            else:
                pass
                #print('skipping...')
        except Exception as e:
            print(e)


