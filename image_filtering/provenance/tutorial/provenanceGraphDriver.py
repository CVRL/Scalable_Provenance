import argparse
import os
import sys
import logging
import traceback
import json
import glob
from resources import Resource

from provenanceGraphConstruction import  provenanceGraphConstruction


def convertNISTJSON(associations):
    nodecount={}
    nodecounter=0
    for association in associations:
       if association['link']['from'] not in nodecount:
          nodecount[association['link']['from']]=nodecounter
          nodecounter=nodecounter+1
       if association['link']['to'] not in nodecount:
          nodecount[association['link']['to']]=nodecounter
          nodecounter=nodecounter+1

    jsonResults={}
    jsonResults['directed']=True

    nodes=[]
    links=[]
    for nodekey in nodecount:
        node={}
        node['id']= str(nodecount[nodekey])
        node['file']='world/'+nodekey
        node['fileid']= os.path.splitext(os.path.basename(nodekey))[0]
        node['nodeConfidenceScore']= str(1)
        nodes.append(node)

    for association in associations:
        link={}
        link['source']=nodecount[association['link']['from']]
        link['target']=nodecount[association['link']['to']]
        link['relationshipConfidenceScore']=association['link']['score']
        links.append(link)
        
    jsonResults['nodes']=nodes
    jsonResults['links']=links
    jsonstring = json.dumps(jsonResults)

    return jsonstring

#look for a quick match to prevent neededing to load the whole world index
def getProbeImage(probeFileId, NISTDatasetPath):
    probeSearchPath = os.path.join(NISTDatasetPath,'world',probeFileId+'*')
    probeFilePath = glob.glob(probeSearchPath)[0]
    probeResource = Resource.from_file(os.path.basename(probeFilePath), probeFilePath)
    return probeResource

def parseFilteringResults(resultsFile,NISTDatasetPath):

    sys.stdout.flush()
    data = json.load(open(resultsFile))
    scores ={}
    resources = {}
    for node in data['nodes']:
      scores[node['fileid']]=node['nodeConfidenceScore']
      filePath=os.path.join(NISTDatasetPath,node['file'])
      #resources[node['fileid']]=Resource.from_file(os.path.basename(filePath),filePath)
      resources[os.path.basename(filePath)]=Resource.from_file(os.path.basename(filePath),filePath)   
    return scores, resources

################################################################
#Start main

parser = argparse.ArgumentParser()

parser.add_argument('--NISTDataset', help='nist dataset directory')
parser.add_argument('--FilteringResults', help='file containing provenance filtering results')
parser.add_argument('--ProvenanceResultsDir', help='file to put the provenance graph results')

args = parser.parse_args()


provenanceresults=open(args.ProvenanceResultsDir,'w')
provenanceresults.write('ProvenanceProbeFileID|ConfidenceScore|ProvenanceOutputFileName|ProvenanceProbeStatus\n')
provenanceFilteringResultsPath = os.path.dirname(args.FilteringResults)

#Greate Graph Processor

provenanceGraphProcessor = provenanceGraphConstruction()

with open(args.FilteringResults) as f:
  probeIndex=-1
  jsonIndex=-1
  optOutStatus=-1
  for line in f:
     line=line.rstrip('\n')
     values = line.split('|') 
     if probeIndex==-1:
         probeIndex=values.index('ProvenanceProbeFileID')
         jsonIndex=values.index('ProvenanceOutputFileName')
         optOutStatusIndex=values.index('ProvenanceProbeStatus')
     else:
         jsonPath = os.path.join(provenanceFilteringResultsPath,values[jsonIndex])
         optOutStatus = values[optOutStatusIndex]
         try:
           if  optOutStatus == "Processed":
             filename=os.path.basename(args.FilteringResults);
             probeResource = getProbeImage(values[probeIndex],args.NISTDataset)

             results, resources = parseFilteringResults(jsonPath,args.NISTDataset)

             graphResult = provenanceGraphProcessor.processGraph(probeResource,resources,results)
             graphOptOut =  graphResult['provenance_graph']['optout']
             associationResult =  graphResult['provenance_graph']['association']
             NISTJson = convertNISTJSON(associationResult)

             #####Write out to nist compatible output
             resultDir = os.path.dirname(args.ProvenanceResultsDir)
             jsonFile='json/'+values[probeIndex]+'.json'
             provenanceresults.write(values[probeIndex]+'|1.0|'+jsonFile+'|Processed\n') 
             jsonPath=os.path.join(resultDir,jsonFile)
             jsonFile = open(jsonPath,'w')
             jsonFile.write( NISTJson)
           else:
             provenanceresults.write(values[probeIndex]+'|1.0||NonProcessed\n') 
         except IOError as e:
            logging.info('skipping '+values[probeIndex])            
            provenanceresults.write(values[probeIndex]+'|1.0||NonProcessed\n') 
         except Exception as e:
            logging.error(traceback.format_exc())
            provenanceresults.write(values[probeIndex]+'|1.0||NonProcessed\n') 





