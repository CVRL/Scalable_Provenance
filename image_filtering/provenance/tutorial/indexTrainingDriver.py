from indexConstruction import indexConstruction
import argparse
import os
import sys
import logging
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--FeatureFileList', help='provenance index file')
parser.add_argument('--IndexOutputFile', help='output file for the Index Training Parameters')

args = parser.parse_args()

indexConstructor = indexConstruction()
indexTrainingParameters = indexConstructor.trainIndex(args.FeatureFileList)
outpath = os.path.join(args.IndexOutputFile)
with open(outpath,'wb') as of:
      of.write(indexTrainingParameters._data)

