from featureExtraction import featureExtraction
import argparse
import os
import sys
import logging
import traceback
from resources import Resource
import concurrent.futures
import progressbar

def process_image(filepath):
    try:
        filename=os.path.basename(filepath);
        worldImageResource= Resource.from_file(filename, filepath)
        logging.info('processing '+filepath);
        featureDict  = featureExtractor.processImage(worldImageResource)
        outpath = os.path.join(args.outputdir,featureDict['supplemental_information']['value'].key)
        #print outpath
        with open(outpath,'wb') as of:
            of.write(featureDict['supplemental_information']['value']._data)
    except IOError as e:
        logging.info('skipping '+filepath);
    except Exception as e:
        logging.error(traceback.format_exc())

    return []

parser = argparse.ArgumentParser()
parser.add_argument('--NISTWorldIndex', help='provenance index file')
parser.add_argument('--NISTDataset', help='nist dataset directory')
parser.add_argument('--outputdir', help='output directory')


args = parser.parse_args()
files = []
featureExtractor = featureExtraction()
with open(args.NISTWorldIndex) as f:
  fileIndex=-1
  bar = progressbar.ProgressBar()
  lines = f.readlines()
for line in lines:
 values = line.split('|')
 if fileIndex==-1:
     fileIndex=values.index('WorldFileName')
 else:
     filepath = os.path.join(args.NISTDataset,values[fileIndex])
     files.append(filepath)

with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
    #for file in files:
    for image_file, output in zip(files, executor.map(process_image, files)):
        print (image_file)
