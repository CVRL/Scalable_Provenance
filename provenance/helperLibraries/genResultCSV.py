import os
import sys

folder = sys.argv[1]
outArr = []
outArr.append('ProvenanceProbeFileID|ConfidenceScore|ProvenanceOutputFileName|ProvenanceProbeStatus')
for f in os.listdir(folder):
    if f.endswith('.json'):
        parts = f.split('.')
        outArr.append(parts[0]+'|1.0|'+os.path.join('json',f)+'|Processed')
with open('results.csv','w') as fp:
    fp.write('\n'.join(outArr))
