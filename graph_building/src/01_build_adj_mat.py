import sys, os
#sys.path.insert(2, '/data/dhenriqu/local/lib/python2.7/dist-packages/') # please, set openCV python library path if necessary
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib/')    # provenance library path

import time, AdjacencyMatrixBuilder

#main
print "Usage: python 01_build_adj_mat.py <probeFilePath> <imgListFilePath> <outputFilePath>"
p1 = sys.argv[1]
p2 = sys.argv[2]
p3 = sys.argv[3]
start_time = time.time()
print "01_build_adj_mat.py", p1, p2, p3
AdjacencyMatrixBuilder.buildAdjMat(p1, p2, p3)
print("--- %s seconds ---" % (time.time() - start_time))
