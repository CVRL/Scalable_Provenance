import sys
#sys.path.insert(2, '/data/dhenriqu/local/lib/python2.7/dist-packages/') # please, set openCV python library path if necessary
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib/')    # provenance library path

import AdjacencyMatrixJoiner

#main
print "Usage: python 02_join_adj_mat.py <npzFilePath> <idx0,idx1,idx2,...,idxn> <outputFilePath>"
p1 = sys.argv[1]
p2 = sys.argv[2].split(',')
p3 = sys.argv[3]
print "02_join_adj_mat.py", p1, p2, p3
AdjacencyMatrixJoiner.buildJointAdjMat(p1, p2, p3)
