import sys
#sys.path.insert(2, '/data/dhenriqu/local/lib/python2.7/dist-packages/') # please, set openCV python library path if necessary
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib/')    # provenance library path

import KruskalGraphBuilder, ClusterGraphBuilder

#main
print "Usage: python 03_build_graph.py <1: type [0: kruskal, 1: cluster]> <2: rankFilePath> <3: npzFilePath> <4: npzMatrixId [ignored if cluster]> <5: outputFilePath>"
p1 = sys.argv[1]
p2 = sys.argv[2]
p3 = sys.argv[3]
p4 = sys.argv[4]
p5 = sys.argv[5]
print "03_build_graph.py", p1, p2, p3, p4, p5

if p1 == 0:
    KruskalGraphBuilder.jsonIt(p2, p3, p4, p5)
else:
    ClusterGraphBuilder.jsonIt(p2, p3, p5)
