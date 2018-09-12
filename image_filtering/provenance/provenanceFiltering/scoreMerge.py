import os
import sys
import numpy as np
import time
from joblib import Parallel,delayed
from multiprocessing import Manager
import progressbar
import math
# q = Manager().queue(1)
def sortRow(dists, inds,k,q=None):
    sortedI = np.argsort(dists)  # [::-1]
    sortedInds = inds[sortedI]
    sortedDists = dists[sortedI]
    if len(sortedInds) < k:
        padSize = k - len(sortedInds)
        np.pad(sortedInds, (padSize), 'constant', constant_values=-1)
        np.pad(sortedDists, (padSize), 'constant', constant_values=-1)
    if q is not None:
        q.put(1)
    # output_inds_sorted[distrow] = sortedInds[:k]
    # output_dists_sorted[distrow] = sortedDists[:k]

    return (sortedInds[:k],sortedDists[:k])

def mergeScorePair(s1,s2):
    s1.mergeScores(s2)
    return s1
def mergeScoreSet(scoresSets,numcores=5):
    ScoreSetstmp = scoresSets
    bar = progressbar.ProgressBar(max_value=math.ceil(math.log2(len(scoresSets))))
    count = 0
    while len(ScoreSetstmp) > 1:
        scorePairs = []
        for i in range(0,len(ScoreSetstmp),2):
            s1 = ScoreSetstmp[i]
            if i+1 < len(ScoreSetstmp):
                s2 = ScoreSetstmp[i+1]
            else:
                s2 = None
            scorePairs.append((s1,s2))
        ScoreSetstmp = Parallel(n_jobs=numcores)(
            delayed(mergeScorePair)(s1,s2) for s1,s2 in scorePairs)
        count += 1
        bar.update(count)
    return ScoreSetstmp[0]


def mergeResultsMatrix(D1,D2,I1,I2,map1,map2,numcores=1,k=50):
    D1 = np.append(D1,D2,axis=1)
    I1max = I1.max()
    I1 = np.append(I1,np.add(I2,int(I1max)),axis=1)
    for m in map2:
        map1[int(m)+I1max] = map2[m]
    # q = Manager().queue(D1.shape[0])
    output_dists_sorted_vals = D1.argsort(axis=1)
    output_inds_sorted = I1[np.arange(output_dists_sorted_vals.shape[0])[:,None],output_dists_sorted_vals]
    output_dists_sorted = D1[np.arange(output_dists_sorted_vals.shape[0])[:, None], output_dists_sorted_vals]
    # if numcores > 1:
    #     allSortedResults = Parallel(n_jobs=numcores)(delayed(sortRow)(di[0], di[1], k,None) for di in zip(D1, I1))
    # else:
    #     # output_inds_sorted = np.zeros((I1.shape[0], k))
    #     # output_dists_sorted = np.zeros((I1.shape[0], k))
    #     allSortedResults = []
    #     bar = progressbar.ProgressBar()
    #     for i in bar(range(0,D1.shape[0])):
    #         allSortedResults.append(sortRow(D1[i],I1[i],k))
    # allSortedResultsSplit = list(zip(*allSortedResults))
    # output_inds_sorted = np.vstack(allSortedResultsSplit[0])
    # output_dists_sorted = np.vstack(allSortedResultsSplit[1])
    outk = int(min(output_inds_sorted.shape[1],k*1.5))
    return (output_inds_sorted[:,:outk],output_dists_sorted[:,:outk],map1)