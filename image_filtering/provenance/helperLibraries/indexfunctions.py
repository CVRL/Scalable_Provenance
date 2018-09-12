import os
import sys
import numpy as np
import faiss
from joblib import Parallel,delayed

def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))

def weightVotes(locs,shape):
    locCoords = np.unravel_index(locs,shape)
    return np.sum(1/(locCoords[1]+1))

def tallyVotes(D, I,numcores=1,useWeights = True):
    ids, unq_inv, votes = np.unique(I,return_inverse=True, return_counts=True)
    unq_inv_s = np.argsort(unq_inv)
    id_locs = np.split(unq_inv_s,np.cumsum(votes[:-1]))
    if useWeights:

        if numcores > 1:
            votes = Parallel(n_jobs=numcores)(
                delayed(weightVotes)(locs,I.shape) for locs in id_locs)
        else:
            votes = []
            for locs in id_locs:
                votes.append(weightVotes(locs,I.shape))


    votes = np.asarray(votes)
    voteOrder = np.argsort(votes)[::-1]
    sortedIDs = ids[voteOrder]
    sortedVotes = votes[voteOrder]
    nIndex = np.where(sortedIDs == -1)[0]
    if len(nIndex) > 0:
        sortedIDs = np.delete(sortedIDs, nIndex[0])
        sortedVotes = np.delete(sortedVotes, nIndex[0])
    maxVoteVal = I.shape[0]* np.sum(1.0 /(np.arange(I.shape[1]) + 1))
    return sortedIDs, sortedVotes, maxVoteVal

def make_vres_vdev(gpu_resources,i0=0, i1=-1,ngpu=0,):
    " return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if i1 == -1:
        i1 = ngpu
    for i in range(int(i0), int(i1)):
        print("i0: " + str(i0) + "i1: " + str(i1))
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev

def featuresFitWithinRam(RAM,dims,float16,tempmem=0):
    ramBytes = float(RAM*1024*1024*1024)-tempmem
    dimBitSize = 32
    if float16:
        dimBitSize = 16
    featuresForRam = ramBytes/float(dims*(dimBitSize/8))
    return int(featuresForRam)

def wake_up_gpus(ngpu,tempmem):
    print("preparing resources for %d GPUs" % ngpu)
    gpu_resources = []
    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)
    return gpu_resources
