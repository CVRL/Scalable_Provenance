import sys
import numpy as np
import faiss
import os
import progressbar

def dumpIndex(indexInMemory,indexOnDiskPath,shardCount):
    if shardCount == 0:
        faiss.write_index(indexInMemory,indexOnDiskPath)
    else:
        ivfs = []
        ivfs.append(indexInMemory.invlists)
        indexInMemory.own_invlists = False
        diskIndex = faiss.read_index(indexOnDiskPath,faiss.IO_FLAG_MMAP)
        ivfs.append(diskIndex.invlists)
        diskIndex.own_invlists = False

        invlists = faiss.OnDiskInvertedLists(diskIndex.nlist,diskIndex.code_size,'mergedIndex_tmp.ivfdata')
        ivf_vector = faiss.InvertedListsPtrVector()
        ivf_vector.push_back(indexInMemory.invlists)
        ivf_vector.push_back(diskIndex.invlists)
        ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())
        indexInMemory.ntotal = ntotal
        indexInMemory.replace_invlists(invlists)
        print('Index on disk now has ',indexInMemory.ntotal,' entries')
        faiss.write_index(indexInMemory,indexOnDiskPath)

def mergeIndexList(indexList,emptyIndexPath,outPath,machineNum=""):
    # merge the images into an on-disk index
    # first load the inverted lists
    ivfs = []
    outDir = os.path.dirname(outPath)
    bar = progressbar.ProgressBar()
    for indexFile in bar(indexList):
        # the IO_FLAG_MMAP is to avoid actually loading the data thus
        # the total size of the inverted lists can exceed the
        # available RAM
        # print("read " + indexFile)
        index = faiss.read_index(indexFile,
                                 faiss.IO_FLAG_MMAP)
        ivfs.append(index.invlists)

        # avoid that the invlists get deallocated with the index
        index.own_invlists = False

    # construct the output index
    index = faiss.read_index(emptyIndexPath)

    # prepare the output inverted lists. They will be written
    # to merged_index.ivfdata
    ivfDataStr = outDir + "merged_index_"+machineNum+"_.ivfdata"
    invlists = faiss.OnDiskInvertedLists(
        index.nlist, index.code_size,
        ivfDataStr)

    # merge all the inverted lists
    ivf_vector = faiss.InvertedListsPtrVector()
    for ivf in ivfs:
        ivf_vector.push_back(ivf)

    print("merge %d inverted lists " % ivf_vector.size())
    ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())

    # now replace the inverted lists in the output index
    index.ntotal = ntotal
    index.replace_invlists(invlists)

    print("write " + outPath)
    faiss.write_index(index, outPath)
    return ivfDataStr

def getListOfIndexes(indexFolder,machineNum):
    indexList = []
    for f in os.listdir(indexFolder):
        if f.endswith('.index'):
            parts = f.split('_')
            if len(parts) == 3:
                mnum = int(parts[1][-3:])
                if mnum == machineNum:
                    indexList.append(os.path.join(indexFolder, f))
    return indexList

if __name__ == "__main__":
    indexFolder = sys.argv[1]
    emptyPath = sys.argv[2]
    outPath = sys.argv[3]
    machineNum = int(sys.argv[4])
    outPath+= 'mergedIndex_machine'+"%03d" % machineNum + '.index'
    indexList = getListOfIndexes(indexFolder,machineNum)
    mergeIndexList(indexList,emptyPath,outPath,str(machineNum))
