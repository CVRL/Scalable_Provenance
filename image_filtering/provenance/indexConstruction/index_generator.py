# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#! /usr/bin/env python2

import numpy as np
import time
import os
import sys
import faiss
import glob
import re
import json
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process, Lock
from multiprocessing import Manager
import threading
import queue
from multiprocessing.managers import BaseManager
# from joblib import Parallel,delayed,load, dump
# from joblib.pool import has_shareable_memory
import fnmatch
from collections import namedtuple
import socket
import psutil
import resource
import getch
from pathlib import Path


####################################################################
# Parse command line
####################################################################

def usage():
    print >>sys.stderr, """

Usage: bench_gpu_1bn.py dataset indextype [options]

dataset: set of vectors to operate on.
   Supported: SIFT1M, SIFT2M, ..., SIFT1000M or Deep1B

indextype: any index type supported by index_factory that runs on GPU.

    General options

-ngpu ngpu         nb of GPUs to use (default = all)
-tempmem N         use N bytes of temporary GPU memory
-nocache           do not read or write intermediate files
-float16           use 16-bit floats on the GPU side

    Add options

-abs N             split adds in blocks of no more than N vectors
-max_add N         copy sharded dataset to CPU each max_add additions
                   (to avoid memory overflows with geometric reallocations)
-altadd            Alternative add function, where the index is not stored
                   on GPU during add. Slightly faster for big datasets on
                   slow GPUs

    Search options

-R R:              nb of replicas of the same dataset (the dataset
                   will be copied across ngpu/R, default R=1)
-noptables         do not use precomputed tables in IVFPQ.
-qbs N             split queries in blocks of no more than N vectors
-nnn N             search N neighbors for each query
-nprobe 4,16,64    try this number of probes
-knngraph          instead of the standard setup for the dataset,
                   compute a k-nn graph with nnn neighbors per element
-oI xx%d.npy       output the search result indices to this numpy file,
                   %d will be replaced with the nprobe
-oD xx%d.npy       output the search result distances to this file

"""
    sys.exit(1)


# default values

dbname = None
index_key = None

ngpu = faiss.get_num_gpus()

replicas = 1  # nb of replicas of sharded dataset
add_batch_size = 32768
query_batch_size = 16384
nprobes = [1 << l for l in range(9)]
knngraph = False
use_precomputed_tables = True
tempmem = -1  # if -1, use system default
max_add = -1
use_float16 = False
use_cache = True
nnn = 10
altadd = False
I_fname = None
D_fname = None
featuredictfile = ""
base_feature_file = ""
query_feature_file = ""
featureCountFile = None
train_size = -1
recursive = False
totalFeatureNum = -1
kp_per_file = -1
featureDictionary = {}
producerTimeout = 0
timeoutMax = 20
producerIsDone = Manager().Value('j',False)
base_prefix = ""
featureDictionaryKeys=None
cacheroot = None
totalTime = 0
preprocTime = 0
quantTime = 0
addTime = 0
kpstrict = False
args = sys.argv[1:]
nproc = 1
mem_buffer = .9
num_jobs = 1
machine_num = 0
featureCountList = None
runEval = False
query_folder = None
keep_running_file_write = Manager().Value('k',True)
index_offset = 0
shard_offset = 0
jsonMapFile = None
tier2search = False
genQueryJsonOnly = False
datasetJsonFile = None
queryJsonFile = None

# myIP = [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]



#################################################################
# Small Utility Functions
#################################################################
def startWatchdog(tm):
    def runwatchdog(tm):
        try:
            t = watchdog_q.get(timeout=tm)
            return
        except:
            print("Watchdog timed out, killing everything!")
            os.system("pkill -9 python")
            os._exit(0)

    p1=Process(target=runwatchdog, args=(tm,), )
    p1.start()


class myThread(Process):
    def __init__(self, threadID, name, cont):
        super(myThread, self).__init__()
        # threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.cont = cont
    def run(self):
        print("Starting kill switch process..." + self.name + "\n")
        char = getch.getch()
        if char == 'k':
            print("Killing everything NOW")
            os._exit(1)
            kill_q.put(1)


def killThread():
    while kill_q.empty():
        time.sleep(1)
    print("Kill signal recieved")
    os._exit(1)

def startQueueServer(ip):
    q = queue.Queue()
    class QueueManager(BaseManager): pass
    QueueManager.register('get_queue', callable=lambda: queue)
    m = QueueManager(address=('', 50000), authkey='abracadabra')
    s = m.get_server()
    s.serve_forever()
    print('serving')

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]

    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

# we mem-map the biggest files to avoid having them in memory all at
# once

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def rate_limited_imap(f, l):
    """A threaded imap that does not produce elements faster than they
    are consumed"""
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i, ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()
def load_qery_from_features_file(file):
    print("Loading query file ", file,'...')
    return np.load(file)
def load_query_from_features_directory(folder,genQueryJsonOnly=False):
    if os.path.exists(queryfeat_cachefile) and not genQueryJsonOnly:
        print("Loading premade query file...")
        qfeats =  np.load(queryfeat_cachefile)
        return (qfeats,None,None,None)


    print("Loading needed files for query..." + query_folder)
    qcount = 0
    allfeats = []
    queryFeatureIDs = []
    qidtoimage = {}
    qimagetoid = {}
    qimageRanges = {}
    files = [file for file in glob.glob(folder + '/**/*.npy', recursive=True)]
    for file_name in files:
        if file_name.endswith('.npy'):
            file = os.path.join(file_name)
            imgname = os.path.basename(file_name)[:-4]
            try:
                features=np.load(file)
                allfeats.append(features)
                queryFeatureIDs.append(np.zeros(features.shape[0],dtype='int32')+qcount)
                qidtoimage[qcount] = imgname
                qimagetoid[imgname] = qcount
            except:
                print("WARNING: Couldn't load a query file ",file)
            qcount+=1
            if qcount%100 == 0:
                    print(qcount)
    print("Concatenating query files...")
    allfeats = np.vstack(allfeats)
    queryFeatureIDs = np.concatenate(queryFeatureIDs)
    print("total of ", allfeats.shape[0], " query features")
    np.save(queryfeat_cachefile,allfeats)
    # if machine_num == 0:
        # print('saving query files to ', imageIDMap_query_cachefile )
        # with open(imageIDMap_query_cachefile,'w') as fp:
        #     json.dump(qimagetoid,fp)
        # with open(IDimageMap_query_cachefile,'w') as fp:
        #     json.dump(qidtoimage,fp)
        # np.save(ID_query_cachefile,queryFeatureIDs)
    return (allfeats,qimagetoid,qidtoimage,queryFeatureIDs)

def load_training_from_file_list(filelist,train_size,kp_per_file,cache_dir):
    if os.path.exists(cache_dir):
        print("Loading premade training file...")
        return np.load(cache_dir)

    if train_size == -1:
        train_size = 50000
    if kp_per_file == -1:
        fname = os.path.join(filelist[0])
        a = np.load(fname)
        kp_per_file = a.shape[0]
    randomKeys = np.random.choice(len(filelist),int(train_size*1.0/kp_per_file),replace=False)
    print("found " + str(len(filelist)) + " keys")
    allfeats = []
    print("Loading needed files for training...")
    for idx in randomKeys:
        key = filelist[idx]
        file = os.path.join(key)
        # print("loading" + file)
        try:
            features=np.load(file)
            allfeats.append(features)
        except:
            print("WARNING: Couldn't load a file")
    print("Concatenatinputg training files...")
    allfeats = np.concatenate(allfeats)
    np.save(cache_dir,allfeats)
    return allfeats

def load_training_from_features_dictionary(dict,train_size,kp_per_file):
    if os.path.exists(trainfeat_cachefile):
        print("Loading premade training file...")
        return np.load(trainfeat_cachefile)
    print("found " + str(len(dict)) + " files")
    keys =  featureDictionaryKeys
    print("found " + str(len(keys)) + " keys")

    if train_size == -1:
        train_size = 50000
    if kp_per_file == -1:
        fname = os.path.join(base_prefix,dict[keys[0]],keys[0])
        a = np.load(fname)
        kp_per_file = a.shape[0]
    randomKeys = np.random.choice(len(keys),int(train_size*1.0/kp_per_file),replace=False)
    print("found " + str(len(keys)) + " keys")
    allfeats = []
    print("Loading needed files for training...")
    for idx in randomKeys:
        key = keys[idx]
        file = os.path.join(base_prefix, dict[key],key)
        # print("loading" + file)
        try:
            features=np.load(file)
            allfeats.append(features)
        except:
            print("WARNING: Couldn't load a file")
    print("Concatenatinputg training files...")
    allfeats = np.concatenate(allfeats)
    np.save(trainfeat_cachefile,allfeats)
    return allfeats

class IdentPreproc:
    """a pre-processor is either a faiss.VectorTransform or an IndentPreproc"""

    def __init__(self, d):
        self.d_in = self.d_out = d

    def apply_py(self, x):
        return x


def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))

def qadd(qu,obj):
    notput = True
    while notput:
        try:
            qu.put(obj,timeout=15)
            notput = False
        except:
            notput = True

class ProducerThread(threading.Thread):
    def dispatch_fileToucher(self,filenames):
        print("Starting file touching thread...")
        for filename in filenames:
            if keep_running_file_write.value:
                path = os.path.join(base_prefix, featureDictionary[filename], filename)
                Path(path).touch()
                qadd(file_Touch_q,path)
            else:
                print("Ending file touching process from call")
                self.fileTouch_is_running.value = False
                return
        self.fileTouch_is_running.value = False
        print("Done with file touch thread")
        return

    def dispatch_fileLoader(self,totalCount):
        # self.cup()
        countdown = totalCount
        totalKeypoints = 0
        fileIDMap = {}
        imageToIDMap = datasetJsonFile['imageIDMap']
        keepgoing = True
        c = 0
        sums =0
        print("Starting feature file loading thread...")
        while (self.fileTouch_is_running.value or file_Touch_q.qsize() > 0) and keepgoing:
            # if not self.fileTouch_is_running.value:
                # if c%15 == 0:
                #     print('all touched, still have more to load')
            if keep_running_file_write.value:

                path = file_Touch_q.get(timeout=240)

                # path = os.path.join(base_prefix, featureDictionary[filename], filename)
                featureSet = None
                try:
                    featureSet = np.load(path)
                except:
                    print("Warning: Producer could not load file ", path)
                    pass
                if featureSet is not None:
                    if kpstrict:
                        featureSet = featureSet[:min(kp_per_file, featureSet.shape[0])]
                    # featureSet = preproc.apply_py(featureSet)
                    # featureSet, np.zeros((featureSet.shape[0]), dtype='int64') + int(c + shard_offset + index_offset)))
                    inds = np.zeros((featureSet.shape[0]), dtype='int64') + int(imageToIDMap[os.path.basename(path)])
                    qadd(file_Load_q, (featureSet,inds ))

                    totalKeypoints += featureSet.shape[0]
                    # bname = os.path.basename(path)[:-4]
                    # fileIDMap[c + index_offset + shard_offset] = bname
                    # imageToIDMap[bname] = c + index_offset
                    qsize = file_Load_q.qsize()
                    sums += featureSet.shape[0]
                    del featureSet
                    del inds
                    # print('file ',c+shard_offset,',',os.path.basename(path),': ',featureSet.shape[0],' ',sums)
                    # if qsize%100==0:
                        # print("file q: ", qsize," preproc q: ",preproc_q.qsize())
                    c += 1

                    # print(preproc_q.qsize())
                    # print("added file to preproc queue: " + str(preproc_q.qsize()))
            else:
                print("Ending file load process from call")
                keepgoing=False
                self.fileLoad_is_running.value = False
                return (totalKeypoints, None, None)
                # return (totalKeypoints, fileIDMap, imageToIDMap)
            countdown -= 1
        self.fileLoad_is_running.value = False
        print('done with parallel, loaded total of ', totalKeypoints, " keypoints, qsize=", preproc_q.qsize())
        # with open(IDimageMap_cachefile,'w') as fp:
        #     json.dump(fileIDMap,fp)
        # with open(imageIDMap_cachefile,'w') as fp:
        #     json.dump(imageToIDMap,fp)
        print("finished reading all relevant files")
        return (totalKeypoints, None, None)
    def dispatch_filePreprocessor(self,ncores):
        keepgoing = True
        print("Starting feature preprocessing thread...")
        c=0
        while (self.fileLoad_is_running.value or file_Load_q.qsize() > 0) and keepgoing:
            # if not self.fileLoad_is_running.value:
            #     if c%15==0:
            #         print('file load is done, but preproc still running')
            if keep_running_file_write.value:
                didGet = False
                waitTime = 5
                tm = 0
                fileFeatsObj = []
                while not didGet and (self.fileLoad_is_running.value or file_Load_q.qsize() > 0) and keep_running_file_write.value and tm < 240:
                    try:
                        fileFeatsObj = file_Load_q.get(timeout=waitTime)
                        didGet = True
                    except:
                        tm += waitTime
                        print('waiting...')
                if didGet:
                    filefeats = fileFeatsObj[0]
                    featureids = fileFeatsObj[1]
                    featureSet = preproc.apply_py(filefeats)
                    qadd(preproc_q,(featureSet, featureids))
                    del filefeats
                    del featureids
                    del featureSet
                    c+=1
                else:
                    print('didget: ',didGet,' fileLoad run: ', self.fileLoad_is_running.value,' ',keep_running_file_write.value)
                # print(file_Load_q.qsize())
            else:
                print("Ending feature preprocessing process from call")
                self.preproc_is_running.value = False
                keepgoing = False
        self.preproc_is_running.value = False
        print("Done with preproc consumer thread")
        return
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None,preproc=None,q_preproc=None, bs=32768,ncores=5,featureKeys=None):
        super(ProducerThread,self).__init__()
        self.target = target
        self.name = name
        self.preproc = preproc
        self.q_preproc = q_preproc
        self.bs = bs
        self.preproc_is_running = Manager().Value('i',True)
        self.fileLoad_is_running = Manager().Value('j',True)
        self.fileTouch_is_running = Manager().Value('l', True)
        # self.manager = BaseManager(address=('', 50000), authkey=bytes('abc',encoding='utf-8'))
        # self.server = self.manager.get_server()
        # self.server.serve_forever()

        self.featureKeys = featureKeys
        # self.counter_lock = Lock()
        print("Starting up " +  str(3) +" file loading threads...")
        self.procthreads = 0
        p0 = Process(target=self.dispatch_fileToucher, args=(self.featureKeys,),)
        p1 = Process(target=self.dispatch_fileLoader, args=(len(self.featureKeys),),)
        p2 = Process(target=self.dispatch_filePreprocessor,args=(1,),)

        p0.start()
        p1.start()
        p2.start()
        print("All background processes started successfully")
        # p0.join()
        # p1.join()
        # p2.join()
        # time.sleep(30)
    batchCounter = 0
    def run(self):
        print("Starting iterator on " + str(len(self.featureKeys)) + " files")
        t1 = time.time()
        d = -1
        fullBatch = None
        fullBatchIDs = np.zeros(add_batch_size,dtype='int64')
        bcount = 0
        tcount = 0
        toadd = 0
        totalCount = 0
        while self.preproc_is_running.value or preproc_q.qsize() > 0:
            # print(self.preproc_is_running,preproc_q.qsize())
            fileFeatsObj = preproc_q.get()

            filefeats = fileFeatsObj[0]
            d = filefeats.shape[1]
            if fullBatch is None:
                fullBatch = np.zeros((add_batch_size, d),dtype='float32')
            featureids = fileFeatsObj[1]
            if bcount+filefeats.shape[0] <= add_batch_size:

                fullBatch[bcount:bcount+filefeats.shape[0]] = filefeats
                fullBatchIDs[bcount:bcount + featureids.shape[0]] = featureids
                bcount += filefeats.shape[0]
                del filefeats
                del featureids

                # print(bcount)
            else:
                toadd = add_batch_size-bcount
                fullBatch[bcount:add_batch_size] = filefeats[:toadd]
                fullBatchIDs[bcount:add_batch_size] = featureids[:toadd]
                t0 = time.time()
                q.put((fullBatch,fullBatchIDs),timeout=240)
                totalCount+=fullBatch.shape[0]
                # print(fullBatchIDs[0], ' ', fullBatchIDs[-1], ' ',fullBatchIDs[-1]-fullBatchIDs[0], ' ',fullBatch.shape[0])
                # qadd(q, (fullBatch, fullBatchIDs))
                t1 = time.time()
                tcount += t1 - t0
                fullBatch = np.zeros((add_batch_size, d),dtype='float32')
                fullBatchIDs = np.zeros(add_batch_size,dtype='int64')
                # print("add time: ",tcount)
                tcount = 0
                fullBatch[0:filefeats.shape[0]-toadd] = filefeats[toadd:]
                fullBatchIDs[0:featureids.shape[0] - toadd] = featureids[toadd:]
                bcount = filefeats.shape[0]-toadd
                del featureids
                del filefeats
        if bcount > 0:

            fullBatch = fullBatch[:bcount]
            fullBatchIDs = fullBatchIDs[:bcount]

            print("Added leftover batch size ", fullBatchIDs.shape[0])
            totalCount += fullBatch.shape[0]
            print(totalCount)
            q.put((fullBatch,fullBatchIDs))
            del fullBatch
            del fullBatchIDs
            # print(fullBatchIDs[0], ' ', fullBatchIDs[-1]," ", fullBatch.shape[0])
        print("preproc queue is empty and preproc is done running")
        producerIsDone.value = True
        print("Producer thread finished")
        return

def dataset_iterator2(x, preproc, bs):
    """ iterate over the lines of x in blocks of size bs"""
    i0 = 0
    while not producerIsDone.value or not q.empty():
        output = None
        i1 = i0
        itStart = time.time()
        # for t in range(0,bs):
        #
        #     try:
        #         output.append(q.get(timeout=3))
        #     except:
        #         print('timeout occurred')
        #         producerIsDone = True
        #         break
        #     i1+=1
        try:
            output_obj = q.get(timeout = 200)
            output=output_obj[0]
            ids = output_obj[1]
            i1 += output.shape[0]
            # print("Getting batch from queue: " + str(q.qsize()))
        except:
            print('timeout occurred')
            producerIsDone.value = True
            break
        i0_orig = i0
        i0=i1
        if output is not None and output.shape[0] > 0:
            itEnd = time.time()
            # print('fetch time: ' + str(itEnd-itStart) + 's for ' + str(output.shape[0]) + 'vectors')
            if q.empty():
                # print("q is empty")
                pass
            yield i0_orig, ids, output
        else:
            yield None,None,None
    yield None, None,None


def dataset_iterator(x, preproc, bs):
    """ iterate over the lines of x in blocks of size bs"""

    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    def prepare_block(i):
        xb = sanitize(x[i[0]:i[1]])
        return i[0], None, preproc.apply_py(xb)

    return rate_limited_imap(prepare_block, block_ranges)



def eval_intersection_measure(gt_I, I):
    """ measure intersection measure (used for knngraph)"""
    inter = 0
    rank = I.shape[1]
    assert gt_I.shape[1] >= rank
    for q in range(nq_gt):
        inter += faiss.ranklist_intersection_size(
            rank, faiss.swig_ptr(gt_I[q, :]),
            rank, faiss.swig_ptr(I[q, :].astype('int64')))
    return inter / float(rank * nq_gt)


def make_vres_vdev(self,gpu_resources, i0=0, i1=-1):
    " return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if i1 == -1:
        i1 = ngpu
    for i in range(int(i0), int(i1)):
        print("i0: " + str(i0) + "i1: " + str(i1))
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev, gpu_res


#################################################################
# Prepare ground truth (for the knngraph)
#################################################################


def compute_GT():
    print ("compute GT")
    t0 = time.time()

    gt_I = np.zeros((nq_gt, gt_sl), dtype='int64')
    gt_D = np.zeros((nq_gt, gt_sl), dtype='float32')
    heaps = faiss.float_maxheap_array_t()
    heaps.k = gt_sl
    heaps.nh = nq_gt
    heaps.val = faiss.swig_ptr(gt_D)
    heaps.ids = faiss.swig_ptr(gt_I)
    heaps.heapify()
    bs = 10 ** 5

    n, d = xb.shape
    xqs = sanitize(xq[:nq_gt])

    db_gt = faiss.IndexFlatL2(d)
    vres, vdev = make_vres_vdev()
    db_gt_gpu = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, db_gt)

    # compute ground-truth by blocks of bs, and add to heaps
    for i0, xsl in dataset_iterator(xb, IdentPreproc(d), bs):
        db_gt_gpu.add(xsl)
        D, I = db_gt_gpu.search(xqs, gt_sl)
        I += i0
        heaps.addn_with_ids(
            gt_sl, faiss.swig_ptr(D), faiss.swig_ptr(I), gt_sl)
        db_gt_gpu.reset()
        print ("\r   %d/%d, %.3f s" % (i0, n, time.time() - t0))

    heaps.reorder()

    print ("GT time: %.3f s" % (time.time() - t0))
    return gt_I




#################################################################
# Prepare the vector transformation object (pure CPU)
#################################################################



def train_preprocessor_api(filelist,train_size,kp_per_file,preproc_str_local,cache_dir):
    xt_local =  load_training_from_file_list(filelist,train_size,kp_per_file,cache_dir)
    preproc = train_preprocessor(preproc_str_local,xt_local)
    return preproc

def train_preprocessor():
    train_preprocessor(preproc_str,xt)

def train_preprocessor(preproc_str_local,xt_local):
    print("train preproc", preproc_str_local)
    d = xt_local.shape[1]
    t0 = time.time()
    if preproc_str_local.startswith('OPQ'):
        fi = preproc_str_local[3:-1].split('_')
        m = int(fi[0])
        dout = int(fi[1]) if len(fi) == 2 else d
        preproc = faiss.OPQMatrix(d, m, dout)
    elif preproc_str_local.startswith('PCAR'):
        dout = int(preproc_str_local[4:-1])
        preproc = faiss.PCAMatrix(d, dout, 0, True)
    else:
        assert False
    preproc.train(sanitize(xt_local[:100000000]))
    print("preproc train done in %.3f s" % (time.time() - t0))
    return preproc

def get_preprocessor():
    return get_preprocessor(preproc_cachefile)

def get_preprocessor(cache_dir):
    preprocstart = time.time()
    if preproc_str:
        if not cache_dir or not os.path.exists(cache_dir):
            preproc = train_preprocessor()
            if cache_dir:
                print ("store", cache_dir)
                faiss.write_VectorTransform(preproc, cache_dir)
        else:
            print ("load", cache_dir)
            preproc = faiss.read_VectorTransform(cache_dir)
    else:
        d = xt.shape[1]
        preproc = IdentPreproc(d)
    preprocend = time.time()
    preprocTime = preprocend-preprocstart
    with open(times_cachefile,'a') as fp:
        fp.write('preproc: ' + str(preprocTime) + '\n')

    return preproc


#################################################################
# Prepare the coarse quantizer
#################################################################


def train_coarse_quantizer(x, k, preproc):
    d = preproc.d_out
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    # clus.niter = 2
    clus.max_points_per_centroid = 10000000

    print ("apply preproc on shape", x.shape, 'k=', k)
    t0 = time.time()
    x = preproc.apply_py(sanitize(x))
    print ("   preproc %.3f s output shape %s" % (
        time.time() - t0, x.shape))

    vres, vdev = make_vres_vdev()
    index = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, faiss.IndexFlatL2(d))

    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)

    return centroids.reshape(k, d)


def prepare_coarse_quantizer(preproc):

    if cent_cachefile and os.path.exists(cent_cachefile):
        print ("load centroids", cent_cachefile)
        centroids = np.load(cent_cachefile)
    else:
        nt = max(1000000, 256 * ncent)
        print ("train coarse quantizer...")
        t0 = time.time()
        centroids = train_coarse_quantizer(xt[:nt], ncent, preproc)
        print ("Coarse train time: %.3f s" % (time.time() - t0))
        if cent_cachefile:
            print ("store centroids", cent_cachefile)
            np.save(cent_cachefile, centroids)

    coarse_quantizer = faiss.IndexFlatL2(preproc.d_out)
    coarse_quantizer.add(centroids)

    return coarse_quantizer


#################################################################
# Make index and add elements to it
#################################################################


def prepare_trained_index(preproc):
    if os.path.exists(codes_cachefile):
        print("loading pretrained codebook")
        return faiss.read_index(codes_cachefile)
    quantstart = time.time()
    coarse_quantizer = prepare_coarse_quantizer(preproc)
    print("preproc d: " + str(preproc.d_out))
    d = preproc.d_out
    if pqflat_str == 'Flat':
        print ("making an IVFFlat index")
        idx_model = faiss.IndexIVFFlat(coarse_quantizer, d, ncent,
                                       faiss.METRIC_L2)
    else:
        m = int(pqflat_str[2:])
        assert m < 56 or use_float16, "PQ%d will work only with -float16" % m
        print ("making an IVFPQ index, m = ", m)
        idx_model = faiss.IndexIVFPQ(coarse_quantizer, d, ncent, m, 8)

    coarse_quantizer.this.disown()
    idx_model.own_fields = True

    # finish training on CPU
    t0 = time.time()
    print ("Training vector codes")
    x = preproc.apply_py(sanitize(xt[:1000000]))
    idx_model.train(x)
    faiss.write_index(idx_model,codes_cachefile)
    print ("  done %.3f s" % (time.time() - t0))

    quantend = time.time()
    quantTime = quantend-quantstart
    with open(times_cachefile,'a') as fp:
        fp.write('PQ Training: ' + str(quantTime) + '\n')
    return idx_model


def compute_populated_index(preproc):
    """Add elements to a sharded index. Return the index and if available
    a sharded gpu_index that contains the same data. """

    indexall = prepare_trained_index(preproc)
    addstart = time.time()
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = use_float16
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = use_precomputed_tables
    co.indicesOptions = faiss.INDICES_CPU

    co.verbose = True
    co.reserveVecs = max_add if max_add > 0 else xb.shape[0]
    co.shard = True
    # indexall.ntotal = xb.shape[0]
    print("running make_vres_vdev")
    vres, vdev = make_vres_vdev()
    print("running cpu to gpu multiple")
    print("Index size: " + str(indexall.ntotal))
    gpu_index = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, indexall, co)
    indexall.ntotal = 0
    print ("add...")
    t0 = time.time()
    nb = xb.shape[0]
    # gpu_index.ntotal = 0
    print("nb: " + str(nb))
    batch_add_count = 0
    if datasetJsonFile is None:
        iterator = dataset_iterator(xb,preproc,add_batch_size)
    else:
        iterator = dataset_iterator2(xb,preproc,add_batch_size)
        itCount = 0
        cpuFlushCount = 0
    tget0=time.time()
    for i0, ids, xs in iterator:
        if i0 is None or xs is None:
            print("Finishing up iterator...")
            break
        i1 = i0 + xs.shape[0]
        tget1 = time.time()
        if ids is not None:
            gpu_index.add_with_ids(xs, ids)
        else:
            print("ids is none")
            gpu_index.add_with_ids(xs, np.arange(i0, i1))

        if max_add > 0 and gpu_index.ntotal > max_add:
            print ("Flush indexes to CPU")
            for i in range(ngpu):
                index_src_gpu = faiss.downcast_index(gpu_index.at(i))
                index_src = faiss.index_gpu_to_cpu(index_src_gpu)
                print ("  index %d size %d" % (i, index_src.ntotal))
                index_src.copy_subset_to(indexall, 0, 0, nb)

                index_src_gpu.reset()
                index_src.reset()
                index_src_gpu.reserveMemory(max_add)
            gpu_index.sync_with_shard_indexes()
            # freeMem = psutil.virtual_memory().total #in kilobytes
            # usedMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss #in kilobytes on linux
            # if usedMem*1.0/freeMem >= mem_buffer:
            #     # we need to save the index to disk and reset it
            #     print("Memory almost full, flushing CPU index to disk")
        itCount +=1
        treport =  time.time() - t0
        progress_percent = (i0)/nb
        totalTimeTake = (treport*nb/(i0+1))/60/60
        timeLeft = totalTimeTake*(1-progress_percent)
        freeMem = psutil.virtual_memory().total/1024  # in kilobytes
        usedMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # in kilobytes on linux
        update = '\r%d/%d (%.3f s, %.3f hours total, %.3f hours left, ntotal=%d %.3f percent complete , %d batches left in queue, %d files left in file queue, %d sets left in preproc queue, %.3f seconds to get batch)  ' % (
            i0, nb,treport,totalTimeTake,timeLeft,gpu_index.ntotal,progress_percent*100,q.qsize(),file_Load_q.qsize(),preproc_q.qsize(),tget1-tget0)
        print(update)
        del xs
        del ids
        del i0
        if batch_add_count%10 == 0:
            with open('machine_'+str(machine_num)+'_progress.txt','a') as fp:
                fp.write(update+'\n')

        sys.stdout.flush()
        batch_add_count+=1
        tget0 = time.time()
    print ("Add time: %.3f s" % (time.time() - t0))

    print ("Aggregate indexes to CPU")
    t0 = time.time()

    for i in range(ngpu):
        index_src = faiss.index_gpu_to_cpu(gpu_index.at(i))
        print ("  index %d size %d" % (i, index_src.ntotal))
        index_src.copy_subset_to(indexall, 0, 0, nb)

    print ("  done in %.3f s" % (time.time() - t0))
    print("Index size: ", indexall.ntotal)
    if max_add > 0:
        # it does not contain all the vectors
        gpu_index = None
    addend = time.time()
    addTime = addend-addstart
    with open(times_cachefile,'a') as fp:
        fp.write('Adding ' + str(xb.shape[0]) + 'feature vectors (machine ' + str(machine_num) + '): ' + str(addTime) + '\n')
    return gpu_index, indexall

def compute_populated_index_2(preproc):

    indexall = prepare_trained_index(preproc)

    # set up a 3-stage pipeline that does:
    # - stage 1: load + preproc
    # - stage 2: assign on GPU
    # - stage 3: add to index
    if not featuredictfile == "":
        stage1 = dataset_iterator2(xb,preproc,add_batch_size)
    else:
        stage1 = dataset_iterator(xb, preproc, add_batch_size)

    vres, vdev = make_vres_vdev()
    coarse_quantizer_gpu = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, indexall.quantizer)

    def quantize(i):
        _, assign = coarse_quantizer_gpu.search(i[1], 1)
        return i[0],i[1], assign.ravel()

    stage2 = rate_limited_imap(quantize, stage1)

    print("add...")
    t0 = time.time()
    if totalFeatureNum == -1: nb = xb.shape[0]
    else: nb = totalFeatureNum

    for i0, xs, assign in stage2:
        i1 = i0 + xs.shape[0]
        if indexall.__class__ == faiss.IndexIVFPQ:
            indexall.add_core_o(i1 - i0, faiss.swig_ptr(xs),
                                None, None, faiss.swig_ptr(assign))
        elif indexall.__class__ == faiss.IndexIVFFlat:
            indexall.add_core(i1 - i0, faiss.swig_ptr(xs), None,
                              faiss.swig_ptr(assign))
        else:
            assert False

        print ('\r%d/%d (%.3f s)  ' % (
            i0, nb, time.time() - t0),
        sys.stdout.flush())
    print ("Add time: %.3f s" % (time.time() - t0))

    return None, indexall



def get_populated_index(preproc):
    print("looking for index at ", index_cachefile)
    if not index_cachefile or not os.path.exists(index_cachefile):
        if not altadd:
            gpu_index, indexall = compute_populated_index(preproc)
        else:
            gpu_index, indexall = compute_populated_index_2(preproc)
        if index_cachefile:
            print ("store", index_cachefile)
            faiss.write_index(indexall, index_cachefile)
    else:
        print("Found cached index!")
        print ("load", index_cachefile)
        indexall = faiss.read_index(index_cachefile)
        gpu_index = None
    if not runEval:
        return None
    if ngpu == 0:
        print("no gpu specified, staying on CPU...")
        return indexall
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = use_float16
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = use_precomputed_tables
    co.indicesOptions = 0
    # if max_add > 0:
    #     co.reserveVecs = max_add
    co.verbose = True
    co.shard = True # the replicas will be made "manually"
    t0 = time.time()
    print ("CPU index contains %d vectors, move to GPU" % indexall.ntotal)
    if replicas == 1:

        if not gpu_index:
            print ("copying loaded index to GPUs")
            vres, vdev = make_vres_vdev()
            try:
                print("starting watchdog...")
                startWatchdog(60*15)
                print("transfer index to gpu..")
                index = faiss.index_cpu_to_gpu_multiple(
                    vres, vdev, indexall, co)
                watchdog_q.put(1)
                print("Index created")
            except:
                print("Exception occured")
                exit(0)
        else:
            index = gpu_index

    else:
        del gpu_index # We override the GPU index

        print ("Copy CPU index to %d sharded GPU indexes" % replicas)

        index = faiss.IndexProxy()
        print("starting watchdog...")
        startWatchdog(60 * 15)
        for i in range(replicas):
            gpu0 = ngpu * i / replicas
            gpu1 = ngpu * (i + 1) / replicas
            vres, vdev = make_vres_vdev(gpu0, gpu1)

            print( "   dispatch to GPUs %d:%d" % (gpu0, gpu1))

            index1 = faiss.index_cpu_to_gpu_multiple(
                vres, vdev, indexall, co)
            index1.this.disown()
            index.addIndex(index1)
        index.own_fields = True
    del indexall
    print( "move to GPU done in %.3f s" % (time.time() - t0))
    return index



#################################################################
# Perform search
#################################################################


def eval_dataset(index, preproc):
    ps = None
    if ngpu >0:
        ps = faiss.GpuParameterSpace()
        ps.initialize(index)
    print ("search...")
    sl = query_batch_size
    nq = xq.shape[0]
    for nprobe in nprobes:
        if ngpu > 0:
            ps.set_index_parameter(index, 'nprobe', nprobe)
        else:
            index.nprobe = nprobe
        t0 = time.time()

        if sl == 0:
            D, I = index.search(preproc.apply_py(sanitize(xq)), nnn)
        else:
            I = np.empty((nq, nnn), dtype='int32')
            D = np.empty((nq, nnn), dtype='float32')

            inter_res = ''
            print("Running query...")
            for i0, ids, xs in dataset_iterator(xq, preproc, sl):
                print ('\r%d/%d (%.3f s Query %s)   ' % (
                    i0, nq, time.time() - t0, inter_res),
                sys.stdout.flush())

                i1 = i0 + xs.shape[0]
                Di, Ii = index.search(xs, nnn)

                I[i0:i1] = Ii
                D[i0:i1] = Di


            print("Done with query, Saving... ",I.shape)
            t1 = time.time()
            try:
                os.makedirs(results_folder)
            except:
                pass

            I_fname_i =  results_Ifile + 'nprobe' + str(nprobe) + tier2string + "_I.npy"
            print ("storing", I_fname_i)
            print("Saving top indexes...")
            np.save(I_fname_i,I)
            D_fname_i = results_Dfile + 'nprobe' + str(nprobe) + tier2string +"_D.npy"
            print("saving top distances...")
            print ("storing", D_fname_i)
            np.save(D_fname_i,D)
    print("All done!")
    return

def generateIndex(dbname,index_key,preproc_str,iv_str,pqflat_str,jsonMapFile,ngpu,use_precomputed_tables,add_batch_size,tempmem,use_cache,altadd,use_float16,max_add,cacheroot,nproc,job_num,total_jobs,):

    #Load necessary files and create threading queues
    with open(jsonMapFile, 'r') as fp:
        datasetJsonFile = json.load(fp)
    preproc_q = Manager().Queue(min(add_batch_size * 1.0 / kp_per_file * 400, 1500))
    file_Load_q = Manager().Queue(add_batch_size * 1.0 / kp_per_file * 400)
    file_Touch_q = Manager().Queue(20)
    watchdog_q = Manager().Queue(1)
    kill_q = Manager().Queue(1)
    q = Manager().Queue(100)
    if cacheroot is None:
        cacheroot = '/tmp/bench_gpu_1bn'
    if not os.path.isdir(cacheroot):
        print("%s does not exist, creating it" % cacheroot)
        os.mkdir(cacheroot)
    ncent = int(ivf_str[3:])
    prefix = ''



#################################################################
# Driver
#################################################################

if __name__ == "__main__":

    while args:
        a = args.pop(0)
        if a == '-h': usage()
        elif a == '-ngpu':      ngpu = int(args.pop(0))
        elif a == '-R':         replicas = int(args.pop(0))
        elif a == '-noptables': use_precomputed_tables = False
        elif a == '-abs':       add_batch_size = int(args.pop(0))
        elif a == '-qbs':       query_batch_size = int(args.pop(0))
        elif a == '-nnn':       nnn = int(args.pop(0))
        elif a == '-tempmem':   tempmem = int(args.pop(0))
        elif a == '-nocache':   use_cache = False
        elif a == '-knngraph':  knngraph = True
        elif a == '-altadd':    altadd = True
        elif a == '-float16':   use_float16 = True
        elif a == '-nprobe':    nprobes = [int(x) for x in args.pop(0).split(',')]
        elif a == '-max_add':   max_add = int(args.pop(0))
        elif a == '-base_feats': base_feature_file = args.pop(0)
        elif a == '-query_file': query_feature_file = args.pop(0)
        elif a == '-train_size': train_size = int(args.pop(0))
        # elif a == '-feature_dictionary': featuredictfile=args.pop(0)
        elif a == '-keypoints_per_file': kp_per_file = int(args.pop(0))
        elif a == '--kpstrict' : kpstrict = True
        # elif a == '-total_features': totalFeatureNum = int(args.pop(0))
        # elif a == '-base_prefix' : base_prefix = args.pop(0)
        elif a == '-cache_root' : cacheroot = args.pop(0)
        elif a == '-nproc' : nproc = int(args.pop(0))
        elif a == '-mem_buffer' : mem_buffer = int(args.pop(0))
        elif a == '-job_num' : machine_num = max(int(args.pop(0))-1,0)
        elif a == '-total_jobs': num_jobs = max(int(args.pop(0)),0)
        # elif a == '-featurecount_file': featureCountFile = args.pop(0)
        elif a == '-evaluate' : runEval = True
        elif a == '-query_folder': query_folder = args.pop(0)
        elif a == '-index_offset': index_offset = int(args.pop(0))
        elif a == '-json_map': jsonMapFile = args.pop(0)
        elif a == '-tier2_search': tier2search = True
        elif a == '-genQueryJsonOnly' : genQueryJsonOnly = True
        elif not dbname:        dbname = a
        elif not index_key:     index_key = a
        else:
            print("argument %s unknown" % a)
            sys.exit(1)
    print("Job id: ", machine_num)
    print('reading json file...')

    with open(jsonMapFile, 'r') as fp:
        datasetJsonFile = json.load(fp)
    preproc_q = Manager().Queue(min(add_batch_size * 1.0 / kp_per_file * 400, 1500))
    file_Load_q = Manager().Queue(add_batch_size * 1.0 / kp_per_file * 400)
    file_Touch_q = Manager().Queue(20)
    watchdog_q = Manager().Queue(1)
    kill_q = Manager().Queue(1)
    q = Manager().Queue(100)
    if cacheroot is None:
        cacheroot = '/tmp/bench_gpu_1bn'
    if not os.path.isdir(cacheroot):
        print("%s does not exist, creating it" % cacheroot)
        os.mkdir(cacheroot)

    totalStart = time.time()
    #################################################################
    # Prepare dataset
    #################################################################
    xt = None
    xb = None
    xq = None
    gt_I = None

    #################################################################
    # Parse index_key and set cache files
    #
    # The index_key is a valid factory key that would work, but we
    # decompose the training to do it faster
    #################################################################


    pat = re.compile('(OPQ[0-9]+(_[0-9]+)?,|PCAR[0-9]+,)?' +
                     '(IVF[0-9]+),' +
                     '(PQ[0-9]+|Flat)')

    matchobject = pat.match(index_key)

    assert matchobject, 'could not parse ' + index_key

    mog = matchobject.groups()

    preproc_str = mog[0]
    ivf_str = mog[2]
    pqflat_str = mog[3]

    ncent = int(ivf_str[3:])

    prefix = ''

    if knngraph:
        gt_cachefile = '%s/BK_gt_%s.npy' % (cacheroot, dbname)
        prefix = 'BK_'
        # files must be kept distinct because the training set is not the
        # same for the knngraph

    if preproc_str:
        preproc_cachefile = '%s/%spreproc_%s_%s.vectrans' % (
            cacheroot, prefix, dbname, preproc_str[:-1])
    else:
        preproc_cachefile = None
        preproc_str = ''

    cent_cachefile = '%s/%scent_%s_%s%s.npy' % (
        cacheroot, prefix, dbname, preproc_str, ivf_str)
    index_cachefolder = '%s%s_%s%s,%s' % (prefix, dbname, preproc_str, ivf_str, pqflat_str)
    index_cachefile = '%s/%s/%s%s_%s%s,%s_part%sOf%s.index' % (
        cacheroot, index_cachefolder, prefix, dbname, preproc_str, ivf_str, pqflat_str, str(machine_num), str(num_jobs))
    times_cachefile = '%s/%s%s_%s%s,%s_times.txt' % (
        cacheroot, prefix, dbname, preproc_str, ivf_str, pqflat_str)
    trainfeat_cachefile = '%s/%s%s_%s%s,%s_trainfeat.npy' % (
        cacheroot, prefix, dbname, preproc_str, ivf_str, pqflat_str)
    queryfeat_cachefile = '%s/%s%s_%s%s,%s_queryfeat.npy' % (
        cacheroot, prefix, dbname, preproc_str, ivf_str, pqflat_str)
    codes_cachefile = '%s/%s%s_%s%s,%s_codes.index' % (
        cacheroot, prefix, dbname, preproc_str, ivf_str, pqflat_str)
    preproc_queue_dump = '%s/%s%s_%s%s,%s_preproc_queue' % (
        cacheroot, prefix, dbname, preproc_str, ivf_str, pqflat_str)
    queue_dump = '%s/%s%s_%s%s,%s_queue' % (
        cacheroot, prefix, dbname, preproc_str, ivf_str, pqflat_str)
    map_folder = '%s/%s/%s' % (cacheroot, index_cachefolder, 'IDMaps')
    imageIDMap_cachefile = '%s/%s%s_%s%s,%s_imagetoid%sOf%s.json' % (
        map_folder, prefix, dbname, preproc_str, ivf_str, pqflat_str, str(machine_num), str(num_jobs))
    IDimageMap_cachefile = '%s/%s%s_%s%s,%s_idtoimage%sOf%s.json' % (
        map_folder, prefix, dbname, preproc_str, ivf_str, pqflat_str, str(machine_num), str(num_jobs))
    progress_cachefile = '%s/%s%s_%s%s,%s_progress.json' % (
        cacheroot, prefix, dbname, preproc_str, ivf_str, pqflat_str)
    results_folder = '%s/%s/%s/' % (
        cacheroot, index_cachefolder, 'results')
    results_map_folder = '%s/%s/' % (
        results_folder, 'IDMaps')
    imageIDMap_query_cachefile = '%s/%s%s_%s%s,%s_imagetoid%sOf%s.json' % (
        results_map_folder, prefix, dbname, preproc_str, ivf_str, pqflat_str, str(machine_num), str(num_jobs))
    IDimageMap_query_cachefile = '%s/%s%s_%s%s,%s_idtoimage%sOf%s.json' % (
        results_map_folder, prefix, dbname, preproc_str, ivf_str, pqflat_str, str(machine_num), str(num_jobs))
    ID_query_cachefile = '%s/%s%s_%s%s,%s_queryData%sOf%s' % (
        results_map_folder, prefix, dbname, preproc_str, ivf_str, pqflat_str, str(machine_num), str(num_jobs))
    try:
        os.makedirs(os.path.dirname(imageIDMap_query_cachefile))
    except:
        pass
    tier2string = ""
    if tier2search:
        results_Ifile = '%s/%s/results_part%sOf%s_' % (
            results_folder, 'tier2', str(machine_num), str(num_jobs))
        results_Dfile = '%s/%s/results_part%sOf%s_' % (
            results_folder, 'tier2', str(machine_num), str(num_jobs))
        try:
            os.makedirs(os.path.dirname(results_Ifile))
        except:
            pass
    else:
        results_Ifile = '%s/results_part%sOf%s_' % (
            results_folder, str(machine_num), str(num_jobs))
        results_Dfile = '%s/results_part%sOf%s_' % (
            results_folder, str(machine_num), str(num_jobs))

    try:
        os.makedirs(map_folder)
    except:
        pass
    try:
        os.makedirs(results_map_folder)
    except:
        pass

    image_to_ID_Map = {}
    ID_to_image_Map = {}

    # if os.path.exists(preproc_queue_dump):
    #     os.unlink(preproc_queue_dump)
    # dump(preproc_q,preproc_queue_dump)
    # preproc_q = load(preproc_queue_dump,mmap_mode='w+')
    #
    # if os.path.exists(queue_dump):
    #     os.unlink(queue_dump)
    # dump(q,queue_dump)
    # q = load(queue_dump,mmap_mode='w+')

    if not use_cache:
        preproc_cachefile = None
        cent_cachefile = None
        index_cachefile = None

    print("cachefiles:")
    print(preproc_cachefile)
    print(cent_cachefile)
    print(index_cachefile)

    print("Preparing dataset", dbname)
    totalFeatureNum = int(datasetJsonFile['totalFeatures'])
    if datasetJsonFile is not None:
        print("loading feature dictionary..." + featuredictfile)
        featureDictionary = datasetJsonFile['nameToPathMap']
        # with open(featuredictfile,'r') as fp:
        #     featureDictionary = json.load(fp)
        print("found " + str(len(featureDictionary)) + " files")
        featureDictionaryKeys = datasetJsonFile['sortedFeatureKeys']
        # featureDictionaryKeys = sorted(fnmatch.filter(list(featureDictionary.keys()), '*.npy'))
        # print(featureDictionaryKeys[1:20])
        span = len(featureDictionaryKeys) / num_jobs
        startnum = int(machine_num * span)
        endnum = int(startnum + span)
        print('Start index: ', startnum, ' end index: ', endnum)
        featureDictionaryKeys = featureDictionaryKeys[startnum:min(endnum, len(featureDictionaryKeys))]
        # if not preproc_cachefile or not os.path.exists(preproc_cachefile):
        if not runEval:
            xt = load_training_from_features_dictionary(featureDictionary, train_size, kp_per_file)
        else:
            xt = np.zeros((0, 0))
        if runEval:
            if query_folder is not None:

                query_out = load_query_from_features_directory(query_folder, genQueryJsonOnly)

                xq = query_out[0]
                queryJsonFile = {}
                if query_out[1] is not None:
                    queryJsonFile['qimageIDMap'] = query_out[1]
                    queryJsonFile['qIDimageMap'] = query_out[2]
                    queryJsonFile['queryFeatureIDs'] = list(query_out[3].tolist())
                    # print(len(query_out[1]),' ',len(query_out[2]), ' ',len(query_out[3]))

                    with open(imageIDMap_query_cachefile, 'w') as fp:
                        json.dump(queryJsonFile, fp)
                if genQueryJsonOnly:
                    print('Query json generated at', imageIDMap_query_cachefile, ', exiting')
                    exit(1)

            elif not query_feature_file == "":
                xq = load_qery_from_features_file(query_feature_file)
            else:
                print("no query folder input!")
        featureCountList = datasetJsonFile['featureCounts']
        # if totalFeatureNum == -1:
        #     totalFeatureNum = kp_per_file*len(featureDictionaryKeys)
        #     totalFeatureNum /= num_jobs
        #     totalFeatureNum = max(int(totalFeatureNum), int(len(featureDictionaryKeys) * kp_per_file / num_jobs))
        # else featureCountFile is not None:
        print("Loading feature counts...")
        # featureCountList = np.load(featureCountFile)
        s1 = 0
        if startnum > 0:
            s1 = featureCountList[startnum - 1]
        shard_offset = startnum
        totalFeatureNum = int(featureCountList[min(endnum - 1, len(featureCountList) - 1)] - s1)

        print('estimated total feature number is: ' + str(totalFeatureNum))
        xb = namedtuple("xb", 'shape')
        xb.shape = (totalFeatureNum, xt.shape[1])
        if xt is not None:
            print("feature train size is " + str(xt.shape[0]))
        else:
            print("Didn't load training features because they were already cached")

    elif not base_feature_file == "" and not query_feature_file == "":
        xb = mmap_fvecs(base_feature_file)
        xq = mmap_fvecs(query_feature_file)
        if train_size == -1:
            train_size = int(xb.shape[0] * .05)
        xb_idx = np.random.choice(xb.shape[0], min(train_size, xb.shape[0]), replace=False)
        xt = xb[xb_idx]
    else:
        if dbname.startswith('SIFT'):
            # SIFT1M to SIFT1000M
            dbsize = int(dbname[4:-1])
            xb = mmap_bvecs('bigann/bigann_base.bvecs')
            xq = mmap_bvecs('bigann/bigann_query.bvecs')
            xt = mmap_bvecs('bigann/bigann_learn.bvecs')

            # trim xb to correct size
            xb = xb[:dbsize * 1000 * 1000]

            gt_I = ivecs_read('bigann/gnd/idx_%dM.ivecs' % dbsize)

        elif dbname == 'Deep1B':
            xb = mmap_fvecs('deep1b/base.fvecs')
            xq = mmap_fvecs('deep1b/deep1B_queries.fvecs')
            xt = mmap_fvecs('deep1b/learn.fvecs')
            # deep1B's train is is outrageously big
            xt = xt[:10 * 1000 * 1000]
            gt_I = ivecs_read('deep1b/deep1B_groundtruth.ivecs')

        else:
            # print() sys.stderr, 'unknown dataset', dbname
            print('unknown dataset ', dbname)
            sys.exit(1)

    if knngraph:
        # convert to knn-graph dataset
        xq = xb
        xt = xb
        # we compute the ground-truth on this number of queries for validation
        nq_gt = 10000
        gt_sl = 100

        # ground truth will be computed below
        gt_I = None

    print("sizes: B %s Q %s T %s gt %s" % (
        xb.shape if xb is not None else None, xq.shape if xq is not None else None, xt.shape,
        gt_I.shape if gt_I is not None else None))

    #################################################################
    # Wake up GPUs
    #################################################################

    print("preparing resources for %d GPUs" % ngpu)

    gpu_resources = []

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    if knngraph:

        if gt_cachefile and os.path.exists(gt_cachefile):
            print("load GT", gt_cachefile)
            gt_I = np.load(gt_cachefile)
        else:
            gt_I = compute_GT()
            if gt_cachefile:
                print("store GT", gt_cachefile)
                np.save(gt_cachefile, gt_I)



    preproc = get_preprocessor()
    print('preproc retrieved')
    producer = None
    if not runEval:
        producer = ProducerThread(name='producer',preproc=preproc,bs=add_batch_size,q_preproc=preproc_q,featureKeys=featureDictionaryKeys,ncores=nproc)
        time.sleep(1)
        producer.start()
        time.sleep(30)
    print('getting populated index..')
    index = get_populated_index(preproc)
    # if index is None:
    #     print("error " + 1)
    print('Index retrieved!')
    keep_running_file_write = False
    if producer:
        producer.preproc_is_running.value = False
    if runEval:
        eval_dataset(index,preproc)
    totalend = time.time()
    totalTime = totalend-totalStart
    with open(times_cachefile, 'a') as fp:
        fp.write('Total: ' + str(totalTime) + '\n')
    # eval_dataset(index, preproc)

    # make sure index is deleted before the resources
    del index
    os.system("pkill -9 python")
    os._exit(0)
