# create a linear time index for example purposes

import os
import gc
import traceback
import pickle
from resources import Resource
import fileutil
# use the logging class to write logs out
import logging
import numpy as np
import time
import os
import sys
import faiss
import json
from multiprocessing import Manager
import indexfunctions
import progressbar
import psutil
import indexMerger

def getMemUsage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

class indexConstruction:
    # Put variable for your code here
    index = {}

    # modify these variables
    algorithmName = "ND_dsurf_5000_filtering"
    preproc_type = ""
    nonBinarizedFeatures = False
    trainSize = "100000"
    algorithmVersion = "1.0"
    onlyBuildIDMap = False
    dbname = None
    index_key = None
    rebuildShards = False

    ngpu = faiss.get_num_gpus()

    replicas = 1  # nb of replicas of sharded dataset
    add_batch_size = 32768
    use_precomputed_tables = False
    tempmem = -1#1536*1024*1024  # if -1, use system default
    max_add = -1
    use_float16 = True
    use_cache = True
    featuredictfile = ""
    base_feature_file = ""
    query_feature_file = ""
    featureCountFile = None
    train_size = 100000
    recursive = False
    totalFeatureNum = -1
    kp_per_file = 5000
    featureDictionary = {}
    producerTimeout = 0
    timeoutMax = 20
    base_prefix = ""
    featureDictionaryKeys = None
    cacheroot = None
    totalTime = 0
    preprocTime = 0
    quantTime = 0
    addTime = 0
    kpstrict = False
    mem_buffer = .9
    num_jobs = 1
    machine_num = 0
    index_offset = 0
    shard_offset = 0
    preproc_str = "OPQ32_96"
    ivf_str = "IVF393216"
    pqflat_str = "PQ32"
    jsonMapFile = None
    tier2search = False
    producerIsDone = None
    keep_running_file_write = None
    featuredimensions = 64
    imCount = 0
    preproc_q = None
    file_Load_q= None
    file_Touch_q = None
    watchdog_q = None
    kill_q = None
    q = None
    ngpu = 1
    index = None
    index_mmap= None
    gpu_index = None
    preproc = None
    IDtoNameMap = {}
    gpuRAM = 7  # in GB
    feats_per_file = 5000
    totalImagesToIndex = -1
    cpuShardCount = 0
    # Set any parameters needed for the model initialization here

    def __init__(self):
        # Set up file processing queues
        self.producerIsDone = Manager().Value('j', False)
        self.keep_running_file_write = Manager().Value('k', True)
        preproc_q = Manager().Queue(min(self.add_batch_size * 1.0 / self.kp_per_file * 400, 1500))
        self.file_Load_q = Manager().Queue(self.add_batch_size * 1.0 / self.kp_per_file * 400)
        self.file_Touch_q = Manager().Queue(20)
        self.watchdog_q = Manager().Queue(1)
        self.kill_q = Manager().Queue(1)
        self.q = Manager().Queue(100)
        self.preproc_dims = int(self.preproc_str[3:].split('_')[1])
        cacheroot = fileutil.getResourcePath(self.algorithmName)
        self.cacheroot = cacheroot
        self.preproc_cachefile = os.path.join(cacheroot, 'preproc.vectrans')
        self.cent_cachefile = os.path.join(cacheroot, 'centroids.npy')
        self.index_cachefile = os.path.join(cacheroot,'index.index')
        self.times_cachefile = os.path.join(cacheroot, 'computetimes.txt')
        self.trainfeat_cachefile = os.path.join(cacheroot, 'trainfeatures.npy')
        self.IDMap_cachefile = os.path.join(cacheroot,'IDMap.json')
        self.codes_cachefile = os.path.join(cacheroot, 'codes.index')
        self.gpu_resources = indexfunctions.wake_up_gpus(self.ngpu,self.tempmem)
        print("gpus found: ",len(self.gpu_resources))
        self.ncent = int(self.ivf_str[3:])
        self.alreadyTrained = False
        # passes file list to the train function (optional needed)
        # create your own class variables to utilize this function
        # I have concerns about making sure we are able to run this in existing RAM dynamically since all features
        # will be needed at run time. Can you use send a portion? Can this be done offline as we begin to scale further
        # without much accuracy impact? Passing a file list here in case it can be used to help optimize
        # for 5 Million images this is not as large a concern, bbut it will be in the future as we try to scale this further.
    def indexSaveNameForShard(self,shardnum,machineNum=1):
        savename = self.index_cachefile[:-6]+"_machine"+"%03d" % self.machine_num+'_shard' + "%03d" % shardnum + '.index'
        return savename
    def load_training_from_file_list(self, filelist):
        import os
        import numpy as np
        if os.path.exists(self.preproc_cachefile):
            print('preproc index already exists, not loading training features...')
            allfeats =  np.load(self.trainfeat_cachefile)
        elif os.path.exists(self.trainfeat_cachefile):
            print('found cached training features, loading..')
            allfeats = np.load(self.trainfeat_cachefile)
        else:
            # if os.path.exists(cache_dir):
            #     print("Loading premade training file...")
            #     return np.load(cache_dir)
            if self.train_size == -1:
                self.train_size = 50000
            if self.kp_per_file == -1:
                fname = os.path.join(filelist[0])
                if self.nonBinarizedFeatures:
                    features = Resource('', np.load(fname), 'application/octet-stream')
                else:
                    features = Resource('', np.fromfile(fname, 'float32'), 'application/octet-stream')
                a = self.deserializeFeatures(features)
            randomKeys = np.random.choice(len(filelist), int(self.train_size * 1.0 / self.kp_per_file), replace=False)
            print("found " + str(len(filelist)) + " keys")
            allfeats = []
            print("Loading needed files for training...")
            bar=progressbar.ProgressBar()
            for idx in bar(randomKeys):
                key = filelist[idx]
                # print(key)
                if self.nonBinarizedFeatures:
                    featureResource = Resource(key, np.load(key), 'application/octet-stream')
                else:
                    featureResource = Resource(key, np.fromfile(key,'float32'), 'application/octet-stream')

                file = os.path.join(key)
                # print("loading" + file)
                try:
                    features = self.deserializeFeatures(featureResource)
                    # features = np.load(file)
                    allfeats.append(features)
                except Exception as e:
                    logging.error(traceback.format_exc())
                    print("WARNING: Couldn't load a file")
            print("Concatenating training files...")
            allfeats = np.concatenate(allfeats)
            print('training size: ', allfeats.shape[0])
            np.save(self.trainfeat_cachefile, allfeats)
        return allfeats

    def train_preprocessor(self, preproc_str_local, xt_local):
        if not self.preproc_cachefile or not os.path.exists(self.preproc_cachefile):
            print("train preproc", preproc_str_local)
            d = xt_local.shape[1]
            t0 = time.time()
            if preproc_str_local.startswith('OPQ'):
                fi = preproc_str_local[3:].split('_')
                m = int(fi[0])
                dout = int(fi[1]) if len(fi) == 2 else d
                preproc = faiss.OPQMatrix(d, m, dout)
            elif preproc_str_local.startswith('PCAR'):
                dout = int(preproc_str_local[4:-1])
                preproc = faiss.PCAMatrix(d, dout, 0, True)
            else:
                assert False
            preproc.train(indexfunctions.sanitize(xt_local[:100000000]))
            print("preproc train done in %.3f s" % (time.time() - t0))
            faiss.write_VectorTransform(preproc,self.preproc_cachefile)
        else:
            print("load preproc ", self.preproc_cachefile)
            preproc = faiss.read_VectorTransform(self.preproc_cachefile)
        return preproc

    def train_coarse_quantizer(self, x, k, preproc):
        d = preproc.d_out
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        # clus.niter = 2
        clus.max_points_per_centroid = 10000000

        print("apply preproc on shape", x.shape, 'k=', k)
        t0 = time.time()
        x = preproc.apply_py(indexfunctions.sanitize(x))
        print("   preproc %.3f s output shape %s" % (
            time.time() - t0, x.shape))
        vres, vdev = indexfunctions.make_vres_vdev(self.gpu_resources,ngpu=self.ngpu)
        index = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, faiss.IndexFlatL2(d))
        clus.train(x, index)
        centroids = faiss.vector_float_to_array(clus.centroids)

        return centroids.reshape(k, d)

    def prepare_coarse_quantizer(self, preproc, xt):

        if self.cent_cachefile and os.path.exists(self.cent_cachefile):
            print("load centroids ", self.cent_cachefile)
            centroids = np.load(self.cent_cachefile)
        else:
            nt = max(1000000, 256 * self.ncent)
            print("train coarse quantizer...")
            t0 = time.time()
            centroids = self.train_coarse_quantizer(xt[:nt], self.ncent, preproc)
            print("Coarse train time: %.3f s" % (time.time() - t0))
            if self.cent_cachefile:
                print("store centroids", self.cent_cachefile)
                np.save(self.cent_cachefile, centroids)
        coarse_quantizer = faiss.IndexFlatL2(preproc.d_out)
        coarse_quantizer.add(centroids)
        return coarse_quantizer

    def prepare_trained_index(self, preproc, coarse_quantizer, xt):
        if os.path.exists(self.codes_cachefile):
            print("load pretrained codebook")
            return faiss.read_index(self.codes_cachefile)
        d = preproc.d_out
        if self.pqflat_str == 'Flat':
            print("making an IVFFlat index")
            idx_model = faiss.IndexIVFFlat(coarse_quantizer, d, self.ncent,
                                           faiss.METRIC_L2)
        else:
            m = int(self.pqflat_str[2:])
            assert m < 56 or self.use_float16, "PQ%d will work only with -float16" % m
            print("making an IVFPQ index, m = ", m)
            idx_model = faiss.IndexIVFPQ(coarse_quantizer, d, self.ncent, m, 8)
        coarse_quantizer.this.disown()
        idx_model.own_fields = True
        # finish training on CPU
        t0 = time.time()
        x = preproc.apply_py(indexfunctions.sanitize(xt[:1000000]))
        idx_model.train(x)
        faiss.write_index(idx_model, self.codes_cachefile)
        return idx_model

    def trainIndexWithFeatures(self, xt):
        # Train the prerpoc transform
        trainedPreproc = self.train_preprocessor(self.preproc_str, xt)
        # Train the coarse quantizer centroids
        coarse_quantizer = self.prepare_coarse_quantizer(trainedPreproc, xt)
        # Train the codebooks for the index model
        trainedIndex = self.prepare_trained_index(trainedPreproc, coarse_quantizer, xt)
        return (trainedPreproc,coarse_quantizer,trainedIndex)
        # with open(times_cachefile, 'a') as fp:
        #     fp.write('PQ Training: ' + str(quantTime) + '\n')

    def trainIndex(self, featurelistfile):
        print('train index')
        filelist = []
        with open(featurelistfile) as featurelist:
            for f in featurelist:
                filelist.append(f.rstrip())
        xt = self.load_training_from_file_list(filelist)
        trainedParams = self.trainIndexWithFeatures(xt)
        data = self.zipBinaryTrainingParams(trainedParams[0],trainedParams[1],trainedParams[2])
        print('finished training')
        return Resource("Parameters", data, 'application/octet-stream')

    # filelist is a list of objects from feature extraction to index
    # if the index is greater then RAM(GB) create a new Index
    # Conforming to the RAM is optional for now
    # file the file based transactions are less than desriable, this is the most general implementation
    # try to use addToIndex and finalizeIndex rather then this function if possible

    def moveCPUtoGPU(self):
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = self.use_float16
        co.useFloat16CoarseQuantizer = False
        co.usePrecomputed = self.use_precomputed_tables
        co.indicesOptions = faiss.INDICES_CPU
        co.verbose = True
        co.reserveVecs = self.max_add
        co.shard = True
        vres, vdev = indexfunctions.make_vres_vdev( self.gpu_resources,ngpu=self.ngpu)
        self.gpu_index = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, self.index, co)

    def moveGPUtoCPU(self,maxSize):
        print("Aggregate indexes to CPU")
        t0 = time.time()
        if self.index is None:
            self.index = faiss.read_index(self.emptyIndexPath)
        for i in range(self.ngpu):
            if self.ngpu > 1:
                index_src = faiss.index_gpu_to_cpu(self.gpu_index.at(i))
            else:
                index_src = faiss.index_gpu_to_cpu(self.gpu_index)
            print("  index %d size %d" % (i, index_src.ntotal))
            index_src.copy_subset_to(self.index, 0, 0, maxSize)

        print("  done in %.3f s" % (time.time() - t0))
        print("Index size: ", self.index.ntotal)
    def flushGPUIndex(self):
        for i in range(self.ngpu):
            if self.ngpu > 1:
                index_src_gpu = faiss.downcast_index(self.gpu_index.at(i))
            else:
                index_src_gpu = faiss.downcast_index(self.gpu_index)
            index_src = faiss.index_gpu_to_cpu(index_src_gpu)
            # print("  index %d size %d" % (i, index_src.ntotal))
            if self.index is None:
                self.index = faiss.read_index(self.emptyIndexPath)
            index_src.copy_subset_to(self.index, 0, 0, self.totalImagesToIndex * self.feats_per_file)
            index_src_gpu.reset()
            index_src.reset()
            index_src_gpu.reserveMemory(self.max_add)
        if self.ngpu > 1:
            self.gpu_index.sync_with_shard_indexes()
    def flushCPUIndex(self,withGPUFlush = True):
        if withGPUFlush:
            self.flushGPUIndex()
        # print('index size: ', self.index.ntotal)
        savename = self.indexSaveNameForShard(self.cpuShardCount,machineNum=self.machine_num)
        # print(savename)

        #indexMerger.dumpIndex(self.index,self.index_cachefile,isFirst=self.cpuShardCount)
        faiss.write_index(self.index,savename)
        self.index.reset()
        del self.index
        gc.collect()
        #print('reading empyt from ',self.emptyIndexPath)
        #self.index = faiss.read_index(self.emptyIndexPath)
        #print('ntotal is ', self.index.ntotal)
        # print('index size: ', self.index.ntotal)
        self.cpuShardCount += 1
    def buildIndex(self, featurelistfile, RAM=16):
        # if not self.alreadyTrained:
        #     self.trainIndex(featurelistfile)
        gpuRamBytes = self.gpuRAM * 1024 * 1024 * 1024.0
        cpuRamBytes = RAM * 1024 * 1024 * 1024.0
        numsize = 32
        if self.use_float16: numsize = 16
        numFeaturesForGPURam = indexfunctions.featuresFitWithinRam(self.gpuRAM, self.preproc_dims,
                                                                   self.use_float16,0)
        numFeaturesForCPURam = indexfunctions.featuresFitWithinRam(RAM, self.featuredimensions, self.use_float16)
        print("Features per GPU to adhear to GPU ram constraint of ", self.gpuRAM, "GB: ", numFeaturesForGPURam)
        print("Features per index shard to adhear to CPU ram constraint of ", RAM, "GB: ", numFeaturesForCPURam)
        self.max_add = numFeaturesForGPURam
        if self.feats_per_file > 0:
            self.max_add = max(self.feats_per_file,numFeaturesForGPURam)
        print('max add: ', self.max_add)
        if self.ngpu > 0:
            self.moveCPUtoGPU()

        with open(featurelistfile) as fp:
            fileCount = 0
            featurelist = fp.readlines()
        #featurelist = featurelist[:int(len(featurelist)/20)]
        if self.rebuildShards:
            featurelist = []
            with open(self.IDMap_cachefile) as fp:
                self.IDtoNameMap = json.load(fp)
                print('loaded ', len(self.IDtoNameMap), 'image IDs')
        self.totalImagesToIndex = len(featurelist)
        # memorystats = psutil.virtual_memory().percent
        print('Total feature files to index:', self.totalImagesToIndex)
        # widgets = ['[', progressbar.Timer(),'] ', progressbar.Bar(),' (',progressbar.ETA(),') ',' [','Memory usage: '+str(memorystats)+'%','] ',]
        widgets = [progressbar.Timer(),progressbar.Percentage(),progressbar.Counter(),progressbar.Bar(),progressbar.ETA(),progressbar.DynamicMessage('mem')]
        bar = progressbar.ProgressBar(widgets=widgets,max_value=len(featurelist))
        counter = 0
        flushSize = 0
        for filepath in featurelist:
            gpuWasFlushed = False
            try:
                filepath = filepath.rstrip('\n')
                filename = os.path.basename(filepath)
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as featurefile:
                        if self.nonBinarizedFeatures:
                            features = Resource(filename, np.load(featurefile), 'application/octet-stream')
                        else:
                            features = Resource(filename, np.fromfile(featurefile,'float32'), 'application/octet-stream')
                    self.addToIndex(features)
                    # print('entries left before flush: ',self.max_add-(self.gpu_index.ntotal))
                    # print('GPU_Index size: ',self.gpu_index.ntotal)
                    # print('Main index size:', self.index.ntotal)
                    memorystats = psutil.virtual_memory().percent
                    freeMemPercent = 100-memorystats

                    if self.gpu_index is not None and self.max_add > 0 and self.gpu_index.ntotal > self.max_add and self.ngpu > 0:
                        # print("\nFlush GPU indexes to CPU")
                        flushSize += self.gpu_index.ntotal
                        print('\nflushing GPU to CPU')
                        self.flushGPUIndex()
                        gpuWasFlushed = True
                    #approxFileIndex = int((self.gpu_index.ntotal+self.index.ntotal)/self.kp_per_file)
                    if (freeMemPercent < 16 and counter%(len(featurelistfile)+1) == 0 and counter > 0) or False:
                        print('\nflushing CPU to disk because memory is ', psutil.virtual_memory().percent, ' full')
                        self.flushCPUIndex(withGPUFlush=(not gpuWasFlushed))
                    #elif self.gpu_index is not None and self.max_add > 0 and self.gpu_index.ntotal+self.index.ntotal > numFeaturesForCPURam and self.ngpu > 0:
                        # print("\nFlush CPU index to Disk")
                     #   self.flushCPUIndex()
                    counter += 1
                    bar.update(fileCount, mem=memorystats)
                else:
                    print('\nWarning: ', filepath, ' Does not exist')


                fileCount += 1
                if not fileCount % 100:
                    if not self.rebuildShards and len(self.IDtoNameMap) > 0 :
                        with open(self.IDMap_cachefile,'w') as fp:
                            json.dump(self.IDtoNameMap,fp)
            except Exception as e:
                logging.error(traceback.format_exc())

        with open(self.IDMap_cachefile,'w') as fp :
            if len(self.IDtoNameMap) > 0:
                json.dump(self.IDtoNameMap,fp)
            else:
                self.IDtoNameMap = json.load(fp)
        if self.ngpu > 0:
            if not self.onlyBuildIDMap:
                self.flushCPUIndex()
            #self.moveGPUtoCPU(len(featurelistfile) * self.feats_per_file)

        if self.max_add > 0:
            # it does not contain all the vectors
            gpu_index = None

        return self.finalizeIndex()
        # rather then providing the buildIndex function, we would ideally like to provide this function
        # if it meets everyone's needs
        # add a feature Resource object to the index

    def addToIndex(self, featureResource):

        features = self.deserializeFeatures(featureResource)
        # print(features.shape)
        ids = np.zeros((features.shape[0]), dtype='int64') + self.imCount
        self.IDtoNameMap[self.imCount] = featureResource.key
        if features is not None and len(features) > 0 and len(features.shape) > 1 and features.shape[1] == 64:
            if not self.onlyBuildIDMap:
                xs = self.preproc.apply_py(features)
                if self.ngpu > 0:
                    self.gpu_index.add_with_ids(xs, ids)
                else:
                    self.index.add_with_ids(xs,ids)
            self.imCount += 1
        else:
            print('Warning, bad features: ', features.shape)

    # added at the request of TA1, not currently called anywhere
    def mergeIndex(self, indexResourceToAdd):
        # do something with the index (example desrializing the resource comign in)
        indexToMerge = pickle.loads(indexResourceToAdd._data)

    def finalizeIndex(self):
        indexFiles = indexMerger.getListOfIndexes(self.cacheroot,self.machine_num)
        if self.index is not None:
            self.index.reset()
            del self.index
        print('merging all ', len(indexFiles),' flushed indexes...')
        print('saving finalized non binary index to ', os.path.abspath(self.index_cachefile))
        mmapFilePath = indexMerger.mergeIndexList(indexFiles,self.emptyIndexPath,self.index_cachefile)
        print('ivf data path:', mmapFilePath) 
        #self.index = faiss.read_index(self.index_cachefile,faiss.IO_FLAG_MMAP)
        featureResource = Resource('index', self.serializeIndex(indexFilePath=self.index_cachefile,mmapPath = mmapFilePath), 'application/octet-stream')
        return self.createOutput(featureResource)

    def createOutput(self, indexResource):
        return {'algorithm': self.createAlgOutput(), 'supplemental_information': self.createIndexOutput(indexResource)}

    def createAlgOutput(self, ):
        return {'name': self.algorithmName.replace(" ", ""), 'version': self.algorithmVersion.replace(" ", "")}

    def createIndexOutput(self, indexResource):
        return {'name': 'provenance_index', 'description': 'index for provenance', 'value': indexResource}


        # Serialize the index
        # useful libraries to serialize include pickle (bad for large strucutres), protobuf, and hdf5
        # this gives your code maximum flexability on how to reproesent data

    def zipBinaryTrainingParams(self,preproc,coarse_quantizer,codesIndex):
        faiss.write_index(codesIndex,'tmp1')
        # faiss.write_ProductQuantizer(coarse_quantizer,'tmp2')
        faiss.write_VectorTransform(preproc,'tmp3')
        with open('tmp1','r+b') as fp:
            bin_index = fp.read()
        # with open('tmp2','r+b') as fp:
        #     bin_coarsequantizer = fp.read()
        with open('tmp3','r+b') as fp:
            bin_preproc = fp.read()
        index_length   = ("%012d"%len(bin_index)).encode('ascii')
        # quantizer_length   = ("%012d"%len(bin_coarsequantizer)).encode('ascii')
        preproc_length = ("%012d" % len(bin_preproc)).encode('ascii')
        data = index_length+bin_index+preproc_length+bin_preproc
        return data

    def serializeIndex(self,indexFilePath=None,mmapPath=None):
        import mmap
        index_tmp_path = 'tmpIndex_%03d' % self.machine_num
        if indexFilePath is None:
            indexFilePath = index_tmp_path
            faiss.write_index(self.index, index_tmp_path)
        with open(indexFilePath, 'r+b') as fp:
            # bin_index = fp.read()
            bin_index_map = mmap.mmap(fp.fileno(), 0)
        with open(self.emptyIndexPath, 'r+b') as fp:
            # bin_codes = fp.read()
            bin_codes_map = mmap.mmap(fp.fileno(), 0)
        if not os.path.exists(self.preproc_cachefile):
            faiss.write_VectorTransform(self.preproc,self.preproc_cachefile)
        with open(self.preproc_cachefile, 'r+b') as fp:
            # bin_preproc = fp.read()
            bin_preproc_map = mmap.mmap(fp.fileno(), 0)
        if mmapPath is not None:
            print('saving ivf mmap data to binary...')
            with open(mmapPath, 'r+b') as fp:
                bin_ivf_mmap_map = mmap.mmap(fp.fileno(),0)
                ivf_length = ("%012d" % len(bin_ivf_mmap_map)).encode('ascii')
                ivf_mmap_path = mmapPath.encode('ascii')
                ivf_mmap_path_length = ("%012d" % len(ivf_mmap_path)).encode('ascii')

        bin_IDtoNameMap = json.dumps(self.IDtoNameMap).encode('ascii')
        index_length = ("%012d" % len(bin_index_map)).encode('ascii')
        codes_length = ("%012d" % len(bin_codes_map)).encode('ascii')
        preproc_length = ("%012d" % len(bin_preproc_map)).encode('ascii')
        map_length = ("%012d" % len(bin_IDtoNameMap)).encode('ascii')
        totalLength = len(index_length)+len(bin_index_map)+len(codes_length)+len(bin_codes_map) + len(preproc_length) + len(bin_preproc_map) + len(map_length) + len(bin_IDtoNameMap)
        if mmapPath is not None:
            totalLength += len(ivf_length)+len(bin_ivf_mmap_map)+len(ivf_mmap_path_length)+len(ivf_mmap_path)
        print('creating final binary file of size ', totalLength/1024/1024, ' MB')
        with open('tmp_binary_index.dat','wb') as fp:
            #final_index_bin_map = mmap.mmap(fp.fileno(),totalLength)
            #final_index_bin_map.seek(0)
            print('writing binary to mmaped file...')
            fp.write(index_length)
            fp.write(bin_index_map)
            fp.write(codes_length)
            fp.write(bin_codes_map)
            fp.write(preproc_length)
            fp.write(bin_preproc_map)
            fp.write(map_length)
            fp.write(bin_IDtoNameMap)
            if mmapPath is not None:
                print('writing ivf mmpa data')
                print('lenght: ',ivf_mmap_path_length)
                fp.write(ivf_mmap_path_length)
                print('path: ', ivf_mmap_path)
                fp.write(ivf_mmap_path)
                print('length: ', ivf_length)
                fp.write(ivf_length)
                print('map size: ', len(bin_ivf_mmap_map))
                fp.write(bin_ivf_mmap_map)

        print('Memory mapping final index file...')
        with open('tmp_binary_index.dat','r+b') as fp:
            final_index_bin_map = mmap.mmap(fp.fileno(),0)
        print('returning final binary')
        #final_index_bin = index_length + bin_index_map[:] + codes_length + bin_codes_map[:] + preproc_length + bin_preproc_map[:] + map_length + bin_IDtoNameMap
        return final_index_bin_map[:]

    # deserialize the features serialized by featureExtraction
    def deserializeFeatures(self, featureResource):
        data = featureResource._data
        if self.nonBinarizedFeatures:
            return data
        return np.reshape(data[:-2],(int(data[-2]),int(data[-1])))

    #only used if you algorithm trains teh indexing, i.e. implemnents (optional)
    #guarenteed to be called prior to running indexing.
    #If indexParametersResource is "None", then algorithm should load default pretrained parameters (not crash and burn)
    #Passed a single Resource object
    def loadIndexParameters(self, indexParametersResource=None):
        index,preproc,emptypath = loadIndexParameters(indexParametersResource)
        self.index = index
        #self.emptyIndex=faiss.clone_index(index)
        #self.invlists = faiss.OnDiskInvertedLists(self.index.nlist,self.index.code_size,'merged_index_'+'%03d' % self.machine_num + '.ivfdata')
        #self.codes = self.emptyIndex
        self.preproc = preproc
        self.emptyIndexPath = emptypath
        if index is None:
            indexParameterData = None
        else:
            self.index.reset()
            gc.collect()

def loadIndexParameters(indexParametersResource):
  if indexParametersResource is not None:
      indexParameterData = indexParametersResource._data #Takes 150 Mb of memory
      bytearrays = fileutil.splitMultiFileByteArray(indexParameterData, 12, 2) #Takes another 150 Mb of memory
      tmppath = 'tmp'
      mv = memoryview(indexParameterData)
      all_tmp_paths = []
      count = 0
      for bytearray in bytearrays:
          p = tmppath + str(count) + '.dat'
          with open(p, 'wb') as fp:
              fp.write(mv[bytearray[0]:bytearray[1]])
          count += 1
          all_tmp_paths.append(p)
      index = faiss.read_index(all_tmp_paths[0]) #WHY 12.5 GB?!!?!
      emptyIndexPath = all_tmp_paths[0]
      preproc = faiss.read_VectorTransform(all_tmp_paths[1])
      return(index,preproc,emptyIndexPath)
  else:
     indexParameterData=None
     return (None,None,None)


def deserializeIndex(self, indexFileResource, id=None):
    # the joined index file contains populated_index,empty_trained_index,preproc,IDMap, and (possible) merged IVF file in that order
    bytearrays = fileutil.splitMultiFileByteArray(indexFileResource._data, 12, 4)
    tmppath = 'tmp'
    mv = memoryview(indexFileResource._data)
    if id is not None:
        tmppath += '_' + str(id)
    all_tmp_paths = []
    count = 0
    for bytearray in bytearrays:
        p = tmppath + str(count) + '.dat'
        with open(p, 'wb') as fp:
            fp.write(mv[bytearray[0]:bytearray[1]])
        count += 1
        all_tmp_paths.append(p)
    index = faiss.read_index(all_tmp_paths[0], faiss.IO_FLAG_MMAP)
    emptyIndex = faiss.read_index(all_tmp_paths[1])
    preproc = faiss.read_VectorTransform(all_tmp_paths[2])
    IDToName = json.loads(bytes(mv[bytearrays[3][0]:bytearrays[3][1]]).decode('ascii'))
    print('index size: ', index.ntotal)
    print('map size:', len(IDToName))
    del bytearrays
    del indexFileResource
    print('initializing index...')
    print('index size: ', self.index.ntotal)
    self.isDeserialized = True
    return (index, emptyIndex,preproc, IDToName,all_tmp_paths)

def mergeIndexes(indexFolder,machineNumber,finalIndexFile):
    indexesToMerge = []
    indexFiles = os.listdir(os.path.dirname(indexFolder))
    for indexFile in indexFiles:
       parts = indexFile.split('_')
       if len(parts) > 2:
           mnum = int(parts[1][-3:])
           if mnum == machineNumber:
               fullFile = os.path.join(os.path.dirname(indexFolder),indexFile)
               print('mmaping ',fullFile)
               print('memory usage: ', psutil.virtual_memory().percent)
               index = faiss.read_index(fullFile,faiss.IO_FLAG_MMAP)
               indexesToMerge.append(index.invlists)
    print('adding final index')
    mainIndexFile = open(finalIndexFile, 'rb')
    mainIndexResource = Resource('indexparameters', mainIndexFile.read(), 'application/octet-stream')
    mainIndex,emptyIndex,preproc,map,all_tmp_paths = deserializeIndex(mainIndexResource)
    indexesToMerge.append(mainIndex.invlists)
    print("Merging " + str(len(indexesToMerge)) + "Index Shards for final index")
    finalIndex = emptyIndex
    invlists = faiss.OnDiskInvertedLists(finalIndex.nlist,index.code_size,os.path.join(os.path.dirname(self.index_cachefile),'merged_index.ivfdata'))
    ivf_vector = faiss.InvertedListsPtrVector()
    bar  = progressbar.ProgressBar()
    for ivf in bar(indexesToMerge):
       ivf_vector.push_back(ivf)
    print("merge %d inverted lists " %ivf_vector.size())
    ntotal = invlists.merge_from(ivf_vector.data(),ivf_vector.size())
    finalIndex.ntotal = ntotal
    finalIndex.replace_invlists(invlists)
    print('ntotal: ', finalIndex.ntotal)
    outName = "finalIndex_%03d" % machineNumber
    outPath = os.path.join(os.path.dirname(finalIndexFile),outName)
    binaryIndex = serializeIndex(finalIndexFile,map,machineNumber,all_tmp_paths)
    with open(outPath, 'wb') as of:
        of.write(binaryIndex)
