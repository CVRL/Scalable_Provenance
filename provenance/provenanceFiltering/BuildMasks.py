import sys
import os
import numpy as np
import cv2 as cv
import math
import imcompare
import time
from skimage import data
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pickle
import json
import rawpy
from scipy import misc

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from skimage import exposure
maxImagePixels = 1920*1080 #We will resize all images to be less than 1080p resolution (doesn't matter dimensions)
dumpDir = "."

def loadImage(imgpath):
    try:
        if imgpath.endswith('.gif'):
            img = misc.imread(imgpath)
        else:
            img = cv.imread(imgpath)
        if img is None or img == []:
            print("Could not open with OpenCV, trying raw codecs")
            img = rawpy.imread(imgpath).postprocess()
        return img
    except:
        print('Could Not load ', imgpath)
        return None

def showImage(im):
    im2 = im
    try:
        if len(im.shape) > 2 and im.shape[2] is not None and im.shape[2] >2 and map.dtype == np.dtype('uint8'):
            im2 = cv.cvtColor(im,cv.COLOR_BGR2RGB)
    except:
        print("")
    plt.imshow(im2);
    plt.show();

def getSURFDist(f):
    return f.distance

def getSURFScore(f1,f2):
    bf = cv.BFMatcher(crossCheck=True)
    try:
        matches = bf.match(f1[1],f2[1])
        matches = sorted(matches,key = lambda x:x.distance)
        M = []
        # matches = matches[:int(round(len(matches)/2))]
        dists = map(getSURFDist,matches)
        distcsum = np.cumsum(dists,dtype=float)/np.sum(dists)
        idx = np.searchSorted(distcsum,.65)
        return np.mean(dists[:idx])
        # avg = float(np.sum(dists))/float(len(dists))
    except:
        return 100000000000

def getSURFFeaturesForFile(imgPath):
    img = loadImage(imgPath)
    img = getScaledImage(img)
    surf = cv.xfeatures2d.SURF_create(400)
    (kps,descs) = surf.detectAndCompute(img,None)
    return (kps,descs,img)

def getSURFFeaturesForFileName(img):
    surf = cv.xfeatures2d.SURF_create(400)
    (kps,descs) = surf.detectAndCompute(img,None)
    return (kps,descs,img)
    # for r in results:
    #     resultFile = os.path.join(baseDir,r)
    #     resultImg = cv.imread(probeFile)
def getScaledImage(img,maxWidth = 512.0):
    w = img.shape[1]
    scale = maxWidth/w
    if scale < 1:
        newWidth = int(w*scale)
        newHeight = int(img.shape[0]*scale)
        return cv.resize(img,(newWidth,newHeight))
    else:
        return img
def generateTransformForFeatures(f1,f2):
    bf = cv.BFMatcher(crossCheck=True)
    if f1[1] is not None and f2[1] is not None and len(f1[1]) > 0 and len(f2[1]) > 0:
        try:
            matches = bf.match(f1[1],f2[1])
            matches = sorted(matches,key = lambda x:x.distance)
            # good = []
            # for m,n in matches:
            #     if m.distance < .75*n.distance:
            #         good.append([m])
            M = []
            matches = matches[:int(round(len(matches)/2))]
            if len(matches) > 4:
                src_pts = np.asarray(np.array([f1[0][m.queryIdx].pt for m in matches],np.float32),np.float32)
                dst_pts = np.asarray(np.array([f2[0][m.trainIdx].pt for m in matches],np.float32),np.float32)
                # M,mask = cv.findHomography(src_pts,dst_pts)
                M = cv.estimateRigidTransform(src_pts,dst_pts,fullAffine=True)
                # M = cv.getAffineTransform(src_pts[:3],dst_pts[:3])
            # img3 = cv.drawMatches(f1[2],f1[0],f2[2],f2[0],matches,None,flags=2)
            # plt.imshow(img3),plt.show()
            # cv.waitKey(5)
        except:
            M = None
    else:
        M=None
    if M is None or M == []:
        M = np.array([[0,pow(10,10),0],[pow(10,10),0,0]])
    return M
    # img3 = []
    # img3 = cv.drawMatches(f1[2],f1[0],f2[2],f2[0],matches[:10],flags=2,outImg=None)
    #
    # plt.imshow(img3),plt.show()
    # cv.waitKey()
def registerImage(img1,img2,M):
    img2_warped = cv.warpAffine(img2,M,dsize=(img1.shape[1],img1.shape[0]))
    return img2_warped

def determineIfRotated90(probeImg,resultImg):
    probeRatio = float(probeImg.shape[0])/float(probeImg.shape[1])
    resultRatio = float(resultImg.shape[0])/float(resultImg.shape[1])
    resultSizeVect = resultImg.shape[:2]
    invResultRatio = 1/resultRatio
    bestRatioDif = abs(probeRatio-resultRatio)
    invertedRatioDif = abs(probeRatio-invResultRatio)
    isRotated = 0
    if bestRatioDif > invertedRatioDif:
        # The result image is most likely flipped on its side (which side is unknown, but most likely not upside down)
        isRotated = 1
        bestRatio = invertedRatioDif
        resultSizeVect = [resultImg.shape[1],resultImg.shape[0]]
    try:
        denom = (np.linalg.norm(probeImg.shape[:2])*np.linalg.norm(resultSizeVect))
        if denom == 0: denom = 1
        inner = abs(np.dot(probeImg.shape[:2],resultSizeVect)/denom)
        angleDif = math.acos(inner)*180/math.pi
        if abs(angleDif) > 25:
            # The best ratio correspondence is still too far off from one another, we can't assue the image is rotated (probably different images)
            isRotated = 0
    except:
        isRotated = 0
    return isRotated

def rotateImage(img,flag):
    if flag == 90: return np.flipud(img.T)
    elif flag == -90: return np.fliplr(img.T,)
    else: return img

def genMasksForProbe(probePath,resultPaths,numberOfRanks,relPath="",rcondsOnly=False):
    p = os.path.basename(probePath)
    print("Probe: " + probePath)
    baseDir = os.path.dirname(probePath)
    results = resultPaths
    tform_Mats = []
    rconds = []
    rotations = []
    probeFile = probePath
    probeImg = loadImage(os.path.join(relPath,probeFile))
    if probeImg is not None and len(probeImg) > 0:
        probeImg_small = probeImg
        resizeFactor = 1
        if probeImg.shape[0]*probeImg.shape[1] > maxImagePixels:
            probeImg_small,resizeFactor = resizeImageToLargeImage(probeImg,maxImagePixels)
        features_p = loadFeatures(p)
        if features_p is None:
            try:
                features_p = rescaleKeypoints(getSURFFeaturesForFileName(probeImg_small),1/resizeFactor)
            except:
                print("ERROR: Could not compute features of a probe image")
                return (np.zeros_like(probeImg),np.zeros_like(probeImg))
            saveFeatures(features_p,p)
        imageData = "ID,rcond,Rotation,isModified,isGobalModification,globalModVal"
        imageDataDict = {}
        # First pass to determine rconds of all results
        if probeFile in results:
            results.remove(probeFile)
        rcount = 0
        for r in results[:min(len(results)-1,numberOfRanks)]:
            if r == probeFile:
                continue
            # print("Result:" + r)
            resultFile = r
            resultImg = loadImage(os.path.join(relPath,resultFile))
            if resultImg is None or resultImg.shape[0] == 0 or resultImg.shape[1] == 0:
                print("Couldn't load image, moving on")
                continue
            resultImg_small = resultImg

            if probeImg.shape[0] * probeImg.shape[1] > maxImagePixels:
                resultImg_small = cv.resize(resultImg,(int(round(resizeFactor*resultImg.shape[1])),int(round(resizeFactor*resultImg.shape[0]))))
            isRotated = determineIfRotated90(probeImg_small, resultImg_small)
            imgRotSet = []
            if r == "/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/world/904a38e52a6ea708d59613c8abd3fcf1.jpg":
                print("test")
            if isRotated:
                # Rotate image CC
                imgRotSet.append(np.flipud(resultImg_small.T))
                # Rotate image clockwise
                imgRotSet.append(np.fliplr(resultImg_small.T))
            else:
                imgRotSet.append(resultImg_small)
            bestImgStruct = (-1, pow(10,-10), 'none', [])
            for i in range(0, len(imgRotSet)):
                rImg = imgRotSet[i]
                features_r = loadFeatures(os.path.basename(r)+"_"+str(i))
                if features_r is None:
                    try:
                        features_r = rescaleKeypoints(getSURFFeaturesForFileName(rImg),1/resizeFactor)
                    except:
                        print("ERROR: Could not compute features of a result image")
                        continue
                    saveFeatures(features_r,os.path.basename(r+"_"+str(i)))
                if len(features_p) > 0 and len(features_r) > 0:
                    M = generateTransformForFeatures(features_p, features_r)
                    tform = np.vstack((M, np.array([0, 0, 1])))
                    rcond = 1 / np.linalg.cond(tform)
                    if rcond > bestImgStruct[1]:
                        brot = 0
                        if i == 0:
                            brot = 90
                        elif i == 0:
                            brot = -90
                        if ~isRotated: brot = 0
                        brcond = rcond
                        bind = i
                        bM = M
                        bestImgStruct = (bind, brcond, brot, bM)

            tform_Mats.append(bestImgStruct[3])
            rconds.append(bestImgStruct[1])
            if bestImgStruct[1] >= .25 and bestImgStruct[1] < .8:
                print("strange")
            # print(str(rcount)+"/"+str(len(results[:min(len(results)-1,numberOfRanks)])) +" rcond: " + str(bestImgStruct[1]))
            rcount+=1
            rotations.append(bestImgStruct[2])
            imageDataDict[r] = {"rcond":bestImgStruct[1],"rotation":bestImgStruct[2], "isModified":False, "isGlobalModification":False,"globalModVal":-1}
        # Sort result images by Rcondition fitness
        newResultOrder = np.argsort(rconds)
        imageDataDict2 = {}
        imageDataDict2['imageData'] = imageDataDict
        imageDataDict2['reRankedOrder'] = newResultOrder
        imageDataDict2['rconds'] = rconds
        imageDataDict = imageDataDict2
        supressedRconds = [i for i in rconds if i > .05]
        # rcond_thresh = filters.threshold_otsu(np.asarray(supressedRconds))*2
        rcond_thresh = .3
        # ret,rcond_thresh = cv.threshold(np.asarray(rconds),0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        # print("Transform condition thresh: " + str(rcond_thresh))
        # Third pass on ordered images to determine the correct masks
        masks = []
        maps = []
        probechans = 1
        finalMapMask = None
        finalMask = None
        if not rcondsOnly:
            if len(probeImg.shape) > 2 and probeImg.shape[2] is not None and probeImg.shape[2] > 1:
                probechans = probeImg.shape[2]
            for sortedIndex in newResultOrder:
                if rconds[sortedIndex] >= rcond_thresh:
                    print("Warping image " + str(sortedIndex))
                    resultFile =  results[sortedIndex]
                    resultImg = loadImage(resultFile)
                    # resultImg = cv.resize(resultImg,(round(resizeFactor * resultImg.shape[1]), round(resizeFactor * resultImg.shape[0])))
                    rotateTo = rotations[sortedIndex]
                    M = tform_Mats[sortedIndex]
                    # Rotate image in 90 degree increments if needed
                    result_warped = rotateImage(resultImg, rotateTo)
                    # Register image accurately to probe
                    result_warped = registerImage(probeImg, result_warped, M)
                    resultchans = 1
                    if len(result_warped.shape) > 2 and result_warped.shape[2] is not None and result_warped.shape[2] > 1:
                        resultchans = result_warped.shape[2]
                    # Compare images now that they are registered
                    if sortedIndex == 1:
                        print("")
                    maxchans = max(probechans,resultchans)
                    # grayscale map
                    map, hasBeenModified, globalMod,globalModVal = imcompare.resPSNR(probeImg, result_warped)
                    # Channel-wise map
                    for i in range(0,maxchans):
                        map_t, hasBeenModified_t,globalMod_t,globalModVal_t = imcompare.resPSNR(probeImg[:,:,min(i,probechans)], result_warped[:,:,min(i,probechans)])
                        map = np.minimum(map,map_t)
                        if hasBeenModified_t:
                            hasBeenModified = True
                        if not globalMod_t:
                            globalMod = False
                        if globalModVal_t > globalModVal:
                            globalModVal = globalModVal_t
                    if hasBeenModified and not globalMod:
                        mask = imcompare.genMask(((map) * 255).astype(np.uint8))
                        masks.append(mask)
                        maps.append(map)

                    imageDataDict[resultFile]["isModified"] = hasBeenModified
                    imageDataDict[resultFile]["isGlobalModification"] = globalMod
                    imageDataDict[resultFile]["globalModVal"] = globalModVal
                    #     Generate average masks and maps
            avgMask = np.zeros((probeImg.shape[0], probeImg.shape[1]), dtype=float)
            avgMap = np.zeros((probeImg.shape[0], probeImg.shape[1]), dtype=float)


            if len(masks) > 0:
                for mask in masks:
                    avgMask += mask
                for map in maps:
                    avgMap += map
                avgMask /= len(masks)
                avgMap /= len(maps)
                maskVals = np.unique(avgMask)
                if len(maskVals) >= 2:
                    try:
                        avgMaskVal = filters.threshold_otsu(avgMask)
                    except:
                        avgMask = 128
                else:
                    avgMaskVal = 128
                avgMask = avgMask < avgMaskVal
                avgMap = (avgMap - np.min(avgMap)) / (np.max(avgMap) - np.min(avgMap))
                avgMask = (avgMask - np.min(avgMask)) / (np.max(avgMask) - np.min(avgMask))
                try:
                    finalMask = imcompare.genMask(((avgMask) * 255).astype(np.uint8))
                except:
                    print("Couldn't generate average mask")
                    finalMask = avgMask
                try:
                    finalMapMask = imcompare.genMask(((avgMap) * 255).astype(np.uint8))
                except:
                    print("Couldn't generate average map mask")
                    finalMapMask = avgMask

            else:
                finalMapMask = avgMask
                finalMask = avgMask
    else:
        print("Error, could not open probe file, " ,os.path.join(relPath,probeFile))
        finalMapMask = []
        finalMask = []
        imagedataDict = []
    return (finalMapMask,finalMask,imageDataDict)

def resizeImageToLargeImage(img,maxPixels):
    oldw = img.shape[1]
    oldh = img.shape[0]
    newh = math.sqrt((oldh*maxPixels)/oldw)
    neww = newh*oldw/oldh
    resizeAmount = neww/oldw
    newImg = cv.resize(img,(int(round(neww)),int(round(newh))))
    return newImg,resizeAmount
def rescaleKeypoints(kps, scale):
    for kp in kps[0]:
        kp.pt = (kp.pt[0]*scale,kp.pt[1]*scale)
    return kps

def pickle_keypoints(keypoints,descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])
        i+=1
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

def saveFeatures(features,key,dir="./featureDump"):
    temp = pickle_keypoints(features[0],features[1])
    temp_array = []
    temp_array.append(temp)
    if not os.path.exists(dir):
        os.makedirs(dir)
    try:
        pickle.dump(temp_array,open(os.path.join(dir,key+".pickle"),"wb"))
    except:
        print("could not dump features to " + os.path.join(dir,key))
def loadFeatures(key,dir=dumpDir):
    path = os.path.join(dir,key+".pickle")
    if os.path.exists(path):
        try:
            features_array = pickle.load(open(path,"rb"))
            kp1,desc1 = unpickle_keypoints(features_array[0])
            return (kp1,desc1,None)
        except:
            return None
    else:
        return None


def main():
    if len(sys.argv) >= 3:
        # read in the top 100 ranks from txt file
        rank100file = sys.argv[1]
        saveDir = sys.argv[2]
        # baseDir = sys.argv[5]
        # if len(sys.argv) < 6:
        #     baseDir = None
        if len(sys.argv) < 5:
            jobName = ""
        else:
            jobName = sys.argv[4]
        if len(sys.argv) < 4:
            numRanks = 100
        else:
            numRanks = int(sys.argv[3])
        if jobName == "" or jobName == " ":
            jobName = "default"
        #     Get the arguments that tell me what job number I am, and how many total jobs there are
        totalTasks = 1
        if len(sys.argv) > 5:
            totalTasks = int(sys.argv[5])
        taskNumber = 0
        if len(sys.argv) > 6:
            taskNumber = int(sys.argv[6])
            if taskNumber >= totalTasks:
                taskNumber = totalTasks - 1
        numCores = 1
        if len(sys.argv) > 7:
            numCores = int(sys.argv[7])
        dumpDir = "./featureDump"
        if len(sys.argv) > 8:
            dumpDir = sys.argv[8]
        SingleFileName = None
        if len(sys.argv) > 9:
            SingleFileName = sys.argv[9]
        saveDir_mapMask = os.path.join(saveDir, jobName+"_MapMasks")
        saveDir_mask = os.path.join(saveDir, jobName+"_Masks")
        saveDir_other = os.path.join(saveDir, jobName+"_otherData")
        try:
            os.makedirs(saveDir_mapMask)
        except:
            None
        try:
            os.makedirs(saveDir_mask)
        except:
            None
        try:
            os.makedirs(saveDir_other)
        except:
            None
        resultsDictionary = {}
        # Load all rank files (each row contains a probe and its top-N rank results)
        with open(rank100file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        # determine the row indexes of which probe/top N result pairs I want to calculate
        myPartitionSize = int(len(content) / totalTasks)
        myPartitionRangeStart = taskNumber * myPartitionSize
        myPartitionRangeEnd = myPartitionRangeStart + myPartitionSize

        for c in content:
            carr = c.split(",")
            resultsDictionary[carr[0]] = carr[1:]
        sortedKeys = resultsDictionary.keys()
        # Sort keys so we know each partition is always working with the same ordered set to partition from
        sortedKeys = sorted(sortedKeys)
        # sortedKeys = ["/afs/crc.nd.edu/group/cvrl/scratch_18/medifor/evaluation/NC2017_evaluation_all/world200_9/f7894ef3d96c0767ba23783d66b1e298.jpg"]
        if numCores > 1:
            Parallel(n_jobs=4)(delayed(outer_generateMaskForProbe)(p, resultsDictionary, numRanks, saveDir_mask, saveDir_mapMask,saveDir_other) for p in sortedKeys[myPartitionRangeStart:myPartitionRangeEnd])
        else:
            for p in sortedKeys[myPartitionRangeStart:myPartitionRangeEnd]:
                # if os.path.basename(p) == "bbfb07e272b66a6be65ca87e20908e53.jpg":
                if os.path.basename(p) == "170303979309eebf5a92c492a84997f6.jpg":
                    outer_generateMaskForProbe(p,resultsDictionary,numRanks,saveDir_mask,saveDir_mapMask,saveDir_other)
        # for p in sortedKeys:
        #     # if os.path.basename(p) == "8b3c9021c7e6dda308cfe7c594dc79e4.jpg":#"c59a64fb6a8f26cdbc15f3408c43ed26.jpg" or True:#"173e754519ea142944dab8c686efa7b3.jpg":
        #     results = resultsDictionary[p]
        #     finalMapMask, finalMask = genMasksForProbe(p, results,numRanks)
        #     savePath_mask = os.path.join(saveDir_mask, os.path.basename(p))
        #     savePath_mapmask = os.path.join(saveDir_mapMask, os.path.basename(p))
        #     cv.imwrite(savePath_mapmask, finalMapMask)
        #     cv.imwrite(savePath_mask, finalMask)

    else:
        print("usage: BuildMasks.py <rankFile> <save Dir> <Number of Ranks=100> <jobname=default> <Total Number of Jobs1=> <Current Job Number=0> <number of cores = 1> <dataDump Directory= ./datadump> ")
def outer_generateMaskForProbe(p,resultsDictionary,numRanks,saveDir_mask,saveDir_mapMask,saveDir_other):
    p_name = os.path.basename(p)
    if not os.path.isfile(os.path.join(saveDir_mask,p_name)):
        results = resultsDictionary[p]
        finalMapMask, finalMask, matchData = genMasksForProbe(p, results, numRanks)
        basename = os.path.basename(p)
        basename = basename.split(".")
        basename = basename[0]
        savePath_mask = os.path.join(saveDir_mask, basename+".png")
        savePath_mapmask = os.path.join(saveDir_mapMask, basename+".png")
        savePath_Other = os.path.join(saveDir_other,basename+".json")
        cv.imwrite(savePath_mapmask, finalMapMask)
        cv.imwrite(savePath_mask, finalMask)
        with open(savePath_Other,'w') as fp:
            json.dump(matchData,fp)
    else:
        print("skipping file...")

if __name__ == "__main__":
    main()





