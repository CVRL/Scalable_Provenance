import os
import sys
import numpy
import cv2
import BuildMasks
import imcompare
from queryIndex import filteringResults
from resources import Resource
from featureExtraction import featureExtraction
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
import rawpy
import progressbar
import indexfunctions
from collections import OrderedDict
import progressbar
from scipy import misc
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
maxImagePixels = 960*540 #We will resize all images to be less than 1080p resolution (doesn't matter dimensions)
dumpDir = "./featureDump"
useResultFeatures = False
# def generateTier2Features(resultScores,currentQuery):
#
#     finalMapMask, finalMask, matchData = genMasksForProbe(p, results, numRanks)

def getTier2Features(queryImageResource,resultResources,numberOfRanks,rcondsOnly=False,rcond_thresh = .1):
    result_manip_data = genMasksForProbe(queryImageResource,resultResources,numberOfRanks,rcondsOnly=rcondsOnly,rcond_thresh = rcond_thresh)

    featureDictionary = extractFeaturesFromMasks(queryImageResource, result_manip_data)
    return featureDictionary


def getRcondOfResult(r, probeImg, probeImg_small, features_p, p, resizeFactor, featureextractor):
    if os.path.basename(r.key) == p:
        return
    resultFile = r.key
    resultImg = featureextractor.deserialize_image(r._data)
    if resultImg is None or resultImg.shape[0] == 0 or resultImg.shape[1] == 0:
        print("Couldn't load image, moving on")
        return
    resultImg_small = resultImg

    if resultImg.shape[0] * resultImg.shape[1] > maxImagePixels:
        resultImg_small, resizeFactor_r = resizeImageToLargeImage(resultImg, maxImagePixels)

        # int(round(resizeFactor * resultImg.shape[1])), int(round(resizeFactor * resultImg.shape[0]))))
    isRotated = determineIfRotated90(probeImg_small, resultImg_small)
    imgRotSet = []
    if isRotated:
        # Rotate image CC
        imgRotSet.append(np.rot90(resultImg_small))
        # Rotate image clockwise
        imgRotSet.append(np.rot90(resultImg_small, k=3))
    else:
        imgRotSet.append(resultImg_small)
    bestImgStruct = (-1, pow(10, -10), 'none', [])
    for i in range(0, len(imgRotSet)):
        rImg = imgRotSet[i]
        features_r = loadFeatures(os.path.basename(r.key) + "_" + str(i))
        if features_r is None:
            try:
                features_r = rescaleKeypoints(getSURFFeaturesForImage(rImg), 1 / resizeFactor)
            except:
                print("ERROR: Could not compute features of a result image")
                continue
            saveFeatures(features_r, os.path.basename(r.key + "_" + str(i)))
        if len(features_p) > 0 and len(features_r) > 0:
            M = generateTransformForFeatures(features_p, features_r)
            tform = np.vstack((M, np.array([0, 0, 1])))
            rcond = 1 / np.linalg.cond(tform)
            if rcond > bestImgStruct[1]:
                brot = 0
                if i == 1:
                    brot = 90
                elif i == 2:
                    brot = -90
                if ~isRotated: brot = 0
                brcond = rcond
                bind = i
                bM = M
                bestImgStruct = (bind, brcond, brot, bM)

    tform = bestImgStruct[3]
    rcond = bestImgStruct[1]
    # if bestImgStruct[1] >= .25 and bestImgStruct[1] < .8:
    #     print("strange")
    # print(str(rcount)+"/"+str(len(results[:min(len(results)-1,numberOfRanks)])) +" rcond: " + str(bestImgStruct[1]))
    rotation = bestImgStruct[2]
    dict = {"resource": r, "rcond": bestImgStruct[1], "rotation": bestImgStruct[2],
            "isModified": False, "isGlobalModification": False, "globalModVal": -1}
    res = [r,bestImgStruct[1],bestImgStruct[2],False,False,-1]
    return res

def genSingleMask(probeImg,result,M,rotateTo):
    # print("Warping image " + str(sortedIndex))
    resultFile = result.key
    featureextractor = featureExtraction()
    resultImg = featureextractor.deserialize_image(result._data)
    # resultImg = cv.resize(resultImg,(round(resizeFactor * resultImg.shape[1]), round(resizeFactor * resultImg.shape[0])))

    # Rotate image in 90 degree increments if needed
    result_warped = rotateImage(resultImg, rotateTo)
    # Register image accurately to probe
    result_warped = registerImage(probeImg, result_warped, M)
    resultchans = 1
    # TODO Resize image
    if len(result_warped.shape) > 2 and result_warped.shape[2] is not None and result_warped.shape[2] > 1:
        resultchans = result_warped.shape[2]
    # Compare images now that they are registered

    # grayscale map
    print('run psnr...')
    map, mask, hasBeenModified, globalMod, globalModVal = imcompare.resPSNR_Norm(probeImg, result_warped, withMask=True,
                                                                                 withMetaData=True)
    rmask = None
    rmap = None
    if hasBeenModified and not globalMod:
        rmask = mask
        rmap = map
    dict = {}
    dict["isModified"] = hasBeenModified
    dict["isGlobalModification"] = globalMod
    dict["globalModVal"] = globalModVal
    dict["map"] = rmap
    dict['mask'] = rmask
    dict['image'] = resultImg
    print('done')
    return(resultFile,mask,map,dict)
    # final_imageDataDict[resultFile] = imageDataDict[resultFile]

def genMasksForProbe(queryImageResource,resultResources,numberOfRanks,rcondsOnly=False,rcond_thresh = .1,numcores=10):
    featureextractor = featureExtraction()
    # resultsKeys = []
    # for r in resultResources:
    #     resultsKeys.append(os.path.basename(r.key))
    tform_Mats = []
    rconds = []
    rotations = []
    probeFile = queryImageResource.key
    p = os.path.basename(probeFile)
    imagedata = queryImageResource._data
    probeImg = featureextractor.deserialize_image(imagedata)
    final_imageDataDict = OrderedDict()
    if probeImg is not None and len(probeImg) > 0:
        probeImg_small = probeImg
        resizeFactor = 1
        if probeImg.shape[0]*probeImg.shape[1] > maxImagePixels:
            probeImg_small,resizeFactor = resizeImageToLargeImage(probeImg,maxImagePixels)
        features_p = loadFeatures(p)
        if features_p is None:
            try:
                features_p = rescaleKeypoints(getSURFFeaturesForImage(probeImg_small),1/resizeFactor)
            except:
                print("ERROR: Could not compute features of a probe image")
                return (np.zeros_like(probeImg),np.zeros_like(probeImg))
            saveFeatures(features_p,p)
        imageDataDict = {}
        # First pass to determine rconds of all results
        # if probeFile in resultsKeys:
        #     resultsKeys.remove(probeFile)
        rcount = 0
        bar = progressbar.ProgressBar()
        print('find rconds')
        for r in bar(resultResources[:min(len(resultResources)-1,numberOfRanks)]):
            if os.path.basename(r.key) == p:
                continue
            # print("Result:" + r)
            resultFile = r.key
            resultImg = featureextractor.deserialize_image(r._data)
            if resultImg is None or resultImg.shape[0] == 0 or resultImg.shape[1] == 0:
                print("Couldn't load image, moving on")
                continue
            resultImg_small = resultImg

            if resultImg.shape[0] * resultImg.shape[1] > maxImagePixels:
                resultImg_small, resizeFactor_r = resizeImageToLargeImage(resultImg, maxImagePixels)
                # resultImg_small = cv.resize(resultImg,(int(round(resizeFactor*resultImg.shape[1])),int(round(resizeFactor*resultImg.shape[0]))))
            isRotated = determineIfRotated90(probeImg_small, resultImg_small)
            imgRotSet = []
            if isRotated:
                # Rotate image CC
                imgRotSet.append(np.rot90(resultImg_small))
                # Rotate image clockwise
                imgRotSet.append(np.rot90(resultImg_small,k=3))
            else:
                imgRotSet.append(resultImg_small)
            bestImgStruct = (-1, pow(10,-10), 'none', [])
            for i in range(0, len(imgRotSet)):
                rImg = imgRotSet[i]
                features_r = loadFeatures(os.path.basename(r.key)+"_"+str(i))
                if features_r is None:
                    try:
                        features_r = rescaleKeypoints(getSURFFeaturesForImage(rImg),1/resizeFactor_r)
                    except:
                        print("ERROR: Could not compute features of a result image")
                        continue
                    saveFeatures(features_r,os.path.basename(r.key+"_"+str(i)))
                if len(features_p) > 0 and len(features_r) > 0:
                    M = generateTransformForFeatures(features_p, features_r)
                    tform = np.vstack((M, np.array([0, 0, 1])))
                    rcond = 1 / np.linalg.cond(tform)
                    if rcond > bestImgStruct[1]:
                        brot = 0
                        if i == 1:
                            brot = 90
                        elif i == 2:
                            brot = -90
                        if ~isRotated: brot = 0
                        brcond = rcond
                        bind = i
                        bM = M
                        bestImgStruct = (bind, brcond, brot, bM)

            tform_Mats.append(bestImgStruct[3])
            rconds.append(bestImgStruct[1])

            # print(str(rcount)+"/"+str(len(results[:min(len(results)-1,numberOfRanks)])) +" rcond: " + str(bestImgStruct[1]))
            rcount+=1
            rotations.append(bestImgStruct[2])
            imageDataDict[resultFile] = {"resource":r,"rcond":bestImgStruct[1],"rotation":bestImgStruct[2], "isModified":False, "isGlobalModification":False,"globalModVal":-1}
        # Sort result images by Rcondition fitness
        newResultOrder = np.argsort(-1*np.asarray(rconds))

        final_imageDataDict = OrderedDict()
        supressedRconds = [i for i in rconds if i > .05]
        # rcond_thresh = filters.threshold_otsu(np.asarray(supressedRconds))*2
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
            bar = progressbar.ProgressBar()
            newResultOrder = np.asarray(newResultOrder)[np.where(newResultOrder >= rcond_thresh)]
            print('Find masks')
            # parallelResources = []
            # for sortedIndex in newResultOrder:
            #     result = resultResources[sortedIndex]
            #     if rconds[sortedIndex] >= rcond_thresh and result.key is not probeFile:
            #         resultFile = result.key
            #         if not resultFile == probeFile:
            #             parallelResources.append((result,rotations[sortedIndex],tform_Mats[sortedIndex]))
            # if len(parallelResources) > 0:
            #     # print('running parallel')
            #     allDictresults = []
            #     allDictresults = Parallel(n_jobs=numcores)(delayed(genSingleMask)(probeImg, result,M, rotateTo) for result,M,rotateTo in parallelResources)
            #     for p in parallelResources:
            #         allDictresults.append(genSingleMask(p[0],p[1],p[2]))
            #     for d in allDictresults:
            #         resultFile = d[0]
            #         resultDict = d[3]
            #         masks.append(d[1])
            #         maps.append(d[2])
            #         imageDataDict[resultFile].update(resultDict)
            #         final_imageDataDict[resultFile] = imageDataDict[resultFile]
            for sortedIndex in bar(newResultOrder):
                result = resultResources[sortedIndex]
                if rconds[sortedIndex] >= rcond_thresh and result.key is not probeFile:
                    #print("Warping image " + str(sortedIndex))
                    resultFile =  result.key
                    if resultFile == probeFile:
                        continue
                    resultImg = featureextractor.deserialize_image(result._data)
                    # resultImg = cv.resize(resultImg,(round(resizeFactor * resultImg.shape[1]), round(resizeFactor * resultImg.shape[0])))
                    rotateTo = rotations[sortedIndex]
                    M = tform_Mats[sortedIndex]
                    # Rotate image in 90 degree increments if needed
                    result_warped = rotateImage(resultImg, rotateTo)
                    # Register image accurately to probe
                    result_warped = registerImage(probeImg, result_warped, M)
                    resultchans = 1
                    #TODO Resize image
                    if len(result_warped.shape) > 2 and result_warped.shape[2] is not None and result_warped.shape[2] > 1:
                        resultchans = result_warped.shape[2]
                    # Compare images now that they are registered

                    maxchans = max(probechans,resultchans)
                    # grayscale map
                    map,mask, hasBeenModified,globalMod,globalModVal = imcompare.resPSNR_Norm(probeImg, result_warped,withMask=True,withMetaData=True)
                    # hasBeenModified = ImgMetadata[0], globalMod = ImgMetadata[1], globalModVal = ImgMetadata[2]
                    # Channel-wise map
                    # for i in range(0,maxchans):
                    #     map_t,maskt_t, hasBeenModified_t,globalMod_t,globalModVal_t = imcompare.resPSNR_Norm(probeImg[:,:,min(i,probechans)], result_warped[:,:,min(i,probechans)],withMask=True,withMetaData=True)
                    #     # hasBeenModified_t = ImgMetadata_t[0], globalMod_t = ImgMetadata_t[1], globalModVal_t = ImgMetadata_t[2]
                    #     map = np.minimum(map,map_t)
                    #     if hasBeenModified_t:
                    #         hasBeenModified = True
                    #     if not globalMod_t:
                    #         globalMod = False
                    #     if globalModVal_t > globalModVal:
                    #         globalModVal = globalModVal_t
                    rmask = None
                    rmap = None
                    if hasBeenModified and not globalMod:
                        rmask = mask
                        rmap = map
                        masks.append(mask)
                        maps.append(map)

                    imageDataDict[resultFile]["isModified"] = hasBeenModified
                    imageDataDict[resultFile]["isGlobalModification"] = globalMod
                    imageDataDict[resultFile]["globalModVal"] = globalModVal
                    imageDataDict[resultFile]["map"] = rmap
                    imageDataDict[resultFile]['mask'] = rmask
                    imageDataDict[resultFile]['image'] = resultImg
                    final_imageDataDict[resultFile] = imageDataDict[resultFile]
            #     Generate average masks and maps

            return final_imageDataDict
    else:
        print("Error, could not open probe file, " ,probeFile)
        finalMapMask = []
        finalMask = []
        imagedataDict = []

    return final_imageDataDict

def segmentMask(img,mask):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,se, iterations = 2)
    sure_bg = cv2.dilate(opening, se, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    uniqueMarkers = np.unique(markers)
    cc_total = []
    finalMask = None
    for m in uniqueMarkers:
        segmentImg = np.uint8(markers.copy())
        segmentImg[segmentImg == m] = 255
        segmentImg[segmentImg != m] = 0
        if finalMask is None: finalMask = segmentImg
        else: finalMask = np.bitwise_or(finalMask,segmentImg)
        cc = cv2.connectedComponentsWithStats(segmentImg, 4, cv2.CV_32S)
        for c in cc[2]:
            if abs(c[2] - img.shape[1]) > img.shape[1] * .1 and abs(c[3] - img.shape[0]) > img.shape[0] * .1:
                cc_total.append(c)
        return (finalMask,cc_total)
def extractFeaturesFromMasks(queryImageResource,imageDataDict,useWatershed=False):
    featureextractor = featureExtraction()
    probeImg = featureextractor.deserialize_image(queryImageResource._data)
    import matplotlib.pyplot as plt
    maskAccumulator_probe = np.zeros(probeImg.shape[:2],dtype=np.uint8)
    featureDictionary = {}
    featureIDList = []
    featureObjectIDList = []
    featureVisDict = {'probe':{},'result':{}}
    saveVis = True
    fullFeatureSet = []
    queryOrResultList = []
    bar = progressbar.ProgressBar()
    keys = list(imageDataDict.keys())
    print('extract features')
    featureSetMap = []
    featureSets = []
    # keys = ['777ea5f5957f58a012bd4521e088357c.jpg']
    for key in bar(keys):
        r = imageDataDict[key]
        if r['isModified'] and r['mask'] is not None:
            resource = r['resource']
            resultFile = resource.key
            featureDictionary[resultFile] = []

            resultImg = featureextractor.deserialize_image(resource._data)
            mask = r['mask']
            mask = 255-np.asarray((mask-mask.min())/(mask.max()-mask.min())*255,dtype=np.uint8)
            if useWatershed:
                mask,components = segmentMask(resultImg,mask)
            else:
                cc = cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)
                components = cc[2]
            # r_rect_img = resultImg.copy()
            # for c in components:
            #     r_rect_img = cv2.rectangle(r_rect_img,(c[0],c[1]),(c[0]+c[2],c[1]+c[3]),(255,0,0),10)
            count = 0
            featureVisDict['probe'][resultFile] = {}
            featureVisDict['result'][resultFile] = {}
            for c in components:
                if abs(c[2]-resultImg.shape[1]) > resultImg.shape[1]*.16 and abs(c[3]-resultImg.shape[0]) > resultImg.shape[0]*.16 and (c[3]-c[1])*(c[2]-c[0]) > 125*125:
                    # crop out the ROI of the suspect object
                    roi = resultImg[c[1]:c[1]+c[3],c[0]:c[0]+c[2]]
                    roi_mask = mask[c[1]:c[1]+c[3],c[0]:c[0]+c[2]]
                    # Make sure we don't find duplicate features from the probe image over and over again
                    maskUnaccountedForInProbe = roi_mask - maskAccumulator_probe[c[1]:c[1] + c[3], c[0]:c[0] + c[2]]
                    maskUnaccountedForInProbe[maskUnaccountedForInProbe < 0 ] = 0
                    maskAccumulator_probe[c[1]:c[1]+c[3],c[0]:c[0]+c[2]] = np.bitwise_or(maskAccumulator_probe[c[1]:c[1]+c[3],c[0]:c[0]+c[2]],roi_mask)
                    # Get features from probe image in the masked area of interest
                    areaUnacountedForInProbe = np.count_nonzero(maskUnaccountedForInProbe)
                    FeaturesForResult = {}
                    if areaUnacountedForInProbe > 50: #If there are more than 50 pixels to pay attention to, get more features from the probe image
                        probeROI = probeImg[c[1]:c[1]+c[3],c[0]:c[0]+c[2]]
                        probeFeatures = getSURFFeaturesForImage(probeROI,shouldScale=False,mask=maskUnaccountedForInProbe)
                        if probeFeatures[1] is not None and len(probeFeatures) > 0:
                            FeaturesForResult['probeFeatures'] = probeFeatures[1]
                            featureIDList.extend([resultFile]*probeFeatures[1].shape[0])
                            featureObjectIDList.extend([count]*probeFeatures[1].shape[0])
                            queryOrResultList.extend(['q']*probeFeatures[1].shape[0])
                            fullFeatureSet.append(probeFeatures[1])
                            featureSetResource = featureextractor.createOutput(Resource(queryImageResource.key, featureextractor.serializeFeature(probeFeatures[1]),'application/octet-stream'))
                            featureSets.append(featureSetResource['supplemental_information']['value'])
                            featureSetMap.append({'type':'query','object':count})
                            if saveVis:
                                vis = probeROI.copy()
                                vis = cv2.drawKeypoints(probeROI,probeFeatures[0],vis)
                                featureVisDict['probe'][resultFile][count] = vis
                    # Get features from result image in the masked area of interest
                    if useResultFeatures:
                        features = getSURFFeaturesForImage(roi,shouldScale=False,mask=roi_mask)
                        if features[1] is not None and len(features) > 0:
                            FeaturesForResult['resultFeatures'] = features[1]
                            FeaturesForResult['Object'] = count
                            featureIDList.extend([resultFile] * features[1].shape[0])
                            featureObjectIDList.extend([count] * features[1].shape[0])
                            queryOrResultList.extend(['r'] * features[1].shape[0])
                            fullFeatureSet.append(features[1])
                            featureSetResource = featureextractor.createOutput(
                                Resource(queryImageResource.key, featureextractor.serializeFeature(features[1]),
                                         'application/octet-stream'))
                            featureSets.append(featureSetResource['supplemental_information']['value'])
                            featureSetMap.append({'type': 'result', 'object': count})
                            if saveVis:
                                vis = roi.copy()
                                vis = cv2.drawKeypoints(roi, features[0], vis)
                                featureVisDict['result'][resultFile][count] = vis
                    featureDictionary[resource.key].append(FeaturesForResult)

                    count += 1
                    # kps = features[0]
                    # descs = features[1]
                    # roi_kp = roi.copy()
                    # roi_kp = cv2.drawKeypoints(roi[:,:,0:3],kps,roi_kp)
                    # roi
    if len(fullFeatureSet) > 0:
        fullFeatureSet = np.concatenate(fullFeatureSet,axis=0)
        fullFeatureResource = featureextractor.createOutput(
            Resource(queryImageResource.key, featureextractor.serializeFeature(fullFeatureSet),
                     'application/octet-stream'))

    else:
        fullFeatureSet = None
        fullFeatureResource = None
    return (fullFeatureResource,featureSets,featureIDList,featureObjectIDList,featureDictionary,queryOrResultList,featureSetMap,featureVisDict)

def getObjectScores(resultScores,featureIDList,featureObjectIDList,featureDictionary,queryOrResultList,numberOfResultsToRetrieve=100,objectWise = False,ignoreIDs = []):
    types = ['q','r']
    talliedResultsForType = {}
    #collect image-wise and object-wise lists of the returned feature index and distance results
    for type in types:
        resultDictionaries = {}
        bar = progressbar.ProgressBar()
        print('separate results')
        for i in bar(range(0,resultScores.I.shape[0])):
            if queryOrResultList[i] == type:
                IndexResults = resultScores.I[i]
                DistanceResults = resultScores.D[i]
                belongsToImage = featureIDList[i]
                belongsToObject = featureObjectIDList[i]
                if belongsToImage not in resultDictionaries:
                    resultDictionaries[belongsToImage] = {}
                    resultDictionaries[belongsToImage]['objects'] = {}
                    resultDictionaries[belongsToImage]['I'] = []
                    resultDictionaries[belongsToImage]['D'] = []
                if belongsToObject not in resultDictionaries[belongsToImage]['objects']:
                    resultDictionaries[belongsToImage]['objects'][belongsToObject] = {}
                    resultDictionaries[belongsToImage]['objects'][belongsToObject]['I'] = []
                    resultDictionaries[belongsToImage]['objects'][belongsToObject]['D'] = []
                resultDictionaries[belongsToImage]['I'].append(IndexResults)
                resultDictionaries[belongsToImage]['D'].append(DistanceResults)
                resultDictionaries[belongsToImage]['objects'][belongsToObject]['I'].append(IndexResults)
                resultDictionaries[belongsToImage]['objects'][belongsToObject]['D'].append(DistanceResults)
        bar = progressbar.ProgressBar()
        print('stack results')
        for k in bar(resultDictionaries):
            resultDictionaries[k]['I'] = np.vstack(resultDictionaries[k]['I'])
            resultDictionaries[k]['D'] = np.vstack(resultDictionaries[k]['D'])
            for k2 in resultDictionaries[k]['objects']:
                resultDictionaries[k]['objects'][k2]['I'] = np.vstack(resultDictionaries[k]['objects'][k2]['I'])
                resultDictionaries[k]['objects'][k2]['D'] = np.vstack(resultDictionaries[k]['objects'][k2]['D'])
        talliedResultsPerImage = {}
        from queryIndex import filteringResults
        bar = progressbar.ProgressBar()
        print('build scores')
        for k in bar(resultDictionaries):
            if objectWise:
                talliedResultsPerImage[k] = {}
                for o in resultDictionaries[k]['objects']:
                    object = resultDictionaries[k]['objects'][o]
                    sortedIDs, sortedVotes,maxvoteval = indexfunctions.tallyVotes(object['D'],
                                                                       object['I'], numcores=12)
                    voteScores = 1.0 * sortedVotes / (1.0 * np.max(sortedVotes))
                    voteScores = 1.0 * sortedVotes / (1.0 * maxvoteval)
                    resultScoresSet = filteringResults()
                    for i in range(0, min(len(sortedIDs), numberOfResultsToRetrieve)):
                        id = sortedIDs[i]
                        if id not in ignoreIDs:
                            id_str = str(id)
                            isInMap = False
                            if id_str in resultScores.map:
                                id_str = id_str
                                isInMap = True
                            elif id in resultScores.map:
                                id_str = id
                                isInMap = True
                            if isInMap:
                                imname = resultScores.map[id_str]
                                score = voteScores[i]
                                resultScoresSet.addScore(imname, score, ID=id)
                    talliedResultsPerImage[k][o] = resultScoresSet
            else:
                sortedIDs, sortedVotes = indexfunctions.tallyVotes(resultDictionaries[k]['D'], resultDictionaries[k]['I'],numcores=12)
                voteScores = 1.0 * sortedVotes / (1.0 * np.max(sortedVotes))
                # talliedResultsPerImage[k] = (sortedIDs,sortedVotes,voteScores)
                resultScoresSet = filteringResults()
                for i in range(0, min(len(sortedIDs), numberOfResultsToRetrieve)):
                  id = sortedIDs[i]
                  id_str = str(id)
                  isInMap = False
                  if id_str in resultScores.map:
                      id_str = id_str
                      isInMap = True
                  elif id in resultScores.map:
                      id_str = id
                      isInMap = True
                  if isInMap:
                      imname = resultScores.map[id_str]
                      score = voteScores[i]
                      resultScoresSet.addScore(imname,score,ID=id)
                talliedResultsPerImage[k] = resultScoresSet
        talliedResultsForType[type] = talliedResultsPerImage
    finalResults = filteringResults()
    # print('merge scores')
    # for r in talliedResultsPerImage:
    #     finalResults.mergeScores(talliedResultsPerImage[r])
    return talliedResultsForType#finalResults
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

def getSURFFeaturesForImage(img,shouldScale=True,mask=None):
    if shouldScale:
        img,scale = getScaledImage(img)
        if mask is not None:
            mask,scale = getScaledImage(mask)
    surf = cv.xfeatures2d.SURF_create(100)
    (kps, descs) = surf.detectAndCompute(img, mask)
    return (kps, descs, img)
def getSURFFeaturesForFile(imgPath):
    img = loadImage(imgPath)
    return getSURFFeaturesForImage(img)
def getScaledImage(img,maxWidth = 512.0):
    w = img.shape[1]
    scale = maxWidth/w
    if scale < 1:
        newWidth = int(w*scale)
        newHeight = int(img.shape[0]*scale)
        return (cv.resize(img,(newWidth,newHeight)),scale)
    else:
        return (img,1)
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