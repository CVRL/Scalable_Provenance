import sys
import os
import cv2
import numpy as np
from skimage.filters import threshold_local
# import scipy.ndimage.measurements as sci
def scaleImage(im,maxVal=1,Sigma=-1):
    imMax=im.max()
    imMin = im.min()
    denom = 1
    if Sigma > 0:
        std = im.std()
        mean = im.mean()
        imMax = min(imMax,mean+std*Sigma)
        imMin = max(imMin,mean-std*Sigma)
        im[im < imMin] = imMin
        im[im > imMax] = imMax
        denom = (imMax-imMin)
        if denom == 0: denom = 1
    normed = (im-imMin)/ denom
    return normed*maxVal
def resPSNR(im1,im2,gs=True):
    differenceThresh = .06
    hasBeenModified = True
    globalModification = False
    sigma = 5;
    ksize = int(2*np.ceil(2*sigma)+1)
    im1g = im1
    im2g = im2
    if gs and len(im1.shape) > 2 and  im1.shape[2] is not None and im1.shape[2] >= 3:
        im1g = cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY)
    if gs and len(im2.shape) > 2 and im2.shape[2] is not None and im2.shape[2] >= 3:
        im2g = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    im1_blur = cv2.GaussianBlur(im1g.astype(float),(ksize,ksize),sigma,borderType=cv2.BORDER_DEFAULT)
    im2_blur = cv2.GaussianBlur(im2g.astype(float),(ksize,ksize),sigma,borderType=cv2.BORDER_DEFAULT)
    imdif = np.abs(im1_blur-im2_blur)
    maxv = np.max(imdif)
    minv = min(0,np.min(imdif)) #Only normalize minimum if min is negative
    maxPossibleDifference = max(abs(np.max(im1_blur)-np.min(im2_blur)),abs(np.max(im2_blur)-np.min(im1_blur)))
    if maxv < maxPossibleDifference*differenceThresh: #If the maximum difference is less than 6% of the possible difference between the images, we consider it not modified.
        hasBeenModified = False
        return np.zeros_like(imdif),hasBeenModified,False,-1
    differentArea = np.where(imdif > maxv-maxPossibleDifference*differenceThresh*3.5)
    if len(differentArea[0]) < 6*6: #Anything smaller than this means it probably is noise, not a modification
        hasBeenModified = False
        return np.zeros_like(imdif), hasBeenModified, False,-1
    if maxv-minv != 0:
        imdif = (imdif - minv) / (maxv - minv)
    map = 10*np.log10(pow(255,2)/(np.power(imdif,2)+1))
    mapDifferentArea = np.where(map < np.min(map) * 1.065)
    globalModVal = len(mapDifferentArea[0])/(im1.shape[0]*im1.shape[1])
    if globalModVal > .5: #a global operation has been performed
        globalModification = True
    maxv = np.max(map)
    minv = min(0,np.min(map))
    if maxv==minv and maxv != 0:
        map_norm = map/maxv
    elif maxv==minv and maxv == 0:
        map_norm = map
    else :
        map_norm = (map-minv)/(maxv-minv)
    map_norm = map_norm.max()-map_norm
    return map_norm,hasBeenModified,globalModification,globalModVal

def genMask(mapIm):
    # th, BW1 = cv2.threshold(mapIm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    BW = threshold_local(mapIm,35,offset=10)
    BW[BW < np.average(BW)-np.std(BW)*0] = 0
    # BW = 255-BW
    # th2,BW = cv2.threshold(mapIm, 253, 255, cv2.THRESH_BINARY_INV)

    # BW = BW.max()-BW
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
    BW_filled = cv2.morphologyEx(BW,cv2.MORPH_CLOSE,se)
    # BW_filled = imFill(BW_filled)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    BW_filled = cv2.morphologyEx(BW_filled, cv2.MORPH_CLOSE, se)
    BW_filled = cv2.morphologyEx(BW_filled, cv2.MORPH_OPEN, se)
    # BW_filled =threshold_local(BW_filled,35,offset=10)
    BWmin = np.min(BW_filled)
    BWmax = np.max(BW_filled)
    BW_filled = scaleImage(BW_filled)
    return BW_filled
    # cv2.imshow("m1",BW_filled)
    # labeled, n = sci.label(BW_filled,np.ones(3,3))
    # print n

def imFill(BW):
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    im_floodfill = BW.copy()
    im_floodfill = cv2.bitwise_not(im_floodfill)
    h, w = BW.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = BW | im_floodfill_inv

    # Display images.
    # cv2.imshow("Thresholded Image", im_th)
    # cv2.imshow("Floodfilled Image", im_floodfill)
    # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # cv2.imshow("Foreground", im_out)
    # cv2.waitKey(0)
    return im_out

def resPSNR_Norm(im1,im2,withMask=False,withMetaData=False):
    dif = resPSNR(im1,im2)
    dif_norm = scaleImage(dif[0],255,Sigma=4)
    mask = None
    if withMask:
        mask = genMask(dif[0])*255
    if withMetaData:
        return (dif_norm,mask,dif[1],dif[2],dif[3])
    else:
        return (dif_norm,mask)
def usage():
    print("Usage: imcompare.py <image1> <image2> <outputDir>")
    print("Output: difference map '<image1>_<image2>_dif'")
    print("Flags: \n--output-type <type>:     What image type to output masks in (e.g. png, jpg, tiff)"
                 "\n--with-mask:              output binary mask along with difference map (default: False)"
                 "\n--quiet:                  Don't output aglorithm's yes/no answers to if image has been modified")
    exit(1)
def main():
    args = sys.argv[1:]
    im1path = None
    im2path = None
    outpath = None
    withMask = False
    printGuesses = True
    outputType = 'png'
    while args:
        a = args.pop(0)
        if a == '-h': usage()
        elif a == '--output-type' : outputType = True
        elif a == '--with-mask' :   withMask = True
        elif a == '--quiet': printGuesses = False
        elif not im1path:        im1path = a
        elif not im2path:     im2path = a
        elif not outpath:   outpath = a
        else:
            print("argument %s unknown" % a)
            usage()
            sys.exit(1)
    if not os.path.exists(outpath):
        try:
            os.makedirs(outpath)
        except:
            print("Could not generate path: ",outpath)
    if outputType.startswith('.'): outputType = outputType[1:]
    im1 = cv2.imread(im1path)
    im2 = cv2.imread(im2path)
    res = resPSNR_Norm(im1,im2,withMask=withMask,withMetaData=True)
    dif_norm = res[0]
    mask = res[1]
    if printGuesses:
        print("Has modification occured: ", res[2][0])
        print("Has global modification occured: ", res[2][1])
    cv2.imwrite(os.path.join(outpath,os.path.basename(im1path)+'_'+os.path.basename(im2path)+'_dif.'+outputType),dif_norm)
    if mask is not None:
        cv2.imwrite(os.path.join(outpath,os.path.basename(im1path)+'_'+os.path.basename(im2path)+'_mask.'+outputType),mask)

if __name__ == "__main__":
    main()
