# Describes images with SURF keypoint detection and description.
import numpy, cv2


def surfDescribe(image, kpCount=2000, hessian=100.0, mask=[[]]):
    if image is None:
        return [], []

    # obtains the gray-scaled version of the given image
    #gsImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gsImage = image
    if len(image.shape) > 2:
        gsImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # creates a mask to ignore eventual black borders
    _, bMask = cv2.threshold(cv2.normalize(gsImage, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX),
                             1, 255, cv2.THRESH_BINARY)
    bMask = cv2.convertScaleAbs(bMask)

    # combines the border mask to an eventual given mask
    if mask != [[]]:
        mask = cv2.bitwise_and(mask, bMask)
    else:
        mask = bMask

    # detects the SURF keypoints
    surfDetectorDescriptor = cv2.SURF(hessian)
    keypoints = surfDetectorDescriptor.detect(gsImage, mask)
    descriptions = []

    # describes the obtained keypoints
    if len(keypoints) > 0:
        # removes the weakest keypoints (according to hessian)
        keypoints = sorted(keypoints, key=lambda match: match.response, reverse=True)
        del keypoints[kpCount:]

        # describes the selected keypoints
        keypoints, descriptions = surfDetectorDescriptor.compute(gsImage, keypoints)

    # returns keypoints and descriptions
    return keypoints, descriptions
