import os, sys

import pdb

import numpy as np
import cv2

from collections import deque



def thresholdByPercentage(img, percentage):
    """Threshold to the given percentage of pixels.
    Compute histogram (sorted array of sorts) and select the lowest
    percentage of pixels to remain.
    
    count from lowest bin until percentage is reached, threshold all
    above that level.
    """
    numCells = img.shape[0]*img.shape[1]
    percCount = 0.0
    threshVal = None
    
    bins = range(256)
    hist = np.histogram(img, bins=bins)
    
    for i in range(len(bins)):
        histbin = hist[0][i]
        percCount += float(histbin)/numCells
        if percCount > percentage:
            threshVal = hist[1][i]
            break
    
    #print threshVal
    #pdb.set_trace()
    
    retval, threshimg = cv2.threshold(img, threshVal, 255, cv2.THRESH_BINARY_INV)
    
    return threshimg


def findPointOnPupil(img):
    """Attempts to find a point on the pupil by taking a thresholded binary image
    and shrinking it until nothing remains, back up a step, and find that point
    """
    history = deque([img.copy(),img.copy()])
    curr = img
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    
    while not isUniform(curr):
        curr = cv2.erode(curr, kernel)
        history.append(curr.copy())
        history.popleft()
    
    prev = history.popleft()
    max = prev.max()
    
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if prev[r,c] == max:
                return (c, r)
    
    
def isUniform(img):
    """Tests if all pixels are uniform.
    compute histogram. If first non-zero bin does not have all pixels, it is
    not uniform.
    """
    #hist = np.histogram(img, bins=range(256))
    #testidx = 0
    #for i in range(256):
    #    if hist[0][i] != 0:
    #        testidx = i
    #        break
    #
    #return hist[0][i] == (img.shape[0]*img.shape[1])
    return img.max() == img.min()

def main():
    IMG = 'eyes.tiff'
    
    img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
    
    cv2.imshow('orig', img)
    
    threshimg = thresholdByPercentage(img, .3)
    
    cv2.imshow('percthresh', threshimg)
    
    cv2.imwrite('eyetest.tiff',threshimg)
    
    pdb.set_trace()



if __name__ == '__main__':
    main()