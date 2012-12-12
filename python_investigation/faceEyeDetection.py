#
#
# CSE x0535 face detector with eyescascade detector
# PJF 11/4/2012
#
# This is a modified version of the samples/python2/facedetect.py demo
# It was modified to (1) remove local dependencies on other python scripts,
# use image source 0 when opening up a capture device,
# capture only 2 images per second, and provide the half-second delay to
# let the user save the image and detected features to a pair of files.
#
# Specifically, if the image capture window is active, and the uses presses 'w'
# when she wants to save the image and detection results, the original image,
# the annotated image, and the text description of detected
# features are written to two PNG files and a text file named N.png,
# N.annotated.png, and N.txt, respectively.
# N starts at 1 and increases by 1 after each use of the 'w' key.
# BEWARE: old .png and .txt files will be overwritten! Make local copies before
# you run the script again!
#
# Random notes: could add a second subface detector to detect another
# type of subface feature

import numpy as np
import cv2
import cv2.cv as cv
#from video import create_capture
#from common import clock, draw_str

from random import randrange

import math

from pupilIsolation import *

import pdb

help_message = '''
USAGE: facedetect.py [--facecascade <cascade_fn>] [--eyescascade-facecascade <cascade_fn>] [<video_source>]
'''



def detect(img, cascade):
    """Detect the cascade feature.
    """
    facerects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(facerects) == 0:
        return []
    facerects[:,2:] += facerects[:,:2]
    return facerects

def draw_rects(img, facerects, color):
    """Draw all rects in the list.
    """
    tag = 0
    for x1, y1, x2, y2 in facerects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img,str(tag),(x1,y2),cv2.FONT_HERSHEY_PLAIN,2.0,(200,0,200),2)
        tag = tag+1

def is_intersection(rect1, rect2):
    """Determines if two rectangles overlap.
    """
    x1s = [rect1[0],rect1[2]]
    x1s.sort()
    x2s = [rect2[0],rect2[2]]
    x2s.sort()
    y1s = [rect1[1],rect1[3]]
    y1s.sort()
    y2s = [rect2[1],rect2[3]]
    y2s.sort()
    
    return not (x1s[1] < x2s[0] or x1s[0] > x2s[1] or y1s[1] < y2s[0] or y1s[0] > y2s[1])

def find_rect_center(rect):
    """Finds the center of a rectangle from two opposite corners.
    """
    x1, y1, x2, y2 = rect
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    return (x, y)

def rectArea(rect):
    """Get area of the supplied rectangle.
    """
    x1, y1, x2, y2 = rect
    return (x2-x1) * (y2-y1)

def combineEyes(left, right):
    """Stack images horizontally.
    """
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    hoff = left.shape[0]
    
    shape = list(left.shape)
    shape[0] = h
    shape[1] = w
    
    comb = np.zeros(tuple(shape),left.dtype)
    
    # left will be on left, aligned top, with right on right
    comb[:left.shape[0],:left.shape[1]] = left
    comb[:right.shape[0],left.shape[1]:] = right
    
    return comb

def processEye(eyesubrects, vis_roi, gray_roi):
    """threshold
    get contours
    find largest
    fit ellipse
    """
    # apply threshold
    thresh = gray_roi.copy()
    #cv2.adaptiveThreshold(leftthresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 25, leftthresh)
    thresh = thresholdByPercentage(thresh, .075)
    
    # find contours from thresholded img
    contours, heirarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    maxarea = 0
    maxidx = 0
    
    # find contour with largest area
    for idx in range(len(contours)):
        a = cv2.contourArea(contours[idx])
        if maxarea < a:
            maxarea = a
            maxidx = idx
    
    cv2.drawContours(vis_roi, contours, maxidx, (0,0,255), -1)
    
    # fit ellipse to the detected contour
    ellipseBox = cv2.fitEllipse(contours[maxidx])
    cv2.ellipse(vis_roi, ellipseBox, (50, 150, 255))
    
    return thresh, ellipseBox

def processFrame(img, facecascade, eyescascade):
    """Process a single frame.
    """
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    facerects = detect(gray, facecascade)
    vis = img.copy()
    draw_rects(vis, facerects, (0, 255, 0))
    
    facerect = None
    eyesubrectsleft, eyesubrectsright = [], []
    eyesubrectleft, eyesubrectright = None, None
    leftEllipseBox, rightEllipseBox = None, None
    
    
    try:
        # find largest face rect and process
        maxsize = 0
        maxidx = 0
        for i in range(len(facerects)):
            a = rectArea(facerects[i])
            if maxsize < a:
                maxsize = a
                maxidx = i
        
        facerect = facerects[maxidx]
        x1, y1, x2, y2 = facerect
        
        # left eye
        roileft = gray[y1:(y1+y2)/2, x1:(x1+x2)/2]
        vis_roileft = vis[y1:(y1+y2)/2, x1:(x1+x2)/2]
        gray_roileft = gray[y1:(y1+y2)/2, x1:(x1+x2)/2]
        eyesubrectsleft = detect(roileft.copy(), eyescascade)
        draw_rects(vis_roileft, eyesubrectsleft, (255, 0, 0))
        
        # right eye
        roiright = gray[y1:(y1+y2)/2, (x1+x2)/2:x2]
        vis_roiright = vis[y1:(y1+y2)/2, (x1+x2)/2:x2]
        gray_roiright = gray[y1:(y1+y2)/2, (x1+x2)/2:x2]
        eyesubrectsright = detect(roiright.copy(), eyescascade)
        draw_rects(vis_roiright, eyesubrectsright, (255, 0, 0))
        
        # if left and right eyes detected
        if len(eyesubrectsleft) > 0 and len(eyesubrectsright) > 0:
            # take first eye detected on each side
            eyesubrectleft = eyesubrectsleft[0] 
            eyesubrectright = eyesubrectsright[0]
            
            # slice from color img
            vis_roileft_eye = vis_roileft[eyesubrectleft[1]:eyesubrectleft[3], eyesubrectleft[0]:eyesubrectleft[2]]
            vis_roiright_eye = vis_roiright[eyesubrectright[1]:eyesubrectright[3], eyesubrectright[0]:eyesubrectright[2]]
            
            # slice from gray img
            gray_roileft_eye = gray_roileft[eyesubrectleft[1]:eyesubrectleft[3], eyesubrectleft[0]:eyesubrectleft[2]]
            gray_roiright_eye = gray_roiright[eyesubrectright[1]:eyesubrectright[3], eyesubrectright[0]:eyesubrectright[2]]
            
            # display combined gray eyes (pre-processed)
            comb = combineEyes(
                                gray_roileft_eye,
                                gray_roiright_eye
                                )
            cv2.imshow('combinedeyes', comb)
            
            leftthresh, leftEllipseBox = processEye(eyesubrectsleft, vis_roileft_eye, gray_roileft_eye)
            rightthresh, rightEllipseBox = processEye(eyesubrectsright, vis_roiright_eye, gray_roiright_eye)
            
            combthresh = combineEyes(leftthresh, rightthresh)
            cv2.imshow('combinedthresh', combthresh)
            
            
            
            ############ Thresholding ###########
            #
            ## threshold left eye
            #leftthresh = gray_roileft_eye.copy()
            ##cv2.adaptiveThreshold(leftthresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 25, leftthresh)
            #leftthresh = thresholdByPercentage(leftthresh, .075)
            #
            ## threshold right eye
            #rightthresh = gray_roiright_eye.copy()
            ##cv2.adaptiveThreshold(rightthresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 25, rightthresh)
            #rightthresh = thresholdByPercentage(rightthresh, .075)
            #
            ## display combined threshes
            #combthresh = combineEyes(leftthresh, rightthresh)
            #cv2.imshow('combinedthresh', combthresh)
            #
            #
            ############ Contours ###########
            #
            ## find contours from thresholded img
            #contours, heirarchy = cv2.findContours(leftthresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #
            #maxarea = 0
            #maxidx = 0
            #
            ## find contour with largest area
            #for idx in range(len(contours)):
            #    a = cv2.contourArea(contours[idx])
            #    if maxarea < a:
            #        maxarea = a
            #        maxidx = idx
            #
            #cv2.drawContours(vis_roileft_eye, contours, maxidx, (0,0,255), -1)
            #
            ##for idx in range(len(contours)):
            ##        
            ##    clr = (randrange(0, 255), randrange(0, 255), randrange(0, 255))
            ##    cv2.drawContours(vis_roileft_eye, contours, idx, clr, -1)
            #
            #
            ############ Ellipse ###########
            #
            ## fit ellipse to the detected contour
            #ellipseBox = cv2.fitEllipse(contours[maxidx])
            #cv2.ellipse(vis_roileft_eye, ellipseBox, (50, 150, 255))
            #
            ##pdb.set_trace()
            #
            ##combvis = combineEyes(
            ##                    vis_roileft_eye,
            ##                    vis_roiright_eye
            ##                  )
            ##cv2.imshow('colorcombinedeyes', combvis)
            #
            ##pdb.set_trace()
    
    except IndexError:
        pass

    cv2.imshow('facedetect', vis)
    
    return vis, gray, facerect, eyesubrectleft, eyesubrectright, leftEllipseBox, rightEllipseBox


def writeline(f,line):
    f.write(line+'\n')


if __name__ == '__main__':
    import sys, getopt
    print help_message

    args, video_src = getopt.getopt(sys.argv[1:], '', ['facecascade=', 'eyescascade-facecascade='])
    try: video_src = video_src[0]
    except: video_src = 0
    args = dict(args)
    cascade_fn = args.get('--facecascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    eyes_fn  = args.get('--eyescascade-facecascade', "data/haarcascades/haarcascade_eye.xml")
    #eyes_fn  = args.get('--eyescascade-facecascade', "data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")
    mouth_fn  = "data/haarcascades/haarcascade_mcs_mouth.xml"

    facecascade = cv2.CascadeClassifier(cascade_fn)
    eyescascade = cv2.CascadeClassifier(eyes_fn)
    mouthcascade = cv2.CascadeClassifier(mouth_fn)
    
    
    #mode = 'picture'
    mode = 'video'
    #source = 'imgs/IMG_20121112_121434.jpg'
    #source = 'imgs/IMG_20121112_121455.jpg'
    #source = 'imgs/IMG_20121112_121522.jpg'
    #source = 'imgs/IMG_20121112_121443.jpg'
    #source = 'imgs/IMG_20121112_121501.jpg'
    #source = 'imgs/IMG_20121112_121525.jpg'
    #source = 'imgs/IMG_20121112_121448.jpg'
    #source = 'imgs/IMG_20121112_121519.jpg'
    source = 'imgs/2.png'
    
    if mode == 'picture':
        img = cv2.imread(source)
        processFrame(img, facecascade, eyescascade)
        
        #pdb.set_trace()
    
    elif mode == 'video':
        
        # assume camera 0, webcam. Could also be a file name, etc.
        cam = cv2.VideoCapture(0)
    
        fileindex = 0
    
        while True:
            ret, img = cam.read()
            
            vis, gray, facerect, eyesubrectleft, eyesubrectright, leftEllipseBox, rightEllipseBox = processFrame(img, facecascade, eyescascade)
            
            #pdb.set_trace()
            
            key = cv2.waitKey(1) & 0xff
            
            if key == ord('w'):
                cv2.imwrite(str(fileindex)+'.png',img)
                cv2.imwrite(str(fileindex)+'.vis.png',vis)
                cv2.imwrite(str(fileindex)+'.gray.png',gray)
                rindex = 0 
                if facerect != None and eyesubrectleft != None and eyesubrectright != None and leftEllipseBox != None and rightEllipseBox != None:
                    with file(str(fileindex)+'.txt','w') as f:
                        writeline(f,'For image file %d.png:' % fileindex)
                        # face
                        writeline(f,'face rect: (%d, %d) -> (%d, %d)' % tuple(facerect))
                        
                        # left
                        writeline(f,'left eye rect: (%d, %d) -> (%d, %d)' % tuple(eyesubrectleft))
                        writeline(f, 'left eye ellipse: center: (%f, %f), size: (%f, %f), angle: %f' % (leftEllipseBox[0][0],leftEllipseBox[0][1],leftEllipseBox[1][0],leftEllipseBox[1][1],leftEllipseBox[2]))
                        
                        # right
                        writeline(f,'right eye rect: (%d, %d) -> (%d, %d)' % tuple(eyesubrectright))
                        writeline(f, 'right eye ellipse: center: (%f, %f), size: (%f, %f), angle: %f' % (rightEllipseBox[0][0],rightEllipseBox[0][1],rightEllipseBox[1][0],rightEllipseBox[1][1],rightEllipseBox[2]))
                        
                        
                        
                        fileindex = fileindex + 1
            elif key == ord('q'):
                break
            else:
                continue

    cv2.destroyAllWindows()

