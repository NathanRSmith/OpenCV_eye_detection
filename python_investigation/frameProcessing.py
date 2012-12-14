import numpy as np
import cv2

import pdb

import pupilIsolation
import gazeFunctions
from calibrationHandler import Calibrator

class FrameProcessor():
    lastGazeLoc = None
    simulatormode = None
    
    def __init__(self, facecascade, lefteyecascade, righteyecascade, tabletDims, outimg, calib, draw=True):
        self.facecascade = facecascade
        self.lefteyecascade = lefteyecascade
        self.righteyecascade = righteyecascade
        self.tabletDims = tabletDims
        self.outimg = outimg
        self.calib = calib
        self.draw = draw
        
        
        
    def drawGazeLoc(self):
        """Draw last gaze location
        """
        if self.lastGazeLoc != None:
            cv2.circle(self.outimg, self.lastGazeLoc, 10, (255,0,0), -1)
        
    def processFrame(self, img, simulatormode):
        """Process a single frame.
        """
        self.simulatormode = simulatormode
        facerect = None
        eyesubrectsleft, eyesubrectsright = [], []
        eyesubrectleft, eyesubrectright = None, None
        leftEllipseBox, rightEllipseBox = None, None
        gazeLoc = None
        
        # get gray version of img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        try:
            # detect face
            facerects = self._detect(gray, self.facecascade)
            vis = img.copy()
            if self.draw:
                self._drawRects(vis, facerects, (0, 255, 0))
            
            # find largest face rect
            facerect = self._getLargestRect(facerects)
            x1, y1, x2, y2 = facerect
            
            # left eye
            roileft = gray[y1:(y1+y2)/2, x1:(x1+x2)/2]
            vis_roileft = vis[y1:(y1+y2)/2, x1:(x1+x2)/2]
            gray_roileft = gray[y1:(y1+y2)/2, x1:(x1+x2)/2]
            eyesubrectsleft = self._detect(roileft.copy(), self.lefteyecascade)
            if self.draw:
                self._drawRects(vis_roileft, eyesubrectsleft, (255, 0, 0))
            
            # right eye
            roiright = gray[y1:(y1+y2)/2, (x1+x2)/2:x2]
            vis_roiright = vis[y1:(y1+y2)/2, (x1+x2)/2:x2]
            gray_roiright = gray[y1:(y1+y2)/2, (x1+x2)/2:x2]
            eyesubrectsright = self._detect(roiright.copy(), self.righteyecascade)
            if self.draw:
                self._drawRects(vis_roiright, eyesubrectsright, (255, 0, 0))
            
            # take largest eye detected on each side
            eyesubrectleft = self._getLargestRect(eyesubrectsleft)
            eyesubrectright = self._getLargestRect(eyesubrectsright)
            
            # slice from color img
            vis_roileft_eye = vis_roileft[eyesubrectleft[1]:eyesubrectleft[3], eyesubrectleft[0]:eyesubrectleft[2]]
            vis_roiright_eye = vis_roiright[eyesubrectright[1]:eyesubrectright[3], eyesubrectright[0]:eyesubrectright[2]]
            
            # slice from gray img
            gray_roileft_eye = gray_roileft[eyesubrectleft[1]:eyesubrectleft[3], eyesubrectleft[0]:eyesubrectleft[2]]
            gray_roiright_eye = gray_roiright[eyesubrectright[1]:eyesubrectright[3], eyesubrectright[0]:eyesubrectright[2]]
            
            if self.draw:
                # display combined gray eyes (pre-processed)
                comb = self._combineEyes(
                                    gray_roileft_eye,
                                    gray_roiright_eye
                                    )
                cv2.imshow('combinedeyes', comb)
            
            leftpupil = self._processEyeByCorners(eyesubrectleft, vis_roileft_eye, gray_roileft_eye, vis, gray)
            rightpupil = self._processEyeByCorners(eyesubrectright, vis_roiright_eye, gray_roiright_eye, vis, gray)
            
            if leftpupil != None and rightpupil != None:
                
                #pdb.set_trace()
                
                # if in CALIBRATION mode
                if self.simulatormode == 'CALIBRATION':
                    self.calib.processPhase(leftpupil, rightpupil)
                else:
                    leftyaw, leftpitch, rightyaw, rightpitch = gazeFunctions.getAnglesFromPupilRelativeCenter(leftpupil, rightpupil, self.calib.calibrationInfo, self.calib.calibrationPoints, fliplr=True)
                    gazeLoc = gazeFunctions.findGazeLocation(leftyaw, leftpitch, rightyaw, rightpitch)
                    if gazeLoc != None:
                        self.lastGazeLoc = gazeLoc
                        print gazeLoc
                    else:
                        print 'Gazing offscreen'
                    
        except IndexError:
            pass
        
        if self.draw:
            cv2.imshow('facedetect', vis)
        return vis, gray, facerect, eyesubrectleft, eyesubrectright, gazeLoc
    
    
    def _processEyeByCorners(self, eyesubrect, vis_roi, gray_roi, vis, gray):
        """Get corners from good features to track (leftmost/rightmost possibly).
        Get iris/pupil center either from hough circles or contour method previous explored.
        Find uncorrected gaze direction from amount below line connecting corners
        """
        
        h, w = gray_roi.shape
        leftmostx, rightmostx = w/2, w/2
        leftmostIdx, rightmostIdx = None, None
        leftCorner, rightCorner = None, None
        pupilRelativeCenter = None
        
        # get corners
        corners = cv2.goodFeaturesToTrack(gray_roi, 20, .001, gray_roi.shape[0]/4, useHarrisDetector=False)
        
        # process corners
        for i in range(len(corners)):
            x, y = corners[i][0]
            if ( x<(w/5) or x>((w/4)*3) ) and (y>int(h*0.45) and y<int(h*0.65)):
                # get leftmost and right most points
                if x < leftmostx:
                    leftmostx = x
                    leftmostIdx = i
                if x > rightmostx:
                    rightmostx = x
                    rightmostIdx = i
                
                if self.draw:
                    cv2.circle(vis_roi, tuple(corners[i][0]), 2, (0,255,0), -1)
        
        # display left and right
        if leftmostIdx != None:
            leftCorner = tuple(corners[leftmostIdx][0])
            if self.draw:
                cv2.circle(vis_roi, leftCorner, 2, (0,0,255), -1)
        if rightmostIdx != None:
            rightCorner = tuple(corners[rightmostIdx][0])
            if self.draw:
                cv2.circle(vis_roi, rightCorner, 2, (0,0,255), -1)
        
        if leftCorner != None and rightCorner != None:
            
            # find pupil/iris center via shrink method & look for circles near that center
            
            # apply threshold
            thresh = gray_roi.copy()
            thresh = pupilIsolation.thresholdByPercentage(thresh, .075)
            
            pupilCenter = pupilIsolation.findPointOnPupil(thresh)
            if self.draw:
                cv2.circle(vis_roi, pupilCenter, 2, (255,0,0), -1)
            
            # get eye corner center x, y
            x = (leftCorner[0] + rightCorner[0])/2
            y = (leftCorner[1] + rightCorner[1])/2
            cornersCenter = (x, y)
            
            pupilRelativeCenter = (
                pupilCenter[0] - cornersCenter[0],
                pupilCenter[1] - cornersCenter[1]
            )
            
        return pupilRelativeCenter
    
    
    
    def _detect(self, img, cascade):
        """Detect the cascade feature.
        """
        facerects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
        if len(facerects) == 0:
            return []
        facerects[:,2:] += facerects[:,:2]
        return facerects
    
    def _getLargestRect(self,rects):
        maxsize = 0
        maxidx = 0
        for i in range(len(rects)):
            a = self._rectArea(rects[i])
            if maxsize < a:
                maxsize = a
                maxidx = i
        return rects[maxidx]
    
    def _rectArea(self,rect):
        """Get area of the supplied rectangle.
        """
        x1, y1, x2, y2 = rect
        return (x2-x1) * (y2-y1)
    
    def _drawRects(self, img, facerects, color):
        """Draw all rects in the list.
        """
        tag = 0
        for x1, y1, x2, y2 in facerects:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img,str(tag),(x1,y2),cv2.FONT_HERSHEY_PLAIN,2.0,(200,0,200),2)
            tag = tag+1
    
    def _combineEyes(self, left, right):
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

