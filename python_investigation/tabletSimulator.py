import os, sys

import numpy as np
import cv2

import pdb

from frameProcessing import FrameProcessor
from calibrationHandler import Calibrator
import gazeFunctions



def main():
    ###### Setup ######
    
    # assume camera 0, webcam. Could also be a file name, etc.
    cam = cv2.VideoCapture(0)
    
    # cascade classifiers
    facecascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
    lefteyecascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_mcs_lefteye.xml')
    righteyecascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_mcs_righteye.xml')
    
    
    
    
    tabletDims = gazeFunctions.tabletDims
    outimg = np.zeros((tabletDims['resolution']['height'],tabletDims['resolution']['width'],3),np.uint8)
    calib = Calibrator(gazeFunctions.calibrationInfo, outimg)
    fp = FrameProcessor(facecascade, lefteyecascade, righteyecascade, tabletDims, outimg, calib, draw=False)
    
    calib.calibrationPoints = gazeFunctions.dummyCalibrationPoints
    
    
    mode = 'CALIBRATION'
    
    fileindex = 0
    
    while True:
        ret, img = cam.read()
        
        outimg[:] = 255
        
        calib.drawCalibrationPoint()
        
        #vis, gray, facerect, eyesubrectleft, eyesubrectright, leftEllipseBox, rightEllipseBox = processFrame(img, facecascade, lefteyecascade, righteyecascade)
        vis, gray, facerect, eyesubrectleft, eyesubrectright, gazeLoc = fp.processFrame(img)
        fp.drawGazeLoc()
        
        
        
        cv2.imshow('TabletScreen',outimg)
        
        #pdb.set_trace()
        
        
        
        
        
        
        
        
        key = cv2.waitKey(2)
        
        
        
        
        
        # pause
        if key == ord('p'):
            pdb.set_trace()
        elif key == ord('n'):
            if mode == 'CALIBRATION':
                calib.setMode('CALIBRATE')
        
        elif key == ord('w'):
            cv2.imwrite(str(fileindex)+'.png',img)
            cv2.imwrite(str(fileindex)+'.vis.png',vis)
            cv2.imwrite(str(fileindex)+'.gray.png',gray)
            rindex = 0 
            #if facerect != None and eyesubrectleft != None and eyesubrectright != None and leftEllipseBox != None and rightEllipseBox != None:
            if facerect != None and eyesubrectleft != None and eyesubrectright != None:
                with file(str(fileindex)+'.txt','w') as f:
                    writeline(f,'For image file %d.png:' % fileindex)
                    # face
                    writeline(f,'face rect: (%d, %d) -> (%d, %d)' % tuple(facerect))
                    
                    # left
                    writeline(f,'left eye rect: (%d, %d) -> (%d, %d)' % tuple(eyesubrectleft))
                    #writeline(f, 'left eye ellipse: center: (%f, %f), size: (%f, %f), angle: %f' % (leftEllipseBox[0][0],leftEllipseBox[0][1],leftEllipseBox[1][0],leftEllipseBox[1][1],leftEllipseBox[2]))
                    
                    # right
                    writeline(f,'right eye rect: (%d, %d) -> (%d, %d)' % tuple(eyesubrectright))
                    #writeline(f, 'right eye ellipse: center: (%f, %f), size: (%f, %f), angle: %f' % (rightEllipseBox[0][0],rightEllipseBox[0][1],rightEllipseBox[1][0],rightEllipseBox[1][1],rightEllipseBox[2]))
                    
                    
                    
                    fileindex = fileindex + 1
        elif key == ord('q'):
            break
        else:
            continue

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()