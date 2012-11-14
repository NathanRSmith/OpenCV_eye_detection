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

import pdb

help_message = '''
USAGE: facedetect.py [--facecascade <cascade_fn>] [--eyescascade-facecascade <cascade_fn>] [<video_source>]
'''

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    tag = 0
    for x1, y1, x2, y2 in rects:
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

# assume camera 0, webcam. Could also be a file name, etc.

    cam = cv2.VideoCapture(0)

    fileindex = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        rects = detect(gray, facecascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        for x1, y1, x2, y2 in rects:
            roileft = gray[y1:(y1+y2)/2, x1:(x1+x2)/2]
            vis_roileft = vis[y1:(y1+y2)/2, x1:(x1+x2)/2]
            eyesubrectsleft = detect(roileft.copy(), eyescascade)
            draw_rects(vis_roileft, eyesubrectsleft, (255, 0, 0))
            
            #pdb.set_trace()
            
            roiright = gray[y1:(y1+y2)/2, (x1+x2)/2:x2]
            vis_roiright = vis[y1:(y1+y2)/2, (x1+x2)/2:x2]
            eyesubrectsright = detect(roiright.copy(), eyescascade)
            draw_rects(vis_roiright, eyesubrectsright, (255, 0, 0))
	    
	    #pdb.set_trace()
	    
#	    mouthsubrects = detect(roi.copy(), mouthcascade)
#	    
#	    # remove mouth detections that overlap with eyes
#	    remove = []
#	    if type(mouthsubrects) != list:
#		for i in range(mouthsubrects.shape[0]):
#		    for rect in eyesubrects:
#			if is_intersection(mouthsubrects[i],rect):
#			    remove.append(i)
#		for i in sorted(set(remove), reverse=True):
#		    mouthsubrects = np.delete(mouthsubrects, (i), axis=0)
#		
#            draw_rects(vis_roi, mouthsubrects, (0, 0, 255))

        cv2.imshow('facedetect', vis)
        
	key = cv2.waitKey(1) & 0xff

	if key == ord('w'):
		cv2.imwrite(str(fileindex)+'.png',img)
		cv2.imwrite(str(fileindex)+'.annotated.png',vis)
		rindex = 0 
		if (len(rects) == 0) and (len(eyesubrects) == 0) and (len(mouthsubrects) == 0):
			continue
		#else
		f = file(str(fileindex)+'.txt','w')
		print >>f, 'For image file '+str(fileindex)+'.png:'
		for x1,y1,x2,y2 in rects:
			print >>f,'face rect ',rindex,': (',x1,',',y1,') -> (',x2,',',y2,'), ',find_rect_center((x1,y1,x2,y2))
			rindex = rindex +1
		rindex = 0
		for x1,y1,x2,y2 in eyesubrects:
			print >>f,'eye rect ',rindex,': (',x1,',',y1,') -> (',x2,',',y2,'), ',find_rect_center((x1,y1,x2,y2))
			rindex = rindex + 1
		rindex = 0
		for x1,y1,x2,y2 in mouthsubrects:
			print >>f,'mouth rect ',rindex,': (',x1,',',y1,') -> (',x2,',',y2,'), ',find_rect_center((x1,y1,x2,y2))
			rindex = rindex + 1
		f.close()
		fileindex = fileindex + 1
	elif key == 27:
		break
	else:
		continue

    cv2.destroyAllWindows()

