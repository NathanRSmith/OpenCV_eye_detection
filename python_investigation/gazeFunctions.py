import math
import pdb


tabletDims = {
    'units': 'cm',
    'width': 25.0,
    'height': 17.0,
    'border': {
        'top': 1.5,
        'bottom': 1.5,
        'left': 1.5,
        'right': 1.5,
    },
    'resolution': {
        'width': 1280,
        'height': 800,
    },
    'camera_position': {
        'x': 12.5,
        'y': .65,
    },
}

face_dist = 30.0
eye_dist = 6.0

# angles
pitchmin = (-1)*math.atan(tabletDims['border']['top']/face_dist)
pitchmax = (-1)*math.atan((tabletDims['height']-tabletDims['border']['top'])/face_dist)
lefteye_leftmax = (-1)*math.atan( ( (tabletDims['width'] - eye_dist - (tabletDims['border']['left'] + tabletDims['border']['right'])) / 2  ) / face_dist )
lefteye_rightmax = math.atan( ( (tabletDims['width'] + eye_dist - (tabletDims['border']['left'] + tabletDims['border']['right'])) / 2  ) / face_dist )
righteye_leftmax = (-1)*lefteye_rightmax
righteye_rightmax = (-1)*lefteye_leftmax

dummyCalibrationPoints = [
    {
        'left_eye': (-20, -10),
        'right_eye': (-20, -10),
    },
    {
        'left_eye': (20, -10),
        'right_eye': (20, -10),
    },
    {
        'left_eye': (0, 0),
        'right_eye': (0, 0),
    },
    {
        'left_eye': (0, -20),
        'right_eye': (0, -20),
    },
]

calibrationInfo = [
    {
        'name': 'left', # left direction is actually right side of img
        'screen_pos_x': 0,
        'screen_pos_y': 400,
        'left_eye_yaw': lefteye_leftmax,
        'left_eye_pitch': (pitchmax+pitchmin)/2,
        'right_eye_yaw': righteye_leftmax,
        'right_eye_pitch': (pitchmax+pitchmin)/2,
    },
    {
        'name': 'right', # right direction is actually left side of img
        'screen_pos_x': 1280,
        'screen_pos_y': 400,
        'left_eye_yaw': lefteye_rightmax,
        'left_eye_pitch': (pitchmax+pitchmin)/2,
        'right_eye_yaw': righteye_rightmax,
        'right_eye_pitch': (pitchmax+pitchmin)/2,
    },
    {
        'name': 'top',
        'screen_pos_x': 640,
        'screen_pos_y': 0,
        'left_eye_yaw': 0,
        'left_eye_pitch': pitchmin,
        'right_eye_yaw': 0,
        'right_eye_pitch': pitchmin,
    },
    {
        'name': 'bottom',
        'screen_pos_x': 640,
        'screen_pos_y': 800,
        'left_eye_yaw': 0,
        'left_eye_pitch': pitchmax,
        'right_eye_yaw': 0,
        'right_eye_pitch': pitchmax,
    }
]

def getAnglesFromPupilRelativeCenter(leftPupil, rightPupil, calibrationInfo, calibrationPoints, fliplr=True):
    """Compare pupil location to calibration points and come up with yaw and pitch.
    Start with left eye
    """
    # if fliplr, switch the eyes and x components of positions
    if fliplr == True:
        ltmp = leftPupil
        leftPupil = ((-1)*rightPupil[0], rightPupil[1])
        rightPupil = ((-1)*ltmp[0], ltmp[1])
    
    
    leftyaw, leftpitch = None, None
    rightyaw, rightpitch = None, None
    
    ###### process leftyaw (x component) ######
    # get slope of relationship between calibration points x and leftyaw (assume linear)
    yawslope = ((calibrationInfo[0]['left_eye_yaw'] - calibrationInfo[1]['left_eye_yaw']) /
                (calibrationPoints[0]['left_eye'][0] - calibrationPoints[1]['left_eye'][0]))
    leftyaw = yawslope*leftPupil[0]
    
    yawslope = ((calibrationInfo[0]['right_eye_yaw'] - calibrationInfo[1]['right_eye_yaw']) /
                (calibrationPoints[0]['right_eye'][0] - calibrationPoints[1]['right_eye'][0]))
    rightyaw = yawslope*rightPupil[0]
    
    #pdb.set_trace()
    
    ###### process leftpitch (y component) ######
    pitchslope = ((calibrationInfo[2]['left_eye_pitch'] - calibrationInfo[3]['left_eye_pitch']) /
                (calibrationPoints[2]['left_eye'][1] - calibrationPoints[3]['left_eye'][1]))
    leftpitch = pitchslope*leftPupil[1]
    
    pitchslope = ((calibrationInfo[2]['right_eye_pitch'] - calibrationInfo[3]['right_eye_pitch']) /
                (calibrationPoints[2]['right_eye'][1] - calibrationPoints[3]['right_eye'][1]))
    rightpitch = pitchslope*rightPupil[1]
    
    return leftyaw, leftpitch, rightyaw, rightpitch


def findGazeLocation(lefteye_yaw, lefteye_pitch, righteye_yaw, righteye_pitch, units='rad'):
    """
    yaw = horizontal rotation
    pitch = vertical rotation
    
    angles in radians
    
    for yaw angles, 0 is aligned with normal angle of surface
    negative is left, positive is right
    
    pitch angles will be negative for on-screen positions
    """
    if units == 'deg':
        lefteye_yaw = lefteye_yaw * math.pi / 180.0
        lefteye_pitch = lefteye_pitch * math.pi / 180.0
        righteye_yaw = righteye_yaw * math.pi / 180.0
        righteye_pitch = righteye_pitch * math.pi / 180.0
    
    #pdb.set_trace()
    
    lefteye_x = tabletDims['width']/2 - eye_dist/2 + face_dist*math.tan(lefteye_yaw)
    righteye_x = tabletDims['width']/2 + eye_dist/2 + face_dist*math.tan(righteye_yaw)
    ave_x = (lefteye_x + righteye_x) / 2
    
    # if the gaze location is outside of screen area
    if ave_x < tabletDims['border']['left'] or ave_x > tabletDims['width'] - tabletDims['border']['right']:
        return None
    
    lefteye_y = (-1) * face_dist * math.tan(lefteye_pitch)
    righteye_y = (-1) * face_dist * math.tan(righteye_pitch)
    ave_y = (lefteye_y + righteye_y) / 2
    
    # if the gaze location is outside of screen area
    if ave_y < tabletDims['border']['top'] or ave_y > tabletDims['height'] - tabletDims['border']['bottom']:
        return None
    
    px_x = ( (ave_x - tabletDims['border']['left']) / (tabletDims['width'] - tabletDims['border']['left'] - tabletDims['border']['right']) ) * tabletDims['resolution']['width']
    px_y = ( (ave_y - tabletDims['border']['top']) / (tabletDims['height'] - tabletDims['border']['top'] - tabletDims['border']['bottom']) ) * tabletDims['resolution']['height']
    
    return (int(px_x), int(px_y))
    

    
    
    


    
if __name__ == '__main__':
    pdb.set_trace()
    
    #leftyaw, leftpitch, rightyaw, rightpitch = getAnglesFromPupilRelativeCenter((11.0, -2.5), (11.0, -2.5),dummyCalibrationPoints) # looking at left side of laptop screen
    leftyaw, leftpitch, rightyaw, rightpitch = getAnglesFromPupilRelativeCenter((20,-20), (20,-20),dummyCalibrationPoints) # looking at left side of laptop screen
    print leftyaw, leftpitch, rightyaw, rightpitch
    print findGazeLocation(leftyaw, leftpitch, rightyaw, rightpitch)
    
    #print findGazeLocation(15.0,25.0,12.0,25.0,'deg')
    #print processEllise(((45.887634, 27.060354), (16.731773, 23.523066), 16.967484))

