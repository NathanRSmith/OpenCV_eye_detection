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
pitchmin = math.atan(tabletDims['border']['top']/face_dist)
pitchmax = math.atan((tabletDims['height']-tabletDims['border']['top'])/face_dist)
lefteye_leftmax = math.atan( ( (tabletDims['width'] - eye_dist - (tabletDims['border']['left'] + tabletDims['border']['right'])) / 2  ) / face_dist )
lefteye_rightmax = math.atan( ( (tabletDims['width'] + eye_dist - (tabletDims['border']['left'] + tabletDims['border']['right'])) / 2  ) / face_dist )
righteye_topmax = math.atan( ( (tabletDims['width'] + eye_dist - (tabletDims['border']['left'] + tabletDims['border']['right'])) / 2  ) / face_dist )
righteye_bottommax = math.atan( ( (tabletDims['width'] - eye_dist - (tabletDims['border']['left'] + tabletDims['border']['right'])) / 2  ) / face_dist )

def findGazeLocation(lefteye_yaw, lefteye_pitch, righteye_yaw, righteye_pitch, units='rad'):
    """
    yaw = horizontal rotation
    pitch = vertical rotation
    
    angles in radians
    
    for yaw angles, 0 is aligned with normal angle of surface
    negative is left, positive is right
    """
    if units == 'deg':
        lefteye_yaw = lefteye_yaw * math.pi / 180.0
        lefteye_pitch = lefteye_pitch * math.pi / 180.0
        righteye_yaw = righteye_yaw * math.pi / 180.0
        righteye_pitch = righteye_pitch * math.pi / 180.0
    
    pdb.set_trace()
    
    lefteye_x = tabletDims['width']/2 - eye_dist/2 + face_dist*math.tan(lefteye_yaw)
    righteye_x = tabletDims['width']/2 + eye_dist/2 + face_dist*math.tan(righteye_yaw)
    ave_x = (lefteye_x + righteye_x) / 2
    
    # if the gaze location is outside of screen area
    if ave_x < tabletDims['border']['left'] or ave_x > tabletDims['width'] - tabletDims['border']['right']:
        return None
    
    lefteye_y = face_dist * math.tan(lefteye_pitch)
    righteye_y = face_dist * math.tan(righteye_pitch)
    ave_y = (lefteye_y + righteye_y) / 2
    
    # if the gaze location is outside of screen area
    if ave_y < tabletDims['border']['top'] or ave_y > tabletDims['height'] - tabletDims['border']['bottom']:
        return None
    
    px_x = ( (ave_x - tabletDims['border']['left']) / (tabletDims['width'] - tabletDims['border']['left'] - tabletDims['border']['right']) ) * tabletDims['resolution']['width']
    px_y = ( (ave_y - tabletDims['border']['top']) / (tabletDims['height'] - tabletDims['border']['top'] - tabletDims['border']['bottom']) ) * tabletDims['resolution']['height']
    
    return (px_x, px_y)
    
    
    
if __name__ == '__main__':
    print findGazeLocation(15.0,25.0,12.0,25.0,'deg')

