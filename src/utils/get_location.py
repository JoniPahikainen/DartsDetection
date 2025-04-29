import cv2
import numpy as np
from shapely.geometry import Polygon

def get_real_location(corners_final, camera_index, prev_tip_point=None, blur=None):
    loc = np.argmax(corners_final, axis=0)
    locationofdart = corners_final[loc]
    
    dart_contour = corners_final.reshape((-1, 1, 2))
    skeleton = cv2.ximgproc.thinning(cv2.drawContours(np.zeros_like(blur), [dart_contour], -1, 255, thickness=cv2.FILLED))
    dart_tip = find_dart_tip(skeleton, prev_tip_point)
    
    if dart_tip is not None:
        locationofdart = dart_tip
    
    return locationofdart, dart_tip

def find_dart_tip(skeleton, prev_tip_point):
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        dart_contour = max(contours, key=cv2.contourArea)
        dart_polygon = Polygon(dart_contour.reshape(-1, 2))
        dart_points = dart_polygon.exterior.coords
        lowest_point = max(dart_points, key=lambda x: x[1])
        tip_point = lowest_point
        return int(tip_point[0]), int(tip_point[1])
    return None