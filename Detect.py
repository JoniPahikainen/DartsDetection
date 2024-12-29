import numpy as np
import cv2
import time
import math
from shapely.geometry import Polygon
import logging
import json
from dartboard_utils import draw_dartboard
from config import (
    NUMBER_OF_CAMERAS, FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS, DARTBOARD_DIAMETER_MM, BULLSEYE_RADIUS_PIXELS, OUTER_BULLSEYE_RADIUS_PIXELS,
    TRIPLE_RING_INNER_RADIUS_PIXELS, TRIPLE_RING_OUTER_RADIUS_PIXELS, DOUBLE_RING_INNER_RADIUS_PIXELS, DOUBLE_RING_OUTER_RADIUS_PIXELS,
    DARTBOARD_CENTER_COORDS, CAMERA_INDEXES
)

dartboard_image = None
score_images = None
perspective_matrices = []
dartboard_image = draw_dartboard(dartboard_image, FRAME_HEIGHT_PIXELS, FRAME_WIDTH_PIXELS, DARTBOARD_DIAMETER_MM, DARTBOARD_CENTER_COORDS)


def setup_json():
    with open('darts_data.json', mode='w') as file:
        json.dump([], file, indent=4)  


def log_to_json(data):
    try:
        
        with open('darts_data.json', mode='r+') as file:
            try:
                file_data = json.load(file)  
            except json.JSONDecodeError:
                logging.warning("JSON file is corrupted or empty. Resetting the file.")
                file_data = []  
            
            file_data.append(data)  
            file.seek(0)           
            json.dump(file_data, file, indent=4)  
    except FileNotFoundError:
        logging.warning("JSON file not found. Creating a new file.")
        setup_json()  
        log_to_json(data)  



def log_dart_data(timestamp, dart_data=None):
    if dart_data:
        
        data = {
            "timestamp": timestamp,
            "dart_group": dart_data  
        }
    else:
        logging.error("Dart data is missing. Please provide the required data.")
    
    log_to_json(data)


def cam2gray(cam, flip=False):
    success, image = cam.read()
    if flip and success:
        image = cv2.flip(image, 0)
    img_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if success else None
    return success, img_g


def initialize_camera(index, width=432, height=432):
    cam = cv2.VideoCapture(index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam if cam.isOpened() else None


def apply_morphology(img, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening


def getThreshold(cam, t, flip=False):
    success, t_plus = cam2gray(cam, flip=flip)
    if not success:
        return None
    dimg = cv2.absdiff(t, t_plus)
    blur = cv2.GaussianBlur(dimg, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
    return apply_morphology(thresh)


def diff2blur(cam, t, flip=False):
    _, t_plus = cam2gray(cam, flip=flip)
    dimg = cv2.absdiff(t, t_plus)
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(dimg, -1, kernel)
    return t_plus, blur


def getCorners(img_in):
    edges = cv2.goodFeaturesToTrack(img_in, 640, 0.0008, 1, mask=None, blockSize=3, useHarrisDetector=1, k=0.06)
    return np.intp(edges)


def filterCorners(corners):
    mean_corners = np.mean(corners, axis=0)
    corners_new = np.array([i for i in corners if abs(mean_corners[0][0] - i[0][0]) <= 180 and abs(mean_corners[0][1] - i[0][1]) <= 120])
    return corners_new


def filterCornersLine(corners, rows, cols):
    [vx, vy, x, y] = cv2.fitLine(corners, cv2.DIST_HUBER, 0, 0.1, 0.1)
    lefty = int((-x[0] * vy[0] / vx[0]) + y[0])
    righty = int(((cols - x[0]) * vy[0] / vx[0]) + y[0])
    corners_final = np.array([i for i in corners if abs((righty - lefty) * i[0][0] - (cols - 1) * i[0][1] + cols * lefty - righty) / np.sqrt((righty - lefty)**2 + (cols - 1)**2) <= 40])
    return corners_final


def getRealLocation(corners_final, mount, prev_tip_point=None, blur=None, kalman_filter=None):
    loc = np.argmax(corners_final, axis=0)
    locationofdart = corners_final[loc]
    
    dart_contour = corners_final.reshape((-1, 1, 2))
    skeleton = cv2.ximgproc.thinning(cv2.drawContours(np.zeros_like(blur), [dart_contour], -1, 255, thickness=cv2.FILLED))
    
    dart_tip = find_dart_tip(skeleton, prev_tip_point, kalman_filter)
    
    if dart_tip is not None:
        tip_x, tip_y = dart_tip
        if blur is not None:
            cv2.circle(blur, (tip_x, tip_y), 1, (0, 255, 0), 1)
        locationofdart = dart_tip
    
    return locationofdart, dart_tip


def calculate_score(distance, angle):
    if angle < 0:
        angle += 2 * np.pi
    sector_scores = [10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6]
    sector_index = int(angle / (2 * np.pi) * 20)
    base_score = sector_scores[sector_index]
    description = ""

    if distance <= BULLSEYE_RADIUS_PIXELS:
        score = 50
        description = "Bullseye"
    elif distance <= OUTER_BULLSEYE_RADIUS_PIXELS:
        score = 25
        description = "Outer Bull"
    elif TRIPLE_RING_INNER_RADIUS_PIXELS < distance <= TRIPLE_RING_OUTER_RADIUS_PIXELS:
        score = base_score * 3
        description = f"{base_score} (Triple)"
    elif DOUBLE_RING_INNER_RADIUS_PIXELS < distance <= DOUBLE_RING_OUTER_RADIUS_PIXELS:
        score = base_score * 2
        description = f"{base_score} (Double)"
    elif distance <= DOUBLE_RING_OUTER_RADIUS_PIXELS:
        score = base_score
        description = str(base_score)
    else:
        score = 0
        description = "Miss"

    return score, description


def calculate_score_from_coordinates(x, y, camera_index, perspective_matrices):
    inverse_matrix = cv2.invert(perspective_matrices[camera_index])[1]
    transformed_coords = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), inverse_matrix)[0][0]
    transformed_x, transformed_y = map(float, transformed_coords)
    dx = transformed_x - DARTBOARD_CENTER_COORDS[0]
    dy = transformed_y - DARTBOARD_CENTER_COORDS[1]
    distance_from_center = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx)
    score, description = calculate_score(distance_from_center, angle)
    logging.debug(f"Camera {camera_index} -Dart location: ({x}, {y}) Transformed coordinates: ({transformed_x}, {transformed_y}), Distance from center: {distance_from_center}, Angle: {angle}, Score: {score}, Zone: {description}")

    return score, description, {
        "camera_index": camera_index,
        "x": x,
        "y": y,
        "transformed_x": transformed_x,
        "transformed_y": transformed_y,
        "distance_from_center": distance_from_center,
        "angle": angle,
        "detected_score": score,
        "zone": description
    }


def load_perspective_matrices():
    perspective_matrices = []
    for camera_index in range(NUMBER_OF_CAMERAS):
        try:
            data = np.load(f'camera_calibration_{camera_index}.npz')
            matrix = data['matrix']
            perspective_matrices.append(matrix)
        except FileNotFoundError:
            logging.error(f"Perspective matrix file not found for camera {camera_index}. Please calibrate the cameras first.")
            exit(1)
    return perspective_matrices


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.A = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = np.array([[(self.dt**2)/2, 0], [0, (self.dt**2)/2], [self.dt, 0], [0, self.dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0], [0, (self.dt**4)/4, 0, (self.dt**3)/2], [(self.dt**3)/2, 0, self.dt**2, 0], [0, (self.dt**3)/2, 0, self.dt**2]]) * self.std_acc**2
        self.R = np.array([[x_std_meas**2, 0], [0, y_std_meas**2]])
        self.P = np.eye(4)
        self.x = np.zeros((4, 1))

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, np.array([[self.u_x], [self.u_y]]))
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.H.shape[1])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)


def find_dart_tip(skeleton, prev_tip_point, kalman_filter):
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        dart_contour = max(contours, key=cv2.contourArea)
        dart_polygon = Polygon(dart_contour.reshape(-1, 2))
        dart_points = dart_polygon.exterior.coords
        lowest_point = max(dart_points, key=lambda x: x[1])
        tip_point = lowest_point
        predicted_tip = kalman_filter.predict()
        kalman_filter.update(np.array([[tip_point[0]], [tip_point[1]]]))
        return int(tip_point[0]), int(tip_point[1])
    return None


def perform_takeout(cams, kalman_filters, takeout_delay=1.0):
    logging.info("Takeout procedure initiated.")

    for kf in kalman_filters:
        kf.x = np.zeros((4, 1))
        kf.P = np.eye(4)

    start_time = time.time()
    while time.time() - start_time < takeout_delay:
        for cam in cams:
            success, _ = cam2gray(cam)
            if not success:
                logging.warning("Camera read failed during takeout.")
        time.sleep(0.1)

    logging.info("Takeout procedure completed.")


def process_camera(thresh, cam, t, flip):
    count = cv2.countNonZero(thresh)
    cv2.putText(thresh, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2)
    time.sleep(0.2)
    _, blur = diff2blur(cam, t, flip)
    
    corners = getCorners(blur)
    corners_f = filterCorners(corners)
    rows, cols = blur.shape[:2]
    corners_final = filterCornersLine(corners_f, rows, cols)
    _, thresh = cv2.threshold(blur, 60, 255, 0)
    return thresh, corners_final, blur


def detection_image(cam, flip, locationdart):
    success, t = cam.read()
    if flip:
        t = cv2.flip(t, 0)

    if not success:
        logging.error("Failed to read camera frame.")
        return None
    
    if isinstance(locationdart, tuple) and len(locationdart) == 2:
        cv2.circle(t, locationdart, 10, (255, 255, 255), 2, 8)
        cv2.circle(t, locationdart, 2, (0, 255, 0), 2, 8)
    return t


def detect_dart(cam_R, cam_L, cam_C, t_R, t_L, t_C, camera_scores, descriptions, kalman_filter_R, kalman_filter_L, kalman_filter_C, prev_tip_point_R, prev_tip_point_L, prev_tip_point_C, perspective_matrices):
    dart_data = []
    motion_detected = False

    while not motion_detected:  
        time.sleep(0.1)
        thresh_R = getThreshold(cam_R, t_R, flip=True)
        thresh_L = getThreshold(cam_L, t_L, flip=True)
        thresh_C = getThreshold(cam_C, t_C, flip=False)

        if thresh_R is None or thresh_L is None or thresh_C is None:
            logging.warning("Failed to process one or more thresholds.")
            continue

        motion_detected = any(500 < cv2.countNonZero(thresh) < 7500 for thresh in [thresh_R, thresh_L, thresh_C])
        if not motion_detected:
            continue

        try:
            results = []
            for cam, thresh, kalman_filter, mount, t in zip(
                [cam_R, cam_L, cam_C], [thresh_R, thresh_L, thresh_C],
                [kalman_filter_R, kalman_filter_L, kalman_filter_C],
                [0, 1, 2], [t_R, t_L, t_C]
            ):
                thresh, corners_final, blur = process_camera(thresh, cam, t, flip=(mount != 2))
                location, _ = getRealLocation(corners_final, mount, None, blur, kalman_filter)

                if isinstance(location, tuple) and len(location) == 2:
                    x, y = location
                    score, description, data = calculate_score_from_coordinates(x, y, mount, perspective_matrices)

                    
                    assert isinstance(data, dict), f"Data is not a dictionary: {data}"
                    assert "detected_score" in data, f"'detected_score' missing in data: {data}"

                    results.append(data)
                    logging.debug(f"Appended data to results: {data}")

                    camera_scores[mount] = score  
                    descriptions[mount] = description  
                    logging.info(f"Camera {mount} - Dart detected at ({x}, {y}). Score: {score}, Zone: {description}")

            if results:
                dart_data.append(results)
                logging.debug(f"Results: {results}")

                
                for res in results:
                    assert isinstance(res, dict), f"Result is not a dictionary: {res}"
                    assert "detected_score" in res, f"Key 'detected_score' missing in result: {res}"

                scores = [res["detected_score"] for res in results]
                logging.debug(f"Scores: {scores}")

                final_score = max(set(scores), key=scores.count)
                logging.debug(f"Final score determined: {final_score}")
                
                """
                majority_camera_index = camera_scores.index(final_score)
                final_description = descriptions[majority_camera_index]
                """

                final_camera_index = -1  
                transformed_x = None
                transformed_y = None
                
                logging.debug(f"final_score: {final_score}, x: {x}, y: {y}")
                for res in results:
                    if res["detected_score"] == final_score:
                        logging.debug(f"Camera index: {res['camera_index']}")
                        final_camera_index = res["camera_index"]
                        transformed_x = res["transformed_x"]
                        transformed_y = res["transformed_y"]
                        break  

                logging.debug(f"Final camera index determined: {final_camera_index}")

                final_description = next(
                    (res["zone"] for res in results if res["detected_score"] == final_score),
                    "Unknown"
                )

                logging.debug(f"Final score: {final_score}")

                dart_data.append({
                    "x_coordinate": transformed_x,
                    "y_coordinate": transformed_y,
                    "detected_score": final_score,
                    "detected_zone": final_description,
                    "final_camera_index": final_camera_index
                })

            _, t_R = cam2gray(cam_R, flip=True)
            _, t_L = cam2gray(cam_L, flip=True)
            _, t_C = cam2gray(cam_C, flip=False)

            logging.debug("Dart detection completed.")
            return dart_data, t_R, t_L, t_C

        except AssertionError as e:
            logging.error(f"Data validation error: {e}")
        except IndexError as e:
            logging.error(f"IndexError in dart detection: {e}. Check list lengths.")
        except Exception as e:
            logging.error(f"Error in dart detection: {e}")

    
    return dart_data, t_R, t_L, t_C


