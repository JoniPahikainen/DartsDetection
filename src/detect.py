import numpy as np
import cv2
import time
import math
from shapely.geometry import Polygon
import logging
import json
import os
from .config import (
    NUMBER_OF_CAMERAS, FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS, DARTBOARD_DIAMETER_MM, BULLSEYE_RADIUS_PIXELS, OUTER_BULLSEYE_RADIUS_PIXELS,
    TRIPLE_RING_INNER_RADIUS_PIXELS, TRIPLE_RING_OUTER_RADIUS_PIXELS, DOUBLE_RING_INNER_RADIUS_PIXELS, DOUBLE_RING_OUTER_RADIUS_PIXELS,
    DARTBOARD_CENTER_COORDS, CAMERA_INDEXES
)

from .utils import get_location, process, get_score


#from dartboard_utils import draw_dartboard
#dartboard_image = None
score_images = None
perspective_matrices = []
#dartboard_image = draw_dartboard(dartboard_image, FRAME_HEIGHT_PIXELS, FRAME_WIDTH_PIXELS, DARTBOARD_DIAMETER_MM, DARTBOARD_CENTER_COORDS)


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


def initialize_camera(index, width=432, height=432):
    cam = cv2.VideoCapture(index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam if cam.isOpened() else None


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


def save_temporary_image(cam, dart_index, camera_index):
    ret, frame = cam.read()
    if ret:
        temp_dir = "images\\temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, f"dart{dart_index}_camera{camera_index}.jpg")
        cv2.imwrite(image_path, frame)
        logging.debug(f"Temporary image saved: {image_path}")
        return image_path
    else:
        logging.warning(f"Failed to capture image for dart {dart_index} from camera {camera_index}")
        return None


def detect_dart(cam_R, cam_L, cam_C, t_R, t_L, t_C, camera_scores, descriptions, prev_tip_point_R, prev_tip_point_L, prev_tip_point_C, perspective_matrices, dart_index):
    dart_data = []
    motion_detected = False

    while not motion_detected:  
        time.sleep(0.1)
        thresh_R = process.get_threshold(cam_R, t_R, flip=True)
        thresh_L = process.get_threshold(cam_L, t_L, flip=True)
        thresh_C = process.get_threshold(cam_C, t_C, flip=False)

        if thresh_R is None or thresh_L is None or thresh_C is None:
            logging.warning("Failed to process one or more thresholds.")
            continue

        motion_detected = any(500 < cv2.countNonZero(thresh) < 7500 for thresh in [thresh_R, thresh_L, thresh_C])
        if not motion_detected:
            continue

        try:
            results = []
            for cam, thresh, camera_index, t in zip(
                [cam_R, cam_L, cam_C], [thresh_R, thresh_L, thresh_C],
                [0, 1, 2], [t_R, t_L, t_C]
            ):
                thresh, corners_final, blur = process.process_camera(thresh, cam, t, flip=(camera_index != 2))
                location, _ = get_location.get_real_location(corners_final, camera_index, None, blur)

                if isinstance(location, tuple) and len(location) == 2:
                    temp_image_path = save_temporary_image(cam, dart_index, camera_index)

                    x, y = location
                    score, description, data = get_score.calculate_score_from_coordinates(x, y, camera_index, perspective_matrices)
                    
                    assert isinstance(data, dict), f"Data is not a dictionary: {data}"
                    assert "detected_score" in data, f"'detected_score' missing in data: {data}"

                    results.append(data)
                    logging.debug(f"Appended data to results: {data}")

                    camera_scores[camera_index] = score  
                    descriptions[camera_index] = description  
                    logging.info(f"Camera {camera_index} - Dart detected at ({x}, {y}). Score: {score}, Zone: {description}")

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

            _, t_R = process.cam_to_gray(cam_R, flip=True)
            _, t_L = process.cam_to_gray(cam_L, flip=True)
            _, t_C = process.cam_to_gray(cam_C, flip=False)

            logging.debug("Dart detection completed.")
            return dart_data, t_R, t_L, t_C

        except AssertionError as e:
            logging.error(f"Data validation error: {e}")
        except IndexError as e:
            logging.error(f"IndexError in dart detection: {e}. Check list lengths.")
        except Exception as e:
            logging.error(f"Error in dart detection: {e}")

    
    return dart_data, t_R, t_L, t_C


