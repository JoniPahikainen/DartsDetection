import numpy as np
import cv2
import time
import math
from shapely.geometry import Polygon
from .core import logger
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


def initialize_camera(index, width=432, height=432):
    cam = cv2.VideoCapture(index)
    if index == 1:
        cam.set(cv2.CAP_PROP_BRIGHTNESS, 100)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam if cam.isOpened() else None


def load_perspective_matrices():
    perspective_matrices = []
    for camera_index in range(NUMBER_OF_CAMERAS):
        try:
            if camera_index == 1:
                test = 2
            elif camera_index == 2:
                test = 1
            else:
                test = camera_index
            data = np.load(f'camera_calibration_{test}.npz')
            matrix = data['matrix']
            perspective_matrices.append(matrix)
        except FileNotFoundError:
            logger.error(f"Perspective matrix file not found for camera {camera_index}. Please calibrate the cameras first.")
            exit(1)
    return perspective_matrices


def detection_image(cam, flip, locationdart):
    success, t = cam.read()
    if flip:
        t = cv2.flip(t, 0)

    if not success:
        logger.error("Failed to read camera frame.")
        return None
    
    if isinstance(locationdart, tuple) and len(locationdart) == 2:
        cv2.circle(t, locationdart, 10, (255, 255, 255), 2, 8)
        cv2.circle(t, locationdart, 2, (0, 255, 0), 2, 8)
    return t


def save_temporary_image(cam, dart_index, camera_index, image_type="after"):
    ret, frame = cam.read()
    if ret:
        temp_dir = "images\\temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, f"dart{dart_index}_camera{camera_index}_{image_type}.jpg")
        cv2.imwrite(image_path, frame)
        logger.debug(f"Temporary {image_type} image saved: {image_path}")
        return image_path
    else:
        logger.warning(f"Failed to capture {image_type} image for dart {dart_index} from camera {camera_index}")
        return None


def detect_dart(cam_R, cam_L, cam_C, t_R, t_L, t_C, camera_scores, descriptions, prev_tip_point_R, prev_tip_point_L, prev_tip_point_C, perspective_matrices, dart_index):
    dart_data = []
    motion_detected = False

    _ = save_temporary_image(cam_R, dart_index, 0, "before")
    _ = save_temporary_image(cam_L, dart_index, 1, "before")
    _ = save_temporary_image(cam_C, dart_index, 2, "before")

    while not motion_detected:
        time.sleep(0.1)
        thresh_R = process.get_threshold(cam_R, t_R, flip=True)
        thresh_L = process.get_threshold(cam_L, t_L, flip=False)
        thresh_C = process.get_threshold(cam_C, t_C, flip=True)

        if thresh_R is None or thresh_L is None or thresh_C is None:
            logger.warning("Failed to process one or more thresholds.")
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
                corners_final, blur = process.process_camera(cam, t, flip=(camera_index != 1), camera_index=camera_index)

                location, _ = get_location.get_real_location(corners_final, camera_index, None, blur)

                if isinstance(location, tuple) and len(location) == 2:
                    temp_image_path = save_temporary_image(cam, dart_index, camera_index, "after")

                    x, y = location
                    score, category, data = get_score.calculate_score_from_coordinates(x, y, camera_index, perspective_matrices)
                    
                    assert isinstance(data, dict), f"Data is not a dictionary: {data}"
                    assert "detected_score" in data, f"'detected_score' missing in data: {data}"

                    results.append(data)
                    logger.debug(f"Appended data to results: {data}")

                    camera_scores[camera_index] = score  
                    descriptions[camera_index] = category  
                    logger.info(f"Camera {camera_index} - Dart detected at ({x}, {y}). Score: {score}, Zone: {category}")

            if results:
                dart_data.append(results)
                logger.debug(f"Results: {results}")

                
                for res in results:
                    assert isinstance(res, dict), f"Result is not a dictionary: {res}"
                    assert "detected_score" in res, f"Key 'detected_score' missing in result: {res}"

                scores = [res["detected_score"] for res in results]
                logger.debug(f"Scores: {scores}")

                final_score = max(set(scores), key=scores.count)
                logger.debug(f"Final score determined: {final_score}")
                
                final_camera_index = -1  
                transformed_x = None
                transformed_y = None
                
                logger.debug(f"final_score: {final_score}, x: {x}, y: {y}")
                for res in results:
                    if res["detected_score"] == final_score:
                        logger.debug(f"Camera index: {res['camera_index']}")
                        final_camera_index = res["camera_index"]
                        transformed_x = res["transformed_x"]
                        transformed_y = res["transformed_y"]
                        break  

                logger.debug(f"Final camera index determined: {final_camera_index}")

                #EDIT THIS SO IT WILL USE LIKE MULTIBLE OR REMOVE 
                final_description = res["category"]


                logger.debug(f"Final score: {final_score}")

                dart_data.append({
                    "x_coordinate": transformed_x,
                    "y_coordinate": transformed_y,
                    "detected_score": final_score,
                    "detected_catecory": final_description,
                    "final_camera_index": final_camera_index
                })

            _, t_R = process.cam_to_gray(cam_R, flip=True)
            _, t_L = process.cam_to_gray(cam_L, flip=False)
            _, t_C = process.cam_to_gray(cam_C, flip=True)

            logger.debug("Dart detection completed.")
            return dart_data, t_R, t_L, t_C

        except AssertionError as e:
            logger.error(f"Data validation error: {e}")
        except IndexError as e:
            logger.error(f"IndexError in dart detection: {e}. Check list lengths.")
        except Exception as e:
            logger.error(f"Error in dart detection: {e}")

    
    return dart_data, t_R, t_L, t_C


