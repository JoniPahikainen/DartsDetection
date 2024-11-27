import numpy as np
import cv2
import time
import numpy as np
import math
from shapely.geometry import Polygon
import logging
import csv
import json
from dartboard_utils import draw_dartboard
from config import (
    NUMBER_OF_CAMERAS, FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS, DARTBOARD_DIAMETER_MM, BULLSEYE_RADIUS_PIXELS, OUTER_BULLSEYE_RADIUS_PIXELS,
    TRIPLE_RING_INNER_RADIUS_PIXELS, TRIPLE_RING_OUTER_RADIUS_PIXELS, DOUBLE_RING_INNER_RADIUS_PIXELS, DOUBLE_RING_OUTER_RADIUS_PIXELS,
    DARTBOARD_CENTER_COORDS
)

dartboard_image = None
score_images = None
perspective_matrices = []
dartboard_image = draw_dartboard(dartboard_image, FRAME_HEIGHT_PIXELS, FRAME_WIDTH_PIXELS, DARTBOARD_DIAMETER_MM, DARTBOARD_CENTER_COORDS)

def setup_json():
    with open('darts_data.json', mode='w') as file:
        json.dump([], file, indent=4)  # Initialize with an empty list

# Function to log data to the JSON file
def log_to_json(data):
    try:
        # Open the JSON file in read+write mode
        with open('darts_data.json', mode='r+') as file:
            try:
                file_data = json.load(file)  # Load existing data
            except json.JSONDecodeError:
                logging.warning("JSON file is corrupted or empty. Resetting the file.")
                file_data = []  # Reset to an empty list if decoding fails
            
            file_data.append(data)  # Append new data
            file.seek(0)           # Move file pointer to the beginning
            json.dump(file_data, file, indent=4)  # Write updated data back to the file
    except FileNotFoundError:
        logging.warning("JSON file not found. Creating a new file.")
        setup_json()  # Initialize a new JSON file
        log_to_json(data)  # Retry logging the data


# Function to log dart data
def log_dart_data(timestamp, dart_data=None):
    if dart_data:
        # If grouped data is provided, log as a list
        data = {
            "timestamp": timestamp,
            "dart_group": dart_data  # Save all collected data for this detection
        }
    else:
        logging.error("Dart data is missing. Please provide the required data.")
    
    # Save to JSON
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


def calculate_score_from_coordinates(x, y, camera_index):
    dd = []
    inverse_matrix = cv2.invert(perspective_matrices[camera_index])[1]
    transformed_coords = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), inverse_matrix)[0][0]
    transformed_x, transformed_y = map(float, transformed_coords)
    dx = transformed_x - DARTBOARD_CENTER_COORDS[0]
    dy = transformed_y - DARTBOARD_CENTER_COORDS[1]
    distance_from_center = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx)

    score, description = calculate_score(distance_from_center, angle)
    logging.debug(f"Camera {camera_index} -Dart location: ({x}, {y}) Transformed coordinates: ({transformed_x}, {transformed_y}), Distance from center: {distance_from_center}, Angle: {angle}, Score: {score}, Zone: {description}")
    

    dd.append({
            "camera_index": camera_index,
            "x": x,
            "y": y,
            "transformed_x": transformed_x,
            "transformed_y": transformed_y,
            "distance_from_center": distance_from_center,
            "angle": angle,
            "detected_score": score,
            "zone": description
        })

    return score, description, dd


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
    print("Takeout procedure initiated.")

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
    print("Takeout procedure completed.")


def correct_score(detected_score, detected_description):
    print(f"Detected Score: {detected_score} ({detected_description})")
    correction = input("Enter corrected score (e.g., D20 for Double 20, S20 for Single 20, or press Enter to keep detected score): ").strip()
    
    if correction:
        try:
            if correction.lower() == "miss":
                corrected_score = 0
                corrected_description = "Miss"
            elif correction.startswith("D"):
                corrected_score = int(correction[1:]) * 2
                corrected_description = f"{correction[1:]} (Double)"
            elif correction.startswith("T"):
                corrected_score = int(correction[1:]) * 3
                corrected_description = f"{correction[1:]} (Triple)"
            elif correction.startswith("S"):
                corrected_score = int(correction[1:])
                corrected_description = correction[1:]
            else:
                raise ValueError("Invalid correction format.")
            
            return corrected_score, corrected_description, True  # Corrected
        except Exception as e:
            print(f"Error in correction: {e}")
            return detected_score, detected_description, False  # Not corrected
    else:
        return detected_score, detected_description, None  # No correction needed


def proses_camera(thresh, cam, t, flip):
    count = cv2.countNonZero(thresh)
    cv2.putText(thresh, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2)
    time.sleep(0.2)
    _, blur = diff2blur(cam, t, flip)
    #cv2.imshow("Dart Detection - blur", blur)
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


def main():
    global dartboard_image, score_images, perspective_matrices
    dart_data = []
    logging.basicConfig(filename='darts_detection_log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    perspective_matrices = load_perspective_matrices()
    cam_R = initialize_camera(0)
    cam_L = initialize_camera(1)
    cam_C = initialize_camera(2)


    success, t_R = cam2gray(cam_R, flip=True)
    _, t_L = cam2gray(cam_L, flip=True)
    _, t_C = cam2gray(cam_C, flip=False)

    prev_tip_point_R = None
    prev_tip_point_L = None
    prev_tip_point_C = None
    
    dt = 1.0 / 30.0 
    u_x = 0
    u_y = 0
    std_acc = 1.0
    x_std_meas = 0.1
    y_std_meas = 0.1
    
    kalman_filter_R = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    kalman_filter_L = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    kalman_filter_C = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    camera_scores = [None] * NUMBER_OF_CAMERAS
    descriptions = [None] * NUMBER_OF_CAMERAS
    dart_coordinates = None

    takeout_threshold = 20000
    takeout_delay = 1.0

    dartboard_image_copy = dartboard_image.copy()

    print("Starting dart detection...")
    while success:
        time.sleep(0.1)
        thresh_R = getThreshold(cam_R, t_R, flip=True)
        thresh_L = getThreshold(cam_L, t_L, flip=True)
        thresh_C = getThreshold(cam_C, t_C, flip=False)
        
        if (cv2.countNonZero(thresh_R) > 500 and cv2.countNonZero(thresh_R) < 7500) or (cv2.countNonZero(thresh_L) > 500 and cv2.countNonZero(thresh_L) < 7500) or (cv2.countNonZero(thresh_C) > 500 and cv2.countNonZero(thresh_C) < 7500):
            thresh_R, corners_final_R, blur_R = proses_camera(thresh_R, cam_R, t_R, True)
            thresh_L, corners_final_L, blur_L = proses_camera(thresh_L, cam_L, t_L, True)
            thresh_C, corners_final_C, blur_C = proses_camera(thresh_C, cam_C, t_C, False)

            logging.info(f"New frame processed, thresholds - R: {cv2.countNonZero(thresh_R)} L: {cv2.countNonZero(thresh_L)} C: {cv2.countNonZero(thresh_C)}")

            if cv2.countNonZero(thresh_R) > 15000 or cv2.countNonZero(thresh_L) > 15000 or cv2.countNonZero(thresh_C) > 15000:
                continue

            try:
                locationofdart_R, prev_tip_point_R = getRealLocation(corners_final_R, "right", prev_tip_point_R, blur_R, kalman_filter_R)
                locationofdart_L, prev_tip_point_L = getRealLocation(corners_final_L, "left", prev_tip_point_L, blur_L, kalman_filter_L)
                locationofdart_C, prev_tip_point_C = getRealLocation(corners_final_C, "center", prev_tip_point_C, blur_C, kalman_filter_C)

                for camera_index, locationofdart in enumerate([locationofdart_R, locationofdart_L, locationofdart_C]):
                    if isinstance(locationofdart, tuple) and len(locationofdart) == 2:
                        x, y = locationofdart
                        score, description, data = calculate_score_from_coordinates(x, y, camera_index)
                        dart_data.append(data)

                        camera_scores[camera_index] = score
                        descriptions[camera_index] = description

                final_score = None
                score_counts = {}
                for score in camera_scores:
                    if score is not None:
                        if score in score_counts:
                            score_counts[score] += 1
                        else:
                            score_counts[score] = 1
                if score_counts:
                    final_score = max(score_counts, key=score_counts.get)

                    majority_camera_index = camera_scores.index(final_score)
                    final_description = descriptions[majority_camera_index]
                    dart_coordinates = (locationofdart_R, locationofdart_L, locationofdart_C)[majority_camera_index]

                    if dart_coordinates is not None:
                        x, y = dart_coordinates
                        logging.info(f'Dart detected at coordinates: {x}, {y} with score: {final_score}')
                        inverse_matrix = cv2.invert(perspective_matrices[majority_camera_index])[1]
                        transformed_coords = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), inverse_matrix)[0][0]
                        dart_coordinates = tuple(map(int, transformed_coords))

                if final_score is not None:
                    logging.info(f"Final Score (Majority Rule): {final_score} ({final_description})")
                    print(f"Final Score: {final_score} ({final_description})")

                    corrected_score, corrected_description, corrected = correct_score(final_score, final_description)
                    x, y = dart_coordinates

                    if corrected is not None:
                        correction_status = "Corrected" if corrected else "Not Corrected"
                        logging.info(f"Score correction: {correction_status}. Final Score: {corrected_score} ({corrected_description})")
                        print(f"Final Score: {corrected_score} ({corrected_description})")
                        cv2.circle(dartboard_image_copy, (int(x), int(y)), 5, (0, 0, 255), -1)
                        cv2.imwrite("images/dartboard_image_copy.jpg", dartboard_image_copy)

                    else:
                        cv2.circle(dartboard_image_copy, (int(x), int(y)), 5, (205, 90, 106), -1)
                        cv2.imwrite("images/dartboard_image_copy.jpg", dartboard_image_copy)

                    dart_data.append({
                        "x_coordinate": x,
                        "y_coordinate": y,
                        "detected_score": final_score,
                        "detected_zone": final_description,
                        "corrected": corrected,
                        "corrected_score": (corrected_score if corrected else score),
                        "corrected_zone": (corrected_description if corrected else description)
                    })

                    log_dart_data(time.time(), dart_data)
                    dart_data.clear()

                else:
                    logging.info("No majority score detected.")

                tt_R = detection_image(cam_R, True, locationofdart_R)
                tt_L = detection_image(cam_L, True, locationofdart_L)
                tt_C = detection_image(cam_C, False, locationofdart_C)

            except Exception as e:
                logging.error(f"Error processing frame: {str(e)}")
                continue

            """
            cv2.imshow("Dart Detection - Right", tt_R)
            cv2.imshow("Dart Detection - Left", tt_L)
            cv2.imshow("Dart Detection - Center", tt_C)
            """

            if camera_scores[0] is not None and camera_scores[1] is not None and camera_scores[2] is not None:
                cv2.putText(tt_R, f"Score: {camera_scores[0]} ({descriptions[0]})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2)
                cv2.putText(tt_L, f"Score: {camera_scores[1]} ({descriptions[1]})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2)
                cv2.putText(tt_C, f"Score: {camera_scores[2]} ({descriptions[2]})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2)
                
                cv2.imwrite("images/dart_detection_R.jpg", tt_R)
                cv2.imwrite("images/dart_detection_L.jpg", tt_L)
                cv2.imwrite("images/dart_detection_C.jpg", tt_C)
            
            success, t_R = cam2gray(cam_R, flip=True)
            _, t_L = cam2gray(cam_L, flip=True)
            _, t_C = cam2gray(cam_C, flip=False)

        else:
            if cv2.countNonZero(thresh_R) > takeout_threshold or cv2.countNonZero(thresh_L) > takeout_threshold or cv2.countNonZero(thresh_C) > takeout_threshold:
                perform_takeout([cam_R, cam_L, cam_C], [kalman_filter_R, kalman_filter_L, kalman_filter_C], takeout_delay)

                success, t_R = cam2gray(cam_R, flip=True)
                _, t_L = cam2gray(cam_L, flip=True)
                _, t_C = cam2gray(cam_C, flip=False)

        if cv2.waitKey(10) == 113:
            break

    cam_R.release()
    cam_L.release()
    cam_C.release()
    cv2.destroyAllWindows()
    logging.info("Dart detection completed.")

if __name__ == "__main__":
    main() 