import cv2
import numpy as np
import time
from typing import List

# Constants
NUM_CAMERAS = 3
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
DARTBOARD_DIAMETER_MM = 451
DOUBLE_RING_OUTER_RADIUS_MM = 170

# Dartboard radii in mm
BULLSEYE_RADIUS_MM = 6.35
OUTER_BULL_RADIUS_MM = 15.9
TRIPLE_RING_INNER_RADIUS_MM = 99
TRIPLE_RING_OUTER_RADIUS_MM = 107
DOUBLE_RING_INNER_RADIUS_MM = 162
DOUBLE_RING_OUTER_RADIUS_MM = 170
PIXELS_PER_MM = IMAGE_HEIGHT / DARTBOARD_DIAMETER_MM
BULLSEYE_RADIUS_PX = int(BULLSEYE_RADIUS_MM * PIXELS_PER_MM)
OUTER_BULL_RADIUS_PX = int(OUTER_BULL_RADIUS_MM * PIXELS_PER_MM)
TRIPLE_RING_INNER_RADIUS_PX = int(TRIPLE_RING_INNER_RADIUS_MM * PIXELS_PER_MM)
TRIPLE_RING_OUTER_RADIUS_PX = int(TRIPLE_RING_OUTER_RADIUS_MM * PIXELS_PER_MM)
DOUBLE_RING_INNER_RADIUS_PX = int(DOUBLE_RING_INNER_RADIUS_MM * PIXELS_PER_MM)
DOUBLE_RING_OUTER_RADIUS_PX = int(DOUBLE_RING_OUTER_RADIUS_MM * PIXELS_PER_MM)

# Global variables
dartboard_image = None
score_images = None
perspective_matrices = []
center = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)


def calibrate():
    global drawn_points
    drawn_points = np.float32([
        [center[0], center[1] - DOUBLE_RING_OUTER_RADIUS_PX],
        [center[0] + DOUBLE_RING_OUTER_RADIUS_PX, center[1]],
        [center[0], center[1] + DOUBLE_RING_OUTER_RADIUS_PX],
        [center[0] - DOUBLE_RING_OUTER_RADIUS_PX, center[1]],
    ])
    
    camera_indexes = [0, 1, 2]
    for camera_index in camera_indexes:
        live_feed_points = calibrate_camera(camera_index)
        if live_feed_points is not None:
            M = cv2.getPerspectiveTransform(drawn_points, live_feed_points)
            perspective_matrices.append(M)
            np.savez(f'camera_calibration_{camera_index}.npz', matrix=M)
        else:
            print(f"Failed to calibrate camera {camera_index}")
            return
        time.sleep(1)

    print("Calibration completed successfully.")


def calibrate_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to capture image from camera {camera_index}")
        cap.release()
        return None
    
    if camera_index in [0, 1]:
        frame = cv2.flip(frame,0)

    if ret:
        window_name = f"Camera {camera_index} - Select 4 Points"
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, frame)
        
        selected_points = []
        cv2.setMouseCallback(window_name, select_points_event, (frame, selected_points, camera_index))
        
        while len(selected_points) < 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        cap.release()

        if len(selected_points) == 4:
            return np.float32(selected_points)
    return None


def select_points_event(event, x, y, flags, param):
    frame, selected_points, camera_index = param
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(f"Camera {camera_index} - Select 4 Points", frame)
        if len(selected_points) == 4:
            cv2.destroyWindow(f"Camera {camera_index} - Select 4 Points")


def main():
    calibrate()


if __name__ == "__main__":
    main()