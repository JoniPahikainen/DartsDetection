import cv2
import numpy as np
import time
from typing import List
import psutil

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
DOUBLE_RING_INNER_RADIUS_PX = int(DOUBLE_RING_INNER_RADIUS_MM * PIXELS_PER_MM)
DOUBLE_RING_OUTER_RADIUS_PX = int(DOUBLE_RING_OUTER_RADIUS_MM * PIXELS_PER_MM)

# Global variables
dartboard_image = None
score_images = None
perspective_matrices = []
center = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)

def calibrate(camera_index):
    print("Starting calibration for camera", camera_index)  # Debug print
    global drawn_points
    drawn_points = np.float32([
        [center[0], center[1] - DOUBLE_RING_OUTER_RADIUS_PX],
        [center[0] + DOUBLE_RING_OUTER_RADIUS_PX, center[1]],
        [center[0], center[1] + DOUBLE_RING_OUTER_RADIUS_PX],
        [center[0] - DOUBLE_RING_OUTER_RADIUS_PX, center[1]],
    ])
    print("Calibration points set:", drawn_points)  # Debug print

    print(f"Attempting to calibrate camera with index {camera_index}...")

    live_feed_points = calibrate_camera(camera_index)
    if live_feed_points is not None:
        print(f"Live feed points for camera {camera_index}: {live_feed_points}")
        try:
            M = cv2.getPerspectiveTransform(drawn_points, live_feed_points)
            print(f"Perspective matrix calculated for camera {camera_index}: {M}")
            perspective_matrices.append(M)

            # Save the perspective matrix
            M = M.astype(np.float64)
            np.savez(f'perspective_matrix_camera_{camera_index}.npz', matrix=M)
            print(f"Perspective matrix successfully processed for camera {camera_index}")
        except Exception as e:
            print(f"Error during perspective transformation for camera {camera_index}: {e}")
    else:
        print(f"Failed to calibrate camera {camera_index}")

    # Ensure resources are released after each calibration
    cv2.destroyAllWindows()
    time.sleep(2)  # Wait time to allow memory cleanup

def calibrate_camera(camera_index):
    print(f"Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Camera {camera_index} could not be opened.")
        return None

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Failed to capture image from camera {camera_index}")
        cap.release()
        return None
    print(f"Captured frame from camera {camera_index}")

    window_name = f"Camera {camera_index} - Select 4 Points"
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, frame)

    selected_points = []
    cv2.setMouseCallback(window_name, select_points_event, (frame, selected_points, camera_index))
    print(f"Waiting for 4 points to be selected on camera {camera_index}...")

    while len(selected_points) < 4:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Calibration aborted by user.")
            break

    # Release camera and destroy windows safely
    cap.release()
    cv2.destroyAllWindows()

    if len(selected_points) == 4:
        print(f"4 points selected on camera {camera_index}: {selected_points}")
        return np.float32(selected_points)
    print(f"Insufficient points selected on camera {camera_index}")
    return None


def select_points_event(event, x, y, flags, param):
    frame, selected_points, camera_index = param
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(f"Camera {camera_index} - Select 4 Points", frame)
        print(f"Point selected at ({x}, {y}) on camera {camera_index}")  # Debug print
        if len(selected_points) == 4:
            print(f"4 points selected for camera {camera_index}, closing window...")  # Debug print
            cv2.destroyWindow(f"Camera {camera_index} - Select 4 Points")
        print("end of select_points_event")

def main():
    print("Starting main function...")  # Debug print
    calibrate(4)
    time.sleep(5)
    calibrate(6)
    time.sleep(5)
    calibrate(8)
    print("All calibrations completed.")


if __name__ == "__main__":
    main()
