import customtkinter as ctk
import cv2
import numpy as np
import time
import os
from PIL import Image
from .detect import detect_dart
from .core import logger, log_to_json
from .config import NUMBER_OF_CAMERAS, CAMERA_INDEXES
from .ui.ui_manager import UIManager

image_paths = [
    "images/camera_0_image.jpg",
    "images/camera_1_image.jpg",
    "images/camera_2_image.jpg"
]

def initialize_cameras():
    cams = [None] * NUMBER_OF_CAMERAS
    print("Initializing cameras...")
    for i, index in enumerate(CAMERA_INDEXES):
        cam = initialize_camera(index)
        if cam is None:
            logger.error(f"Failed to initialize camera at index {index}")
        cams[i] = cam
    print("Cameras initialized.")
    return cams

def initialize_camera(index, width=432, height=432):
    try:
        cam = cv2.VideoCapture(index)
        if not cam.isOpened():
            logger.error(f"Camera at index {index} could not be opened")
            return None
        if index == 1:
            cam.set(cv2.CAP_PROP_BRIGHTNESS, 100)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cam
    except Exception as e:
        logger.error(f"Error initializing camera at index {index}: {e}")
        return None

def save_dart_data(dart_group):
    timestamp = time.time()
    data = {
        "timestamp": timestamp,
        "dart_group": dart_group
    }
    log_to_json(data)

def load_perspective_matrices():
    perspective_matrices = []
    calibration_dir = os.path.join(os.path.dirname(__file__), "../data/calibration")
    calibration_dir = os.path.abspath(calibration_dir)
    for camera_index in range(NUMBER_OF_CAMERAS):
        try:
            file_path = os.path.join(calibration_dir, f'camera_calibration_{camera_index}.npz')
            data = np.load(file_path)
            matrix = data['matrix']
            perspective_matrices.append(matrix)
        except FileNotFoundError:
            logger.error(f"Perspective matrix file not found for camera {camera_index}. Please calibrate the cameras first.")
            exit(1)
    return perspective_matrices

def create_placeholder_image(file_path):
    img = np.full((200, 200, 3), 211, dtype=np.uint8)
    cv2.putText(img, "No Image", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(file_path, img)

def cam_to_gray(cam, flip=False):
    if cam is None or not cam.isOpened():
        logger.error("Camera is not initialized or not opened")
        return False, None
    success, image = cam.read()
    if not success:
        logger.error("Failed to read from camera")
        return False, None
    if flip:
        image = cv2.flip(image, 0)
    img_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return success, img_g

def detection_image(cam_image, locationdart):
    if cam_image is None:
        return None
    image_with_circle = cv2.flip(cam_image, 0)
    if isinstance(locationdart, tuple) and len(locationdart) == 2:
        cv2.circle(image_with_circle, locationdart, 10, (255, 255, 255), 2, 8)
        cv2.circle(image_with_circle, locationdart, 2, (0, 255, 0), 2, 8)
    return image_with_circle

def cleanup_cameras(cams):
    for cam in cams:
        if cam is not None:
            cam.release()
    cv2.destroyAllWindows()
    logger.info("Cameras released and cleaned up.")

def main():
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.title("Dart Score Detection")
    root.geometry("900x600")
    root.resizable(False, False)

    # Initialize cameras
    cams = initialize_cameras()
    cam_R, cam_L, cam_C = cams if len(cams) == 3 else (None, None, None)

    # Initialize UI manager
    ui_manager = UIManager(root, {
        'initialize_cameras': initialize_cameras,
        'load_perspective_matrices': load_perspective_matrices,
        'cam_to_gray': cam_to_gray,
        'detect_dart': detect_dart,
        'detection_image': detection_image,
        'create_placeholder_image': create_placeholder_image,
        'save_dart_data': save_dart_data,
        'cleanup_cameras': lambda: cleanup_cameras(cams),
        'image_paths': image_paths,
        'NUMBER_OF_CAMERAS': NUMBER_OF_CAMERAS,
        'cam_R': cam_R,
        'cam_L': cam_L,
        'cam_C': cam_C
    })

    # Start with the main UI
    ui_manager.show_ui("main")
    root.mainloop()

if __name__ == "__main__":
    main()