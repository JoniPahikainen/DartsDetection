import customtkinter as ctk
from PIL import Image
from Detect import detect_dart
import logging
from config import (NUMBER_OF_CAMERAS, CAMERA_INDEXES)
import cv2
import numpy as np
import json
import time


# Image paths for testing
image_paths = [
    "images/camera_0_image.jpg",
    "images/camera_1_image.jpg",
    "images/camera_2_image.jpg"
]


def clear_fields():
    """Clear all images and score fields."""
    for i in range(3):
        image_labels[i].configure(image=None)
        detected_score_vars[i].set("")
        corrected_score_vars[i].set("")


def preload_images():
    """Preload images on startup."""
    for index, file_path in enumerate(image_paths):
        if file_path:
            image = ctk.CTkImage(
                light_image=Image.open(file_path),  # Handle images for light and dark themes
                dark_image=Image.open(file_path),
                size=(200, 200)  # Resize to fit the GUI
            )
            image_labels[index].configure(image=image)
            image_labels[index].image = image
            detected_score_vars[index].set(f"Detected: {50 + index}")  # Example detected score


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


def save_dart_data(dart_group):
    """
    Save a group of dart data, including corrections, to the JSON file.
    """
    timestamp = time.time()
    data = {
        "timestamp": timestamp,
        "dart_group": dart_group
    }
    log_to_json(data)
    print(f"Data saved for timestamp: {timestamp}")

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

def update_gui_with_dart_data(index, dart_data, image_path):
    """
    Updates the GUI with detected dart data and replaces the image.
    """
    # Check if dart_data contains the summary dictionary
    if isinstance(dart_data, list) and len(dart_data) > 1:
        summary_data = dart_data[-1]  # The last element appears to be the summary dictionary
    else:
        summary_data = dart_data  # If not a list, use it directly

    # Extract final score and zone
    detected_score = summary_data.get('detected_score', "N/A")
    detected_zone = summary_data.get('detected_zone', "N/A")

    # Update detected score in the GUI
    detected_score_vars[index].set(
        f"Score: {detected_score} (Zone: {detected_zone})"
    )

    # Update the image in the GUI
    image = ctk.CTkImage(
        light_image=Image.open(image_path),
        dark_image=Image.open(image_path),
        size=(200, 200)
    )
    image_labels[index].configure(image=image)
    image_labels[index].image = image



def collect_and_save_data():
    """
    Collect data from the GUI, apply corrections if provided, and save to JSON.
    """

    dart_group = []
    for i in range(3):
        corrected = True
        detected_score = detected_score_vars[i].get()
        corrected_score = corrected_score_vars[i].get()

        # If correction is empty, use detected score
        if corrected_score.strip() == "":
            corrected_score = detected_score
            corrected = False

        dart_data_for_dart = dart_data[i] if isinstance(dart_data, list) and i < len(dart_data) else {}

        dart_group.append({
            "dart_data": dart_data_for_dart,
            "detected_score": detected_score,
            "corrected_score": corrected_score,
            "corrected": corrected,
            "image": f"camera_{i}_image.jpg"
        })

    save_dart_data(dart_group)
    print("Dart data collected and saved. Ready for next detection.")
    clear_fields()  # Clear fields for the next round

def detection_image(cam_image, locationdart):
    """
    Draws a circle on the provided image at the specified location.
    """
    if cam_image is None:
        return None

    # Flip the image if needed (optional based on camera setup)
    image_with_circle = cv2.flip(cam_image, 0)

    # Draw the circle if coordinates are provided
    if isinstance(locationdart, tuple) and len(locationdart) == 2:
        cv2.circle(image_with_circle, locationdart, 10, (255, 255, 255), 2, 8)
        cv2.circle(image_with_circle, locationdart, 2, (0, 255, 0), 2, 8)

    return image_with_circle



def run_dart_detection():
    global dartboard_image, score_images, perspective_matrices, dart_data  
    dart_data = []
    perspective_matrices = load_perspective_matrices()

    logging.basicConfig(filename='darts_detection_log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    cams = [initialize_camera(index) for index in CAMERA_INDEXES]
    cam_R, cam_L, cam_C = cams

    success, t_R = cam2gray(cam_R, flip=True)
    _, t_L = cam2gray(cam_L, flip=True)
    _, t_C = cam2gray(cam_C, flip=False)

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

    print("Starting dart detection...")

    for i in range(3):  # Detect three darts
        print(f"Detecting dart {i+1}...")
        dart_result, t_R, t_L, t_C = detect_dart(
            cam_R, cam_L, cam_C, t_R, t_L, t_C,
            camera_scores, descriptions,
            kalman_filter_R, kalman_filter_L, kalman_filter_C,
            None, None, None,
            perspective_matrices
        )

        dart_data.append(dart_result)

        if dart_result:
            summary_data = dart_result[-1] if isinstance(dart_result, list) and len(dart_result) > 1 else {}

            # Get coordinates
            x_coordinate = summary_data.get("x_coordinate", "N/A")
            y_coordinate = summary_data.get("y_coordinate", "N/A")
            coords = (x_coordinate, y_coordinate) if isinstance(x_coordinate, int) and isinstance(y_coordinate, int) else None

            # Draw detection point on the image
            processed_image = detection_image(t_L, coords)

            # Save processed image with detection
            image_path = f"images/dart_detection_{i+1}.jpg"
            cv2.imwrite(image_path, processed_image)

            # Log and print coordinates
            logging.info(f"Dart detected for attempt {i+1}: x={x_coordinate}, y={y_coordinate}")
            print(f"Coordinates for dart {i+1}: x={x_coordinate}, y={y_coordinate}")

            # Update the GUI with new data and image
            update_gui_with_dart_data(i, dart_result, image_path)
        else:
            logging.warning(f"No dart detected for attempt {i+1}")
            update_gui_with_dart_data(i, {"detected_score": "N/A", "detected_zone": "N/A"}, image_paths[i])  # Fallback to original image

    print("All darts detected. Waiting for user confirmation.")

    # Cleanup
    cam_R.release()
    cam_L.release()
    cam_C.release()
    cv2.destroyAllWindows()
    logging.info("Dart detection completed.")


# Create the main application window
ctk.set_appearance_mode("Dark")  # Set appearance mode (Dark, Light, System)
ctk.set_default_color_theme("blue")  # Set color theme (blue, green, dark-blue)

root = ctk.CTk()  # Initialize the main window
root.title("Dart Score Detection")  # Set window title
root.geometry("900x600")  # Set window size
root.resizable(False, False)  # Disable resizing

# Frame for content
frame = ctk.CTkFrame(root, corner_radius=10)  # Create a frame within the main window
frame.pack(fill="both", expand=True, padx=20, pady=20)  # Pack the frame with padding

# Image placeholders and fields
image_labels = []
detected_score_vars = []
corrected_score_vars = []

for i in range(3):
    # Image placeholder
    image_frame = ctk.CTkFrame(frame, corner_radius=10)  # Create a subframe for each dart
    image_frame.grid(row=0, column=i, padx=10, pady=10)  # Grid position for subframes

    image_label = ctk.CTkLabel(image_frame, text="www", width=200, height=200, fg_color="gray")
    image_label.grid(row=0, column=0, padx=10, pady=10)  # Placeholder for images
    image_labels.append(image_label)

    # Detected score field
    detected_score_var = ctk.StringVar(value="Detected: ")
    detected_score_label = ctk.CTkLabel(image_frame, textvariable=detected_score_var)
    detected_score_label.grid(row=1, column=0, pady=5)  # Display detected score
    detected_score_vars.append(detected_score_var)

    # Corrected score input
    corrected_score_var = ctk.StringVar()
    corrected_score_entry = ctk.CTkEntry(image_frame, textvariable=corrected_score_var, width=200)
    corrected_score_entry.grid(row=2, column=0, pady=5)  # Input for corrections
    corrected_score_vars.append(corrected_score_var)

# Submit button to save data
submit_button = ctk.CTkButton(frame, text="Submit", command=collect_and_save_data, fg_color="green")
submit_button.grid(row=2, column=1, pady=20)  # Place the submit button

# Start detection button
start_button = ctk.CTkButton(frame, text="Start Detection", command=run_dart_detection, fg_color="blue")
start_button.grid(row=2, column=2, pady=20)  # Place the start detection button

# Preload images on application start
preload_images()

# Run the application
root.mainloop()

