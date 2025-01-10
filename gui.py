import customtkinter as ctk
from Detect import detect_dart
from PIL import Image
import logging
from config import (NUMBER_OF_CAMERAS, CAMERA_INDEXES)
import cv2
import numpy as np
import json
import time
import os


file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)  

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)

image_paths = [
    "images/camera_0_image.jpg",
    "images/camera_1_image.jpg",
    "images/camera_2_image.jpg"
]
cam_R, cam_L, cam_C = None, None, None


def initialize_cameras():
    global cam_R, cam_L, cam_C
    if not cam_R or not cam_L or not cam_C:  
        print("Initializing cameras...")
        cams = [initialize_camera(index) for index in CAMERA_INDEXES]
        cam_R, cam_L, cam_C = cams
        print("Cameras initialized.")
    else:
        logging.debug("Cameras already initialized.")


def initialize_camera(index, width=432, height=432):
    cam = cv2.VideoCapture(index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam if cam.isOpened() else None


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


def save_dart_data(dart_group):
    timestamp = time.time()
    data = {
        "timestamp": timestamp,
        "dart_group": dart_group
    }
    log_to_json(data)


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


def clear_fields():
    for i in range(3):
        image_labels[i].configure(image=None)
        detected_score_vars[i].set("")
        corrected_score_vars[i].set("")
        detected_zone_vars[i].set("")


def create_placeholder_image(file_path):
    img = np.full((200, 200, 3), 211, dtype=np.uint8)
    cv2.putText(img, "No Image", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(file_path, img)


def preload_images():
    for index, file_path in enumerate(image_paths):
        if not os.path.exists(file_path):
            logging.info(f"Creating placeholder image for {file_path}")
            create_placeholder_image(file_path)
        try:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (200, 200))
            img = Image.fromarray(img)

            image = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))

            image_labels[index].configure(image=image)
            image_labels[index].image = image
            detected_score_vars[index].set(f"Detected: {50 + index}")
        except Exception as e:
            logging.error(f"Error loading image {file_path}: {e}")
            detected_score_vars[index].set("Error Loading Image")



def cam2gray(cam, flip=False):
    success, image = cam.read()
    if flip and success:
        image = cv2.flip(image, 0)
    img_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if success else None
    return success, img_g


def run_dart_detection():
    global dartboard_image, score_images, perspective_matrices, dart_data, cam_R, cam_L, cam_C  
    dart_data = []
    
    process_start = time.time()
    
    perspective_start = time.time()
    perspective_matrices = load_perspective_matrices()
    perspective_time = time.time() - perspective_start

    camera_init_start = time.time()
    initialize_cameras()
    camera_init_time = time.time() - camera_init_start
    
    capture_start = time.time()
    success, t_R = cam2gray(cam_R, flip=True)
    _, t_L = cam2gray(cam_L, flip=True)
    _, t_C = cam2gray(cam_C, flip=False)
    capture_time = time.time() - capture_start
    
    kalman_start = time.time()
    dt = 1.0 / 30.0 
    u_x = 0
    u_y = 0
    std_acc = 1.0
    x_std_meas = 0.1
    y_std_meas = 0.1
    
    kalman_filter_R = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    kalman_filter_L = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    kalman_filter_C = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    kalman_time = time.time() - kalman_start

    logging.debug("Init Summary: Perspective Load: %.2fs, Camera Init: %.2fs, Frame Capture: %.2fs, Kalman Init: %.2fs", perspective_time, camera_init_time, capture_time, kalman_time)

    camera_scores = [None] * NUMBER_OF_CAMERAS
    descriptions = [None] * NUMBER_OF_CAMERAS

    

    for i in range(3):  
        logging.info(f"Detecting dart {i+1}...")
        dart_start = time.time()
        
        detect_start = time.time()
        dart_result, t_R, t_L, t_C = detect_dart(
            cam_R, cam_L, cam_C, t_R, t_L, t_C,
            camera_scores, descriptions,
            kalman_filter_R, kalman_filter_L, kalman_filter_C,
            None, None, None,
            perspective_matrices, dart_index=i
        )
        detect_time = time.time() - detect_start
        detected_score = ()

        dart_data.append(dart_result)
        save_time, gui_update_time = 0, 0
        if dart_result:
            save_start = time.time()
            summary_data = dart_result[-1] if isinstance(dart_result, list) and len(dart_result) > 1 else {}
            detected_score = summary_data.get('detected_score', "N/A")
            x_coordinate = summary_data.get("x_coordinate", "N/A")
            y_coordinate = summary_data.get("y_coordinate", "N/A")
            final_camera_index = summary_data.get("final_camera_index", None)
            coords = ((int(x_coordinate), int(y_coordinate)) if isinstance(x_coordinate, int) and isinstance(y_coordinate, int) else None)
            detect_cam = [cam_R, cam_L, cam_C][final_camera_index] if final_camera_index is not None else None
            detect_image = cv2.flip(detect_cam.read()[1], 0) if detect_cam else None
            processed_image = detection_image(detect_image, coords)
            image_path = f"images/dart_detection_{i+1}.jpg"
            cv2.imwrite(image_path, processed_image)
            save_time = time.time() - save_start
            
            logging.debug(f"Dart detected for attempt {i+1}: x={x_coordinate}, y={y_coordinate}")
            gui_update_start = time.time()
            update_gui_with_dart_data(i, dart_result, image_path)
            gui_update_time = time.time() - gui_update_start
        else:
            logging.warning(f"No dart detected for attempt {i+1}")
            update_gui_with_dart_data(i, {"detected_score": "N/A", "detected_zone": "N/A"}, image_paths[i])

        dart_total_time = time.time() - dart_start
        logging.info(f"Dart {i+1} Score: {detected_score} | Detection: {detect_time:.2f}s, Save: {save_time:.2f}s, GUI Update: {gui_update_time:.2f}s, Total: {dart_total_time:.2f}s")
    
    total_process_time = time.time() - process_start
    logging.debug(f"Total Detection Process: {total_process_time:.2f}s")


def detection_image(cam_image, locationdart):
    if cam_image is None:
        return None
    
    image_with_circle = cv2.flip(cam_image, 0)
    
    if isinstance(locationdart, tuple) and len(locationdart) == 2:
        cv2.circle(image_with_circle, locationdart, 10, (255, 255, 255), 2, 8)
        cv2.circle(image_with_circle, locationdart, 2, (0, 255, 0), 2, 8)

    return image_with_circle


def parse_correction_input(correction_input):
    multiplier_mapping = {'S': 1, 'D': 2, 'T': 3}

    if correction_input.isdigit():
        correction_input = f"S{correction_input}"

    if len(correction_input) < 2:
        return None, None

    multiplier = correction_input[0].upper()
    try:
        number = int(correction_input[1:])
    except ValueError:
        return None, None

    if multiplier not in multiplier_mapping or not (1 <= number <= 20) and number != 25:
        return None, None

    score = multiplier_mapping[multiplier] * number
    zone = f"{number} ({'Single' if multiplier == 'S' else 'Double' if multiplier == 'D' else 'Triple'})"

    return score, zone


def update_gui_with_dart_data(index, dart_data, image_path):
    if isinstance(dart_data, list) and len(dart_data) > 1:
        summary_data = dart_data[-1]
    else:
        summary_data = dart_data

    detected_score = summary_data.get('detected_score', "N/A")
    detected_zone = summary_data.get('detected_zone', "N/A")

    detected_score_vars[index].set(detected_score)
    detected_zone_vars[index].set(f"{detected_score} ({detected_zone})")

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    img = Image.fromarray(img)

    image = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))

    image_labels[index].configure(image=image)
    image_labels[index].image = image



def collect_and_save_data():
    dart_group = []
    corrected_darts = []

    for i in range(3):
        corrected = True
        detected_score = detected_score_vars[i].get()
        corrected_input = corrected_score_vars[i].get()
        detected_zone = detected_zone_vars[i].get()
        
        if corrected_input.strip():
            corrected_score, corrected_zone = parse_correction_input(corrected_input)
            if corrected_score is None:
                corrected_score = detected_score
                corrected_zone = detected_zone
                corrected = False
        else:
            corrected_score = detected_score
            corrected_zone = detected_zone
            corrected = False

        if str(corrected_score).strip() == "":
            corrected_score = detected_score
            corrected = False

        dart_data_for_dart = dart_data[i] if isinstance(dart_data, list) and i < len(dart_data) else {}
        corrected_darts.append(corrected)

        dart_group.append({
            "dart_data": dart_data_for_dart,
            "detected_score": int(detected_score),
            "detected_zone": detected_zone,
            "corrected_score": int(corrected_score),
            "corrected_zone": corrected_zone,
            "corrected": corrected,
            "dart_index": i + 1,
        })

    finalized_image_paths = finalize_images(corrected_darts=corrected_darts)
    
    for i in range(len(dart_group)):
        if i < len(finalized_image_paths):
            dart_group[i]["image_path"] = finalized_image_paths[i]
        else:
            dart_group[i]["image_path"] = ""

    save_dart_data(dart_group)
    logging.info("Dart data collected and saved.")

    if stop_after_submit_var.get():
        logging.info("Stopping detection after submit...")
        cleanup_cameras()
    else:
        logging.info("Continuing detection...")
        clear_fields()
        run_dart_detection()


def cleanup_cameras():
    global cam_R, cam_L, cam_C
    if cam_R:
        cam_R.release()
    if cam_L:
        cam_L.release()
    if cam_C:
        cam_C.release()
    cv2.destroyAllWindows()
    logging.info("Cameras released and cleaned up.")


def stop_detection():
    cleanup_cameras()
    logging.info("Detection stopped.")


def finalize_images(temp_dir="images\\temp_images", final_dir="images\\corrected", corrected_darts=None):
    os.makedirs(final_dir, exist_ok=True)
    temp_images = os.listdir(temp_dir)
    finalized_image_paths = [] 

    for filename in temp_images:
        try:
            dart_number = int(filename.split("_")[0].replace("dart", ""))
            temp_path = os.path.join(temp_dir, filename)
            logging.debug(f"Processing file {filename} for dart {dart_number}...")

            if corrected_darts:
                if corrected_darts[dart_number - 1]:
                    timestamp = int(time.time())
                    name, ext = os.path.splitext(filename)
                    final_filename = f"{timestamp}_{name.replace(f'dart{dart_number - 1}', f'dart{dart_number}')}{ext}"
                    final_path = os.path.join(final_dir, final_filename)
                    os.rename(temp_path, final_path)
                    logging.info(f"Finalized image: {final_path}")
                    finalized_image_paths.append(final_path)
                else:
                    os.remove(temp_path)
                    logging.debug(f"Deleted temporary image: {temp_path}")
            else:
                logging.warning(f"Skipping file {filename} due to mismatch with corrected_darts list.")
        except (ValueError, IndexError) as e:
            logging.error(f"Error processing file {filename}: {e}")
    return finalized_image_paths


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


ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("blue")  

root = ctk.CTk()  
root.title("Dart Score Detection")  
root.geometry("900x600")  
root.resizable(False, False)  

frame = ctk.CTkFrame(root, corner_radius=10)  
frame.pack(fill="both", expand=True, padx=20, pady=20)  

image_labels = []
detected_score_vars = [ctk.StringVar(value="Detected: ") for _ in range(len(image_paths))]
corrected_score_vars = [ctk.StringVar() for _ in range(len(image_paths))]
detected_zone_vars = [ctk.StringVar(value="") for _ in range(len(image_paths))]

for i in range(len(image_paths)):
    
    image_frame = ctk.CTkFrame(frame, corner_radius=10)  
    image_frame.grid(row=0, column=i, padx=10, pady=10)  

    image_label = ctk.CTkLabel(image_frame, text="No Image", width=200, height=200, fg_color="gray")
    image_label.grid(row=0, column=0, padx=10, pady=10)  
    image_labels.append(image_label)
    
    detected_score_label = ctk.CTkLabel(image_frame, textvariable=detected_score_vars[i])
    detected_score_label.grid(row=1, column=0, pady=5)  
    
    corrected_score_entry = ctk.CTkEntry(image_frame, textvariable=corrected_score_vars[i], width=200)
    corrected_score_entry.grid(row=2, column=0, pady=5)  

stop_after_submit_var = ctk.BooleanVar(value=False)  
stop_after_submit_checkbox = ctk.CTkCheckBox(
    frame,
    text="Stop After Submit",
    variable=stop_after_submit_var
)
stop_after_submit_checkbox.grid(row=2, column=0, pady=20)  

submit_button = ctk.CTkButton(frame, text="Submit", command=collect_and_save_data, fg_color="green")
submit_button.grid(row=2, column=1, pady=20)  

stop_button = ctk.CTkButton(frame, text="Stop Detection", command=stop_detection, fg_color="red")
stop_button.grid(row=3, column=2, pady=20)  

start_button = ctk.CTkButton(frame, text="Start Detection", command=run_dart_detection, fg_color="blue")
start_button.grid(row=2, column=2, pady=20)  

preload_images()
root.mainloop()