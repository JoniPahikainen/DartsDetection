import customtkinter as ctk
import cv2
import numpy as np
import os
from PIL import Image
from ..core import logger

class DetectUI(ctk.CTkFrame):
    def __init__(self, parent, ui_manager, functions):
        super().__init__(parent, corner_radius=10)
        self.ui_manager = ui_manager
        self.functions = functions
        self.image_paths = functions['image_paths']
        self.dart_data = []

        # Initialize variables
        self.image_labels = []
        self.detected_score_vars = [ctk.StringVar(value="Detected: ") for _ in range(len(self.image_paths))]
        self.corrected_score_vars = [ctk.StringVar() for _ in range(len(self.image_paths))]
        self.detected_zone_vars = [ctk.StringVar(value="") for _ in range(len(self.image_paths))]
        self.stop_after_submit_var = ctk.BooleanVar(value=False)
        self.submit_count = ctk.IntVar(value=0)
        self.error_label = None

        # Setup UI
        self.setup_ui()
        self.preload_images()

    def setup_ui(self):
        # Navigation buttons
        nav_frame = ctk.CTkFrame(self, corner_radius=10)
        nav_frame.pack(fill="x", padx=20, pady=10)

        main_button = ctk.CTkButton(nav_frame, text="Main Menu", command=lambda: self.ui_manager.show_ui("main"), fg_color="blue")
        main_button.pack(side="left", padx=5)

        calibrate_button = ctk.CTkButton(nav_frame, text="Calibrate", command=lambda: self.ui_manager.show_ui("calibrate"), fg_color="blue")
        calibrate_button.pack(side="left", padx=5)

        detect_button = ctk.CTkButton(nav_frame, text="Detect", command=lambda: self.ui_manager.show_ui("detect"), fg_color="blue", state="disabled")
        detect_button.pack(side="left", padx=5)

        # Error message label
        self.error_label = ctk.CTkLabel(self, text="", text_color="red", font=("Arial", 12))
        self.error_label.pack(pady=5)

        # Image and score display
        image_frame = ctk.CTkFrame(self, corner_radius=10)
        image_frame.pack(fill="both", expand=True, padx=20, pady=20)

        for i in range(len(self.image_paths)):
            cam_frame = ctk.CTkFrame(image_frame, corner_radius=10)
            cam_frame.grid(row=0, column=i, padx=10, pady=10)

            image_label = ctk.CTkLabel(cam_frame, text="No Image", width=200, height=200, fg_color="gray")
            image_label.grid(row=0, column=0, padx=10, pady=10)
            self.image_labels.append(image_label)

            detected_score_label = ctk.CTkLabel(cam_frame, textvariable=self.detected_score_vars[i])
            detected_score_label.grid(row=1, column=0, pady=5)

            corrected_score_entry = ctk.CTkEntry(cam_frame, textvariable=self.corrected_score_vars[i], width=200)
            corrected_score_entry.grid(row=2, column=0, pady=5)

        # Controls
        control_frame = ctk.CTkFrame(self, corner_radius=10)
        control_frame.pack(fill="x", padx=20, pady=20)

        stop_after_submit_checkbox = ctk.CTkCheckBox(
            control_frame, text="Stop After Submit", variable=self.stop_after_submit_var
        )
        stop_after_submit_checkbox.grid(row=0, column=0, pady=20)

        submit_button = ctk.CTkButton(control_frame, text="Submit", command=self.collect_and_save_data, fg_color="green")
        submit_button.grid(row=0, column=1, pady=20)

        submit_count_label = ctk.CTkLabel(control_frame, textvariable=self.submit_count, width=30, font=("Arial", 14))
        submit_count_label.grid(row=0, column=2, pady=20, padx=(0, 10))

        stop_button = ctk.CTkButton(control_frame, text="Stop Detection", command=self.stop_detection, fg_color="red")
        stop_button.grid(row=0, column=3, pady=20)

        start_button = ctk.CTkButton(control_frame, text="Start Detection", command=self.run_dart_detection, fg_color="blue")
        start_button.grid(row=0, column=4, pady=20)

    def preload_images(self):
        for index, file_path in enumerate(self.image_paths):
            if not os.path.exists(file_path):
                logger.info(f"Creating placeholder image for {file_path}")
                self.functions['create_placeholder_image'](file_path)
            try:
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (200, 200))
                img = Image.fromarray(img)
                image = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))
                self.image_labels[index].configure(image=image)
                self.image_labels[index].image = image
                self.detected_score_vars[index].set(f"Detected: {50 + index}")
            except Exception as e:
                logger.error(f"Error loading image {file_path}: {e}")
                self.detected_score_vars[index].set("Error Loading Image")

    def clear_fields(self):
        for i in range(3):
            self.image_labels[i].configure(image=None)
            self.detected_score_vars[i].set("")
            self.corrected_score_vars[i].set("")
            self.detected_zone_vars[i].set("")
        self.error_label.configure(text="")

    def run_dart_detection(self):
        # Check if cameras are initialized
        if not all([self.functions['cam_R'], self.functions['cam_L'], self.functions['cam_C']]):
            self.error_label.configure(text="Error: One or more cameras not initialized. Please check camera connections.")
            logger.error("Dart detection aborted: One or more cameras not initialized")
            return

        perspective_matrices = self.functions['load_perspective_matrices']()
        camera_scores = [None] * self.functions['NUMBER_OF_CAMERAS']
        descriptions = [None] * self.functions['NUMBER_OF_CAMERAS']

        success_R, t_R = self.functions['cam_to_gray'](self.functions['cam_R'], flip=True)
        success_L, t_L = self.functions['cam_to_gray'](self.functions['cam_L'], flip=False)
        success_C, t_C = self.functions['cam_to_gray'](self.functions['cam_C'], flip=True)

        if not all([success_R, success_L, success_C]):
            self.error_label.configure(text="Error: Failed to read from one or more cameras.")
            logger.error("Dart detection aborted: Failed to read from one or more cameras")
            return

        for i in range(3):
            logger.info(f"Detecting dart {i+1}...")
            dart_result, t_R, t_L, t_C = self.functions['detect_dart'](
                self.functions['cam_R'], self.functions['cam_L'], self.functions['cam_C'],
                t_R, t_L, t_C, camera_scores, descriptions, None, None, None,
                perspective_matrices, dart_index=i
            )
            self.dart_data.append(dart_result)
            if dart_result:
                summary_data = dart_result[-1] if isinstance(dart_result, list) and len(dart_result) > 1 else {}
                x_coordinate = summary_data.get("x_coordinate", "N/A")
                y_coordinate = summary_data.get("y_coordinate", "N/A")
                final_camera_index = summary_data.get("final_camera_index", None)
                coords = ((int(x_coordinate), int(y_coordinate)) if isinstance(x_coordinate, (int, float)) and isinstance(y_coordinate, (int, float)) else None)
                detect_cam = [self.functions['cam_R'], self.functions['cam_L'], self.functions['cam_C']][final_camera_index] if final_camera_index is not None else None
                if detect_cam:
                    _, detect_image = detect_cam.read()
                    detect_image = cv2.flip(detect_image, 0) if detect_image is not None else None
                else:
                    detect_image = None
                processed_image = self.functions['detection_image'](detect_image, coords)
                image_path = f"images/dart_detection_{i+1}.jpg"
                if processed_image is not None:
                    cv2.imwrite(image_path, processed_image)
                self.update_gui_with_dart_data(i, dart_result, image_path)
            else:
                logger.warning(f"No dart detected for attempt {i+1}")
                self.update_gui_with_dart_data(i, {"detected_score": "N/A", "detected_zone": "N/A"}, self.image_paths[i])

    def update_gui_with_dart_data(self, index, dart_data, image_path):
        if isinstance(dart_data, list) and len(dart_data) > 1:
            summary_data = dart_data[-1]
        else:
            summary_data = dart_data
        detected_score = summary_data.get('detected_score', "N/A")
        detected_zone = summary_data.get('detected_zone', "N/A")
        self.detected_score_vars[index].set(detected_score)
        self.detected_zone_vars[index].set(f"{detected_score} ({detected_zone})")
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (200, 200))
            img = Image.fromarray(img)
            image = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))
            self.image_labels[index].configure(image=image)
            self.image_labels[index].image = image
        except Exception as e:
            logger.error(f"Error updating GUI with image {image_path}: {e}")
            self.image_labels[index].configure(text="Image Error")

    def parse_correction_input(self, correction_input):
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

    def collect_and_save_data(self):
        self.submit_count.set(self.submit_count.get() + 1)
        dart_group = []
        corrected_darts = []
        for i in range(3):
            corrected = True
            detected_score = self.detected_score_vars[i].get()
            corrected_input = self.corrected_score_vars[i].get()
            detected_zone = self.detected_zone_vars[i].get()
            if corrected_input.strip():
                corrected_score, corrected_zone = self.parse_correction_input(corrected_input)
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
            dart_data_for_dart = self.dart_data[i] if isinstance(self.dart_data, list) and i < len(self.dart_data) else {}
            corrected_darts.append(corrected)
            dart_group.append({
                "dart_data": dart_data_for_dart,
                "detected_score": int(detected_score) if detected_score.replace("Detected: ", "").isdigit() else 0,
                "detected_zone": detected_zone,
                "corrected_score": int(corrected_score) if isinstance(corrected_score, (int, str)) and str(corrected_score).isdigit() else 0,
                "corrected_zone": corrected_zone,
                "corrected": corrected,
                "dart_index": i + 1,
            })
        finalized_image_paths = self.finalize_images(corrected_darts=corrected_darts)
        for i in range(len(dart_group)):
            dart_group[i]["image_path"] = finalized_image_paths[i] if i < len(finalized_image_paths) else ""
        self.functions['save_dart_data'](dart_group)
        logger.info("Dart data collected and saved.")
        if self.stop_after_submit_var.get():
            logger.info("Stopping detection after submit...")
            self.functions['cleanup_cameras']()
        else:
            logger.info("Continuing detection...")
            self.clear_fields()
            self.run_dart_detection()

    def stop_detection(self):
        self.functions['cleanup_cameras']()
        logger.info("Detection stopped.")

    def finalize_images(self, temp_dir="images\\temp_images", final_dir="images\\corrected", corrected_darts=None):
        os.makedirs(final_dir, exist_ok=True)
        temp_images = os.listdir(temp_dir) if os.path.exists(temp_dir) else []
        finalized_image_paths = []
        for filename in temp_images:
            try:
                parts = filename.split("_")
                dart_number = int(parts[0].replace("dart", ""))
                camera_index = parts[1].replace("camera", "")
                image_type = parts[-1].split(".")[0]
                temp_path = os.path.join(temp_dir, filename)
                logger.debug(f"Processing file {filename} for dart {dart_number} ({image_type})...")
                if corrected_darts and dart_number - 1 < len(corrected_darts):
                    if corrected_darts[dart_number - 1]:
                        timestamp = int(time.time())
                        final_filename_base = f"{timestamp}_dart{dart_number}_camera{camera_index}_{image_type}"
                        ext = os.path.splitext(filename)[1]
                        final_filename = f"{final_filename_base}{ext}"
                        final_path = os.path.join(final_dir, final_filename)
                        os.rename(temp_path, final_path)
                        logger.info(f"Finalized {image_type} image: {final_path}")
                        finalized_image_paths.append(final_path)
                    else:
                        os.remove(temp_path)
                        logger.debug(f"Deleted temporary {image_type} image: {temp_path}")
                else:
                    logger.warning(f"Skipping file {filename} due to mismatch with corrected_darts list.")
            except (ValueError, IndexError) as e:
                logger.error(f"Error processing file {filename}: {e}")
        return finalized_image_paths