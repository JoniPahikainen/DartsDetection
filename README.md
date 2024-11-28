# Dart Detection System - School Thesis Project

## Overview

This project focuses on detecting darts and calculating scores on a dartboard using a multi-camera setup. The system employs **computer vision techniques**, **Kalman filters**, and **perspective transformations** to accurately track and determine dart positions and corresponding scores.

The thesis aims to showcase the integration of programming, machine learning, and real-world applications in the field of Information and Communications Technology (ICT).

---

## Features

- **Multi-Camera Integration**:
  - Utilizes 3 cameras to capture dartboard activity from different angles.
  - Cameras are calibrated with perspective matrices to ensure precise coordinate transformations.

- **Real-Time Detection**:
  - Processes video frames in real time to detect dart locations and calculate scores.

- **Score Calculation**:
  - Accurately identifies dartboard zones (e.g., Bullseye, Double, Triple) based on distance from the center and dart angle.

- **Error Correction**:
  - Allows manual correction of detected scores to enhance reliability.

- **Data Logging**:
  - Logs dart data, including coordinates, score, and zone, in JSON format for analysis and debugging.

- **Robust Dart Tracking**:
  - Uses Kalman filters to handle motion prediction and noise reduction.

---

## System Requirements

- **Programming Language**: Python 3.8 or later
- **Libraries**:
  - OpenCV
  - NumPy
  - Shapely
  - JSON
  - Logging
- **Hardware**:
  - 3 cameras
  - Computer with adequate processing power for real-time video processing

---

# Setup Instructions

Follow the steps below to set up the project and run the dart detection system.

## 1. Clone the Repository

Clone the repository to your local machine:
```bash
git clone https://github.com/JoniPahikainen/DartsDetection.git
cd dart-detection
```

## 2. Create a Virtual Environment
It’s recommended to use a virtual environment to manage dependencies.

```bash
On Linux/Mac:
  python3 -m venv venv
  source venv/bin/activate

On Windows:
  python -m venv venv
  venv\Scripts\activate
```

## 3. Install Dependencies
  Install the required libraries using the requirements.txt file:

 ```bash
pip install -r requirements.txt
```

## 4. Edit Camera Indexes
To configure the system to use your specific cameras, edit the `CAMERA_INDEXES` in `config.py`:

```python
# config.py
CAMERA_INDEXES = [0, 1, 2]  # Replace these with the indexes of your cameras
```

## 5. Calibrate Cameras
  To ensure accurate detection, calibrate your cameras by running calibrate.py. During the calibration process, select the points corresponding to the white dots shown in the example image below:

 ```bash
python calibrate.py
```
### Example Calibration Image:
<img src="docs/images/dartboard.png" alt="Calibration Example" width="400" />

