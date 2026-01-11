# Dart Detection System

## Overview

This project focuses on detecting darts and calculating scores on a dartboard using a multi-camera setup. The system employs computer vision techniques and perspective transformations to accurately track and determine dart positions and corresponding scores.

This project was originally developed as a Bachelor's Thesis for an Information and Communications Technology (ICT) degree, showcasing the integration of programming, machine learning, and real-world applications.

---

## Thesis Publication

The academic phase of this project is complete. The full thesis documentation can be found at the following link:

**Path:** https://www.theseus.fi/handle/10024/885921

---

## Project Status

**Current Status: On Break**

Development is currently on hold. While the project is no longer strictly tied to thesis requirements, it remains an active personal project that will be updated in the future according to the roadmap below.

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

---

## System Requirements

- **Programming Language:** Python 3.8 or later
- **Software Libraries:**
  - `customtkinter==5.2.2` – Modern Tkinter GUI framework
  - `darkdetect==0.8.0` – Detect system dark mode
  - `numpy==2.2.1` – Numerical computations
  - `opencv-contrib-python==4.10.0.84` – Computer vision
  - `packaging==24.2` – Package version handling
  - `pillow==11.1.0` – Image processing
  - `shapely==2.0.6` – Geometric operations
  - `json` and `logging` – (Python built-in modules)

- **Hardware:**
  - 3 cameras
  - Computer with sufficient processing power for real-time video processing

---

## Future Roadmap

### Phase 1: GUI and Game Modes
- Finalize the Graphical User Interface to a production-ready state.
- Implement "Play versus Bot" functionality.
- Add training modules for practice and skill assessment.

### Phase 2: Analytics
- Integration of performance statistics and graphical data visualization.
- Implementation of skill tracking over time.

### Phase 3: Online and Web Integration
- Development of a dedicated website for the system.
- Implementation of online multiplayer capabilities.
