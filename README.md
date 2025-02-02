# Dart Detection System - School Thesis Project

## Overview

This project focuses on detecting darts and calculating scores on a dartboard using a multi-camera setup. The system employs **computer vision techniques** and **perspective transformations** to accurately track and determine dart positions and corresponding scores.

The thesis aims to showcase the integration of programming, machine learning, and real-world applications in the field of Information and Communications Technology (ICT).

---

## Project Status

**Development in Progress – Updates Are Coming Slowly**

This project is actively being developed as part of my thesis. However, updates are currently less frequent as I focus on completing the thesis documentation. Despite the slower pace, improvements and new features will continue to be added over time.

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
- **Libraries:**  
  - `customtkinter==5.2.2` – Modern Tkinter GUI framework  
  - `darkdetect==0.8.0` – Detect system dark mode  
  - `numpy==2.2.1` – Numerical computations  
  - `opencv-contrib-python==4.10.0.84` – Computer vision (with extra modules)  
  - `packaging==24.2` – Package version handling  
  - `pillow==11.1.0` – Image processing  
  - `shapely==2.0.6` – Geometric operations  
  - `json` and `logging` – *(Python built-in modules)*  

- **Hardware:**  
  - 3 cameras  
  - Computer with sufficient processing power for real-time video processing  


---



