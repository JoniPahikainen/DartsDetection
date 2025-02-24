import cv2
import numpy as np
import math
import logging
from ..config import (
    NUMBER_OF_CAMERAS, FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS, DARTBOARD_DIAMETER_MM, BULLSEYE_RADIUS_PIXELS, OUTER_BULLSEYE_RADIUS_PIXELS,
    TRIPLE_RING_INNER_RADIUS_PIXELS, TRIPLE_RING_OUTER_RADIUS_PIXELS, DOUBLE_RING_INNER_RADIUS_PIXELS, DOUBLE_RING_OUTER_RADIUS_PIXELS,
    DARTBOARD_CENTER_COORDS, CAMERA_INDEXES
)

def calculate_score_from_coordinates(x, y, camera_index, perspective_matrices):
    inverse_matrix = cv2.invert(perspective_matrices[camera_index])[1]
    transformed_coords = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), inverse_matrix)[0][0]
    transformed_x, transformed_y = map(float, transformed_coords)
    dx = transformed_x - DARTBOARD_CENTER_COORDS[0]
    dy = transformed_y - DARTBOARD_CENTER_COORDS[1]
    distance_from_center = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx)
    score, category = calculate_score(distance_from_center, angle)
    logging.debug(f"Camera {camera_index} -Dart location: ({x}, {y}) Transformed coordinates: ({transformed_x}, {transformed_y}), Distance from center: {distance_from_center}, Angle: {angle}, Score: {score}, Zone: {category}")

    return score, category, {
        "camera_index": camera_index,
        "x": x,
        "y": y,
        "transformed_x": transformed_x,
        "transformed_y": transformed_y,
        "distance_from_center": distance_from_center,
        "angle": angle,
        "detected_score": score,
        "category": category
    }


def calculate_score(distance, angle):
    if angle < 0:
        angle += 2 * np.pi
    sector_scores = [10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6]
    sector_index = int(angle / (2 * np.pi) * 20)
    base_score = sector_scores[sector_index]

    if distance <= BULLSEYE_RADIUS_PIXELS:
        score = 50
        category = 5  # Bullseye
    elif distance <= OUTER_BULLSEYE_RADIUS_PIXELS:
        score = 25
        category = 4  # Outer Bull
    elif TRIPLE_RING_INNER_RADIUS_PIXELS < distance <= TRIPLE_RING_OUTER_RADIUS_PIXELS:
        score = base_score * 3
        category = 3  # Triple
    elif DOUBLE_RING_INNER_RADIUS_PIXELS < distance <= DOUBLE_RING_OUTER_RADIUS_PIXELS:
        score = base_score * 2
        category = 2  # Double
    elif distance <= DOUBLE_RING_OUTER_RADIUS_PIXELS:
        score = base_score
        category = 1  # Single
    else:
        score = 0
        category = 0  # Miss

    return score, category