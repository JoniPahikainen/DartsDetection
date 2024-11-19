# dartboard_utils.py
import numpy as np
import cv2
import math
from config import (FRAME_HEIGHT_PIXELS, FRAME_WIDTH_PIXELS, BULLSEYE_RADIUS_PIXELS, OUTER_BULLSEYE_RADIUS_PIXELS, TRIPLE_RING_INNER_RADIUS_PIXELS, TRIPLE_RING_OUTER_RADIUS_PIXELS, DOUBLE_RING_INNER_RADIUS_PIXELS, DOUBLE_RING_OUTER_RADIUS_PIXELS)


def draw_point_at_angle(image, center, angle_degrees, radius, color, point_radius):
    angle_radians = np.radians(angle_degrees)
    x = int(center[0] + radius * np.cos(angle_radians))
    y = int(center[1] - radius * np.sin(angle_radians))
    cv2.circle(image, (x, y), point_radius, color, -1)


def draw_segment_text(image, center, start_angle, end_angle, radius, text):
    angle = (start_angle + end_angle) / 2
    text_x = int(center[0] + radius * np.cos(angle))
    text_y = int(center[1] + radius * np.sin(angle))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_origin = (text_x - text_size[0] // 2, text_y + text_size[1] // 2)
    cv2.putText(image, text, text_origin, font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)


def draw_dartboard(dartboard_image, IMAGE_HEIGHT, IMAGE_WIDTH, DARTBOARD_DIAMETER_MM, center):
    dartboard_image = np.ones((FRAME_HEIGHT_PIXELS, FRAME_WIDTH_PIXELS, 3), dtype=np.uint8) * 255
    cv2.circle(dartboard_image, center, BULLSEYE_RADIUS_PIXELS, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(dartboard_image, center, OUTER_BULLSEYE_RADIUS_PIXELS, (255, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(dartboard_image, center, TRIPLE_RING_INNER_RADIUS_PIXELS, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(dartboard_image, center, TRIPLE_RING_OUTER_RADIUS_PIXELS, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(dartboard_image, center, DOUBLE_RING_INNER_RADIUS_PIXELS, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(dartboard_image, center, DOUBLE_RING_OUTER_RADIUS_PIXELS, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    for angle in np.linspace(0, 2 * np.pi, 21)[:-1]:
        start_x = int(center[0] + np.cos(angle) * DOUBLE_RING_OUTER_RADIUS_PIXELS)
        start_y = int(center[1] + np.sin(angle) * DOUBLE_RING_OUTER_RADIUS_PIXELS)
        end_x = int(center[0] + np.cos(angle) * OUTER_BULLSEYE_RADIUS_PIXELS)
        end_y = int(center[1] + np.sin(angle) * OUTER_BULLSEYE_RADIUS_PIXELS)
        cv2.line(dartboard_image, (start_x, start_y), (end_x, end_y), (0, 0, 0), 1, lineType=cv2.LINE_AA)

    text_radius_px = int((TRIPLE_RING_OUTER_RADIUS_PIXELS + TRIPLE_RING_INNER_RADIUS_PIXELS) / 2)
    sector_scores = [10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6]
    for i, score in enumerate(sector_scores):
        start_angle = (i * 360 / 20 - 0) * np.pi / 180
        end_angle = ((i + 1) * 360 / 20 - 0) * np.pi / 180
        draw_segment_text(dartboard_image, center, start_angle, end_angle, text_radius_px, str(score))

    return dartboard_image
