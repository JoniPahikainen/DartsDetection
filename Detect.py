import numpy as np
import cv2
import time
import numpy as np
import math
from shapely.geometry import Point, LineString, Polygon
import shapely

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
TRIPLE_RING_OUTER_RADIUS_PX = int(TRIPLE_RING_OUTER_RADIUS_MM * PIXELS_PER_MM)
DOUBLE_RING_INNER_RADIUS_PX = int(DOUBLE_RING_INNER_RADIUS_MM * PIXELS_PER_MM)
DOUBLE_RING_OUTER_RADIUS_PX = int(DOUBLE_RING_OUTER_RADIUS_MM * PIXELS_PER_MM)

# Global variables
dartboard_image = None
score_images = None
perspective_matrices = []
center = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)

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


def apply_morphology(img, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening


def getThreshold(cam, t, flip=False):
    success, t_plus = cam2gray(cam, flip=flip)
    if not success:
        return None
    dimg = cv2.absdiff(t, t_plus)
    blur = cv2.GaussianBlur(dimg, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
    return apply_morphology(thresh)


def diff2blur(cam, t, flip=False):
    _, t_plus = cam2gray(cam, flip=flip)
    dimg = cv2.absdiff(t, t_plus)
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(dimg, -1, kernel)
    return t_plus, blur


def getCorners(img_in):
    edges = cv2.goodFeaturesToTrack(img_in, 640, 0.0008, 1, mask=None, blockSize=3, useHarrisDetector=1, k=0.06)
    return np.intp(edges)


def filterCorners(corners, na, original_image=None):
    mean_corners = np.mean(corners, axis=0)
    corners_new = np.array([i for i in corners if abs(mean_corners[0][0] - i[0][0]) <= 180 and abs(mean_corners[0][1] - i[0][1]) <= 120])
    if original_image is not None:
        testimg = original_image.copy()
        for i in corners_new:
            xl, yl = i.ravel()
            cv2.circle(testimg, (xl, yl), 3, (255, 0, 0), -1)
        cv2.imwrite(f"images/corners_marked_{na}.jpg", testimg)
    return corners_new


def filterCornersLine(corners, rows, cols):
    [vx, vy, x, y] = cv2.fitLine(corners, cv2.DIST_HUBER, 0, 0.1, 0.1)
    lefty = int((-x[0] * vy[0] / vx[0]) + y[0])
    righty = int(((cols - x[0]) * vy[0] / vx[0]) + y[0])
    corners_final = np.array([i for i in corners if abs((righty - lefty) * i[0][0] - (cols - 1) * i[0][1] + cols * lefty - righty) / np.sqrt((righty - lefty)**2 + (cols - 1)**2) <= 40])
    return corners_final


def getRealLocation(corners_final, mount, prev_tip_point=None, blur=None, kalman_filter=None):
    loc = np.argmax(corners_final, axis=0)
    locationofdart = corners_final[loc]
    
    # Skeletonize the dart contour
    dart_contour = corners_final.reshape((-1, 1, 2))
    skeleton = cv2.ximgproc.thinning(cv2.drawContours(np.zeros_like(blur), [dart_contour], -1, 255, thickness=cv2.FILLED))
    
    # Detect the dart tip using skeletonization and Kalman filter
    dart_tip = find_dart_tip(skeleton, prev_tip_point, kalman_filter)
    
    if dart_tip is not None:
        tip_x, tip_y = dart_tip
        if blur is not None:
            cv2.circle(blur, (tip_x, tip_y), 1, (0, 255, 0), 1)
        
        locationofdart = dart_tip
    
    return locationofdart, dart_tip


def calculate_score(distance, angle):
    if angle < 0:
        angle += 2 * np.pi
    sector_scores = [10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6]
    sector_index = int(angle / (2 * np.pi) * 20)
    print(f"Sector index: {sector_index}")
    base_score = sector_scores[sector_index]
    if distance <= BULLSEYE_RADIUS_PX:
        return 50
    elif distance <= OUTER_BULL_RADIUS_PX:
        return 25
    elif TRIPLE_RING_INNER_RADIUS_PX < distance <= TRIPLE_RING_OUTER_RADIUS_PX:
        return base_score * 3
    elif DOUBLE_RING_INNER_RADIUS_PX < distance <= DOUBLE_RING_OUTER_RADIUS_PX:
        return base_score * 2
    elif distance <= DOUBLE_RING_OUTER_RADIUS_PX:
        return base_score
    else:
        return 0


def calculate_score_from_coordinates(x, y, camera_index):
    print(f"Calculating score for coordinates ({x}, {y}) from camera {camera_index}...")
    inverse_matrix = cv2.invert(perspective_matrices[camera_index])[1]
    print(f"Inverse matrix: {inverse_matrix}")
    transformed_coords = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), inverse_matrix)[0][0]
    print(f"Transformed coordinates: {transformed_coords}")
    transformed_x, transformed_y = transformed_coords

    print(f"Transformed coordinates: ({transformed_x}, {transformed_y})")
    dx = transformed_x - center[0]
    dy = transformed_y - center[1]
    distance_from_center = math.sqrt(dx**2 + dy**2)
    print(f"Distance from center: {distance_from_center}")
    angle = math.atan2(dy, dx)
    print(f"Angle: {angle}")
    score = calculate_score(distance_from_center, angle)
    return score


def load_perspective_matrices():
    perspective_matrices = []
    for camera_index in range(NUM_CAMERAS):
        try:
            data = np.load(f'camera_calibration_{camera_index}.npz')
            matrix = data['matrix']
            perspective_matrices.append(matrix)
        except FileNotFoundError:
            print(f"Perspective matrix file not found for camera {camera_index}. Please calibrate the cameras first.")
            exit(1)
    return perspective_matrices


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc

        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.B = np.array([[(self.dt**2)/2, 0],
                           [0, (self.dt**2)/2],
                           [self.dt, 0],
                           [0, self.dt]])

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        self.Q = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                           [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                           [(self.dt**3)/2, 0, self.dt**2, 0],
                           [0, (self.dt**3)/2, 0, self.dt**2]]) * self.std_acc**2

        self.R = np.array([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])

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
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        

def find_dart_tip(skeleton, prev_tip_point, kalman_filter):
    # Find the contour of the skeleton
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Find the contour with the maximum area (assuming it represents the dart)
        dart_contour = max(contours, key=cv2.contourArea)
        
        # Convert the contour to a Shapely Polygon
        dart_polygon = Polygon(dart_contour.reshape(-1, 2))
        
        # Find the lowest point of the dart contour
        dart_points = dart_polygon.exterior.coords
        lowest_point = max(dart_points, key=lambda x: x[1])
        
        # Consider the lowest point as the dart tip
        tip_point = lowest_point
        
        # Predict the dart tip position using the Kalman filter
        predicted_tip = kalman_filter.predict()
        
        # Update the Kalman filter with the observed dart tip position
        kalman_filter.update(np.array([[tip_point[0]], [tip_point[1]]]))
        
        return int(tip_point[0]), int(tip_point[1])
    
    return None


def main():
    global dartboard_image, score_images, perspective_matrices

    perspective_matrices = load_perspective_matrices()
    cam_R = initialize_camera(0)
    cam_L = initialize_camera(1)
    cam_C = initialize_camera(2)

    # Read first image twice to start loop
    _, _ = cam2gray(cam_R, flip=True)
    _, _ = cam2gray(cam_L, flip=True)
    _, _ = cam2gray(cam_C, flip=False)
    time.sleep(0.1)
    success, t_R = cam2gray(cam_R, flip=True)
    _, t_L = cam2gray(cam_L, flip=True)
    _, t_C = cam2gray(cam_C, flip=False)

    prev_tip_point_R = None
    prev_tip_point_L = None
    prev_tip_point_C = None
    
    # Initialize Kalman filters for each camera
    dt = 1.0 / 30.0  # Assuming 30 FPS
    u_x = 0
    u_y = 0
    std_acc = 1.0
    x_std_meas = 0.1
    y_std_meas = 0.1
    
    kalman_filter_R = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    kalman_filter_L = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    kalman_filter_C = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    camera_scores = [None] * NUM_CAMERAS  # Initialize camera_scores list
    majority_score = None
    dart_coordinates = None

    takeout_threshold = 20000
    takeout_delay = 1.0

    while success:
        print("success")
        time.sleep(0.1)
        thresh_R = getThreshold(cam_R, t_R, flip=True)
        thresh_L = getThreshold(cam_L, t_L, flip=True)
        thresh_C = getThreshold(cam_C, t_C, flip=False)

        cv2.imshow("Dart Detection - thresh_R", thresh_R)
        cv2.imshow("Dart Detection - thresh_L", thresh_L)
        cv2.imshow("Dart Detection - thresh_C", thresh_C)

        print(f"r: {cv2.countNonZero(thresh_R)}, l: {cv2.countNonZero(thresh_L)}, c: {cv2.countNonZero(thresh_C)},")

        if (cv2.countNonZero(thresh_R) > 500 and cv2.countNonZero(thresh_R) < 7500) or (cv2.countNonZero(thresh_L) > 500 and cv2.countNonZero(thresh_L) < 7500) or (cv2.countNonZero(thresh_C) > 500 and cv2.countNonZero(thresh_C) < 7500):

            count_R = cv2.countNonZero(thresh_R)
            count_L = cv2.countNonZero(thresh_L)
            count_C = cv2.countNonZero(thresh_C)

            cv2.putText(thresh_R, f"Count: {count_R}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2)
            cv2.putText(thresh_L, f"Count: {count_L}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2)
            cv2.putText(thresh_C, f"Count: {count_C}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2)

            time.sleep(0.2)

            t_plus_R, blur_R = diff2blur(cam_R, t_R, True)
            t_plus_L, blur_L = diff2blur(cam_L, t_L, True)
            t_plus_C, blur_C = diff2blur(cam_C, t_C, False)
            cv2.imshow("Dart Detection - blur_R", blur_R)
            cv2.imshow("Dart Detection - blur_L", blur_L)
            cv2.imshow("Dart Detection - blur_C", blur_C)

            corners_R = getCorners(blur_R)
            corners_L = getCorners(blur_L)
            corners_C = getCorners(blur_C)

            if corners_R.size < 40 and corners_L.size < 40 and corners_C.size < 40:
                print("### dart not detected")
                continue

            success_R, ttt_R = cam_R.read()
            success_L, ttt_L = cam_L.read()
            success_C, ttt_C = cam_C.read()
            ttt_R = cv2.flip(ttt_R, 0)
            ttt_L = cv2.flip(ttt_L, 0)

            corners_f_R = filterCorners(corners_R, "r", ttt_R)
            corners_f_L = filterCorners(corners_L, "l", ttt_L)
            corners_f_C = filterCorners(corners_C, "c", ttt_C)

            if corners_f_R.size < 30 and corners_f_L.size < 30 and corners_f_C.size < 30:
                print("### dart not detected")
                continue

            rows, cols = blur_R.shape[:2]
            corners_final_R = filterCornersLine(corners_f_R, rows, cols)
            corners_final_L = filterCornersLine(corners_f_L, rows, cols)
            corners_final_C = filterCornersLine(corners_f_C, rows, cols)

            _, thresh_R = cv2.threshold(blur_R, 60, 255, 0)
            _, thresh_L = cv2.threshold(blur_L, 60, 255, 0)
            _, thresh_C = cv2.threshold(blur_C, 60, 255, 0)

            print(f"thresh r:{cv2.countNonZero(thresh_R)} l:{cv2.countNonZero(thresh_L)} c:{cv2.countNonZero(thresh_C)}")

            if cv2.countNonZero(thresh_R) > 15000 or cv2.countNonZero(thresh_L) > 15000 or cv2.countNonZero(thresh_C) > 15000:
                continue

            print("Dart detected")

            try:
                locationofdart_R, prev_tip_point_R = getRealLocation(corners_final_R, "right", prev_tip_point_R, blur_R, kalman_filter_R)
                locationofdart_L, prev_tip_point_L = getRealLocation(corners_final_L, "left", prev_tip_point_L, blur_L, kalman_filter_L)
                locationofdart_C, prev_tip_point_C = getRealLocation(corners_final_C, "center", prev_tip_point_C, blur_C, kalman_filter_C)

                for camera_index, locationofdart in enumerate([locationofdart_R, locationofdart_L, locationofdart_C]):
                    print("ggaa")
                    if isinstance(locationofdart, tuple) and len(locationofdart) == 2:
                        print(f"sssss: {locationofdart}")
                        x, y = locationofdart
                        print(f"x: {x}")
                        score = calculate_score_from_coordinates(x, y, camera_index)
                        print(f"Camera {camera_index} - Dart Location: {locationofdart}, Score: {score}")

                        # Store the score in the camera_scores list
                        camera_scores[camera_index] = score

                print("Camera111")
                final_score = None
                score_counts = {}
                for score in camera_scores:
                    if score is not None:
                        if score in score_counts:
                            score_counts[score] += 1
                        else:
                            score_counts[score] = 1

                print("Camera12211")
                if score_counts:
                    final_score = max(score_counts, key=score_counts.get)
                    majority_score = final_score

                    # Find the camera with the majority score
                    majority_camera_index = camera_scores.index(final_score)
                    dart_coordinates = (locationofdart_R, locationofdart_L, locationofdart_C)[majority_camera_index]

                    # Transform the dart coordinates to match the drawn dartboard
                    if dart_coordinates is not None:
                        x, y = dart_coordinates
                        inverse_matrix = cv2.invert(perspective_matrices[majority_camera_index])[1]
                        transformed_coords = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), inverse_matrix)[0][0]
                        dart_coordinates = tuple(map(int, transformed_coords))
                print("faasdf")

                if final_score is not None:
                    print(f"Final Score (Majority Rule): {final_score}")
                else:
                    print("No majority score found.")



                success_R, tt_R = cam_R.read()
                success_L, tt_L = cam_L.read()
                success_C, tt_C = cam_C.read()
                tt_R = cv2.flip(tt_R, 0)
                tt_L = cv2.flip(tt_L, 0)

                if not (success_R and success_L and success_C):
                    print("Failed to read one or more camera frames.")
                    continue

                if isinstance(locationofdart_R, tuple) and len(locationofdart_R) == 2:
                    cv2.circle(tt_R, locationofdart_R, 10, (255, 255, 255), 2, 8)
                    cv2.circle(tt_R, locationofdart_R, 2, (0, 255, 0), 2, 8)
                    print(f"Right Camera - Dart Location: {locationofdart_R}")

                if isinstance(locationofdart_L, tuple) and len(locationofdart_L) == 2:
                    cv2.circle(tt_L, locationofdart_L, 10, (255, 255, 255), 2, 8)
                    cv2.circle(tt_L, locationofdart_L, 2, (0, 255, 0), 2, 8)
                    print(f"Left Camera - Dart Location: {locationofdart_L}")

                if isinstance(locationofdart_C, tuple) and len(locationofdart_C) == 2:
                    cv2.circle(tt_C, locationofdart_C, 10, (255, 255, 255), 2, 8)
                    cv2.circle(tt_C, locationofdart_C, 2, (0, 255, 0), 2, 8)
                    print(f"Center Camera - Dart Location: {locationofdart_C}")

            except Exception as e:
                print(f"Something went wrong in finding the dart's location: {str(e)}")
                break
                #continue

            cv2.imshow("Dart Detection - Right", tt_R)
            cv2.imshow("Dart Detection - Left", tt_L)
            cv2.imshow("Dart Detection - Center", tt_C)
            
            if camera_scores[0] is not None:
                cv2.putText(tt_R, f"Score: {camera_scores[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite("images/tt_R.jpg", tt_R)

            if camera_scores[1] is not None:
                cv2.putText(tt_L, f"Score: {camera_scores[1]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite("images/tt_L.jpg", tt_L)

            if camera_scores[2] is not None:
                cv2.putText(tt_C, f"Score: {camera_scores[2]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite("images/tt_C.jpg", tt_C)

            # Update the reference frames after a dart has been detected
            success, t_R = cam2gray(cam_R, flip=True)
            _, t_L = cam2gray(cam_L, flip=True)
            _, t_C = cam2gray(cam_C, flip=False)

        else:
            if cv2.countNonZero(thresh_R) > takeout_threshold or cv2.countNonZero(thresh_L) > takeout_threshold or cv2.countNonZero(thresh_C) > takeout_threshold:
                print("Takeout procedure initiated.")
                # Perform takeout actions here, such as resetting variables or updating the reference frames
                prev_tip_point_R = None
                prev_tip_point_L = None
                prev_tip_point_C = None

                # Wait for the specified delay to allow hand removal
                start_time = time.time()
                while time.time() - start_time < takeout_delay:
                    success, t_R = cam2gray(cam_R)
                    _, t_L = cam2gray(cam_L)
                    _, t_C = cam2gray(cam_C)
                    time.sleep(0.1)

                print("Takeout procedure completed.")

        if cv2.waitKey(10) == 113: #113 == 'q'
            break

    cam_R.release()
    cam_L.release()
    cam_C.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main() 