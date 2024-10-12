import numpy as np
import cv2

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

def find_dart_location(image_clear, image_with_dart, kalman_filter=None):
    # Convert both images to grayscale
    gray_clear = cv2.cvtColor(image_clear, cv2.COLOR_BGR2GRAY)
    gray_with_dart = cv2.cvtColor(image_with_dart, cv2.COLOR_BGR2GRAY)

    # Get the difference between the two images
    diff = cv2.absdiff(gray_clear, gray_with_dart)

    # Threshold to get the dart area
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Find contours to locate the dart
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    MIN_CONTOUR_AREA = 50  # Set a minimum contour area (this needs experimentation)
    MAX_CONTOUR_AREA = 500
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.5:  # Assuming narrow aspect ratio is for the steel tip
                dart_center = (x + w // 2, y + h // 2)
                print(f"Dart (likely steel tip) detected at: {dart_center}")
                return dart_center
            else:
                print("Detected contour is likely the flights, not the steel tip.")
    return None

def capture_and_process(camera_index, clear_image_path, kalman_filter):
    # Initialize the camera
    cam = cv2.VideoCapture(camera_index)
    
    # Set camera properties
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Check if the camera opened successfully
    if not cam.isOpened():
        print(f"Failed to open camera {camera_index}")
        return None

    # Load the clear reference image for this camera
    image_clear = cv2.imread(clear_image_path)
    if image_clear is None:
        print(f"Error: Could not load the clear reference image for camera {camera_index}")
        return None

    # Capture a frame
    ret, frame = cam.read()
    if not ret:
        print(f"Failed to capture image from camera {camera_index}")
        cam.release()
        return None

    # Find the dart location
    dart_location = find_dart_location(image_clear, frame, kalman_filter)

    # Annotate the frame if a dart is detected
    if dart_location is not None:
        cv2.circle(frame, dart_location, 10, (0, 255, 0), -1)
        print(f"Dart detected in Camera {camera_index} at: {dart_location}")
    else:
        print(f"No dart detected in Camera {camera_index}")

    # Display the frame
    cv2.imshow(f"Camera {camera_index}", frame)

    # Release the camera
    cam.release()
    return dart_location

def selected_points_event(event, x, y, flags, param):
    frame, selected_points, camera_index = param
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(f"Camera {camera_index} - Select 4 Points", frame)
        if len(selected_points) == 4:
            cv2.destroyWindow(f"Camera {camera_index} - Select 4 Points")

def calibrate(camera):
    image = cv2.VideoCapture(camera)
    ret, frame = image.read()
    if  ret:
        window_name = f"Camera {camera} - Select 4 points"
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, frame)

        selected_points = []
        cv2.setMouseCallback(window_name, selected_points_event, (frame, selected_points, camera))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        image.release()

        if len(selected_points) == 4:
            print(selected_points)
            return np.float32(selected_points)

    return None

def main():

    calibrate(4)
    """
    # Paths to the clear dartboard images for each camera
    clear_image_paths = {
        4: 'images/test/clear_4.jpg',  # Right camera
        6: 'images/test/clear_6.jpg',  # Center camera
        8: 'images/test/clear_8.jpg'   # Left camera
    }

    # Initialize Kalman filters for each camera
    dt = 1.0
    kalman_filters = {
        4: KalmanFilter(dt, 0, 0, 1.0, 0.1, 0.1),
        6: KalmanFilter(dt, 0, 0, 1.0, 0.1, 0.1),
        8: KalmanFilter(dt, 0, 0, 1.0, 0.1, 0.1)
    }

    while True:
        # Process each camera one at a time
        for camera_index in [4, 6, 8]:
            # Call the function to capture and process the camera
            capture_and_process(camera_index, clear_image_paths[camera_index], kalman_filters[camera_index])

            # Wait for a short time before switching to the next camera
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    """

if __name__ == "__main__":
    main()
