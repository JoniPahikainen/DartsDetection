import os
import cv2
import numpy as np
import argparse

image_folder = "images"

if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    print(f"Folder '{image_folder}' created.")
else:
    print(f"Folder '{image_folder}' already exists.")

def test_cameras(save=False):
    open = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} is working, saving frame as camera_{i}_image.jpg")
                open.append(i)
                if save:
                    cv2.imwrite(f'images/test/camera_{i}_image.jpg', frame)  # Save the frame instead of showing it
            else:
                print(f"Camera {i} could not capture a frame.")
        else:
            print(f"Camera {i} is not available.")
        cap.release()

    print(f"Done testing cameras. Open cameras are: {open}")


def test_canny(camera_number, width=1280, height=720):
    camera = cv2.VideoCapture(camera_number)
    if not camera.isOpened():
        print(f"Camera {camera_number} failed to open.")
        return None

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, frame = camera.read()
    if not ret:
        print(f"No ret on camera {camera_number}.")
        return None
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{image_folder}/test/camera_gray_{camera_number}_image.jpg', img_gray)

    # Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    cv2.imwrite(f'{image_folder}/test/camera_blur_{camera_number}_image.jpg', img_blur)

    # Apply Canny Edge Detection with different thresholds
    threshold_values = [(100, 200), (150, 250), (200, 300)]

    for i, (t1, t2) in enumerate(threshold_values):
        edges = cv2.Canny(image=img_blur, threshold1=t1, threshold2=t2)
        cv2.imwrite(f'{image_folder}/test/camera_edges_{camera_number}_t{t1}_t{t2}_image.jpg', edges)
        print(f'Saved edges with thresholds {t1} and {t2}')

    # Apply Canny without blur
    edges_no_blur = cv2.Canny(image=img_gray, threshold1=150, threshold2=250)
    cv2.imwrite(f'{image_folder}/test/camera_edges_no_blur_{camera_number}_image.jpg', edges_no_blur)
    print(f'Saved edges without blur for camera {camera_number}')

    return frame


def test_other(camera_number, width=1280, height=720):
    # Open the camera
    camera = cv2.VideoCapture(camera_number)
    if not camera.isOpened():
        print(f"Camera {camera_number} failed to open.")
        return None

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, frame = camera.read()
    if not ret:
        print(f"No frame captured from camera {camera_number}.")
        return None

    print("Frame captured, starting processing...")

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("Converted to grayscale.")

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    print("Applied Gaussian blur.")

    # Use adaptive thresholding to handle lighting variations
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    print("Applied adaptive thresholding.")

    # Apply a series of morphological transformations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    print("Applied morphological transformations.")

    # Perform edge detection
    edges = cv2.Canny(morph, 50, 150)
    cv2.imshow("Edge Image", edges)
    print("Performed edge detection.")

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours.")

    # Adjust the contour size limits
    min_area = 5000  # Decrease to capture smaller relevant contours
    max_area = 500000  # Increase the max size to include larger contours like dartboard

    ellipse_count = 0  # Counter for ellipse images

    # Process each contour
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Filter based on contour area size
        if min_area < area < max_area:
            print(f"Contour {i}: Area {area} - Processing...")
            if len(cnt) >= 5:  # fitEllipse requires at least 5 points
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    ellipse_count += 1
                    
                    # Make a copy of the frame to draw the ellipse
                    ellipse_img = frame.copy()

                    # Draw the ellipse on the image
                    cv2.ellipse(ellipse_img, ellipse, (255, 0, 0), 2)
                    
                    # Save the image
                    file_name = f'camera_{camera_number}_ellipse_{ellipse_count}.jpg'
                    success = cv2.imwrite(file_name, ellipse_img)

                    if success:
                        print(f"Saved {file_name}")
                    else:
                        print(f"Failed to save {file_name}")

                    # Optionally display the ellipse
                    cv2.imshow(f'Ellipse {ellipse_count}', ellipse_img)
                except Exception as e:
                    print(f"Error fitting ellipse to contour {i}: {e}")
            else:
                print(f"Contour {i} does not have enough points for ellipse fitting.")
        else:
            print(f"Contour {i}: Area {area} - Skipped due to size.")

    # Wait for a key press to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Release the camera
    camera.release()

def save_clear(camera_number, width=1280, height=720):
    camera = cv2.VideoCapture(camera_number)
    if not camera.isOpened():
        print(f"Camera {camera_number} failed to open.")
        return None

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, frame = camera.read()
    if not ret:
        print(f"No frame captured from camera {camera_number}.")
        return None

    print("Frame captured, starting processing...")

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("Converted to grayscale.") 
    cv2.imwrite(f"images/test/clear_{camera_number}.jpg", gray)
    print(f"Saved clear_{camera_number}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darts Detection Tests")
    parser.add_argument("task", choices=["cameras", "canny", "other", "clear"], help="Choose a task to run.")
    parser.add_argument("--save", action="store_true", help="Save images if specified.")  # Add save argument
    
    args = parser.parse_args()

    if args.task == "cameras":
        test_cameras(save=args.save)
    elif args.task == "canny":
        test_canny(4, 1280, 720)
        test_canny(6, 1280, 720)
        test_canny(8, 1280, 720)
    elif args.task == "other":
        test_other(4)
        test_other(6)
        test_other(8)
    elif args.task == "clear":
        save_clear(4, 1280, 720)
        save_clear(6, 1280, 720)
        save_clear(8, 1280, 720)