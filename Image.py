import cv2
import time
import os
import numpy as np

image_folder = "images"
test_folder = os.path.join(image_folder, "test")

if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    print(f"Folder '{image_folder}' created.")
else:
    print(f"Folder '{image_folder}' already exists.")

if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    print(f"Folder '{test_folder}' created inside '{image_folder}'.")
else:
    print(f"Folder '{test_folder}' already exists inside '{image_folder}'.")
    

def edge_gapture(camera_number, width=1280, height=720):
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
    
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    
    t1 = 100
    t2 = 200
    edges = cv2.Canny(image=img_blur, threshold1=t1, threshold2=t2)

    detected_circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 1, maxRadius = 40) 

    cv2.imwrite(f'images/camera_detected_circles_{camera_number}_t{t1}_t{t2}_image.jpg', detected_circles)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for circle in detected_circles[0, :]:
            x, y, radius = circle
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

    cv2.imwrite(f'images/camera_circle_{camera_number}_t{t1}_t{t2}_image.jpg', frame)

def camera_to_gray(camera):
    ret, frame = camera.read()
    if not ret or frame is None:
        print("Error: Couldn't read from the camera.")
        return None, None
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return ret, img_gray

def getThreshold(camera, reference_image, camera_number):
    ret, current_image = camera_to_gray(camera)
    if not ret or current_image is None:
        print("Error: Couldn't capture a valid frame.")
        return None

    clear_board = cv2.imread(f'images/test/clear_{camera_number}.jpg', cv2.COLOR_BGR2GRAY)
    img_diff = cv2.absdiff(clear_board, current_image)
    img_blur = cv2.GaussianBlur(img_diff, (5, 5), 0)
    _, thresh = cv2.threshold(img_blur, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    return opening 

def run(number, width=1280, height=720):
    camera = cv2.VideoCapture(number)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, reference_image = camera_to_gray(camera)
    if not ret or reference_image is None:
        print("Error: Could't capture reference image")
        return
    
    cv2.imwrite(f"images/reference_image_{number}.jpg", reference_image)
    opening = getThreshold(camera, reference_image, number)

    if opening is not None:
        cv2.imwrite(f'images/openong_image_{number}.jpg', opening)
        print("Image saved successfully.")
    else:
        print("Failed to process the threshold.")
    return opening

def main():
    test1 = run(4)
    test2 = run(6)
    test3 = run(8)

    cv2.imshow("4", test1)
    cv2.imshow("6", test2)
    cv2.imshow("8", test3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()