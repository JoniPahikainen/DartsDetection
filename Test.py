import os
import cv2
image_folder = "images"

if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    print(f"Folder '{image_folder}' created.")
else:
    print(f"Folder '{image_folder}' already exists.")

def test_cameras():
    for i in range(10):  # Loop through video devices from 0 to 9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} is working, saving frame as camera_{i}_image.jpg")
                cv2.imwrite(f'images/test/camera_{i}_image.jpg', frame)  # Save the frame instead of showing it
            else:
                print(f"Camera {i} could not capture a frame.")
        else:
            print(f"Camera {i} is not available.")
        cap.release()

    print("Done testing cameras.")


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


frame_4 = test_canny(5, 1280, 720)
frame_6 = test_canny(7, 1280, 720)
frame_8 = test_canny(9, 1280, 720)