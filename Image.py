import cv2
import time
import os

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
    cv2.imwrite(f'images/camera_edges_{camera_number}_t{t1}_t{t2}_image.jpg', edges)



frame_4 = edge_gapture(5, 1280, 720)
frame_6 = edge_gapture(7, 1280, 720)
frame_8 = edge_gapture(9, 1280, 720)


"""

def test_this(camera_number):
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(camera_number)
    ret, frame = cap.read()
    if ret:
        print(f"Camera {camera_number} is working, saving frame")
        cv2.imwrite(f'images/camera_{camera_number}.jpg', frame)
    else:
        print(f"Failed to capture from camera {camera_number}")
    cap.release()

test_this(5)
test_this(7)
test_this(9)
"""