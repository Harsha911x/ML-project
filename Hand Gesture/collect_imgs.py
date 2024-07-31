import os
import cv2

DATA_DIR = 'data'  # Changed from '/data' to 'data' for local directory
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 5
dataset_size = 200

# Try different camera indices, starting with 0
camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Cannot open camera with index {camera_index}. Trying index 1...")
    camera_index = 1
    cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Cannot open camera with index {camera_index}. Trying index 2...")
    camera_index = 2
    cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Cannot open any camera. Please check the connection and try again.")
    exit()

for j in range(4,number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for the user to be ready
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame. Retrying...")
            continue
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame. Retrying...")
            continue
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
