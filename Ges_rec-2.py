# To illustrate, here are some example gestures that MediaPipe's hand gesture recognizer might support:

#     Thumbs Up
#     Thumbs Down
#     Pointing
#     Open Hand
#     Closed Fist
#     Victory Sign
#     OK Sign


import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ensure BaseOptions is defined before use
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'gesture_recognizer.task'
base_options = BaseOptions(model_asset_path=model_path)

# Global variable to store recognized gesture
recognized_gesture = 'Unknown'

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global recognized_gesture
    recognized_gesture = 'Unknown'
    if result.gestures:
        recognized_gesture = result.gestures[0][0].category_name
        print('Detected gestures:', [gesture.category_name for gesture in result.gestures[0]])

options = GestureRecognizerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Initialize the gesture recognizer
recognizer = GestureRecognizer.create_from_options(options)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Track the hands
mphands = mp.solutions.hands

# Access webcam
cap = cv2.VideoCapture(0)
hands = mphands.Hands()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, mark the image as not writeable to pass by reference
    image.flags.writeable = False
    # Process the image and store the results
    results = hands.process(image)

    # Convert the frame to a MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Process the frame for gesture recognition
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    recognizer.recognize_async(mp_image, timestamp)

    # Draw hand landmarks
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mphands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    # Display the recognized gesture on the top right corner
    cv2.putText(image, f'Gesture: {recognized_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Gesture Recognition', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close the recognizer
recognizer.close()
