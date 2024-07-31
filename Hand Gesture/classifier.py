import pickle
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import time

# Load the trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Define the labels
labels_dict = {0: 'Victory', 1: 'Thumbs Down', 2: 'Thumbs Up',3: 'Hello',4: 'All Ok'}  # Update with your actual labels

# Define the confidence threshold
confidence_threshold = 0.98  # Set this to your desired threshold

# Function to normalize landmarks
def normalize_landmarks(landmarks, img_width, img_height):
    x_coords = np.array([landmark.x * img_width for landmark in landmarks])
    y_coords = np.array([landmark.y * img_height for landmark in landmarks])
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    max_dist = np.max(np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2))
    
    normalized_landmarks = []
    for landmark in landmarks:
        normalized_landmarks.append((landmark.x * img_width - center_x) / max_dist)
        normalized_landmarks.append((landmark.y * img_height - center_y) / max_dist)
    
    return normalized_landmarks

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Variable to keep track of the last spoken prediction
last_prediction = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [landmark for landmark in hand_landmarks.landmark]
            normalized_landmarks = normalize_landmarks(landmarks, W, H)
            data_aux = np.asarray(normalized_landmarks).reshape(1, -1)

            #return label after prediction
            prediction = model.predict(data_aux)
            confidence = model.predict_proba(data_aux).max()
            
            # Debug: Print prediction and confidence
            print(f'Prediction: {prediction[0]}, Confidence: {confidence}')
            
            if confidence < confidence_threshold:
                predicted_character = 'Unknown'
            else:
                predicted_label = prediction[0]
                predicted_character = labels_dict[int(predicted_label)]
            
            print(f'Predicted Character: {predicted_character}')

            # Speak the prediction if it is different from the last spoken prediction
            # if last_prediction != predicted_character:
            #     engine.say(predicted_character)
            #     engine.runAndWait()
            #     last_prediction = predicted_character

            # Draw the hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            x_coords = [int(lm.x * W) for lm in landmarks]
            y_coords = [int(lm.y * H) for lm in landmarks]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)

            # Draw bounding box and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f'{predicted_character} ({confidence:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        

    cv2.imshow('Hand Sign Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
