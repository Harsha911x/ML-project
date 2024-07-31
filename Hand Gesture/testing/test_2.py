import pickle
import cv2
import numpy as np
import mediapipe as mp

# Load the trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the labels
labels_dict = {0: 'Victory', 1: 'Thumbs Down', 2: 'Thumbs Up', 3: 'Hello', 4: 'I Love You'}  # Update with your actual labels

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

# Function to apply sepia filter
def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(image, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Apply dim effect
    frame = cv2.convertScaleAbs(frame, alpha=0.75, beta=30)

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [landmark for landmark in hand_landmarks.landmark]
            normalized_landmarks = normalize_landmarks(landmarks, W, H)
            data_aux = np.asarray(normalized_landmarks).reshape(1, -1)

            prediction = model.predict(data_aux)
            confidence = model.predict_proba(data_aux).max()
            
            if confidence < confidence_threshold:
                predicted_character = 'Unknown'
            else:
                predicted_label = prediction[0]
                predicted_character = labels_dict[int(predicted_label)]
            
            # Draw the hand landmarks and info
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

    # Apply sepia filter
    sepia_frame = apply_sepia(frame)

    # Apply edge detection to the original frame
    edges = cv2.Canny(frame, 100, 200)

    # Apply a texture effect (simple example using bitwise_and with edges)
    texture = cv2.bitwise_and(sepia_frame, sepia_frame, mask=edges)

    # Stack the texture and original frame horizontally
    combined_frame = np.hstack((frame, texture))

    # Overlay text
    cv2.putText(combined_frame, 'Press "q" to Quit', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Hand Sign Detection', combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
