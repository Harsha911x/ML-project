import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Use dynamic image mode for better real-time tracking
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

DATA_DIR = './data'
BACKGROUND_DIR = './backgrounds'

data = []
labels = []

# Function to normalize landmarks with padding
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

# Augment image with random backgrounds
def augment_with_random_background(img, background_dir=BACKGROUND_DIR):
    if not os.path.exists(background_dir):
        print(f"Background directory {background_dir} not found. Skipping augmentation.")
        return img
    height, width, _ = img.shape
    background_files = os.listdir(background_dir)
    if not background_files:
        print(f"No background images found in {background_dir}. Skipping augmentation.")
        return img
    background_path = os.path.join(background_dir, random.choice(background_files))
    background = cv2.imread(background_path)
    background = cv2.resize(background, (width, height))
    alpha = 0.5
    augmented_img = cv2.addWeighted(img, alpha, background, 1 - alpha, 0)
    return augmented_img

# Iterate through each directory in the data directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue
    for img_path in os.listdir(dir_path):
        data_aux = []

        img = cv2.imread(os.path.join(dir_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Augment image with random background
        img_rgb = augment_with_random_background(img_rgb)

        img_height, img_width, _ = img.shape

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                normalized_landmarks = normalize_landmarks(hand_landmarks.landmark, img_width, img_height)
                data_aux.extend(normalized_landmarks)

            data.append(data_aux)
            labels.append(dir_)

# Save the data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data collection complete. Saved to data.pickle.")
