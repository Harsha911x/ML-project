# sign-language-detector-python

Sign language detector with Python, OpenCV and Mediapipe !



Hand Sign Recognition

This repository contains a hand sign recognition project using MediaPipe Hands, OpenCV, and a Gradient Boosting Classifier. The project involves collecting image data, normalizing hand landmarks, augmenting images with random backgrounds, and training a classifier to recognize hand signs. Additionally, real-time hand sign detection is implemented using a webcam and integrated with text-to-speech for spoken predictions.

The project uses MediaPipe Hands for hand landmark detection and OpenCV for image processing.
Augmentation with random backgrounds is applied to the images to improve robustness.
The trained model is a Gradient Boosting Classifier.
Real-time hand sign detection integrates text-to-speech for spoken predictions.


Steps to Run the Project



    Install Required Packages:
    Ensure you have the necessary libraries installed. You can install them using pip:

pip install opencv-python mediapipe numpy Pillow scikit-learn pyttsx3

1. Collect Images:
Run the collect_imgs.py script to collect hand sign images for different classes. This script will save images to the data directory:


python collect_imgs.py

2. Create Dataset:
Run the create_dataset.py script to process the collected images, normalize landmarks, and save the data to a pickle file:


python create_dataset.py

3. Train Classifier:
Run the train_classifier.py script to train a Gradient Boosting Classifier using the processed dataset and save the trained model to a pickle file:


python train_classifier.py

4. Real-time Hand Sign Detection:
Run the classifier.py script to start real-time hand sign detection using the webcam. This script will load the trained model and use it to predict hand signs:


python classifier.py
