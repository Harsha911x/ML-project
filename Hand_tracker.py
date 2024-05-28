import cv2
import mediapipe as mp

#drawing inter-connecting lines in landmark
mp_drawing=mp.solutions.drawing_utils
#styling the lines
mp_drawing_style=mp.solutions.drawing_styles

#track the hands
mphands=mp.solutions.hands

#access webcam
cap=cv2.VideoCapture(0)
hands=mphands.Hands()

while True:
    data,image=cap.read()
    #Flip the image
    image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
    #storing the results
    results=hands.process(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,mphands.HAND_CONNECTIONS
            )
    cv2.imshow('Handtracker',image)
    cv2.waitKey(1)
