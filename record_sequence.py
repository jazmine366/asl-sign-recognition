import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
from datetime import datetime

#Config
WINDOW_SIZE = 30
FEATURE_SIZE = 126
DATA_DIR = "data"

#MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#Countdown 321 before record
def countdown(frame, seconds=3):
    for i in range(seconds, 0, -1):
        temp = frame.copy()
        cv2.putText(
            temp,
            f"Recording in {i}",
            (200, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4
        )
        cv2.imshow("ASL Data Recorder", temp)
        cv2.waitKey(1000)

#Landmark extraction
def extract_landmarks(results):
    """
    Returns a (126,) vector:
    [left_hand(63), right_hand(63)]
    Missing hands are zero-padded.
    """
    left = np.zeros(63)
    right = np.zeros(63)

    if not results.multi_hand_landmarks:
        return np.concatenate([left, right])

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        coords = np.array(coords)

        label = handedness.classification[0].label
        if label == "Left":
            left = coords
        else:
            right = coords

    return np.concatenate([left, right])

#Save sequence
def save_sequence(sequence, label):
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f.npy")
    path = os.path.join(DATA_DIR, label, filename)
    np.save(path, sequence)
    print(f"[SAVED] {path}")

#Main 
cap = cv2.VideoCapture(0)
buffer = deque(maxlen=WINDOW_SIZE)

print("Press [S] to record a sample")
print("Press [Q] to quit")

LABEL = input("Enter label (word or letter): ").strip()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    features = extract_landmarks(results)
    buffer.append(features)

    #landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(
        frame,
        f"Label: {LABEL}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Data Recorder", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        countdown(frame, 3)

        buffer.clear()

        while len(buffer) < WINDOW_SIZE:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            features = extract_landmarks(results)
            buffer.append(features)

            cv2.putText(
                frame,
                "Recording...",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2
            )

            cv2.imshow("ASL Data Recorder", frame)
            cv2.waitKey(1)

        sequence = np.array(buffer)
        save_sequence(sequence, LABEL)

#exit q
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
