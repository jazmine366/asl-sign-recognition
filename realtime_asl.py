import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pyttsx3
import time
from collections import deque, Counter
from sklearn.preprocessing import LabelEncoder

#Config
WINDOW_SIZE = 20
CONF_THRESHOLD = 0.65
WORD_CONF_THRESHOLD = 0.5   #words lower confidence
COOLDOWN = 1.0

STABLE_FRAMES_WORD = 3
STABLE_FRAMES_LETTER = 5
MOTION_THRESHOLD = 0.005

MODEL_PATH = "asl_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Labels 
LETTERS = {"A", "B", "M", "N"}
WORDS = {"Hi", "My", "Name"}
LABELS = sorted(list(LETTERS | WORDS))

print("Using labels:", LABELS)

encoder = LabelEncoder()
encoder.fit(LABELS)

#Model
class ASLModel(nn.Module):
    def __init__(self, input_size=126, hidden_size=128, num_classes=len(LABELS)):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

model = ASLModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Text-to-speech
tts = pyttsx3.init()
tts.setProperty("rate", 160)

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark extraction
def extract_landmarks(results):
    left = np.zeros(63)
    right = np.zeros(63)

    if not results.multi_hand_landmarks:
        return np.concatenate([left, right]), 0

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        coords = np.array(coords)

        if handedness.classification[0].label == "Left":
            left = coords
        else:
            right = coords

    return np.concatenate([left, right]), len(results.multi_hand_landmarks)

# Runtime state
buffer = deque(maxlen=WINDOW_SIZE)
recent_preds = deque(maxlen=5)
sentence = []

last_word = None
last_time = 0
prev_features = None

# Webcam
cap = cv2.VideoCapture(0)
print("Real-time ASL running. Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    features, hand_count = extract_landmarks(results)

    #No hand no guess
    if hand_count == 0:
        buffer.clear()
        recent_preds.clear()
        prev_features = None

        cv2.imshow("ASL Real-Time", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    buffer.append(features)

    #landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    motion = 0.0
    if prev_features is not None:
        motion = np.mean(np.abs(features - prev_features))
    prev_features = features

    #Inference
    if len(buffer) == WINDOW_SIZE:
        x = torch.tensor([buffer], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = np.argmax(probs)
        confidence = probs[idx]
        word = encoder.inverse_transform([idx])[0]

        #Name
        if word == "Name" and hand_count < 2:
            continue

        #Motion gate
        if word in WORDS and motion < MOTION_THRESHOLD:
            continue

        #Confidence gate
        required_conf = WORD_CONF_THRESHOLD if word in WORDS else CONF_THRESHOLD
        if confidence < required_conf:
            continue

        #Stability gate
        recent_preds.append(word)

        needed = STABLE_FRAMES_LETTER if word in LETTERS else STABLE_FRAMES_WORD

        if len(recent_preds) >= needed:
            most_common, count = Counter(recent_preds).most_common(1)[0]
            now = time.time()

            if (
                count >= needed
                and most_common != last_word
                and now - last_time >= COOLDOWN
            ):
                sentence.append(most_common)
                last_word = most_common
                last_time = now
                recent_preds.clear()

                print(f"Recognized: {most_common} | conf={confidence:.2f} motion={motion:.4f}")
                tts.say(most_common)
                tts.runAndWait()

    #Display
    cv2.putText(
        frame,
        " ".join(sentence),
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Real-Time", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
