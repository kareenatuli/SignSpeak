# preprocess.py

import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

DATA_DIR = 'alphabet_recognition\data'
SAVE_FILE = 'cnn_landmarks.pickle'
NUM_HANDS = 2
LANDMARKS = 21

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)

data = []
labels = []

for label in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir): continue

    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_tensors = []
            for hand in results.multi_hand_landmarks[:NUM_HANDS]:
                x = [lm.x for lm in hand.landmark]
                y = [lm.y for lm in hand.landmark]
                hand_data = [[lm.x - min(x), lm.y - min(y)] for lm in hand.landmark]
                hand_tensors.append(hand_data)

            while len(hand_tensors) < NUM_HANDS:
                hand_tensors.append([[0.0, 0.0]] * LANDMARKS)

            sample = np.array(hand_tensors)
            if sample.shape == (2, 21, 2):
                data.append(sample)
                labels.append(label)

with open(SAVE_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)