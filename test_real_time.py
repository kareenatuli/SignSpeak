# test_realtime_buffered.py

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
from collections import deque, Counter
import pyttsx3

inference_times = []


# Load model and label encoder
#model = tf.keras.models.load_model('isl_landmark_cnn.h5')
model = tf.keras.models.load_model('isl_landmark_cnn_grok.keras')

with open('label_encoder.pickle', 'rb') as f:
    le = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5)

# Configurations
CONFIDENCE_THRESHOLD = 0.85
LETTER_DELAY = 2.0  # seconds to wait before accepting next letter
BUFFER_SIZE = 15    # for smoothing predictions
SPACE_HOLD_TIME = 2.0  # how long to hold "no hands" for space
BACKSPACE_HOLD_TIME = 2.0  # how long to hold "fist" for backspace

tts = pyttsx3.init()

# State
sentence = ""
last_time = time.time()
prediction_buffer = deque(maxlen=BUFFER_SIZE)
space_timer = None
backspace_timer = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and sentence.strip() != "":
        print(f"Speaking: {sentence}")
        
        # Freeze the screen display
        frozen_frame = frame.copy()

        # Speak the sentence
        tts.say(sentence)
        tts.runAndWait()

        # Show confirmation overlay
        cv2.putText(frozen_frame, "Spoken. Resetting...", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow('ISL Translator - Real Time', frozen_frame)
        cv2.waitKey(1000)

        # Reset everything
        sentence = ""
        prediction_buffer.clear()
        space_timer = None
        backspace_timer = None
        last_time = time.time()
        continue  # Skip further processing this loop

    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_count = 0
    landmarks = []

    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)

        for hand in results.multi_hand_landmarks[:2]:
            x_vals = [lm.x for lm in hand.landmark]
            y_vals = [lm.y for lm in hand.landmark]
            normed = [[lm.x - min(x_vals), lm.y - min(y_vals)] for lm in hand.landmark]
            landmarks.append(normed)
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    while len(landmarks) < 2:
        landmarks.append([[0.0, 0.0]] * 21)

    current_time = time.time()

    # === SPACE ===
    if hand_count == 0:
        if space_timer is None:
            space_timer = current_time
        elapsed = current_time - space_timer

        if elapsed >= SPACE_HOLD_TIME:
            sentence += " "
            print("[+] Space added")
            space_timer = None
            last_time = current_time
            prediction_buffer.clear()
        else:
            # Show SPACE countdown overlay
            remaining = SPACE_HOLD_TIME - elapsed
            cv2.putText(frame, f"SPACE incoming in {remaining:.1f}s",
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        space_timer = None  # reset if any hand appears

        # === BACKSPACE ===
        if hand_count == 1:
            hand = results.multi_hand_landmarks[0]
            # Check "fist" based on spread of points
            spread = np.mean([
                np.linalg.norm(np.array([lm.x, lm.y]) - np.array([hand.landmark[0].x, hand.landmark[0].y]))
                for lm in hand.landmark[1:]
            ])

            if spread < 0.05:
                if backspace_timer is None:
                    backspace_timer = current_time
                elif current_time - backspace_timer >= BACKSPACE_HOLD_TIME:
                    sentence = sentence[:-1]
                    print("[-] Backspace triggered")
                    backspace_timer = None
                    last_time = current_time
                    prediction_buffer.clear()
                continue
            else:
                backspace_timer = None
        else:
            backspace_timer = None

        # === Prediction ===
        sample = np.array(landmarks).reshape(1, 2, 21, 2)
        start = time.time()
        pred = model.predict(sample, verbose=0)
        end = time.time()
        inference_times.append((end - start) * 1000)  # convert to ms

        class_id = np.argmax(pred)
        confidence = np.max(pred)
        label = le.inverse_transform([class_id])[0]
        prediction_buffer.append(label)

        if confidence > CONFIDENCE_THRESHOLD:
            most_common_label, count = Counter(prediction_buffer).most_common(1)[0]
            if count > (BUFFER_SIZE * 0.6) and (current_time - last_time > LETTER_DELAY):
                if most_common_label == "BACKSPACE":
                    if len(sentence) > 0:
                        sentence = sentence[:-1]
                        print("[-] BACKSPACE triggered →", sentence)
                else:
                    sentence += most_common_label
                    print(f"[+] Added '{most_common_label}' → {sentence}")
                last_time = current_time
                prediction_buffer.clear()


        # Show live prediction
        cv2.putText(frame, f'Prediction: {label} ({confidence:.2f})',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Show sentence buffer
    cv2.putText(frame, f'Sentence: {sentence}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)

    cv2.imshow('ISL Translator - Real Time', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


if inference_times:
    avg_time = sum(inference_times) / len(inference_times)
    print(f"\nAverage inference time per frame: {avg_time:.2f} ms")


cap.release()
cv2.destroyAllWindows()