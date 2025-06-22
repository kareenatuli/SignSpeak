import os
import cv2
import time

# === Adjust path to match your existing dataset folder ===
DATA_DIR = './data'
BACKSPACE_DIR = os.path.join(DATA_DIR, 'BACKSPACE')
os.makedirs(BACKSPACE_DIR, exist_ok=True)

# Set webcam resolution (helps with focus sometimes)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Try 1280x720 for better focus
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

dataset_size = 100

print('\nPreparing to collect data for class "BACKSPACE"')

# Wait for user to press Q
while True:
    ret, frame = cap.read()
    cv2.putText(frame, 'Get ready for "BACKSPACE" - Press "Q"', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Countdown
for countdown in range(3, 0, -1):
    ret, frame = cap.read()
    cv2.putText(frame, f'Starting in {countdown}...', (150, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv2.imshow('frame', frame)
    cv2.waitKey(1000)

print(f'Capturing {dataset_size} images for "BACKSPACE"...')

counter = 0
while counter < dataset_size:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    image_path = os.path.join(BACKSPACE_DIR, f'{counter}.jpg')
    cv2.imwrite(image_path, frame)
    counter += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

print("\n Backspace data collection completed.")
cap.release()
cv2.destroyAllWindows()
