#to create alphabet datasetn - data_self

import os
import cv2
import string
import time

# Directory to store data
DATA_DIR = './dataself'

# Create base directory
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of images to capture per class
dataset_size = 100

# 26 alphabet classes
classes = list(string.ascii_uppercase)

# Initialize webcam (change 2 to 0 or 1 depending on your setup)
cap = cv2.VideoCapture(0)

for label in classes:
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'\nPreparing to collect data for class "{label}"')

    # Wait for user readiness
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Get ready for "{label}" - Press "Q"', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 3-second countdown before starting capture
    for countdown in range(3, 0, -1):
        ret, frame = cap.read()
        cv2.putText(frame, f'Starting in {countdown}...', (150, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)  # wait for 1 second

    print(f'Capturing {dataset_size} images for class "{label}"')

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        # Save frame
        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)
        counter += 1

        # Optional delay if you want to slow down capture rate
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

print("\nDataset collection completed.")
cap.release()
cv2.destroyAllWindows()