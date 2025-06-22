# SignSpeak
An **Indian Sign Language to Voice translator** that uses MediaPipe (Python Library) and Machine Learning algorithms.

This project allows users to sign the alphabets in real-time and converts these alphabets to text and audio. This project recognises all 26 alphabets (A-Z) and includes two additional gestures for inserting space and for backspacing. Essentially, it is like texting with ISL and reading that text out loud.

1. Dataset - created manually using webcam and OpenCV. It has 27 gesture classes, this includes A-Z alphabets (26) and one class for the gesture of backspacing [NOTE : The gesture for inserting a space is not showing any hands for a specified duration of time.].
2. Programming Language - Python 3.9.x (code is run on VS Code IDE)
3. Python Libraries used - OpenCV, MediaPipe, Tensorflow/Keras, Numpy, scikit-learn, pyttsx3
4. Performance Evaluation - precision, recall, f1-score, accuracy
5. ML Model used - 
