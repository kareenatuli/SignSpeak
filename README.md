# **SignSpeak**
An **Indian Sign Language to Voice translator** that uses OpenCv (Python Library for webcam input) & MediaPipe (Python Library for hand landmark detection) and Machine Learning algorithms.

This project allows users to sign the alphabets in real-time and converts these alphabets to text and audio. It recognises all 26 alphabets (A-Z) and includes two additional gestures for inserting space and for backspacing. Essentially, it is like texting with ISL, and reading that text out loud.

## Dataset and Tools Used
1. **Dataset** - created manually using webcam and OpenCV. It has 27 gesture classes, this includes A-Z alphabets (26) and one class for the gesture of backspacing _[NOTE : The gesture for inserting a space is - not showing any hands for a specified duration of time.]._
2. **Programming Language** - Python 3.9.x (code is run on VS Code IDE)
3. **Python Libraries used** - OpenCV, MediaPipe, Tensorflow/Keras, Numpy, scikit-learn, pyttsx3
4. **Performance Evaluation parameters** - precision, recall, f1-score, accuracy
5. **ML Model used** - CNN

## Steps to run this project on your own device

1. Ensure you have all libraries for python installed and python version is compatable with the libraries. (version used here Python 3.9.x)
2. The dataset is provided with each gesture as a .zip file (27 .zip files). You can download this and save it all in a folder titled data. If you use the dataset as provided in the repo, you do not need to run the two python files titled - createData.py & createBackspace.py
3. If you would like to create your own dataset, then begin by running the file createData.py and createBackspace.py. In this case, you do not need to download the .zip files of the gestures.
4. Run the file preprocess_alphabet.py. This will create a .pickle file that stores all landmarks of gestures.
5. Now run train_alphabet_model.py. This will train your model and store it (here it is stored as a .keras file). You can update this code to change the model. Model used here is CNN, this may be updated to train a different model.
6. Finally, run the test_real_time.py file, this will turn on your webcam and translate your gestures in real time.
