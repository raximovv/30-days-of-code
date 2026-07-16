# Day 27: SignalFlags

Recognizes semaphore flag gestures through the webcam and spells out letters. It trains a small classifier on a landmark dataset, then reads your hand positions in real time. It uses scikit-learn for the classifier and MediaPipe for hand tracking.

## Run
```
python day27.py
```

## Install
```
pip install opencv-python mediapipe numpy pandas scikit-learn
```

## Controls
Press C to clear the current word. Press Q to quit.

## Notes
The training data (`semaphore_landmarks.csv`) is already included in this folder.
