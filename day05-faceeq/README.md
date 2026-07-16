# Day 05: FaceEQ

Your head becomes the playback control for a music track. Turn your head to seek through the song, and tilt it to change the speed. The webcam tracks your face with MediaPipe.

## Run
```
python day05.py
```

## Install
```
pip install opencv-python mediapipe numpy sounddevice soundfile
```

## Controls
Space toggles play and pause. Press R to reset to the start and play. Press Q to quit.

## Notes
Put an audio file named `track.wav` (or `track.mp3`) in this folder. If you only have an mp3, you can convert it first with ffmpeg: `ffmpeg -i track.mp3 track.wav`.
