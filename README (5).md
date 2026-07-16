# Day 03: VolumeKnuckle

Control your system volume with your fist. Raise your fist to make it louder, lower it to make it quieter. The webcam tracks your hand with MediaPipe.

## Run
```
python day03.py
```

## Install
```
pip install opencv-python mediapipe numpy pycaw comtypes
```

## Controls
Show an open hand first so MediaPipe can find it, then close your fist. Fist up is louder, fist down is quieter. Press Q to quit.

## Notes
The volume control uses pycaw and comtypes, which only work on Windows. On Mac or Linux the hand tracking still runs, but the volume will not change.
