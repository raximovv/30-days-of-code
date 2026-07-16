# Day 06: BreathClock

Measures your breathing rate in real time from the laptop microphone. It runs the mic signal through a low pass filter and counts your breaths.

## Run
```
python day06.py
```

## Install
```
pip install matplotlib numpy pyaudio scipy
```

## Controls
Breathe slowly near the microphone. Close the plot window to quit.

## Notes
pyaudio can be awkward to install. On Mac: `brew install portaudio`, then `pip install pyaudio`. On Linux: `sudo apt-get install portaudio19-dev`, then `pip install pyaudio`.
