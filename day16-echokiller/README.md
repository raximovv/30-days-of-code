# Day 16: EchoKiller

Removes echo from an audio signal using an adaptive FIR filter. If you do not give it a file, it makes a synthetic echo to clean instead. It saves the cleaned audio as `cleaned_output.wav`.

## Run
```
python day16.py
```

## Install
```
pip install matplotlib numpy scipy sounddevice soundfile
```

## Notes
To clean your own audio, put a `.wav` file in this folder. Use any name except `echoed_input.wav` or `cleaned_output.wav`, which are reserved for the program's own output.
