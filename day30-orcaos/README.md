# Day 30: OrcaOS

A single terminal dashboard that pulls together the ideas from the whole challenge. It shows live panels for gesture input, microphone and system sensors, a local language model, and a system pulse, all in one screen. It uses Textual for the interface and Ollama with qwen2.5:3b.

## Run
```
python day30.py
```

## Install
```
pip install opencv-python mediapipe numpy psutil pyaudio textual
```
Also install Ollama (https://ollama.com):
```
ollama pull qwen2.5:3b
```

## Notes
Any panel whose libraries are missing simply switches off, so it still runs if you skip some.
