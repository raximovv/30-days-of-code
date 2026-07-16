# Day 29: SilentAssistant

A hands free assistant that reads your lips. The webcam watches your mouth, a vision model describes what you mouthed, a language model answers, and the answer is spoken back. It uses Ollama with moondream and qwen2.5:3b.

## Run
```
python day29.py
```

## Install
```
pip install opencv-python mediapipe pyttsx3 requests
```
Also install Ollama (https://ollama.com):
```
ollama pull moondream
ollama pull qwen2.5:3b
```

## Controls
Press Q to quit.
