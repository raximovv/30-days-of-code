# Day 28: MorseDecoder

Tap Morse code on your spacebar and it decodes it live. It first calibrates to your tapping speed, converts your taps to letters, then sends the decoded text to a local model. It uses Ollama with qwen2.5:3b.

## Run
```
python day28.py
```

## Install
```
pip install pynput rich
```
Also install Ollama (https://ollama.com):
```
ollama pull qwen2.5:3b
```

## Controls
Follow the on screen calibration first, then tap Morse on the spacebar.
