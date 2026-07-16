# Day 11: MoodSynth

Turns a plain description of a mood into live ambient sound. You type something like "calm rainy morning", a local model turns it into sound settings, and the program plays it. It uses Ollama with qwen2.5:3b.

## Run
```
python day11.py
```

## Install
```
pip install numpy sounddevice
```
Also install Ollama (https://ollama.com):
```
ollama pull qwen2.5:3b
```
