# Day 22: CircuitWhisperer

Reads a hand drawn circuit diagram and describes it. Point the webcam at a schematic, or drop an image in the folder, and a local vision model interprets it. It uses Ollama with moondream. It saves its reading to `analysis_result.txt`.

## Run
```
python day22.py
```

## Install
```
pip install pillow opencv-python numpy
```
Also install Ollama (https://ollama.com):
```
ollama pull moondream
```

## Controls
Press Q to quit.

## Notes
To analyze an image instead of the webcam, put a file named `circuit.png`, `schematic.png`, or `test_circuit.png` in this folder.
