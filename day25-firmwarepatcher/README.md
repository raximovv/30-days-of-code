# Day 25: FirmwarePatcher

Inspects a firmware binary. It shows a hex view, looks for known patterns and readable strings, and asks a local model to flag anything risky. It uses Ollama with llama3.

## Run
```
python day25.py
```

## Install
```
pip install ollama rich
```
Also install Ollama (https://ollama.com):
```
ollama pull llama3
```

## Notes
It reads `assets/firmware_blob.bin`. A sample file is already included. If you remove it, the program generates a simulated one.
