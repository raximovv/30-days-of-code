# Day 14: RegisterBot

A tiny CPU simulator written in plain Python. It runs a fetch, decode, execute cycle over a small set of registers and an ALU, and includes a demo that computes a factorial. A local model explains what each instruction does. It uses Ollama with qwen2.5:3b.

## Run
```
python day14.py
```

## Install
```
pip install rich
```
Also install Ollama (https://ollama.com):
```
ollama pull qwen2.5:3b
```
