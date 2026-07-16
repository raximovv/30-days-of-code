# Day 08: PocketAgent

A command line agent that runs a small language model fully on your laptop. No cloud, no API key. It uses Ollama with the qwen2.5:3b model.

## Run
```
python day08.py
```

## Install
This one needs Ollama, not extra Python packages. Install Ollama (https://ollama.com), start it, then pull the model:
```
ollama serve
ollama pull qwen2.5:3b
```

## Notes
Keep `ollama serve` running in another terminal while you use it.
