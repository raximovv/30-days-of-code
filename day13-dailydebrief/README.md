# Day 13: DailyDebrief

Collects what you did in the last 24 hours (recent git commits and editor activity) and asks a local model to write a short summary. It uses Ollama with qwen2.5:3b.

## Run
```
python day13.py
```

## Install
```
pip install gitpython rich
```
Also install Ollama (https://ollama.com):
```
ollama pull qwen2.5:3b
```

## Notes
If it says no data was found, make a few code edits or commits first, then run it again.
