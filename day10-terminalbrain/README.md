# Day 10: TerminalBrain

Wraps a shell command and watches it run. When the command throws an error, it sends the error to a local model and prints a suggested fix. It uses Ollama with llama3.

## Run
```
python day10.py <command>
```
For example:
```
python day10.py "python broken_script.py"
```

## Install
```
pip install ollama rich
```
Also install Ollama (https://ollama.com):
```
ollama pull llama3
```
