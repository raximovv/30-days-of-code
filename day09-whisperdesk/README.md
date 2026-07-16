# Day 09: WhisperDesk

Offline speech to text from your microphone. It uses faster-whisper to transcribe what you say, with an Ollama model as a fallback.

## Run
```
python day09.py
```

## Install
```
pip install faster-whisper numpy pyaudio
```
Also install Ollama (https://ollama.com) for the fallback:
```
ollama pull qwen2.5:3b
```

## Notes
The first run downloads the whisper model, so give it a minute.
