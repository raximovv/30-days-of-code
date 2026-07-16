# Day 24: HardwareTA

A local question and answer system for hardware datasheets. It reads PDF datasheets, indexes them, and answers your questions using only what is in them. It uses Ollama with llama3, plus a local embedding model for search.

## Run
```
python day24.py
```

## Install
```
pip install chromadb pymupdf ollama sentence-transformers
```
Also install Ollama (https://ollama.com):
```
ollama pull llama3
```

## Notes
Put one or more datasheet PDFs in this folder before running. The first run downloads the embedding model.
