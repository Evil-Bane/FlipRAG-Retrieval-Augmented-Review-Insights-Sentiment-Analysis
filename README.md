# FlipRAG-Retrieval-Augmented-Review-Insights-Sentiment-Analysis
RAG-based product review analyzer powered by local LLM (Ollama + Dolphin3) — scrapes, summarizes, and chats with product reviews using retrieval-augmented generation.
## Overview
This repository contains a single `main.py` which runs the project's pipeline and communicates with a local Ollama LLM server.  
The app expects Ollama to be installed locally and a Dolphin model available in Ollama. Use this README to set up Ollama, download the Dolphin model, start the server, and run `main.py`.

---

## Why Ollama + Dolphin?
- **Ollama** provides an easy way to run LLMs locally (CLI + local server) and exposes a local API (usually at `http://localhost:11434`). It supports pulling and running community models and is suitable for demos and offline development. :contentReference[oaicite:4]{index=4}  
- **Dolphin** is a family of high-quality instruct-tuned local models (variants include`dolphin-mistral`, etc.).

---

## System recommendations
- 16 GB RAM recommended for comfortable use; 8 GB may work for smaller Dolphin variants but could be slow. GPU will speed up inference. :contentReference[oaicite:6]{index=6}

---

## Install Ollama (macOS / Linux / Windows)
1. Visit Ollama downloads / docs and follow the installer for your OS.  
   Official installation & start instructions are in Ollama docs. :contentReference[oaicite:7]{index=7}

2. Verify installation (example):
in a terminal:
ollama version
or check that Ollama service can be started (see next steps)
3. Download (pull) a Dolphin model

 To pull a model:

example: pull dolphin-mistral
ollama pull dolphin-mistral


4. Start the Ollama server

Start the Ollama background server which main.py will contact:

open a terminal and run:
ollama serve


By default Ollama serves locally (the docs reference the default local endpoint and port). Keep this terminal open. 
Ollama Documentation
+1

