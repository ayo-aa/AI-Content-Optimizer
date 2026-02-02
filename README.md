---
title: AI Content Optimizer
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# AI Content Optimizer

An end-to-end AI video editing tool that takes an uploaded MP4 and returns an edited MP4 plus an edit report.  
The system uses fast local speech recognition (word timestamps) and semantic sentence scoring (learned embeddings) to remove low-value or redundant speech while preserving coherence (cuts happen only at sentence boundaries).

This repository includes:
- a user-facing web app (upload → edited video)
- a reproducible pipeline (scripts) for research and iteration
- a Docker deployment that runs locally and on Hugging Face Spaces

---

## What it does

Given a video, the optimizer:

1) **Transcribes speech with timestamps**  
2) **Converts the transcript into sentence-level edit units**  
3) **Scores each sentence for semantic value** (relevance and novelty, penalize redundancy and filler)  
4) **Plans cuts across the full timeline** under coherence constraints  
5) **Exports an edited MP4** and a **JSON report** showing what was kept/removed and why

The result is a tighter cut that stays intelligible (no mid-sentence starts) and removes unnecessary sentences throughout the video (not just head and tail).

---

## Demo

This project can be hosted as a Hugging Face Space so anyone can use it via a link (no cloning required).  
If you are running locally, see “Run the app”.

---

## Run the app

### Option A: Docker (recommended)
Prereqs:
- Docker Desktop installed and running

```bash
git clone https://github.com/ayo-aa/AI-Content-Optimizer.git
cd AI-Content-Optimizer
docker build -t ai-content-optimizer .
docker run --rm -p 7860:7860 ai-content-optimizer
