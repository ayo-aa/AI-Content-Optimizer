FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir gradio faster-whisper sentence-transformers scikit-learn \
 && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

EXPOSE 7860
CMD ["python", "app.py"]
