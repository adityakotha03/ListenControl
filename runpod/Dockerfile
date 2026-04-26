# GPU base: Python 3.11, CUDA 12.4, PyTorch 2.5
# Build from runpod/ with: docker build -t listencontrol .
# Required before build: weights/best_bidir_cross_transformer.pt, render/flame/*.pkl, render/flame/*.npy
# Optional for pipeline 6: weights/best_emo_transformer.pt
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# System deps: ffmpeg for audio merge, git for FLAME_PyTorch clone
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone FLAME_PyTorch (required by render_pipeline)
RUN git clone https://github.com/soubhiksanyal/FLAME_PyTorch.git /app/FLAME_PyTorch

# Copy project code and assets
COPY requirements.txt /app/
COPY handler.py /app/
COPY main.py /app/
COPY model/ /app/model/
COPY render/ /app/render/
COPY weights/ /app/weights/

# Install Python deps (torch/torchaudio already in base)
RUN pip install --no-cache-dir -r requirements.txt

# Install pytorch3d with GPU support (must match base torch/CUDA)
RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git"

# FLAME assets must exist in render/flame (generic_model.pkl, flame_static_embedding.pkl, flame_dynamic_embedding.npy)
# Default weights must exist in weights/best_bidir_cross_transformer.pt
# Pipeline 6 text-control weights should exist in weights/best_emo_transformer.pt when selected.
# These are baked at build time via COPY above

ENV PYTHONPATH=/app:/app/FLAME_PyTorch
ENV LISTEN_WEIGHTS_PATH=/app/weights/best_bidir_cross_transformer.pt

CMD ["python", "-u", "handler.py"]
