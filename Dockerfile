# Svarog Training Dockerfile
FROM nvidia/cuda:12.8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/svarog_venv
ENV PATH="/opt/svarog_venv/bin:$PATH"

# Upgrade pip and install PyTorch with CUDA support
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install \
    transformers==4.35.0 \
    datasets==2.14.0 \
    sentencepiece==0.1.99 \
    deepspeed==0.11.1 \
    accelerate==0.24.0

# Install monitoring and utilities
RUN pip install \
    wandb==0.15.0 \
    mlflow==2.8.0 \
    numpy==1.24.3 \
    pandas==2.1.0 \
    tqdm==4.66.0 \
    psutil==5.9.0 \
    pyyaml==6.0.1

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p data processed models tokenizer checkpoints logs

# Make scripts executable
RUN chmod +x train_svarog.py src/*.py

# Default command
CMD ["python", "train_svarog.py", "--help"]
