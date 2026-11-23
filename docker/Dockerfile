# Reproduce every result on any NVIDIA machine:   docker build -f docker/Dockerfile -t freenerf .
#   docker run --gpus all -v $PWD/data:/app/data -v $PWD/results:/app/results freenerf --only E1 --iters 30000
# Build on a Mac for a cloud GPU host with:  docker buildx build --platform linux/amd64 -f docker/Dockerfile -t freenerf .
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip git ffmpeg colmap zip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.cuda.lock* requirements.txt ./
RUN pip3 install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 \
 && (pip3 install -r requirements.cuda.lock 2>/dev/null || pip3 install -r requirements.txt)
COPY . .
RUN pip3 install -e .
ENTRYPOINT ["bash", "scripts/run_all.sh"]
