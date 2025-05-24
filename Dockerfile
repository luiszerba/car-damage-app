FROM python:3.10-slim

WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    gcc \
    g++ \
    libglib2.0-0 \
    build-essential \
    ninja-build \
    && apt-get clean

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip

# Install Torch first (needed for detectron2)
RUN pip install torch==2.0.1 torchvision==0.15.2

# Install other dependencies
RUN pip install \
    fastapi \
    uvicorn \
    opencv-python \
    numpy \
    requests

# Install Detectron2 from GitHub
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
