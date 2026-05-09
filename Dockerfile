# ============================================================
# VoxSense Online AI - Dockerfile
# ============================================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO model
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data/chroma logs

# Expose ports
EXPOSE 8000 8501

# Startup script
RUN chmod +x start.sh
CMD ["./start.sh"]
