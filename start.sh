#!/bin/bash
# ============================================================
# VoxSense Online AI - Startup Script
# ============================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          VoxSense Online AI - Startup                ║"
echo "║     AI Voice Assistant for Blind Users               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Check .env
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from example..."
    cp .env.example .env
    echo "📝 Please edit .env with your API keys, then re-run this script."
    echo ""
    echo "Required API keys:"
    echo "  - GROQ_API_KEY       → https://console.groq.com"
    echo "  - OPENAI_API_KEY     → https://platform.openai.com"  
    echo "  - DEEPGRAM_API_KEY   → https://console.deepgram.com"
    exit 1
fi

# Create data directories
mkdir -p data/chroma logs

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt -q

# Download YOLO model if needed
echo "📦 Ensuring YOLO model is available..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || echo "⚠️  YOLO model will be downloaded on first detection."

echo ""
echo "🚀 Starting VoxSense services..."
echo ""

# Start FastAPI backend in background
echo "📡 Starting FastAPI backend on port 8000..."
uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --workers 1 &

BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "⏳ Waiting for backend to start..."
sleep 3

# Check backend health
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy!"
else
    echo "⚠️  Backend may still be starting..."
fi

echo ""
echo "🌐 Starting Streamlit frontend on port 8501..."
streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.enableCORS false \
    --server.enableXsrfProtection false &

FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  VoxSense is running!                                ║"
echo "║                                                      ║"
echo "║  🌐 Frontend:  http://localhost:8501                 ║"
echo "║  📡 Backend:   http://localhost:8000                 ║"
echo "║  📚 API Docs:  http://localhost:8000/docs            ║"
echo "║                                                      ║"
echo "║  Press Ctrl+C to stop all services                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Wait and handle shutdown
trap "echo ''; echo '👋 Shutting down VoxSense...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait
