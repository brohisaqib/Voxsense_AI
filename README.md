# 👁️ VoxSense Online AI
### AI Voice Assistant for Blind & Visually Impaired Users

> **"An Online Jarvis for Blind People"** — Real-time AI chat, voice interaction, camera-based object detection, and scene description, all in a accessible web app.

---

## 🚀 Features

| Feature | Technology | Status |
|---------|-----------|--------|
| AI Chat (streaming) | Groq llama3-70b + OpenAI fallback | ✅ |
| Speech-to-Text | Deepgram nova-2 | ✅ |
| Text-to-Speech | Edge-TTS (20+ voices) | ✅ |
| Object Detection | YOLOv8n + OpenCV | ✅ |
| AI Scene Description | OpenAI GPT-4o Vision | ✅ |
| Conversation Memory | ChromaDB | ✅ |
| Live Camera Streaming | streamlit-webrtc | ✅ |
| WebSocket Real-time | FastAPI WebSockets | ✅ |
| Mobile Camera Support | WebRTC (back camera) | ✅ |
| Dark Accessible UI | Streamlit + Custom CSS | ✅ |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VoxSense Online AI                       │
│                                                             │
│  ┌─────────────────────┐    ┌──────────────────────────┐   │
│  │  Streamlit Frontend  │    │    FastAPI Backend        │   │
│  │  (Port 8501)         │◄──►│    (Port 8000)           │   │
│  │                      │    │                          │   │
│  │  - Chat Panel        │    │  REST Endpoints:         │   │
│  │  - Voice Panel       │    │  POST /chat/stream       │   │
│  │  - Camera Panel      │    │  POST /voice/transcribe  │   │
│  │  - Detection Panel   │    │  POST /voice/speak       │   │
│  │  - Settings          │    │  POST /vision/detect     │   │
│  └─────────────────────┘    │  POST /vision/describe   │   │
│                              │                          │   │
│                              │  WebSockets:             │   │
│                              │  WS /ws/chat/{id}        │   │
│                              │  WS /ws/voice/{id}       │   │
│                              │  WS /ws/detection/{id}   │   │
│                              └──────────────────────────┘   │
│                                         │                   │
│              ┌──────────────────────────┼──────────────┐    │
│              │         AI Services      │              │    │
│              ▼                          ▼              ▼    │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────┐  │
│  │  Groq LLM       │  │  OpenAI GPT-4o   │  │  YOLOv8n  │  │
│  │  llama3-70b     │  │  Vision + Chat   │  │  Detection│  │
│  └─────────────────┘  └──────────────────┘  └───────────┘  │
│              │                          │                   │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │  Deepgram STT   │  │  Edge-TTS        │                  │
│  │  nova-2 model   │  │  30+ voices      │                  │
│  └─────────────────┘  └──────────────────┘                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              ChromaDB Memory                        │    │
│  │  Persistent conversation history + semantic search  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

- Python 3.10+
- pip
- API Keys (see below)

---

## 🔑 Required API Keys

| Service | Purpose | Free Tier | Link |
|---------|---------|-----------|------|
| **Groq** | Primary LLM (fast!) | ✅ Yes | [console.groq.com](https://console.groq.com) |
| **OpenAI** | Vision + Fallback LLM | Paid | [platform.openai.com](https://platform.openai.com) |
| **Deepgram** | Speech-to-Text | ✅ $200 credit | [console.deepgram.com](https://console.deepgram.com) |

> 💡 The app works with just Groq for chat. OpenAI is needed for vision. Deepgram for real voice transcription.

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-org/voxsense-ai
cd voxsense-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

# Install Playwright browsers (for automation features)
playwright install chromium
```

### 3. Configure Environment

```bash
cp .env.example .env
nano .env  # or use any editor
```

Fill in:
```env
GROQ_API_KEY=gsk_your_groq_key
OPENAI_API_KEY=sk-your_openai_key
DEEPGRAM_API_KEY=your_deepgram_key
```

### 4. Start VoxSense

**Option A - Auto start (recommended):**
```bash
chmod +x start.sh
./start.sh
```

**Option B - Manual start:**
```bash
# Terminal 1 - Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Frontend
streamlit run frontend/app.py --server.port 8501
```

### 5. Open the App

- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

---

## 📱 Mobile Usage

1. Find your computer's local IP address:
   ```bash
   # Linux/Mac
   hostname -I
   # Windows
   ipconfig
   ```

2. Open on mobile browser:
   ```
   http://YOUR_IP:8501
   ```

3. Allow camera/microphone when prompted

4. Use the **Camera & Vision** tab for real-time object detection

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t voxsense .

# Run
docker run -p 8000:8000 -p 8501:8501 \
  -e GROQ_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  -e DEEPGRAM_API_KEY=your_key \
  voxsense
```

---

## 🌐 Cloud Deployment

### Render.com (Recommended)

1. Push to GitHub
2. Create new Web Service on Render
3. Add environment variables
4. Deploy

### Railway.app

```bash
npm install -g railway
railway login
railway init
railway up
```

### Fly.io

```bash
fly launch
fly secrets set GROQ_API_KEY=your_key
fly deploy
```

---

## 📁 Project Structure

```
voxsense/
├── frontend/
│   └── app.py                  # Streamlit main app
├── backend/
│   ├── main.py                 # FastAPI app + all routes + WebSockets
│   ├── services/
│   │   └── llm_service.py      # Groq + OpenAI LLM service
│   ├── voice/
│   │   ├── stt_service.py      # Deepgram STT
│   │   └── tts_service.py      # Edge-TTS
│   ├── vision/
│   │   └── detection_service.py # YOLOv8 object detection
│   ├── memory/
│   │   └── memory_service.py   # ChromaDB memory
│   └── utils/
│       └── logger.py           # Loguru logger
├── config/
│   ├── config.yaml             # App configuration
│   └── settings.py             # Pydantic settings
├── data/
│   └── chroma/                 # ChromaDB persistence
├── logs/                       # Application logs
├── requirements.txt
├── .env.example
├── start.sh
└── README.md
```

---

## 🔧 API Reference

### Chat
```http
POST /chat
{"message": "Hello", "session_id": "abc123"}

POST /chat/stream  (Server-Sent Events)
{"message": "Navigate me safely"}
```

### Voice
```http
POST /voice/transcribe
Content-Type: multipart/form-data
audio: <file>

POST /voice/speak
{"text": "Person ahead, be careful", "voice": "en-US-JennyNeural"}
```

### Vision
```http
POST /vision/detect
Content-Type: multipart/form-data
image: <file>

POST /vision/describe
{"image_base64": "...", "prompt": "Describe this scene"}
```

### WebSockets
```
WS /ws/chat/{session_id}       → Real-time streaming chat
WS /ws/voice/{session_id}      → Audio in, transcript+speech out
WS /ws/detection/{session_id}  → Frames in, detections out
```

---

## ♿ Accessibility Features

- Large, high-contrast UI buttons
- Dark mode by default
- Screen-reader friendly layout
- Keyboard navigation support
- Auto-play voice responses
- Minimal cognitive load design
- Mobile-first camera access

---

## 📄 License

MIT License — Free to use, modify, and distribute.

---

## 🤝 Contributing

PRs welcome! Please read CONTRIBUTING.md first.

---

*Built with ❤️ for the blind and visually impaired community.*
