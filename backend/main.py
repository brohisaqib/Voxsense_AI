# ============================================================
# VoxSense - FastAPI Main Application
# ============================================================

import asyncio
import base64
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, HTTPException,
    UploadFile, File, Form, BackgroundTasks, Depends, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel

import sys
sys.path.insert(0, '/home/claude/voxsense')
from config.settings import settings, config
from backend.utils.logger import logger
from backend.services.llm_service import get_llm_service, LLMService
from backend.voice.stt_service import get_stt_service, STTService
from backend.voice.tts_service import get_tts_service, TTSService
from backend.vision.detection_service import get_detection_service, ObjectDetectionService
from backend.memory.memory_service import get_memory_service, MemoryService


# ============================================================
# Lifespan (startup/shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 VoxSense API starting up...")
    # Pre-initialize all services
    get_llm_service()
    get_stt_service()
    get_tts_service()
    get_detection_service()
    get_memory_service()
    logger.info("✅ All services initialized")
    yield
    logger.info("👋 VoxSense API shutting down...")


# ============================================================
# App Setup
# ============================================================

app = FastAPI(
    title="VoxSense Online AI API",
    description="AI Voice Assistant API for Blind and Visually Impaired Users",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models
# ============================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = True

class ChatResponse(BaseModel):
    response: str
    session_id: str

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None

class VisionRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = "Describe this scene clearly for a blind person."

class DetectionResponse(BaseModel):
    detections: List[dict]
    voice_alerts: List[str]
    annotated_image: Optional[str] = None


# ============================================================
# Dependency Injection
# ============================================================

def llm_dep() -> LLMService: return get_llm_service()
def stt_dep() -> STTService: return get_stt_service()
def tts_dep() -> TTSService: return get_tts_service()
def det_dep() -> ObjectDetectionService: return get_detection_service()
def mem_dep() -> MemoryService: return get_memory_service()


# ============================================================
# Health & Info Routes
# ============================================================

@app.get("/", tags=["Health"])
async def root():
    return {
        "app": "VoxSense Online AI",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/chat", "/voice/transcribe", "/voice/speak", "/vision/detect", "/vision/describe"],
    }

@app.get("/health", tags=["Health"])
async def health_check(memory: MemoryService = Depends(mem_dep)):
    return {
        "status": "healthy",
        "services": {
            "llm_groq": bool(settings.groq_api_key),
            "llm_openai": bool(settings.openai_api_key),
            "stt_deepgram": bool(settings.deepgram_api_key),
            "tts_edge": True,
            "memory": memory.get_stats(),
        },
        "timestamp": time.time(),
    }


# ============================================================
# Chat Routes
# ============================================================

@app.post("/chat", tags=["Chat"])
async def chat(
    request: ChatRequest,
    llm: LLMService = Depends(llm_dep),
    memory: MemoryService = Depends(mem_dep),
):
    """Non-streaming chat endpoint."""
    session_id = request.session_id or str(uuid.uuid4())

    # Get conversation history
    history = memory.get_conversation_messages(session_id)

    # Get AI response
    response = await llm.complete(request.message, history)

    # Store in memory
    memory.add_message(session_id, "user", request.message)
    memory.add_message(session_id, "assistant", response)

    return ChatResponse(response=response, session_id=session_id)


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    llm: LLMService = Depends(llm_dep),
    memory: MemoryService = Depends(mem_dep),
):
    """Streaming chat endpoint using Server-Sent Events."""
    session_id = request.session_id or str(uuid.uuid4())
    history = memory.get_conversation_messages(session_id)

    async def generate():
        full_response = ""
        try:
            async for chunk in llm.stream_response(request.message, history):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk, 'session_id': session_id})}\n\n"

            # Store in memory after complete
            memory.add_message(session_id, "user", request.message)
            memory.add_message(session_id, "assistant", full_response)
            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================
# Voice Routes
# ============================================================

@app.post("/voice/transcribe", tags=["Voice"])
async def transcribe_audio(
    audio: UploadFile = File(...),
    stt: STTService = Depends(stt_dep),
):
    """Transcribe uploaded audio file to text."""
    try:
        audio_bytes = await audio.read()
        mimetype = audio.content_type or "audio/wav"
        transcript = await stt.transcribe_audio(audio_bytes, mimetype)
        return {"transcript": transcript, "words": len(transcript.split())}
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/speak", tags=["Voice"])
async def text_to_speech(
    request: TTSRequest,
    tts: TTSService = Depends(tts_dep),
):
    """Convert text to speech and return audio bytes (MP3)."""
    try:
        if request.voice:
            tts.set_voice(request.voice)
        audio_bytes = await tts.synthesize_to_bytes(request.text)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"},
        )
    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/speak/stream", tags=["Voice"])
async def text_to_speech_stream(
    request: TTSRequest,
    tts: TTSService = Depends(tts_dep),
):
    """Stream audio as it's being synthesized."""
    if request.voice:
        tts.set_voice(request.voice)

    return StreamingResponse(
        tts.stream_audio(request.text),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache"},
    )


# ============================================================
# Vision Routes
# ============================================================

@app.post("/vision/detect", tags=["Vision"])
async def detect_objects(
    image: UploadFile = File(...),
    det: ObjectDetectionService = Depends(det_dep),
):
    """Run YOLO object detection on uploaded image."""
    try:
        import numpy as np
        import cv2

        image_bytes = await image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        result = det.detect_frame(frame)

        detections_out = [
            {
                "class_name": d.class_name,
                "confidence": round(d.confidence, 3),
                "bbox": list(d.bbox),
                "position": d.position,
                "distance": d.distance,
                "guidance": d.guidance,
            }
            for d in result.detections
        ]

        annotated_b64 = ""
        if result.annotated_frame is not None:
            annotated_b64 = det.frame_to_base64(result.annotated_frame)

        return {
            "detections": detections_out,
            "voice_alerts": result.voice_alerts,
            "count": len(result.detections),
            "annotated_image": annotated_b64,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vision/describe", tags=["Vision"])
async def describe_scene(
    request: VisionRequest,
    llm: LLMService = Depends(llm_dep),
):
    """Use AI Vision to describe a scene from base64 image."""
    try:
        description = await llm.describe_image(request.image_base64, request.prompt)
        return {"description": description}
    except Exception as e:
        logger.error(f"❌ Vision describe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Memory Routes
# ============================================================

@app.get("/memory/{session_id}", tags=["Memory"])
async def get_session_memory(session_id: str, memory: MemoryService = Depends(mem_dep)):
    """Get conversation history for a session."""
    history = memory.get_session_history(session_id)
    return {"session_id": session_id, "messages": history, "count": len(history)}


@app.delete("/memory/{session_id}", tags=["Memory"])
async def clear_session_memory(session_id: str, memory: MemoryService = Depends(mem_dep)):
    """Clear all messages for a session."""
    success = memory.clear_session(session_id)
    return {"success": success, "session_id": session_id}


@app.get("/memory/{session_id}/search", tags=["Memory"])
async def search_memory(session_id: str, q: str, memory: MemoryService = Depends(mem_dep)):
    """Semantic search in session memory."""
    results = memory.search_similar(q, session_id=session_id)
    return {"query": q, "results": results}


# ============================================================
# WebSocket: Chat Streaming
# ============================================================

class ConnectionManager:
    """Manage active WebSocket connections."""

    def __init__(self):
        self.active: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active[client_id] = websocket
        logger.info(f"🔌 WS connected: {client_id}")

    def disconnect(self, client_id: str):
        self.active.pop(client_id, None)
        logger.info(f"🔌 WS disconnected: {client_id}")

    async def send_json(self, client_id: str, data: dict):
        ws = self.active.get(client_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception as e:
                logger.error(f"❌ WS send error: {e}")


manager = ConnectionManager()


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time streaming chat.
    Receives: {"message": "user input"}
    Sends: {"type": "chunk|done|error", "content": "..."}
    """
    await manager.connect(websocket, session_id)
    llm = get_llm_service()
    memory = get_memory_service()

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "").strip()

            if not user_message:
                continue

            logger.info(f"💬 WS [{session_id}]: {user_message[:50]}...")

            # Get history and stream response
            history = memory.get_conversation_messages(session_id)
            full_response = ""

            await websocket.send_json({"type": "start", "session_id": session_id})

            async for chunk in llm.stream_response(user_message, history):
                full_response += chunk
                await websocket.send_json({"type": "chunk", "content": chunk})

            await websocket.send_json({"type": "done", "content": full_response})

            # Store in memory
            memory.add_message(session_id, "user", user_message)
            memory.add_message(session_id, "assistant", full_response)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"❌ WS error [{session_id}]: {e}")
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass
        manager.disconnect(session_id)


@app.websocket("/ws/voice/{session_id}")
async def websocket_voice(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for voice pipeline.
    Receives: binary audio data chunks
    Sends: {"type": "transcript|response|audio|error", "content": "..."}
    """
    await manager.connect(websocket, f"voice_{session_id}")
    stt = get_stt_service()
    llm = get_llm_service()
    tts = get_tts_service()
    memory = get_memory_service()

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                # Audio chunk received — transcribe
                audio_bytes = data["bytes"]
                if len(audio_bytes) < 100:
                    continue

                transcript = await stt.transcribe_audio(audio_bytes, "audio/webm")

                if not transcript.strip():
                    continue

                await websocket.send_json({"type": "transcript", "content": transcript})

                # Get AI response
                history = memory.get_conversation_messages(session_id)
                response_text = await llm.complete(transcript, history)

                await websocket.send_json({"type": "response", "content": response_text})

                # Generate TTS
                audio_response = await tts.synthesize_to_bytes(response_text)
                if audio_response:
                    await websocket.send_bytes(audio_response)
                    await websocket.send_json({"type": "audio_ready"})

                # Store in memory
                memory.add_message(session_id, "user", transcript)
                memory.add_message(session_id, "assistant", response_text)

            elif "text" in data:
                # Text command
                msg = json.loads(data["text"])
                if msg.get("command") == "ping":
                    await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(f"voice_{session_id}")
    except Exception as e:
        logger.error(f"❌ Voice WS error: {e}")
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass


@app.websocket("/ws/detection/{session_id}")
async def websocket_detection(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time object detection.
    Receives: base64 encoded frame
    Sends: detection results + voice alerts
    """
    await manager.connect(websocket, f"det_{session_id}")
    det = get_detection_service()

    try:
        while True:
            data = await websocket.receive_json()
            frame_b64 = data.get("frame", "")

            if not frame_b64:
                continue

            result = det.detect_from_base64(frame_b64)

            response = {
                "detections": [
                    {
                        "class_name": d.class_name,
                        "confidence": round(d.confidence, 3),
                        "position": d.position,
                        "distance": d.distance,
                        "guidance": d.guidance,
                    }
                    for d in result.detections
                ],
                "voice_alerts": result.voice_alerts,
                "count": len(result.detections),
            }

            if result.annotated_frame is not None:
                response["annotated_frame"] = det.frame_to_base64(result.annotated_frame)

            await websocket.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(f"det_{session_id}")
    except Exception as e:
        logger.error(f"❌ Detection WS error: {e}")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
        log_level="info",
    )
