# ============================================================
# VoxSense AI - Streamlit Frontend (Image-matched Design)
# Drop-in replacement — backend functions untouched
# ============================================================

import sys
import os
from pathlib import Path
from PIL import Image as PILImage

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import asyncio, base64, json, time, uuid, threading, queue
from io import BytesIO
from typing import Optional

import httpx
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import av, numpy as np, cv2

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="VoxSense AI",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ============================================================
# CSS — Mobile-first design matching reference image
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:           #0a0a10;
    --bg-card:      #111118;
    --bg-card2:     #16161f;
    --bg-card3:     #1c1c27;
    --accent:       #7c5bf7;
    --accent-light: #a78bfa;
    --accent-glow:  rgba(124,91,247,0.2);
    --green:        #22d3a8;
    --red:          #f87171;
    --yellow:       #fbbf24;
    --blue:         #60a5fa;
    --text1:        #f0ecff;
    --text2:        #9990bb;
    --text3:        #504a6e;
    --border:       rgba(120,100,200,0.12);
    --border2:      rgba(120,100,200,0.22);
    --radius:       14px;
    --radius-lg:    18px;
}

* { box-sizing: border-box; }

.stApp {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text1) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 0 !important; max-width: 480px !important; margin: 0 auto !important; }

/* Hide sidebar toggle */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

/* ─── TOP NAVBAR ─── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background: rgba(10,10,16,0.96);
    border-bottom: 1px solid var(--border2);
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(12px);
}
.topbar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem; font-weight: 700;
    color: var(--text1); letter-spacing: -0.02em;
}
.topbar-logo span { color: var(--accent-light); }
.topbar-page { font-size: 0.82rem; color: var(--text2); font-weight: 500; }
.topbar-icons { display: flex; gap: 8px; align-items: center; }
.topbar-icon {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; color: var(--text2); cursor: pointer;
}
.topbar-avatar {
    width: 36px; height: 36px; border-radius: 50%;
    background: var(--bg-card3);
    border: 1.5px solid var(--border2);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; cursor: pointer;
}

/* ─── CONTENT WRAPPER ─── */
.main-wrap { padding: 0 20px 90px; }

/* ─── HERO ─── */
.hero {
    text-align: center;
    padding: 26px 0 20px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 700;
    letter-spacing: -0.03em; margin-bottom: 6px;
}
.hero-title span { color: var(--accent-light); }
.hero-sub { font-size: 0.88rem; color: var(--text2); }

/* ─── ORB ─── */
.orb-wrap {
    display: flex; justify-content: center;
    align-items: center; padding: 24px 0 20px;
}
.orb-outer {
    width: 190px; height: 190px;
    border-radius: 50%; position: relative;
    display: flex; align-items: center; justify-content: center;
}
.orb-ring3 {
    position: absolute; width: 190px; height: 190px; border-radius: 50%;
    background: radial-gradient(circle at 50% 55%, rgba(96,165,250,0.07) 0%, rgba(124,91,247,0.05) 45%, transparent 70%);
    animation: orbpulse3 3s ease-in-out infinite;
}
.orb-ring2 {
    position: absolute; width: 160px; height: 160px; border-radius: 50%;
    background: radial-gradient(circle at 50% 55%, rgba(96,165,250,0.09) 0%, rgba(124,91,247,0.11) 50%, transparent 75%);
    animation: orbpulse2 2.5s ease-in-out infinite 0.3s;
}
.orb-inner {
    position: absolute; width: 130px; height: 130px; border-radius: 50%;
    background: radial-gradient(circle at 42% 38%, rgba(155,125,255,0.92) 0%, rgba(96,165,250,0.75) 55%, rgba(60,130,240,0.55) 100%);
    animation: orbpulse1 2s ease-in-out infinite 0.15s;
    box-shadow: 0 0 40px rgba(124,91,247,0.45), 0 0 80px rgba(96,165,250,0.2), inset 0 -20px 40px rgba(60,100,220,0.25);
    display: flex; align-items: center; justify-content: center;
}
.orb-mic-svg {
    width: 48px; height: 54px;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,0.4));
}
@keyframes orbpulse1 { 0%,100%{transform:scale(1)} 50%{transform:scale(1.03)} }
@keyframes orbpulse2 { 0%,100%{transform:scale(1);opacity:.8} 50%{transform:scale(1.06);opacity:1} }
@keyframes orbpulse3 { 0%,100%{transform:scale(1);opacity:.6} 50%{transform:scale(1.1);opacity:1} }

/* ─── WAKE WORD BOX ─── */
.wake-box {
    background: var(--bg-card);
    border: 1px solid var(--border2);
    border-radius: 16px; padding: 16px 20px;
    text-align: center; margin-bottom: 10px;
}
.wake-label {
    font-size: 0.72rem; color: var(--text3); letter-spacing: 0.1em;
    font-weight: 500; text-transform: uppercase;
    margin-bottom: 6px;
    display: flex; align-items: center; justify-content: center; gap: 8px;
}
.wake-waves { display: flex; gap: 3px; align-items: center; }
.ww { width: 3px; background: var(--accent); border-radius: 2px; animation: wwave 0.8s ease-in-out infinite; }
.ww:nth-child(1){height:8px;animation-delay:0s}
.ww:nth-child(2){height:14px;animation-delay:.1s}
.ww:nth-child(3){height:10px;animation-delay:.2s}
.ww:nth-child(4){height:16px;animation-delay:.3s}
.ww:nth-child(5){height:10px;animation-delay:.4s}
@keyframes wwave { 0%,100%{transform:scaleY(.6);opacity:.6} 50%{transform:scaleY(1);opacity:1} }
.wake-word {
    font-family: 'Syne', sans-serif;
    font-size: 1.65rem; font-weight: 700;
    color: var(--accent-light); letter-spacing: -0.01em;
}

/* ─── LISTEN STATUS ─── */
.listen-status {
    background: var(--bg-card2); border: 1px solid var(--border);
    border-radius: 24px; padding: 9px 16px;
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 0.84rem; color: var(--text2);
    margin: 0 auto 20px; display: flex;
    width: fit-content; margin: 0 auto 20px;
}
.ldot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--accent);
    animation: dotpulse 1.5s ease-in-out infinite;
}
@keyframes dotpulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.8)} }

/* ─── AI CARD ─── */
.ai-card {
    background: var(--bg-card); border: 1px solid var(--border2);
    border-radius: 16px; padding: 16px 18px; margin-bottom: 20px;
}
.ai-card-hdr {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 10px;
}
.ai-name {
    display: flex; align-items: center; gap: 7px;
    font-weight: 600; font-size: 0.9rem; color: var(--accent-light);
}
.ai-spk {
    width: 32px; height: 32px; border-radius: 8px;
    background: var(--bg-card3); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; font-size: 0.9rem; color: var(--text2);
}
.ai-text { font-size: 0.95rem; line-height: 1.65; color: var(--text1); }
.ai-time { font-size: 0.72rem; color: var(--text3); margin-top: 8px; }

/* ─── ACTIONS GRID ─── */
.actions-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 10px; margin-bottom: 24px;
}
.action-btn {
    background: var(--bg-card); border: 1px solid var(--border2);
    border-radius: 14px; padding: 14px 8px; text-align: center;
    cursor: pointer; display: flex; flex-direction: column;
    align-items: center; gap: 8px;
    transition: all .18s;
}
.action-btn:hover { background: var(--bg-card3); border-color: rgba(124,91,247,.35); }
.act-icon {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.act-label { font-size: 0.7rem; color: var(--text2); font-weight: 500; line-height: 1.3; }

/* ─── BOTTOM NAV ─── */
.bottom-nav {
    position: fixed; bottom: 0; left: 50%; transform: translateX(-50%);
    width: 100%; max-width: 480px; height: 72px;
    background: rgba(12,12,18,0.97);
    border-top: 1px solid var(--border2);
    backdrop-filter: blur(16px);
    display: flex; align-items: center;
    justify-content: space-around;
    padding: 8px 4px 12px; z-index: 100;
}
.bn-item {
    display: flex; flex-direction: column; align-items: center;
    gap: 3px; cursor: pointer; padding: 6px 10px;
    border-radius: 12px; min-width: 56px; flex: 1;
    transition: all .15s;
}
.bn-item.active { background: rgba(124,91,247,.12); }
.bn-icon { font-size: 1.2rem; color: var(--text3); line-height: 1; }
.bn-item.active .bn-icon { color: var(--accent-light); }
.bn-label { font-size: 0.6rem; color: var(--text3); font-weight: 500; white-space: nowrap; }
.bn-item.active .bn-label { color: var(--accent-light); }

/* ─── DETECTION BOXES ─── */
.det-box {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 13px; border-radius: 9px; margin-bottom: 7px;
    font-size: 0.84rem; font-weight: 500;
}
.det-danger { background: rgba(248,113,113,.1); border: 1px solid rgba(248,113,113,.3); color: #f87171; }
.det-warn   { background: rgba(251,191,36,.1);  border: 1px solid rgba(251,191,36,.3);  color: #fbbf24; }
.det-safe   { background: rgba(34,211,168,.08); border: 1px solid rgba(34,211,168,.25); color: #22d3a8; }

/* ─── CHAT BUBBLES ─── */
.chat-you {
    display: flex; justify-content: flex-end; margin-bottom: 10px;
}
.chat-you-inner {
    background: rgba(124,91,247,.18); border: 1px solid rgba(124,91,247,.3);
    border-radius: 14px 14px 3px 14px; padding: 10px 14px;
    max-width: 80%; font-size: 0.9rem; line-height: 1.6;
}
.chat-ai { display: flex; margin-bottom: 10px; }
.chat-ai-inner {
    background: var(--bg-card); border: 1px solid var(--border2);
    border-radius: 14px 14px 14px 3px; padding: 10px 14px;
    max-width: 82%; font-size: 0.9rem; line-height: 1.6;
}
.chat-role { font-size: 0.7rem; font-weight: 600; margin-bottom: 4px; }

/* ─── PANEL CARD ─── */
.panel-card {
    background: var(--bg-card); border: 1px solid var(--border2);
    border-radius: var(--radius-lg); padding: 16px; margin-bottom: 14px;
}
.panel-title {
    font-size: 0.86rem; font-weight: 600; margin-bottom: 12px;
    display: flex; align-items: center; gap: 6px;
}
.section-head {
    font-size: 0.75rem; color: var(--text3);
    letter-spacing: 0.08em; text-transform: uppercase;
    font-weight: 500; margin-bottom: 10px; margin-left: 2px;
}

/* ─── STREAMLIT OVERRIDES ─── */
.stButton > button {
    background: rgba(124,91,247,.14) !important;
    border: 1px solid rgba(124,91,247,.4) !important;
    color: var(--accent-light) !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: .65rem 1rem !important;
    width: 100% !important;
    font-size: .88rem !important;
    transition: all .15s !important;
}
.stButton > button:hover {
    background: rgba(124,91,247,.25) !important;
    transform: translateY(-1px) !important;
}
.stTextInput input, .stTextArea textarea {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text1) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: var(--bg-card2) !important;
    border-color: var(--border2) !important;
    color: var(--text1) !important;
}
.stAlert { border-radius: 10px !important; }
[data-testid="stImage"] { border-radius: 12px; overflow: hidden; }
.stFileUploader {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 12px !important;
}
.stSlider > div { color: var(--text2) !important; }

/* Divider */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State — UNCHANGED from original
# ============================================================
def init_state():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "active_page": "dashboard",
        "is_listening": False,
        "is_speaking": False,
        "transcript_history": [],
        "ai_response": "I'm here to help you. What can I do for you?",
        "detection_results": [],
        "voice_alerts": [],
        "recent_commands": [
            {"cmd": "What is in front of me?", "time": "10:32 AM"},
            {"cmd": "Read text", "time": "10:31 AM"},
            {"cmd": "Describe the scene", "time": "10:29 AM"},
        ],
        "last_audio": None,
        "tts_voice": "en-US-JennyNeural",
        "frame_count": 0,
        "detection_frame": None,
        "last_detection_time": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ============================================================
# Async Helpers — UNCHANGED from original
# ============================================================
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

async def call_chat(message: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(f"{BACKEND_URL}/chat", json={"message": message, "session_id": st.session_state.session_id})
            return r.json().get("response", "")
    except Exception as e:
        return f"Backend connection error: {e}"

async def call_tts(text: str) -> Optional[bytes]:
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(f"{BACKEND_URL}/voice/speak", json={"text": text, "voice": st.session_state.tts_voice})
            return r.content if r.status_code == 200 else None
    except:
        return None

async def call_stt(audio_bytes: bytes) -> str:
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(f"{BACKEND_URL}/voice/transcribe", files={"audio": ("audio.wav", audio_bytes, "audio/wav")})
            return r.json().get("transcript", "")
    except:
        return ""

async def call_detect(img_bytes: bytes) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{BACKEND_URL}/vision/detect", files={"image": ("frame.jpg", img_bytes, "image/jpeg")})
            return r.json()
    except:
        return {"detections": [], "voice_alerts": [], "count": 0}

async def call_describe(img_b64: str, prompt: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(f"{BACKEND_URL}/vision/describe", json={"image_base64": img_b64, "prompt": prompt})
            return r.json().get("description", "")
    except:
        return "Vision service unavailable"

async def call_health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=4) as c:
            r = await c.get(f"{BACKEND_URL}/health")
            return r.json()
    except:
        return {"status": "offline"}


# ============================================================
# Voice Command Processor — UNCHANGED from original
# ============================================================
def process_voice_command(text: str) -> str:
    text_lower = text.lower().strip()
    if any(w in text_lower for w in ["kya hai", "what is", "describe", "batao", "bolo", "what do you see"]):
        if st.session_state.detection_frame is not None:
            buf = BytesIO()
            img = PILImage.fromarray(cv2.cvtColor(st.session_state.detection_frame, cv2.COLOR_BGR2RGB))
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode()
            response = run_async(call_describe(b64, "Describe what you see for a blind person. Focus on obstacles, people, and navigation."))
        else:
            response = run_async(call_chat(text))
    elif any(w in text_lower for w in ["object", "detect", "kya kya", "cheez", "obstacles"]):
        if st.session_state.detection_results:
            items = [d.get("guidance", "") for d in st.session_state.detection_results]
            response = "I can see: " + ", ".join(items[:5])
        else:
            response = run_async(call_chat("What objects might I encounter while walking?"))
    elif any(w in text_lower for w in ["emergency", "help", "madad", "sos"]):
        response = "Emergency mode activated. Please call 15 for police, 115 for ambulance, or 16 for fire emergency."
    elif any(w in text_lower for w in ["read text", "text parho", "likha hua"]):
        if st.session_state.detection_frame is not None:
            buf = BytesIO()
            img = PILImage.fromarray(cv2.cvtColor(st.session_state.detection_frame, cv2.COLOR_BGR2RGB))
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode()
            response = run_async(call_describe(b64, "Read and transcribe any text visible in this image."))
        else:
            response = "Please enable camera mode so I can read the text."
    else:
        response = run_async(call_chat(text)) or "I didn't catch that. Please try again."
    return response or "No response received. Please try again."


# ============================================================
# YOLO Transformer — UNCHANGED from original
# ============================================================
class YOLOTransformer(VideoTransformerBase):
    def _load_service(self):
        try:
            from backend.vision.detection_service import get_detection_service
            self.detection_service = get_detection_service()
        except Exception as e:
            self.detection_service = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        st.session_state.detection_frame = img.copy()
        st.session_state.frame_count += 1
        now = time.time()
        if hasattr(self, 'detection_service') and self.detection_service and (now - st.session_state.last_detection_time) > 2.0:
            try:
                result = self.detection_service.detect_frame(img)
                st.session_state.detection_results = [
                    {"class_name": d.class_name, "confidence": d.confidence,
                     "position": d.position, "distance": d.distance,
                     "guidance": d.guidance, "bbox": list(d.bbox)}
                    for d in result.detections
                ]
                st.session_state.voice_alerts = result.voice_alerts
                st.session_state.last_detection_time = now
                if result.annotated_frame is not None:
                    img = result.annotated_frame
            except:
                pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
# SHARED UI COMPONENTS
# ============================================================

def render_topbar(page_name="Dashboard"):
    health = run_async(call_health())
    is_online = health.get("status") == "healthy"
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-logo">VoxSense <span>AI</span></div>
        <div class="topbar-page">{page_name}</div>
        <div class="topbar-icons">
            <div class="topbar-icon">🌙</div>
            <div class="topbar-avatar">👤</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_bottom_nav():
    pages = [
        ("🏠", "Dashboard", "dashboard"),
        ("🎙", "Voice", "voice"),
        ("📷", "Camera", "camera"),
        ("💬", "AI Chat", "chat"),
        ("⚙", "Settings", "settings"),
    ]
    active = st.session_state.active_page
    nav_html = '<div class="bottom-nav">'
    for icon, label, pid in pages:
        cls = "bn-item active" if active == pid else "bn-item"
        nav_html += f'<div class="{cls}"><div class="bn-icon">{icon}</div><div class="bn-label">{label}</div></div>'
    nav_html += '</div>'
    st.markdown(nav_html, unsafe_allow_html=True)

    # Actual clickable buttons (hidden via JS trick using columns)
    cols = st.columns(len(pages))
    for i, (icon, label, pid) in enumerate(pages):
        with cols[i]:
            if st.button(f"{icon}", key=f"nav_{pid}", help=label):
                st.session_state.active_page = pid
                st.rerun()

    # Hide the visible buttons — we already rendered the custom nav above
    st.markdown("""
    <style>
    [data-testid="column"] .stButton button {
        position: fixed !important; bottom: 0 !important;
        height: 72px !important; opacity: 0 !important;
        border-radius: 0 !important; border: none !important;
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# ORB SVG
# ============================================================
ORB_HTML = """
<div class="orb-wrap">
  <div class="orb-outer">
    <div class="orb-ring3"></div>
    <div class="orb-ring2"></div>
    <div class="orb-inner">
      <svg class="orb-mic-svg" viewBox="0 0 52 60" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="16" y="2" width="20" height="34" rx="10" fill="white" opacity="0.95"/>
        <path d="M6 28C6 40 14 46 26 46C38 46 46 40 46 28"
              stroke="white" stroke-width="3.5" stroke-linecap="round" fill="none"/>
        <line x1="26" y1="46" x2="26" y2="58" stroke="white" stroke-width="3.5" stroke-linecap="round"/>
        <line x1="16" y1="58" x2="36" y2="58" stroke="white" stroke-width="3.5" stroke-linecap="round"/>
      </svg>
    </div>
  </div>
</div>
"""

WAKE_HTML = """
<div class="wake-box">
  <div class="wake-label">
    <div class="wake-waves">
      <div class="ww"></div><div class="ww"></div><div class="ww"></div>
      <div class="ww"></div><div class="ww"></div>
    </div>
    SAY WAKE UP WORD
  </div>
  <div class="wake-word">"Computer"</div>
</div>
<div style="display:flex;justify-content:center;margin-bottom:20px;">
  <div class="listen-status">
    <div class="ldot"></div> Listening for wake word...
  </div>
</div>
"""


# ============================================================
# PAGE: DASHBOARD
# ============================================================
if st.session_state.active_page == "dashboard":
    render_topbar("Dashboard")
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Hello, I'm <span>VoxSense</span></div>
        <div class="hero-sub">Your AI Voice Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    # Orb + Wake word
    st.markdown(ORB_HTML, unsafe_allow_html=True)
    st.markdown(WAKE_HTML, unsafe_allow_html=True)

    # AI Response Card
    cur_time = time.strftime("%I:%M %p")
    st.markdown(f"""
    <div class="ai-card">
      <div class="ai-card-hdr">
        <div class="ai-name"><span style="color:var(--accent-light)">✦</span> VoxSense</div>
        <div class="ai-spk">🔊</div>
      </div>
      <div class="ai-text">{st.session_state.ai_response}</div>
      <div class="ai-time">{cur_time}</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Actions
    st.markdown('<div class="section-head">Quick Actions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="actions-grid">
      <div class="action-btn">
        <div class="act-icon" style="background:rgba(96,165,250,.15)">💬</div>
        <div class="act-label">Describe Scene</div>
      </div>
      <div class="action-btn">
        <div class="act-icon" style="background:rgba(34,211,168,.15)">📄</div>
        <div class="act-label">Read Text</div>
      </div>
      <div class="action-btn">
        <div class="act-icon" style="background:rgba(251,191,36,.15)">🎯</div>
        <div class="act-label">Find Objects</div>
      </div>
      <div class="action-btn">
        <div class="act-icon" style="background:rgba(248,113,113,.15)">🧭</div>
        <div class="act-label">Navigate Me</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Functional quick action buttons
    qa_cols = st.columns(4)
    quick_actions = [
        ("📄 Read Text", "Read any text visible in front of me"),
        ("🖼 Describe", "Describe the scene around me in detail"),
        ("🎯 Find Objects", "What objects are around me?"),
        ("🧭 Navigate", "Help me navigate safely"),
    ]
    for i, (label, prompt) in enumerate(quick_actions):
        with qa_cols[i]:
            if st.button(label, key=f"qa_{i}"):
                with st.spinner("..."):
                    resp = process_voice_command(prompt)
                    st.session_state.ai_response = resp
                    st.session_state.transcript_history.append({"role": "user", "text": prompt})
                    st.session_state.transcript_history.append({"role": "ai", "text": resp})
                    audio = run_async(call_tts(resp))
                    if audio:
                        st.session_state.last_audio = audio
                st.rerun()

    if st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format="audio/mp3", autoplay=True)

    # Live Transcript (collapsible)
    if st.session_state.transcript_history:
        st.markdown('<div class="panel-card" style="margin-top:8px">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📋 Live Transcript</div>', unsafe_allow_html=True)
        for item in st.session_state.transcript_history[-4:]:
            if item["role"] == "user":
                st.markdown(f'<div style="font-size:.88rem;color:var(--text2);margin-bottom:5px"><strong style="color:var(--text1)">You:</strong> {item["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-size:.88rem;color:var(--accent-light);margin-bottom:5px"><strong>VoxSense:</strong> {item["text"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    render_bottom_nav()


# ============================================================
# PAGE: VOICE ASSISTANT
# ============================================================
elif st.session_state.active_page == "voice":
    render_topbar("Voice Assistant")
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

    st.markdown(ORB_HTML, unsafe_allow_html=True)
    st.markdown(WAKE_HTML, unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🎤 Upload Audio (Speech-to-Text)</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload WAV/MP3/WebM", type=["wav","mp3","webm","ogg","m4a"], label_visibility="collapsed")
    if uploaded:
        st.audio(uploaded)
        if st.button("🎙 Transcribe & Respond", key="do_transcribe"):
            audio_bytes = uploaded.read()
            with st.spinner("Transcribing..."):
                transcript = run_async(call_stt(audio_bytes))
            if transcript:
                st.success(f"📝 {transcript}")
                with st.spinner("Getting response..."):
                    response = process_voice_command(transcript)
                    audio_resp = run_async(call_tts(response))
                st.markdown(f'<div class="ai-card"><div class="ai-name">✦ VoxSense</div><div class="ai-text" style="margin-top:8px">{response}</div></div>', unsafe_allow_html=True)
                if audio_resp:
                    st.audio(audio_resp, format="audio/mp3", autoplay=True)
                st.session_state.transcript_history.append({"role":"user","text":transcript})
                st.session_state.transcript_history.append({"role":"ai","text":response})
            else:
                st.error("❌ Transcription failed. Check Deepgram API key.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card" style="margin-top:12px">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔊 Text-to-Speech</div>', unsafe_allow_html=True)
    tts_text = st.text_area("Type text to speak:", height=100, placeholder="VoxSense will say this...", label_visibility="collapsed")
    voice_map = {"Jenny (US Female)":"en-US-JennyNeural","Guy (US Male)":"en-US-GuyNeural","Aria (US Female)":"en-US-AriaNeural","Sonia (UK Female)":"en-GB-SoniaNeural"}
    selected_voice = st.selectbox("Voice:", list(voice_map.keys()), label_visibility="collapsed")
    st.session_state.tts_voice = voice_map[selected_voice]
    if st.button("🔊 Speak", key="speak_now") and tts_text:
        with st.spinner("Generating audio..."):
            audio = run_async(call_tts(tts_text))
        if audio:
            st.audio(audio, format="audio/mp3", autoplay=True)
        else:
            st.error("❌ Backend unavailable.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    render_bottom_nav()


# ============================================================
# PAGE: CAMERA VISION
# ============================================================
elif st.session_state.active_page == "camera":
    render_topbar("Camera Vision")
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span style="color:var(--green);background:rgba(34,211,168,.15);padding:2px 8px;border-radius:4px;font-size:.72rem;letter-spacing:.08em;font-weight:700">LIVE</span> &nbsp;Camera Vision + YOLO</div>', unsafe_allow_html=True)
    try:
        webrtc_streamer(
            key="camera-main",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=YOLOTransformer,
            media_stream_constraints={"video":{"width":{"ideal":640},"height":{"ideal":480},"facingMode":"environment"},"audio":False},
            async_processing=True,
        )
    except Exception:
        st.info("📷 Install streamlit-webrtc for camera access.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Detection results
    if st.session_state.detection_results:
        st.markdown('<div class="panel-card" style="margin-top:12px">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🎯 Detected Objects</div>', unsafe_allow_html=True)
        for det in st.session_state.detection_results[:6]:
            dist = det.get("distance","")
            css = "det-danger" if dist=="very close" else ("det-warn" if dist=="close" else "det-safe")
            st.markdown(f'<div class="det-box {css}"><span>🎯 {det.get("class_name","").title()} — {det.get("position","")}</span><span>{dist}</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Upload fallback
    st.markdown('<div class="panel-card" style="margin-top:12px">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📤 Upload Image</div>', unsafe_allow_html=True)
    img_up = st.file_uploader("Image:", type=["jpg","jpeg","png"], key="cam_upload", label_visibility="collapsed")
    if img_up:
        img = PILImage.open(img_up)
        st.image(img, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🤖 Describe Scene", key="cam_desc"):
                buf = BytesIO(); img.save(buf, format="JPEG", quality=80)
                b64 = base64.b64encode(buf.getvalue()).decode()
                with st.spinner("Analyzing..."):
                    desc = run_async(call_describe(b64, "Describe this scene for a blind person. Mention obstacles, people, distances."))
                    audio = run_async(call_tts(desc))
                st.session_state.ai_response = desc
                st.markdown(f'<div class="ai-card"><div class="ai-text">{desc}</div></div>', unsafe_allow_html=True)
                if audio: st.audio(audio, format="audio/mp3", autoplay=True)
        with c2:
            if st.button("🎯 Detect Objects", key="cam_det"):
                buf = BytesIO(); img.save(buf, format="JPEG", quality=80)
                with st.spinner("Detecting..."):
                    result = run_async(call_detect(buf.getvalue()))
                st.session_state.detection_results = result.get("detections",[])
                if result.get("voice_alerts"):
                    audio = run_async(call_tts(". ".join(result["voice_alerts"])))
                    if audio: st.audio(audio, format="audio/mp3", autoplay=True)
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    render_bottom_nav()


# ============================================================
# PAGE: AI CHAT
# ============================================================
elif st.session_state.active_page == "chat":
    render_topbar("AI Chat")
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

    # Messages
    if not st.session_state.transcript_history:
        st.markdown("""
        <div style="text-align:center;padding:50px 20px;color:var(--text3)">
            <div style="font-size:2.5rem;margin-bottom:12px">🎙</div>
            <div style="font-size:.9rem">Ask VoxSense anything — I'm here to help</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.transcript_history[-12:]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-you"><div class="chat-you-inner"><div class="chat-role" style="color:var(--text3)">YOU</div>{msg["text"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai"><div class="chat-ai-inner"><div class="chat-role" style="color:var(--accent-light)">VOXSENSE</div>{msg["text"]}</div></div>', unsafe_allow_html=True)

    # Input row
    col_inp, col_send = st.columns([4, 1])
    with col_inp:
        user_msg = st.text_input("Message", placeholder="Ask anything...", label_visibility="collapsed", key="chat_msg")
    with col_send:
        if st.button("➤ Send", key="chat_send"):
            if user_msg.strip():
                with st.spinner("..."):
                    resp = process_voice_command(user_msg)
                    audio = run_async(call_tts(resp))
                st.session_state.ai_response = resp
                st.session_state.transcript_history.append({"role":"user","text":user_msg})
                st.session_state.transcript_history.append({"role":"ai","text":resp})
                st.session_state.recent_commands.append({"cmd":user_msg[:40],"time":time.strftime("%I:%M %p")})
                if audio: st.session_state.last_audio = audio
                st.rerun()

    if st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format="audio/mp3")

    # Quick commands
    st.markdown('<div class="section-head" style="margin-top:16px">Quick Commands</div>', unsafe_allow_html=True)
    qcols = st.columns(2)
    quick = ["What is in front of me?", "Navigate me safely", "Emergency help", "Describe the room", "Find the stairs", "Read text near me"]
    for i, q in enumerate(quick):
        with qcols[i % 2]:
            if st.button(q, key=f"qc_{i}"):
                with st.spinner("..."):
                    resp = process_voice_command(q)
                    audio = run_async(call_tts(resp))
                st.session_state.transcript_history.append({"role":"user","text":q})
                st.session_state.transcript_history.append({"role":"ai","text":resp})
                if audio: st.session_state.last_audio = audio
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    render_bottom_nav()


# ============================================================
# PAGE: SETTINGS
# ============================================================
elif st.session_state.active_page == "settings":
    render_topbar("Settings")
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔑 API Keys</div>', unsafe_allow_html=True)
    try:
        from config.settings import settings as app_settings
        keys = {"Groq API (LLM)":bool(app_settings.groq_api_key),"OpenAI API (Vision)":bool(app_settings.openai_api_key),"Deepgram (STT)":bool(app_settings.deepgram_api_key)}
        for name, ok in keys.items():
            color = "var(--green)" if ok else "var(--red)"
            icon = "✓" if ok else "✗"
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:10px;background:var(--bg-card2);border-radius:9px;margin-bottom:6px;border:1px solid var(--border)"><span style="font-size:.85rem">{name}</span><span style="color:{color};font-weight:600">{icon} {"Set" if ok else "Missing"}</span></div>', unsafe_allow_html=True)
    except ImportError:
        st.info("config.settings not found.")

    if st.button("🔄 Test Backend"):
        h = run_async(call_health())
        if h.get("status") == "healthy": st.success("✅ Backend connected!")
        else: st.error("❌ Backend offline — run: uvicorn backend.main:app --port 8000")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card" style="margin-top:12px">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔊 Voice Settings</div>', unsafe_allow_html=True)
    voice_options = {"Jenny (US Female)":"en-US-JennyNeural","Guy (US Male)":"en-US-GuyNeural","Aria (US Female)":"en-US-AriaNeural","Davis (US Male)":"en-US-DavisNeural","Sonia (UK Female)":"en-GB-SoniaNeural"}
    sel = st.selectbox("TTS Voice:", list(voice_options.keys()))
    st.session_state.tts_voice = voice_options[sel]
    test_text = st.text_input("Test text:", value="Hello! I am VoxSense. I am here to help you.")
    if st.button("🔊 Test Voice"):
        audio = run_async(call_tts(test_text))
        if audio: st.audio(audio, format="audio/mp3", autoplay=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card" style="margin-top:12px">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🎯 Detection Settings</div>', unsafe_allow_html=True)
    st.slider("Min Confidence Threshold", 0.3, 0.9, 0.5, 0.05)
    st.slider("Alert Cooldown (seconds)", 1, 10, 3)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card" style="margin-top:12px">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🚀 Start Commands</div>', unsafe_allow_html=True)
    st.code("uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload", language="bash")
    st.code("streamlit run frontend/app.py --server.port 8501", language="bash")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    render_bottom_nav()

else:
    render_topbar(st.session_state.active_page.title())
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-card"><div class="panel-title">{st.session_state.active_page.title()}</div><div style="color:var(--text3)">Coming soon...</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    render_bottom_nav()