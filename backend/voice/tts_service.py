# ============================================================
# VoxSense - Text-to-Speech Service (Edge-TTS)
# ============================================================

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional, AsyncGenerator

import edge_tts

import sys
sys.path.insert(0, '/home/claude/voxsense')
from config.settings import settings, config
from backend.utils.logger import logger


class TTSService:
    """
    Text-to-Speech service using Microsoft Edge-TTS.
    Supports streaming audio output.
    """

    def __init__(self):
        tts_cfg = config.get("speech", {}).get("tts", {})
        self.voice = tts_cfg.get("voice", "en-US-JennyNeural")
        self.rate = tts_cfg.get("rate", "+0%")
        self.pitch = tts_cfg.get("pitch", "+0Hz")
        self.volume = tts_cfg.get("volume", "+0%")
        logger.info(f"🔊 TTS Service initialized | Voice: {self.voice}")

    async def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Convert text to audio bytes (MP3).
        Returns complete audio buffer.
        """
        if not text.strip():
            return b""

        try:
            logger.info(f"🔊 Synthesizing: '{text[:60]}...' " if len(text) > 60 else f"🔊 Synthesizing: '{text}'")

            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                pitch=self.pitch,
                volume=self.volume,
            )

            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])

            audio_bytes = audio_buffer.getvalue()
            logger.info(f"✅ TTS generated {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            logger.error(f"❌ TTS synthesis error: {e}")
            return b""

    async def stream_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks as they are generated.
        Yields bytes chunks.
        """
        if not text.strip():
            return

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                pitch=self.pitch,
                volume=self.volume,
            )

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        except Exception as e:
            logger.error(f"❌ TTS stream error: {e}")

    async def synthesize_to_file(self, text: str, output_path: str) -> bool:
        """Save TTS audio to a file."""
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                pitch=self.pitch,
                volume=self.volume,
            )
            await communicate.save(output_path)
            logger.info(f"✅ TTS saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ TTS save error: {e}")
            return False

    async def list_voices(self) -> list:
        """List available Edge-TTS voices."""
        try:
            voices = await edge_tts.list_voices()
            return [v for v in voices if v.get("Locale", "").startswith("en-")]
        except Exception as e:
            logger.error(f"❌ Failed to list voices: {e}")
            return []

    def set_voice(self, voice: str):
        """Update the TTS voice."""
        self.voice = voice
        logger.info(f"🔊 TTS voice updated to: {voice}")

    def set_rate(self, rate: str):
        """Update speech rate (e.g., '+20%', '-10%')."""
        self.rate = rate

    def set_pitch(self, pitch: str):
        """Update pitch (e.g., '+5Hz', '-10Hz')."""
        self.pitch = pitch


# Singleton
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
