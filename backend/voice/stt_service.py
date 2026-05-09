# ============================================================
# VoxSense - Speech-to-Text Service (Deepgram)
# ============================================================

import asyncio
import json
from typing import Optional, Callable, AsyncGenerator

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    PrerecordedOptions,
    FileSource,
)

import sys
sys.path.insert(0, '/home/claude/voxsense')
from config.settings import settings, config
from backend.utils.logger import logger


class STTService:
    """
    Speech-to-Text service using Deepgram.
    Supports both prerecorded and live streaming transcription.
    """

    def __init__(self):
        if not settings.deepgram_api_key:
            logger.warning("⚠️ Deepgram API key not set. STT will be unavailable.")
            self.client = None
        else:
            options = DeepgramClientOptions(verbose=False)
            self.client = DeepgramClient(settings.deepgram_api_key, options)

        stt_cfg = config.get("speech", {}).get("stt", {})
        self.model = stt_cfg.get("model", "nova-2")
        self.language = stt_cfg.get("language", "en-US")
        self.smart_format = stt_cfg.get("smart_format", True)
        logger.info(f"🎙️ STT Service initialized | Model: {self.model}")

    async def transcribe_audio(self, audio_data: bytes, mimetype: str = "audio/wav") -> str:
        """
        Transcribe a complete audio buffer (prerecorded).
        Returns transcript string.
        """
        if not self.client:
            return ""

        try:
            logger.info(f"🎙️ Transcribing {len(audio_data)} bytes of audio")
            source: FileSource = {"buffer": audio_data, "mimetype": mimetype}

            options = PrerecordedOptions(
                model=self.model,
                language=self.language,
                smart_format=self.smart_format,
                punctuate=True,
                diarize=False,
                utterances=False,
            )

            response = await asyncio.to_thread(
                self.client.listen.prerecorded.v("1").transcribe_file,
                source,
                options
            )

            transcript = response.results.channels[0].alternatives[0].transcript
            logger.info(f"✅ Transcript: '{transcript[:80]}...' " if len(transcript) > 80 else f"✅ Transcript: '{transcript}'")
            return transcript

        except Exception as e:
            logger.error(f"❌ STT transcription error: {e}")
            return ""

    async def create_live_connection(
        self,
        on_transcript: Callable[[str, bool], None],
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Create a live WebSocket connection to Deepgram for real-time STT.
        
        Args:
            on_transcript: Callback receiving (text, is_final)
            on_error: Optional error callback
        Returns:
            Deepgram live connection object
        """
        if not self.client:
            logger.error("❌ Deepgram client not initialized")
            return None

        try:
            connection = self.client.listen.asynclive.v("1")

            async def on_message(self_inner, result, **kwargs):
                try:
                    sentence = result.channel.alternatives[0].transcript
                    is_final = result.is_final
                    if sentence.strip():
                        await on_transcript(sentence, is_final)
                except Exception as e:
                    logger.error(f"❌ STT message handler error: {e}")

            async def on_error_handler(self_inner, error, **kwargs):
                logger.error(f"❌ Deepgram live error: {error}")
                if on_error:
                    await on_error(Exception(str(error)))

            connection.on(LiveTranscriptionEvents.Transcript, on_message)
            connection.on(LiveTranscriptionEvents.Error, on_error_handler)

            options = LiveOptions(
                model=self.model,
                language=self.language,
                smart_format=self.smart_format,
                interim_results=True,
                utterance_end_ms="1000",
                vad_events=True,
                endpointing=300,
            )

            started = await connection.start(options)
            if started:
                logger.info("🎙️ Deepgram live connection established")
            else:
                logger.error("❌ Failed to start Deepgram live connection")
                return None

            return connection

        except Exception as e:
            logger.error(f"❌ Failed to create live STT connection: {e}")
            return None


# Singleton
_stt_service: Optional[STTService] = None


def get_stt_service() -> STTService:
    global _stt_service
    if _stt_service is None:
        _stt_service = STTService()
    return _stt_service
