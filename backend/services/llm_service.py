# ============================================================
# VoxSense - LLM Service (Groq Primary + OpenAI Fallback)
# ============================================================

import asyncio
from typing import AsyncGenerator, List, Dict, Optional

from groq import AsyncGroq
from openai import AsyncOpenAI

import sys
sys.path.insert(0, '/home/claude/voxsense')
from config.settings import settings, config
from backend.utils.logger import logger


SYSTEM_PROMPT = config.get("llm", {}).get("system_prompt", """
You are VoxSense, an AI assistant for blind and visually impaired users. 
Be concise, clear, and always prioritize safety in your responses.
""").strip()


class LLMService:
    """
    Manages LLM interactions with Groq as primary and OpenAI as fallback.
    Supports streaming responses.
    """

    def __init__(self):
        self.groq_client = AsyncGroq(api_key=settings.groq_api_key) if settings.groq_api_key else None
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.groq_model = config.get("llm", {}).get("groq", {}).get("model", "llama3-70b-8192")
        self.openai_model = config.get("llm", {}).get("openai", {}).get("model", "gpt-4o")
        self.max_tokens = config.get("llm", {}).get("groq", {}).get("max_tokens", 1024)
        self.temperature = config.get("llm", {}).get("groq", {}).get("temperature", 0.7)
        logger.info(f"🤖 LLM Service initialized | Groq: {'✓' if self.groq_client else '✗'} | OpenAI: {'✓' if self.openai_client else '✗'}")

    def _build_messages(self, user_message: str, history: Optional[List[Dict]] = None) -> List[Dict]:
        """Build message list with system prompt and history."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-10:])  # last 10 turns
        messages.append({"role": "user", "content": user_message})
        return messages

    async def stream_response(
        self,
        user_message: str,
        history: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream AI response, trying Groq first then falling back to OpenAI.
        Yields text chunks as they arrive.
        """
        messages = self._build_messages(user_message, history)

        # Try Groq first
        if self.groq_client:
            try:
                logger.info(f"📡 Streaming response via Groq ({self.groq_model})")
                async with self.groq_client.chat.completions.stream(
                    model=self.groq_model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                ) as stream:
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return
            except Exception as e:
                logger.warning(f"⚠️ Groq failed, falling back to OpenAI: {e}")

        # Fallback to OpenAI
        if self.openai_client:
            try:
                logger.info(f"📡 Streaming response via OpenAI ({self.openai_model})")
                stream = await self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=True,
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                logger.error(f"❌ OpenAI also failed: {e}")

        yield "I'm sorry, I'm having trouble connecting to my AI services. Please check your API keys."

    async def complete(
        self,
        user_message: str,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """Get complete (non-streaming) response."""
        full_response = ""
        async for chunk in self.stream_response(user_message, history):
            full_response += chunk
        return full_response

    async def describe_image(self, image_base64: str, prompt: str = "Describe this scene for a blind person.") -> str:
        """
        Use OpenAI Vision to describe an image.
        Returns a natural language description.
        """
        if not self.openai_client:
            return "Vision service unavailable. Please provide an OpenAI API key."

        try:
            logger.info("🔍 Sending image to OpenAI Vision")
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant helping blind users understand their surroundings. Describe scenes clearly, concisely, and focus on safety-relevant details."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                max_tokens=500,
            )
            description = response.choices[0].message.content
            logger.info("✅ Vision description received")
            return description
        except Exception as e:
            logger.error(f"❌ Vision API error: {e}")
            return f"Unable to analyze the image: {str(e)}"


# Singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
