# ============================================================
# VoxSense - Memory Service (ChromaDB)
# ============================================================

import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any

import sys
sys.path.insert(0, '/home/claude/voxsense')
from config.settings import settings, config
from backend.utils.logger import logger


class MemoryService:
    """
    Conversation memory using ChromaDB for semantic search.
    Stores and retrieves conversation history and user preferences.
    """

    def __init__(self):
        self.client = None
        self.collection = None
        self._init_chroma()

    def _init_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            persist_dir = settings.chroma_persist_dir
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            collection_name = config.get("memory", {}).get("chroma", {}).get(
                "collection_name", "voxsense_memory"
            )

            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"🧠 Memory Service initialized | Collection: {collection_name}")
        except ImportError:
            logger.warning("⚠️ chromadb not installed. Memory features unavailable.")
        except Exception as e:
            logger.error(f"❌ ChromaDB initialization error: {e}")

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Store a conversation message."""
        if not self.collection:
            return ""

        try:
            msg_id = str(uuid.uuid4())
            meta = {
                "session_id": session_id,
                "role": role,
                "timestamp": time.time(),
                "content_preview": content[:100],
            }
            if metadata:
                meta.update(metadata)

            self.collection.add(
                ids=[msg_id],
                documents=[content],
                metadatas=[meta],
            )
            logger.debug(f"🧠 Stored message {msg_id[:8]}... [{role}]")
            return msg_id

        except Exception as e:
            logger.error(f"❌ Memory add error: {e}")
            return ""

    def search_similar(
        self,
        query: str,
        session_id: Optional[str] = None,
        n_results: int = 5,
    ) -> List[Dict]:
        """
        Semantic search for similar past messages.
        Returns list of relevant memories.
        """
        if not self.collection:
            return []

        try:
            where = {"session_id": session_id} if session_id else None
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where=where,
            )

            memories = []
            if results["documents"]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    memories.append({
                        "content": doc,
                        "role": meta.get("role", "unknown"),
                        "timestamp": meta.get("timestamp", 0),
                        "session_id": meta.get("session_id", ""),
                    })

            logger.debug(f"🧠 Found {len(memories)} similar memories")
            return memories

        except Exception as e:
            logger.error(f"❌ Memory search error: {e}")
            return []

    def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history for a session."""
        if not self.collection:
            return []

        try:
            results = self.collection.get(
                where={"session_id": session_id},
                limit=limit,
            )

            history = []
            if results["documents"]:
                pairs = zip(results["documents"], results["metadatas"])
                sorted_pairs = sorted(pairs, key=lambda x: x[1].get("timestamp", 0))
                for doc, meta in sorted_pairs:
                    history.append({
                        "role": meta.get("role", "user"),
                        "content": doc,
                        "timestamp": meta.get("timestamp", 0),
                    })

            return history

        except Exception as e:
            logger.error(f"❌ Session history error: {e}")
            return []

    def get_conversation_messages(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get messages formatted for LLM context."""
        history = self.get_session_history(session_id, limit=limit)
        return [
            {"role": item["role"], "content": item["content"]}
            for item in history
            if item["role"] in ("user", "assistant")
        ]

    def clear_session(self, session_id: str) -> bool:
        """Delete all messages for a session."""
        if not self.collection:
            return False

        try:
            results = self.collection.get(where={"session_id": session_id})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"🧠 Cleared session {session_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Session clear error: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get memory collection statistics."""
        if not self.collection:
            return {"available": False}
        try:
            return {
                "available": True,
                "total_messages": self.collection.count(),
            }
        except Exception:
            return {"available": False}


# Singleton
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
