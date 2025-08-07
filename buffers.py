# buffers.py
import asyncio
from typing import Dict

class BufferManager:
    def __init__(self):
        # 各阶段队列
        self.audio_buffer = asyncio.Queue(maxsize=10)         # Audio Source → ASR
        self.text_buffer = asyncio.Queue(maxsize=10)          # ASR → LLM
        self.llm_response_buffer = asyncio.Queue(maxsize=20)  # LLM → TTS
        self.playback_buffer = asyncio.Queue(maxsize=20)      # TTS → Player

    def get_queue(self, name: str):
        return getattr(self, f"{name}_buffer")