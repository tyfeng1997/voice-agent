# components.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any

class AudioSource(ABC):
    @abstractmethod
    async def stream_audio(self) -> AsyncGenerator[bytes, None]:
        """Yield audio chunks (e.g., 100ms PCM)"""
        pass


class ASRInterface(ABC):
    @abstractmethod
    async def transcribe_stream(self, audio_queue: 'asyncio.Queue') -> AsyncGenerator[str, None]:
        """
        Consume audio chunks from queue, yield partial/final text
        Yields: partial text, final text (marked if needed)
        """
        pass


class LLMInterface(ABC):
    @abstractmethod
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Take full user utterance, yield partial responses
        """
        pass


class TTSInterface(ABC):
    @abstractmethod
    async def synthesize_stream(self, text_queue: 'asyncio.Queue') -> AsyncGenerator[bytes, None]:
        """
        Consume partial text chunks, yield audio chunks
        """
        pass


class AudioPlayer(ABC):
    @abstractmethod
    async def play_audio(self, audio_queue: 'asyncio.Queue') -> None:
        """
        Consume audio chunks and play them
        """
        pass