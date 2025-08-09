# components.py - Abstract Base Classes for Voice Agent Pipeline
"""
This module defines the abstract interfaces for all components in the voice agent pipeline.
These interfaces ensure modularity, extensibility, and consistent API design across
all implementations.

The interfaces define the contract for:
- Audio input sources
- Speech recognition systems  
- Language model processors
- Text-to-speech synthesizers
- Audio output players
"""

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any

class AudioSource(ABC):
    """
    Abstract base class for audio input sources.
    
    Implementations should capture audio from various sources like:
    - Real-time microphone input
    - Audio files
    - Network audio streams
    """
    
    @abstractmethod
    async def stream_audio(self) -> AsyncGenerator[bytes, None]:
        """
        Stream audio data as chunks.
        
        Yields:
            bytes: Audio chunks in PCM format (typically 100ms duration)
        """
        pass


class ASRInterface(ABC):
    """
    Abstract base class for Automatic Speech Recognition systems.
    
    Implementations should convert audio streams to text transcripts
    with support for real-time streaming recognition.
    """
    
    @abstractmethod
    async def transcribe_stream(self, audio_queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """
        Convert streaming audio to text transcriptions.
        
        Args:
            audio_queue: Queue containing audio chunks to transcribe
            
        Yields:
            str: Text transcriptions (partial or final results)
        """
        pass


class LLMInterface(ABC):
    """
    Abstract base class for Large Language Model processors.
    
    Implementations should generate conversational responses from user input
    with support for streaming output and conversation history.
    """
    
    @abstractmethod
    async def generate_stream(self, text_queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """
        Generate conversational responses from user input.
        
        Args:
            text_queue: Queue containing user text input
            
        Yields:
            str: Response text chunks for streaming output
        """
        pass


class TTSInterface(ABC):
    """
    Abstract base class for Text-to-Speech synthesis systems.
    
    Implementations should convert text to natural speech audio
    with support for streaming synthesis.
    """
    
    @abstractmethod
    async def synthesize_stream(self, text_queue: asyncio.Queue, audio_queue: asyncio.Queue) -> AsyncGenerator[bytes, None]:
        """
        Convert text to synthesized speech audio.
        
        Args:
            text_queue: Queue containing text chunks to synthesize
            
        Yields:
            bytes: Audio chunks in PCM format
        """
        pass


class AudioPlayer(ABC):
    """
    Abstract base class for audio output players.
    
    Implementations should handle real-time playback of audio streams
    with buffering and low-latency requirements.
    """
    
    @abstractmethod
    async def play_audio(self, audio_queue: asyncio.Queue) -> None:
        """
        Play audio chunks from a queue in real-time.
        
        Args:
            audio_queue: Queue containing audio chunks to play
        """
        pass