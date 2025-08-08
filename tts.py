# tts.py - Text-to-Speech Synthesis using Cartesia API
"""
This module implements text-to-speech functionality using Cartesia's streaming TTS service.

The CartesiaTTS class provides real-time speech synthesis with:
- Streaming text input processing
- Intelligent text accumulation for natural speech
- High-quality voice synthesis using Cartesia's Sonic model
- Real-time audio output suitable for conversational AI
- Support for multiple audio formats and sample rates

The implementation accumulates text fragments into complete sentences before synthesis
to ensure natural-sounding speech output.
"""

import asyncio
import numpy as np
from cartesia import AsyncCartesia
from components import TTSInterface
from typing import AsyncGenerator
import os

class CartesiaTTS(TTSInterface):
    """
    Cartesia-powered Text-to-Speech synthesis implementation.
    
    This class provides real-time speech synthesis using Cartesia's streaming TTS API.
    It intelligently accumulates text fragments into complete sentences before
    synthesis to ensure natural-sounding speech output.
    
    Features:
    - Real-time streaming text-to-speech conversion
    - Intelligent sentence boundary detection
    - High-quality voice synthesis with Cartesia's Sonic model
    - Configurable voice selection and audio parameters
    - WebSocket-based communication for low latency
    """
    
    def __init__(self, sample_rate: int = 24000, voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091"):
        """
        Initialize the Cartesia TTS client.
        
        Args:
            sample_rate: Output audio sample rate in Hz (24kHz for high quality)
            voice_id: Cartesia voice ID for synthesis
        """
        self.api_key = os.getenv("CARTESIA_API_KEY")
        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY not found in environment")
        
        self.sample_rate = sample_rate
        self.voice_id = voice_id
        self.client = AsyncCartesia(api_key=self.api_key)

    async def synthesize_stream(self, text_queue: asyncio.Queue) -> AsyncGenerator[bytes, None]:
        """
        Convert streaming text to synthesized speech audio.
        
        This method consumes text chunks from the LLM and accumulates them into
        complete sentences before sending them for synthesis. This approach
        ensures more natural-sounding speech output.
        
        Args:
            text_queue: Queue containing text chunks from LLM and END signal
            
        Yields:
            bytes: Audio chunks in PCM format for real-time playback
        """
        buffer = ""
        
        while True:
            try:
                # Get next text chunk from LLM
                text = await text_queue.get()
                
                if text == "END":
                    # End signal received - synthesize any remaining text
                    if buffer.strip():
                        async for chunk in self._synthesize_sentence(buffer):
                            yield chunk
                    break
                
                # Accumulate text into buffer
                buffer += " " + text.strip() if buffer else text.strip()
                print(f"[TTS] Accumulated text: '{buffer}'")
                
                # Check if we should synthesize the current buffer
                if self._should_send(buffer):
                    async for audio_chunk in self._synthesize_sentence(buffer):
                        yield audio_chunk
                    buffer = ""  # Reset buffer after synthesis
            
            except Exception as e:
                print(f"[TTS] Error in text accumulation: {e}")
                continue
        
        print("[TTS] Synthesis completed.")
    def _should_send(self, text: str) -> bool:
        """
        Determine if accumulated text should be sent for synthesis.
        
        This method checks for sentence-ending punctuation to decide when
        to synthesize the accumulated text. This ensures natural speech
        patterns and prevents synthesis of incomplete thoughts.
        
        Args:
            text: The accumulated text buffer
            
        Returns:
            bool: True if text should be synthesized now
        """
        return text.rstrip().endswith(('.', '!', '?', '。', '！', '？'))

    async def _synthesize_sentence(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize a complete sentence using Cartesia's TTS API.
        
        This method sends the text to Cartesia's Sonic model for synthesis
        and yields the resulting audio chunks for real-time playback.
        
        Args:
            text: Complete sentence or phrase to synthesize
            
        Yields:
            bytes: Audio chunks in PCM float32 little-endian format
        """
        try:
            # Establish WebSocket connection to Cartesia TTS service
            ws = await self.client.tts.websocket()
            
            # Send synthesis request and stream audio response
            async for output in await ws.send(
                model_id="sonic-2",           # Cartesia's Sonic model
                transcript=text,              # Text to synthesize
                voice={"id": self.voice_id},  # Selected voice
                stream=True,                  # Enable streaming output
                output_format={
                    "container": "raw",       # Raw audio format
                    "encoding": "pcm_f32le",  # 32-bit float PCM little-endian
                    "sample_rate": self.sample_rate,  # 24kHz sample rate
                },
            ):
                audio_bytes = output.audio  # Raw audio data
                if audio_bytes:
                    yield audio_bytes
            
            await ws.close()
            print(f"[TTS] Synthesized: '{text}'")
        
        except Exception as e:
            print(f"[TTS] Error synthesizing '{text}': {e}")
