# asr.py - Automatic Speech Recognition using Cartesia API
"""
This module implements speech recognition functionality using Cartesia's streaming ASR service.

The CartesiaASR class provides real-time speech-to-text conversion with support for:
- Streaming audio input via WebSocket
- Real-time transcription with partial and final results
- Word-level timestamps for precise audio-text alignment
- Optimized for Chinese language processing
- Low-latency streaming suitable for conversational AI

The implementation uses Cartesia's "ink-whisper" model which is based on OpenAI's Whisper
but optimized for real-time streaming applications.
"""

import asyncio
import os
from cartesia import AsyncCartesia
from components import ASRInterface
from typing import AsyncGenerator

from conversation_manager import ConversationManager, PipelineState
class CartesiaASR(ASRInterface):
    """
    Cartesia-powered Automatic Speech Recognition implementation.
    
    This class provides real-time speech recognition using Cartesia's streaming ASR API.
    It converts audio streams to text with low latency, making it suitable for
    real-time voice applications.
    
    Features:
    - Real-time streaming transcription
    - Support for partial and final transcription results
    - Word-level timing information
    - Optimized for Chinese language processing
    - WebSocket-based communication for low latency
    """
    
    def __init__(self,
                 conversation_manager: ConversationManager
                 ):
        """
        Initialize the Cartesia ASR client.
        
        Requires CARTESIA_API_KEY environment variable to be set.
        """
        self.conversation_manager = conversation_manager
        self.current_session_id=0
        self.client = AsyncCartesia(api_key=os.getenv("CARTESIA_API_KEY"))

    async def transcribe_stream(self, audio_queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """
        Transcribe streaming audio to text using Cartesia's ASR service.
        
        This method consumes audio chunks from a queue and yields text transcriptions
        as they become available. It uses WebSocket for real-time communication
        with the Cartesia ASR service.
        
        Args:
            audio_queue: Queue containing audio chunks (bytes) and END signal
            
        Yields:
            str: Text transcriptions (final results only for pipeline efficiency)
        """
        # Establish WebSocket connection to Cartesia ASR service
        ws = await self.client.stt.websocket(
            model="ink-whisper",          # Cartesia's streaming Whisper model
            language="zh",                # Chinese language optimization
            encoding="pcm_s16le",         # 16-bit PCM little-endian format
            sample_rate=16000,            # 16kHz sample rate
            min_volume=0.15,              # Volume threshold for voice activity detection. Audio below this threshold will be considered silence. Range: 0.0-1.0.
            max_silence_duration_secs=2.0, # Maximum duration of silence (in seconds) before the system considers the utterance complete and triggers endpointing. Higher values allow for longer pauses within utterances.

        )

        async def sender():
            """
            Send audio chunks to the ASR service via WebSocket.
            
            This coroutine continuously reads audio chunks from the queue
            and sends them to the Cartesia ASR service for processing.
            """
            try:
                while True:
                    chunk = await audio_queue.get()
                    await ws.send(chunk)
            except Exception as e:
                print(f"[ASR Sender] Error: {e}")

        async def receiver():
            """
            Receive transcription results from the ASR service.
            
            This coroutine processes the streaming results from Cartesia,
            including partial transcriptions, final transcriptions, and
            word-level timing information.
            """
            full_transcript = ""
            all_word_timestamps = []
            try:
                async for result in ws.receive():
                    if result['type'] == 'transcript':
                        text = result['text'].strip()
                        is_final = result.get('is_final', False)
                        if text:
                            if self.conversation_manager:
                                current_state = self.conversation_manager.state
                                if current_state == PipelineState.RESPONDING:
                                    print(f"[ASR] User speaking detected during response, triggering interrupt...")
                                    await self.conversation_manager.trigger_interrupt()
                                    # 🔥 等待中断完成后再继续
                                    print(f"[ASR] Interrupt completed, continuing with new input...")
                                self.current_session_id = self.conversation_manager.get_current_session_id()

                        # Process word-level timestamps if available
                        if 'words' in result and result['words']:
                            word_timestamps = result['words']
                            all_word_timestamps.extend(word_timestamps)
                            
                            if is_final:
                                print("Word-level timestamps:")
                                for word_info in word_timestamps:
                                    word = word_info['word']
                                    start = word_info['start']
                                    end = word_info['end']
                                    print(f"  '{word}': {start:.2f}s - {end:.2f}s")
                        
                        # Only process non-empty text
                        if not text:
                            continue
                        if is_final:
                            full_transcript += text + " "
                            if self.conversation_manager:
                                if self.current_session_id == self.conversation_manager.get_current_session_id():
                                    self.conversation_manager.set_state(PipelineState.PROCESSING)
                            yield text
                        else:
                            # Optional: could yield partial results for real-time feedback
                            # Currently skipped to reduce noise in the pipeline
                            pass
                    elif result['type'] == 'done':
                        # ASR session completed
                        break
            except Exception as e:
                print(f"[ASR Receiver] Error: {e}")
            finally:
                await ws.close()

        # Run sender and receiver concurrently
        # This allows simultaneous audio sending and result receiving
        sender_task = asyncio.create_task(sender())
        async for text in receiver():
            yield text

        await sender_task
        await self.client.close()
