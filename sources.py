# sources.py - Audio Input Sources for Voice Agent Pipeline
"""
This module implements audio input sources for capturing real-time audio data.

The main implementation is RealTimeMicrophoneSource which captures audio from
the system microphone using sounddevice and provides it as a stream of audio chunks
suitable for real-time speech recognition.

Key features:
- Low-latency audio capture (100ms chunks)
- Cross-platform compatibility via sounddevice
- Thread-safe buffering for real-time streaming
- Configurable audio parameters (sample rate, channels, etc.)
"""

import asyncio
import os
import sounddevice as sd
import numpy as np
from components import AudioSource
from typing import AsyncGenerator, Any
import threading
from collections import deque
class RealTimeMicrophoneSource(AudioSource):
    """
    Real-time microphone audio source implementation.
    
    Captures audio from the system microphone and provides it as a continuous stream
    of audio chunks. Uses sounddevice for cross-platform audio input with low latency.
    
    The audio is captured in a separate thread and buffered in a thread-safe queue
    for consumption by the async audio processing pipeline.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000, 
                 channels: int = 1,
                 chunk_duration_ms: int = 100,  # 100ms chunks
                 device: int = None):
        """
        Initialize the real-time microphone source.
        
        Args:
            sample_rate: Audio sample rate in Hz (16kHz is optimal for speech)
            channels: Number of audio channels (1 for mono, 2 for stereo)
            chunk_duration_ms: Duration of each audio chunk in milliseconds
            device: Audio device ID (None for system default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.device = device
        
        # Calculate samples per chunk based on sample rate and duration
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        # Calculate bytes per chunk for 16-bit PCM audio
        self.chunk_bytes = self.chunk_samples * channels * 2
        
        # Thread-safe audio buffer for storing captured audio chunks
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.recording = False
        self.stream = None
        
        print(f"[MicSource] Initialized: {sample_rate}Hz, {channels}ch, "
              f"{chunk_duration_ms}ms chunks ({self.chunk_samples} samples)")

    def _audio_callback(self, indata, frames, time, status):
        """
        Sounddevice audio input callback function.
        
        This callback is called by sounddevice in a separate thread whenever
        new audio data is available from the microphone.
        
        Args:
            indata: Input audio data as numpy array
            frames: Number of audio frames
            time: Time information
            status: Stream status flags
        """
        if status:
            print(f"[MicSource] Recording status: {status}")
        
        if self.recording:
            # Convert from float32 [-1.0, 1.0] to int16 PCM format
            audio_int16 = (indata * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Thread-safe addition to buffer
            with self.buffer_lock:
                self.audio_buffer.append(audio_bytes)

    async def stream_audio(self) -> AsyncGenerator[bytes, None]:
        """
        Start audio recording and stream audio data continuously.
        
        This is the main interface for the audio source. It starts the microphone
        recording and yields audio chunks as they become available.
        
        Yields:
            bytes: Audio chunks in 16-bit PCM format
        """
        try:
            # Initialize and start the audio input stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=self.chunk_samples,
                device=self.device,
                latency='low'  # Optimize for low latency
            )
            
            self.stream.start()
            self.recording = True
            print("[MicSource] Started recording...")
            
            # Continuously yield audio chunks as they become available
            while self.recording:
                with self.buffer_lock:
                    if self.audio_buffer:
                        audio_chunk = self.audio_buffer.popleft()
                        yield audio_chunk
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
        
        except Exception as e:
            print(f"[MicSource] Recording error: {e}")
        finally:
            self.stop_recording()

    def stop_recording(self):
        """
        Stop audio recording and clean up resources.
        
        This method stops the audio stream, closes it, and resets the recording state.
        It's automatically called when stream_audio() exits.
        """
        self.recording = False
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("[MicSource] Recording stopped")

    async def record_for_duration(self, duration_seconds: float) -> AsyncGenerator[bytes, None]:
        """
        Record audio for a specific duration.
        
        This is a utility method for recording a fixed amount of audio,
        useful for testing or batch processing scenarios.
        
        Args:
            duration_seconds: How long to record in seconds
            
        Yields:
            bytes: Audio chunks during the recording period
        """
        import time
        start_time = time.time()
        
        async for chunk in self.stream_audio():
            yield chunk
            
            # Check if we've exceeded the specified duration
            if time.time() - start_time >= duration_seconds:
                self.stop_recording()
                break