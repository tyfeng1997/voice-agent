# audio_player.py - Real-time Audio Playback using SoundDevice
"""
This module implements real-time audio playback functionality for the voice agent.

The SoundDeviceAudioPlayer class provides:
- Real-time audio streaming playback with intelligent buffering
- Cross-platform audio output using sounddevice
- Configurable latency and buffer management
- Underrun detection and handling for smooth playback
- Support for multiple audio formats (float32, int16)

The implementation uses a separate audio thread for playback while maintaining
thread-safe communication with the main async pipeline.
"""

import asyncio
import numpy as np
import sounddevice as sd
from components import AudioPlayer
from typing import Optional
import threading
from collections import deque

class SoundDeviceAudioPlayer(AudioPlayer):
    """
    Real-time audio player implementation using sounddevice.
    
    This class handles real-time playback of synthesized speech audio with
    intelligent buffering to prevent audio dropouts and ensure smooth playback.
    
    Features:
    - Real-time streaming audio playback
    - Intelligent buffer management with configurable thresholds
    - Underrun detection and recovery
    - Cross-platform compatibility via sounddevice
    - Support for multiple audio formats and sample rates
    - Thread-safe operation with async pipeline
    """
    
    def __init__(self, 
                 sample_rate: int = 24000, 
                 channels: int = 1,
                 dtype: str = 'float32',
                 buffer_size: int = 2048,  # Audio callback buffer size
                 device: Optional[int] = None,
                 min_buffer_samples: int = 4800):  # Minimum buffer before playback (200ms)
        """
        Initialize the real-time audio player.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            dtype: Audio data type ('float32' or 'int16')
            buffer_size: Size of sounddevice callback buffer
            device: Audio device ID (None for system default)
            min_buffer_samples: Minimum samples to buffer before starting playback
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.buffer_size = buffer_size
        self.device = device
        self.min_buffer_samples = min_buffer_samples
        
        # Thread-safe audio buffer queue
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        # Playback state and statistics
        self.is_playing = False
        self.stream = None
        self.total_samples_buffered = 0
        self.underrun_count = 0
        self.started_playback = False
        
        print(f"[AudioPlayer] Initialized with sample_rate={sample_rate}, "
              f"channels={channels}, dtype={dtype}, buffer_size={buffer_size}")
        print(f"[AudioPlayer] Min buffer: {min_buffer_samples} samples ({min_buffer_samples/sample_rate*1000:.1f}ms)")

    def _audio_callback(self, outdata, frames, time, status):
        """
        Sounddevice audio output callback function.
        
        This callback is called by sounddevice in a separate audio thread
        whenever the system needs more audio data for playback. It manages
        the audio buffer and implements intelligent playback logic.
        
        Args:
            outdata: Output buffer to fill with audio data
            frames: Number of audio frames requested
            time: Timing information
            status: Stream status flags
        """
        if status:
            print(f"[AudioPlayer] Status: {status}")
        
        with self.buffer_lock:
            # Wait for minimum buffer before starting playback
            if not self.started_playback and self.total_samples_buffered < self.min_buffer_samples:
                # Not enough buffered data yet - output silence
                outdata.fill(0)
                return
            
            # Start playback once we have sufficient buffer
            self.started_playback = True
            
            if self.audio_buffer:
                samples_needed = frames
                output_pos = 0
                
                # Fill output buffer from audio queue
                while samples_needed > 0 and self.audio_buffer:
                    try:
                        audio_chunk = self.audio_buffer[0]  # Peek at first chunk
                        
                        if len(audio_chunk) <= samples_needed:
                            # Use entire chunk
                            chunk_size = len(audio_chunk)
                            outdata[output_pos:output_pos + chunk_size] = audio_chunk.reshape(-1, self.channels)
                            
                            # Remove used chunk from buffer
                            self.audio_buffer.popleft()
                            self.total_samples_buffered -= chunk_size
                            
                            output_pos += chunk_size
                            samples_needed -= chunk_size
                        else:
                            # Use partial chunk
                            outdata[output_pos:output_pos + samples_needed] = audio_chunk[:samples_needed].reshape(-1, self.channels)
                            
                            # Keep remaining data
                            self.audio_buffer[0] = audio_chunk[samples_needed:]
                            self.total_samples_buffered -= samples_needed
                            
                            output_pos += samples_needed
                            samples_needed = 0
                            
                    except Exception as e:
                        print(f"[AudioPlayer] Error in callback processing: {e}")
                        break
                
                # Fill remaining with silence if needed
                if samples_needed > 0:
                    outdata[output_pos:] = 0
                    self.underrun_count += 1
                    if self.underrun_count % 10 == 1:  # Log every 10th underrun
                        print(f"[AudioPlayer] Buffer underrun #{self.underrun_count}, missing {samples_needed} samples")
            else:
                # No data available - fill with silence
                outdata.fill(0)
                if self.started_playback:
                    self.underrun_count += 1

    def _start_stream(self):
        """
        Start the audio output stream.
        
        Initializes and starts the sounddevice output stream with the
        configured parameters and callback function.
        """
        if self.stream is None or not self.stream.active:
            try:
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    callback=self._audio_callback,
                    blocksize=self.buffer_size,
                    device=self.device,
                    latency='high'  # Use high latency to reduce buffer underruns
                )
                self.stream.start()
                self.is_playing = True
                print(f"[AudioPlayer] Audio stream started with blocksize={self.buffer_size}")
            except Exception as e:
                print(f"[AudioPlayer] Failed to start stream: {e}")
                raise

    def _stop_stream(self):
        """
        Stop the audio output stream and clean up resources.
        """
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_playing = False
            print("[AudioPlayer] Audio stream stopped")

    def _add_audio_chunk(self, audio_data: np.ndarray):
        """
        Add an audio chunk to the playback buffer.
        
        This method safely adds audio data to the buffer queue for
        consumption by the audio callback thread.
        
        Args:
            audio_data: Audio samples as numpy array
        """
        with self.buffer_lock:
            self.audio_buffer.append(audio_data)
            self.total_samples_buffered += len(audio_data)

    async def play_audio(self, audio_queue: asyncio.Queue) -> None:
        """
        Main audio playback method that consumes audio chunks and plays them.
        
        This method runs the main playback loop that:
        1. Starts the audio output stream
        2. Continuously consumes audio chunks from the queue
        3. Processes and buffers the audio data
        4. Handles format conversion as needed
        5. Manages graceful shutdown
        
        Args:
            audio_queue: Queue containing audio chunks to play
        """
        try:
            self._start_stream()
            
            while True:
                try:
                    # Get next audio chunk from TTS
                    audio_chunk = await audio_queue.get()
                    # Process audio data based on input format
                    if isinstance(audio_chunk, bytes):
                        # Convert bytes to numpy array based on expected format
                        if self.dtype == 'float32':
                            audio_array = np.frombuffer(audio_chunk, dtype='<f4')
                        elif self.dtype == 'int16':
                            # Convert float32 input to int16 if needed
                            float_array = np.frombuffer(audio_chunk, dtype='<f4')
                            audio_array = (float_array * 32767).astype(np.int16)
                        else:
                            audio_array = np.frombuffer(audio_chunk, dtype=self.dtype)
                    else:
                        audio_array = audio_chunk
                    
                    # Add to playback buffer
                    if len(audio_array) > 0:
                        self._add_audio_chunk(audio_array)
                        print(f"[AudioPlayer] Added {len(audio_array)} samples to buffer")
                    
                    audio_queue.task_done()
                
                except asyncio.CancelledError:
                    print("[AudioPlayer] Play task cancelled")
                    break
                except Exception as e:
                    print(f"[AudioPlayer] Error processing audio: {e}")
                    continue
            
            # Wait for buffer to drain before stopping
            print("[AudioPlayer] Waiting for buffer to drain...")
            while self.audio_buffer:
                await asyncio.sleep(0.1)
            
        finally:
            self._stop_stream()
            print("[AudioPlayer] Audio playback finished")

    async def flush_and_stop(self):
        """
        Immediately clear the buffer and stop playback.
        
        Useful for interrupting current playback or resetting the player state.
        """
        with self.buffer_lock:
            self.audio_buffer.clear()
        self._stop_stream()

    def get_buffer_info(self):
        """
        Get current buffer status information for debugging.
        
        Returns:
            dict: Buffer statistics including queue length, total samples,
                  buffer duration, playback status, and underrun count
        """
        with self.buffer_lock:
            buffer_duration_ms = (self.total_samples_buffered / self.sample_rate) * 1000
            return {
                'buffer_chunks': len(self.audio_buffer),
                'total_samples': self.total_samples_buffered,
                'buffer_duration_ms': buffer_duration_ms,
                'is_playing': self.is_playing,
                'started_playback': self.started_playback,
                'underrun_count': self.underrun_count
            }