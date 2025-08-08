# sources.py
import asyncio
import os
import sounddevice as sd
import numpy as np
from components import AudioSource
from typing import AsyncGenerator, Any
import threading
from collections import deque
class RealTimeMicrophoneSource(AudioSource):
    def __init__(self, 
                 sample_rate: int = 16000, 
                 channels: int = 1,
                 chunk_duration_ms: int = 100,  # 100ms chunks
                 device: int = None):
        """
        实时麦克风音频源
        
        Args:
            sample_rate: 采样率 (Hz)
            channels: 声道数
            chunk_duration_ms: 每个音频块的时长 (毫秒)
            device: 音频设备ID，None为默认设备
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.device = device
        
        # 计算每个chunk的样本数
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        # 每个chunk的字节数 (16-bit PCM)
        self.chunk_bytes = self.chunk_samples * channels * 2
        
        # 音频缓冲区
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.recording = False
        self.stream = None
        
        print(f"[MicSource] Initialized: {sample_rate}Hz, {channels}ch, "
              f"{chunk_duration_ms}ms chunks ({self.chunk_samples} samples)")

    def _audio_callback(self, indata, frames, time, status):
        """sounddevice 录音回调"""
        if status:
            print(f"[MicSource] Recording status: {status}")
        
        if self.recording:
            # 转换为 int16 PCM 格式
            audio_int16 = (indata * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            with self.buffer_lock:
                self.audio_buffer.append(audio_bytes)

    async def stream_audio(self) -> AsyncGenerator[bytes, None]:
        """开始录音并流式返回音频数据"""
        try:
            # 启动录音流
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=self.chunk_samples,
                device=self.device,
                latency='low'
            )
            
            self.stream.start()
            self.recording = True
            print("[MicSource] Started recording...")
            
            # 持续产生音频块
            while self.recording:
                with self.buffer_lock:
                    if self.audio_buffer:
                        audio_chunk = self.audio_buffer.popleft()
                        yield audio_chunk
                
                # 小延迟避免忙等
                await asyncio.sleep(0.01)
        
        except Exception as e:
            print(f"[MicSource] Recording error: {e}")
        finally:
            self.stop_recording()

    def stop_recording(self):
        """停止录音"""
        self.recording = False
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("[MicSource] Recording stopped")

    async def record_for_duration(self, duration_seconds: float) -> AsyncGenerator[bytes, None]:
        """录制指定时长的音频"""
        import time
        start_time = time.time()
        
        async for chunk in self.stream_audio():
            yield chunk
            
            # 检查是否超过指定时长
            if time.time() - start_time >= duration_seconds:
                self.stop_recording()
                break