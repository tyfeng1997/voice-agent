# sources.py
import asyncio
import os
from components import AudioSource
from typing import AsyncGenerator, Any
class MicrophoneSource(AudioSource):
    def __init__(self, audio_file: str = "./audio.wav", chunk_size: int = 3200, sample_rate: int = 16000):
        self.audio_file = audio_file
        self.chunk_size = chunk_size  #  100ms at 16kHz, 16-bit: 16000 * 0.1 * 2 = 3200
        self.sample_rate = sample_rate

    async def stream_audio(self) -> AsyncGenerator[bytes, None]:
        with open(self.audio_file, "rb") as f:
            audio_data = f.read()

        for i in range(0, len(audio_data), self.chunk_size):
            chunk = audio_data[i:i + self.chunk_size]
            if chunk:
                yield chunk
                # await asyncio.sleep(0.07)  # 模拟实时流
