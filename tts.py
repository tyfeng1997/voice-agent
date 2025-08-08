# tts.py
import asyncio
import numpy as np
from cartesia import AsyncCartesia
from components import TTSInterface
from typing import AsyncGenerator
import os

class CartesiaTTS(TTSInterface):
    def __init__(self, sample_rate: int = 24000, voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091"):
        self.api_key = os.getenv("CARTESIA_API_KEY")
        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY not found in environment")
        
        self.sample_rate = sample_rate
        self.voice_id = voice_id
        self.client = AsyncCartesia(api_key=self.api_key)

    async def synthesize_stream(self, text_queue: asyncio.Queue) -> AsyncGenerator[bytes, None]:
        """
        从 text_queue receive partial text，accumulate into complete sentences
        yield: PCM audio chunks (bytes)
        """
        buffer = ""
        
        while True:
            try:
               
                text = await text_queue.get()
                
                if text == "END":
                    # 结束信号
                    if buffer.strip():
                        async for chunk in self._synthesize_sentence(buffer):
                            yield chunk
                    break
                
                buffer += " " + text.strip() if buffer else text.strip()
                print(f"[TTS] Accumulated text: '{buffer}'")
                
                if self._should_send(buffer):
                    async for audio_chunk in self._synthesize_sentence(buffer):
                        yield audio_chunk
                    buffer = ""  
            
            except Exception as e:
                print(f"[TTS] Error in text accumulation: {e}")
                continue
        
        print("[TTS] Synthesis completed.")

    def _should_send(self, text: str) -> bool:
        return text.rstrip().endswith(('.', '!', '?', '。', '！', '？'))

    async def _synthesize_sentence(self, text: str) -> AsyncGenerator[bytes, None]:
        """调 Cartesia API and return audio chunks"""
        try:
            ws = await self.client.tts.websocket()
            
            #  await
            async for output in await ws.send(
                model_id="sonic-2",
                transcript=text,
                voice={"id": self.voice_id},
                stream=True,
                output_format={
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate": self.sample_rate,
                },
            ):
                audio_bytes = output.audio  # bytes (PCM f32 little-endian)
                if audio_bytes:
                    yield audio_bytes
            
            await ws.close()
            print(f"[TTS] Played: '{text}'")
        
        except Exception as e:
            print(f"[TTS] Error synthesizing '{text}': {e}")
