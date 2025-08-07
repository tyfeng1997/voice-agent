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

if __name__ == "__main__":
    from dotenv import load_dotenv
    import wave
    load_dotenv()
    output_wav_path = "tts_output.wav"
    wav_file = wave.open(output_wav_path, "wb")
    wav_file.setnchannels(1)                   
    wav_file.setsampwidth(2)                    
    wav_file.setframerate(24000)             
    
    
    async def example_usage():
        tts = CartesiaTTS()
        text_queue = asyncio.Queue()
        
        
        await text_queue.put("Hello")
        await text_queue.put("world.")
        await text_queue.put("END")  
        
        print("[TTS] Starting synthesis and saving to file...")
        total_samples = 0
        async for audio_chunk in tts.synthesize_stream(text_queue):
            float32_buffer = np.frombuffer(audio_chunk, dtype='<f4')  # float32
            int16_buffer = (float32_buffer * 32767).astype(np.int16)  # to int16
            wav_file.writeframes(int16_buffer.tobytes())
            total_samples += len(int16_buffer)
            print(f"[TTS] Wrote {len(int16_buffer)} int16 samples")
        
        wav_file.close()
        duration = total_samples / 24000
        print(f"[TTS] Saved audio to '{output_wav_path}' ({duration:.2f}s)")

    asyncio.run(example_usage())