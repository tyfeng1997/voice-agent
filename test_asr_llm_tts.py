# test_asr_to_llm.py
import asyncio
import os
from dotenv import load_dotenv
import wave
import numpy as np

from sources import MicrophoneSource
from asr import CartesiaASR
from llm import StreamingLLM
from tts import CartesiaTTS

async def test_asr_to_llm():
    audio_queue = asyncio.Queue(maxsize=10)
    text_queue = asyncio.Queue(maxsize=10)
    tts_queue = asyncio.Queue(maxsize=20)
  
    
    output_wav_path = "tts_output.wav"
    wav_file = wave.open(output_wav_path, "wb")
    wav_file.setnchannels(1)                   
    wav_file.setsampwidth(2)                    
    wav_file.setframerate(24000)                
    
    
    audio_source = MicrophoneSource("./audio.wav", chunk_size=3200)
    asr = CartesiaASR()
    llm = StreamingLLM(timeout=1.0)
    tts = CartesiaTTS()

   
    async def feed_audio():
        async for chunk in audio_source.stream_audio():
            await audio_queue.put(chunk)
        await audio_queue.put("END")
        print("[Audio] Feed completed.")
    async def run_asr():
        async for text in asr.transcribe_stream(audio_queue):
            print(f"[ASR] Final chunk: '{text}'")
            await text_queue.put(text)
        await text_queue.put("END")
        print("[ASR] Completed.")
        
    async def run_llm():
        async for partial in llm.generate_stream(text_queue):
            print(f"[LLM] {partial}")
            await tts_queue.put(partial)
        await tts_queue.put("END")
        print("[LLM] Response completed.")
    
    async def run_tts():
        print("[TTS] Starting synthesis and saving to file...")
        total_samples = 0

        async for audio_chunk in tts.synthesize_stream(tts_queue):
            # audio_chunk is bytes (PCM f32 little-endian)
            float32_buffer = np.frombuffer(audio_chunk, dtype='<f4')  # float32
            int16_buffer = (float32_buffer * 32767).astype(np.int16)  # to int16
            wav_file.writeframes(int16_buffer.tobytes())
            total_samples += len(int16_buffer)
            print(f"[TTS] Wrote {len(int16_buffer)} int16 samples")

        
        wav_file.close()
        duration = total_samples / 24000
        print(f"[TTS] Saved audio to '{output_wav_path}' ({duration:.2f}s)")
            
    
    try:
        await asyncio.gather(
            feed_audio(),
            run_asr(),
            run_llm(),
            run_tts(),
        )
    except Exception as e:
        print(f"[Error] {e}")
    finally:
        
        if not wav_file.closed:
            wav_file.close()
            
            
if __name__ == "__main__":
    load_dotenv()  
    asyncio.run(test_asr_to_llm())