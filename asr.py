# asr.py
import asyncio
import os
from cartesia import AsyncCartesia
from components import ASRInterface
from typing import AsyncGenerator
class CartesiaASR(ASRInterface):
    def __init__(self):
        self.client = AsyncCartesia(api_key=os.getenv("CARTESIA_API_KEY"))

    async def transcribe_stream(self, audio_queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        ws = await self.client.stt.websocket(
            model="ink-whisper",
            language="en",
            encoding="pcm_s16le",
            sample_rate=16000,
            min_volume=0.15,
            max_silence_duration_secs=0.5,
        )

        async def sender():
            try:
                while True:
                    chunk = await audio_queue.get()
                    if chunk == "END":
                        await ws.send("finalize")
                        await ws.send("done")
                        break
                    await ws.send(chunk)
                    await asyncio.sleep(0.02)
            except Exception as e:
                print(f"[ASR Sender] Error: {e}")

        async def receiver():
            full_transcript = ""
            all_word_timestamps = []
            try:
                async for result in ws.receive():
                    if result['type'] == 'transcript':
                        text = result['text'].strip()
                        is_final = result.get('is_final', False)
                        
                        
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
                        
                        
                        if not text:
                            continue
                        if is_final:
                            full_transcript += text + " "
                            # Finalized text sent to next stage
                            await asyncio.sleep(0)  # yield control
                            # yield full_transcript.strip()
                            yield text
                        else:
                            # Optional: send partial updates
                            # yield text
                            pass
                    elif result['type'] == 'done':
                        break
            except Exception as e:
                print(f"[ASR Receiver] Error: {e}")
            finally:
                await ws.close()

        # 并行发送和接收
        sender_task = asyncio.create_task(sender())
        async for text in receiver():
            yield text

        await sender_task
        await self.client.close()

