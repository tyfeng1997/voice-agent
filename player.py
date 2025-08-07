# player.py
import asyncio
import numpy as np
import sounddevice as sd
from typing import Callable

class AsyncAudioPlayer:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.queue = asyncio.Queue()
        self.playing = False

    async def play_audio(self):
        """持续从队列读取 audio chunk 并非阻塞播放"""
        self.playing = True
        while self.playing:
            try:
                audio_chunk = await self.queue.get()
                if audio_chunk == "END":
                    break

                # 转成 numpy float32 数组
                audio_array = np.frombuffer(audio_chunk, dtype='<f4')

                if len(audio_array) > 0:
                    # 非阻塞播放，使用回调
                    def callback(outdata, frames, time, status):
                        if status:
                            print(status)
                        outdata[:] = audio_array.reshape(-1, 1)

                    stream = sd.OutputStream(
                        samplerate=self.sample_rate,
                        channels=1,
                        callback=callback,
                        blocksize=len(audio_array)
                    )
                    with stream:
                        await asyncio.sleep(len(audio_array) / self.sample_rate)  # 粗略等待播放完成

            except Exception as e:
                print(f"[Player] Error: {e}")

        print("[Player] Stopped.")

    def stop(self):
        self.playing = False
        self.queue.put_nowait("END")