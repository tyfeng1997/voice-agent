# audio_player.py
import asyncio
import numpy as np
import sounddevice as sd
from components import AudioPlayer
from typing import Optional
import threading
from collections import deque

class SoundDeviceAudioPlayer(AudioPlayer):
    def __init__(self, 
                 sample_rate: int = 24000, 
                 channels: int = 1,
                 dtype: str = 'float32',
                 buffer_size: int = 1024,
                 device: Optional[int] = None):
        """
        初始化实时音频播放器
        
        Args:
            sample_rate: 采样率
            channels: 声道数
            dtype: 数据类型 ('float32' 或 'int16')
            buffer_size: 播放缓冲区大小
            device: 音频设备ID，None为默认设备
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.buffer_size = buffer_size
        self.device = device
        
        # 音频缓冲队列
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        # 播放状态
        self.is_playing = False
        self.stream = None
        
        print(f"[AudioPlayer] Initialized with sample_rate={sample_rate}, "
              f"channels={channels}, dtype={dtype}")

    def _audio_callback(self, outdata, frames, time, status):
        """sounddevice 的回调函数，用于填充播放缓冲区"""
        if status:
            print(f"[AudioPlayer] Status: {status}")
        
        with self.buffer_lock:
            if self.audio_buffer:
                # 从缓冲队列中取出音频数据
                try:
                    audio_chunk = self.audio_buffer.popleft()
                    
                    # 确保数据长度匹配
                    if len(audio_chunk) >= frames:
                        outdata[:] = audio_chunk[:frames].reshape(-1, self.channels)
                        
                        # 如果有剩余数据，放回队列开头
                        if len(audio_chunk) > frames:
                            remaining = audio_chunk[frames:]
                            self.audio_buffer.appendleft(remaining)
                    else:
                        # 数据不够，用零填充
                        outdata[:len(audio_chunk)] = audio_chunk.reshape(-1, self.channels)
                        outdata[len(audio_chunk):] = 0
                        
                except Exception as e:
                    print(f"[AudioPlayer] Error in callback: {e}")
                    outdata.fill(0)
            else:
                # 没有数据时填充静音
                outdata.fill(0)

    def _start_stream(self):
        """启动音频流"""
        if self.stream is None or not self.stream.active:
            try:
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    callback=self._audio_callback,
                    blocksize=self.buffer_size,
                    device=self.device,
                    latency='low'
                )
                self.stream.start()
                self.is_playing = True
                print("[AudioPlayer] Audio stream started")
            except Exception as e:
                print(f"[AudioPlayer] Failed to start stream: {e}")
                raise

    def _stop_stream(self):
        """停止音频流"""
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_playing = False
            print("[AudioPlayer] Audio stream stopped")

    def _add_audio_chunk(self, audio_data: np.ndarray):
        """添加音频块到缓冲队列"""
        with self.buffer_lock:
            self.audio_buffer.append(audio_data)

    async def play_audio(self, audio_queue: asyncio.Queue) -> None:
        """
        从队列中消费音频块并实时播放
        
        Args:
            audio_queue: 包含音频数据的异步队列
        """
        try:
            self._start_stream()
            
            while True:
                try:
                    # 从队列获取音频数据
                    audio_chunk = await audio_queue.get()
                    
                    if audio_chunk == "END":
                        print("[AudioPlayer] Received END signal")
                        break
                    
                    # 处理音频数据
                    if isinstance(audio_chunk, bytes):
                        # 假设输入是 float32 PCM 数据
                        if self.dtype == 'float32':
                            audio_array = np.frombuffer(audio_chunk, dtype='<f4')
                        elif self.dtype == 'int16':
                            # 如果输入是 float32 但需要 int16
                            float_array = np.frombuffer(audio_chunk, dtype='<f4')
                            audio_array = (float_array * 32767).astype(np.int16)
                        else:
                            audio_array = np.frombuffer(audio_chunk, dtype=self.dtype)
                    else:
                        audio_array = audio_chunk
                    
                    # 添加到播放缓冲区
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
            
            # 等待缓冲区播放完成
            print("[AudioPlayer] Waiting for buffer to drain...")
            while self.audio_buffer:
                await asyncio.sleep(0.1)
            
        finally:
            self._stop_stream()
            print("[AudioPlayer] Audio playback finished")

    async def flush_and_stop(self):
        """清空缓冲区并停止播放"""
        with self.buffer_lock:
            self.audio_buffer.clear()
        self._stop_stream()

    def get_buffer_info(self):
        """获取缓冲区信息用于调试"""
        with self.buffer_lock:
            return {
                'buffer_length': len(self.audio_buffer),
                'total_samples': sum(len(chunk) for chunk in self.audio_buffer) if self.audio_buffer else 0,
                'is_playing': self.is_playing
            }


# 更新后的测试代码
if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    from dotenv import load_dotenv
    from tts import CartesiaTTS
    
    load_dotenv()
    
    async def test_realtime_playback():
        """测试实时音频播放"""
        # 初始化 TTS 和音频播放器
        tts = CartesiaTTS(sample_rate=24000)
        audio_player = SoundDeviceAudioPlayer(
            sample_rate=24000,
            channels=1,
            dtype='float32',  # 匹配 Cartesia TTS 输出格式
            buffer_size=1024
        )
        
        # 创建队列
        text_queue = asyncio.Queue()
        audio_queue = asyncio.Queue()
        
        # 准备测试文本
        test_texts = [
            "Hello there!",
            "This is a test of real-time audio synthesis and playback.",
            "The audio should start playing as soon as the first chunks are generated.",
            "Pretty cool, right?"
        ]
        
        # 添加文本到队列
        for text in test_texts:
            await text_queue.put(text)
        await text_queue.put("END")
        
        print("[Test] Starting real-time TTS and playback...")
        
        async def tts_producer():
            """TTS 生产者任务"""
            try:
                async for audio_chunk in tts.synthesize_stream(text_queue):
                    await audio_queue.put(audio_chunk)
                await audio_queue.put("END")
                print("[Test] TTS generation completed")
            except Exception as e:
                print(f"[Test] TTS error: {e}")
                await audio_queue.put("END")
        
        async def audio_consumer():
            """音频播放消费者任务"""
            try:
                await audio_player.play_audio(audio_queue)
                print("[Test] Audio playback completed")
            except Exception as e:
                print(f"[Test] Audio playback error: {e}")
        
        # 并行运行 TTS 生成和音频播放
        try:
            await asyncio.gather(
                tts_producer(),
                audio_consumer()
            )
        except Exception as e:
            print(f"[Test] Error in main tasks: {e}")
        finally:
            # 清理
            await audio_player.flush_and_stop()
            print("[Test] Test completed!")
    
    # 运行测试
    print("Starting real-time audio test...")
    print("Make sure your speakers/headphones are connected!")
    asyncio.run(test_realtime_playback())