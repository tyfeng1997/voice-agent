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
                 buffer_size: int = 2048,  # 增加默认缓冲区大小
                 device: Optional[int] = None,
                 min_buffer_samples: int = 4800):  # 最小缓冲样本数 (200ms)
        """
        初始化实时音频播放器
        
        Args:
            sample_rate: 采样率
            channels: 声道数
            dtype: 数据类型 ('float32' 或 'int16')
            buffer_size: sounddevice 回调缓冲区大小
            device: 音频设备ID，None为默认设备
            min_buffer_samples: 开始播放前的最小缓冲样本数
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.buffer_size = buffer_size
        self.device = device
        self.min_buffer_samples = min_buffer_samples
        
        # 音频缓冲队列
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        # 播放状态和统计
        self.is_playing = False
        self.stream = None
        self.total_samples_buffered = 0
        self.underrun_count = 0
        self.started_playback = False
        
        print(f"[AudioPlayer] Initialized with sample_rate={sample_rate}, "
              f"channels={channels}, dtype={dtype}, buffer_size={buffer_size}")
        print(f"[AudioPlayer] Min buffer: {min_buffer_samples} samples ({min_buffer_samples/sample_rate*1000:.1f}ms)")

    def _audio_callback(self, outdata, frames, time, status):
        """sounddevice 的回调函数，用于填充播放缓冲区"""
        if status:
            print(f"[AudioPlayer] Status: {status}")
        
        with self.buffer_lock:
            # 检查是否有足够的数据播放
            if not self.started_playback and self.total_samples_buffered < self.min_buffer_samples:
                # 还没有足够的缓冲，输出静音
                outdata.fill(0)
                return
            
            # 开始播放
            self.started_playback = True
            
            if self.audio_buffer:
                samples_needed = frames
                output_pos = 0
                
                while samples_needed > 0 and self.audio_buffer:
                    try:
                        audio_chunk = self.audio_buffer[0]  # 查看但不移除
                        
                        if len(audio_chunk) <= samples_needed:
                            # 整个 chunk 都可以用
                            chunk_size = len(audio_chunk)
                            outdata[output_pos:output_pos + chunk_size] = audio_chunk.reshape(-1, self.channels)
                            
                            # 移除已使用的 chunk
                            self.audio_buffer.popleft()
                            self.total_samples_buffered -= chunk_size
                            
                            output_pos += chunk_size
                            samples_needed -= chunk_size
                        else:
                            # 只使用 chunk 的一部分
                            outdata[output_pos:output_pos + samples_needed] = audio_chunk[:samples_needed].reshape(-1, self.channels)
                            
                            # 保留剩余数据
                            self.audio_buffer[0] = audio_chunk[samples_needed:]
                            self.total_samples_buffered -= samples_needed
                            
                            output_pos += samples_needed
                            samples_needed = 0
                            
                    except Exception as e:
                        print(f"[AudioPlayer] Error in callback processing: {e}")
                        break
                
                # 如果还有未填充的部分，用静音填充
                if samples_needed > 0:
                    outdata[output_pos:] = 0
                    self.underrun_count += 1
                    if self.underrun_count % 10 == 1:  # 每10次打印一次
                        print(f"[AudioPlayer] Buffer underrun #{self.underrun_count}, missing {samples_needed} samples")
            else:
                # 没有数据时填充静音
                outdata.fill(0)
                if self.started_playback:
                    self.underrun_count += 1

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
                    latency='high'  # 改为 high 减少 buffer underrun
                )
                self.stream.start()
                self.is_playing = True
                print(f"[AudioPlayer] Audio stream started with blocksize={self.buffer_size}")
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
            self.total_samples_buffered += len(audio_data)

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
            buffer_duration_ms = (self.total_samples_buffered / self.sample_rate) * 1000
            return {
                'buffer_chunks': len(self.audio_buffer),
                'total_samples': self.total_samples_buffered,
                'buffer_duration_ms': buffer_duration_ms,
                'is_playing': self.is_playing,
                'started_playback': self.started_playback,
                'underrun_count': self.underrun_count
            }