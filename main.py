# main.py
import asyncio
from buffers import BufferManager
from sources import MicrophoneSource
from asr import CartesiaASR
from llm import MockLLM
from tts import CartesiaTTS
from player import PyAudioPlayer

async def main():
    buffers = BufferManager()

    # 初始化组件
    audio_source = MicrophoneSource("./audio.wav")
    asr = CartesiaASR()
    llm = MockLLM()
    tts = CartesiaTTS()
    player = PyAudioPlayer()

    async def run_audio_source():
        async for chunk in audio_source.stream_audio():
            await buffers.audio_buffer.put(chunk)
        await buffers.audio_buffer.put("END")  # 结束标志

    async def run_asr():
        async for text in asr.transcribe_stream(buffers.audio_buffer):
            print(f"[ASR] Final transcript: {text}")
            await buffers.text_buffer.put(text)
            await buffers.text_buffer.put("END")  # 假设只处理一次
            break

    async def run_llm():
        while True:
            text = await buffers.text_buffer.get()
            if text == "END":
                await buffers.llm_response_buffer.put("END")
                break
            print(f"[LLM] Generating response for: {text}")
            async for partial in llm.generate_stream(text):
                await buffers.llm_response_buffer.put(partial)

    async def run_tts():
        async for partial_text in tts.synthesize_stream(buffers.llm_response_buffer):
            await buffers.playback_buffer.put(partial_text)
        await buffers.playback_buffer.put("END")

    async def run_player():
        await player.play_audio(buffers.playback_buffer)

    # 并发运行所有模块
    await asyncio.gather(
        run_audio_source(),
        run_asr(),
        run_llm(),
        run_tts(),
        run_player(),
    )

if __name__ == "__main__":
    asyncio.run(main())