import asyncio
from dotenv import load_dotenv
from sources import RealTimeMicrophoneSource
from asr import CartesiaASR
from llm import AnthropicLLM
from tts import CartesiaTTS
from audio_player import SoundDeviceAudioPlayer
load_dotenv()
if __name__ == "__main__":
    
    async def test_asr():
        audio_source = RealTimeMicrophoneSource()
        asr = CartesiaASR()
        llm = AnthropicLLM()
        tts = CartesiaTTS()
        audio_queue = asyncio.Queue()
        text_queue = asyncio.Queue()
        response_queue = asyncio.Queue()
        tts_chunk_queue = asyncio.Queue()
        player = SoundDeviceAudioPlayer()
        
        
        async def audio_producer():
            async for chunk in audio_source.stream_audio():
                await audio_queue.put(chunk)
            await audio_queue.put("END")

        async def text_producer():
            async for text in asr.transcribe_stream(audio_queue):
                await text_queue.put(text)
            await text_queue.put("END")

        async def llm_interface():
            async for response_text in llm.generate_stream(text_queue):
                print(f"[LLM] Generated response: {response_text}")
                await response_queue.put(response_text)
            await response_queue.put("END")

        async def tts_interface():
            async for audio_chunk in tts.synthesize_stream(response_queue):
                print(f"[TTS] Generated audio chunk, {len(audio_chunk)}")
                await tts_chunk_queue.put(audio_chunk)
        
        async def player_interface():
            await player.play_audio(tts_chunk_queue)

        # 启动生产者和转录器
        await asyncio.gather(audio_producer(), text_producer(), llm_interface(), 
                             tts_interface(),
                             player_interface()
                             )

    asyncio.run(test_asr())