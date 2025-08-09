# main.py - Voice Agent Pipeline Main Entry Point
"""
This is the main orchestrator for the voice agent pipeline.
It creates and coordinates all components of the real-time voice conversation system:
- Audio capture from microphone
- Speech-to-text transcription
- LLM response generation
- Text-to-speech synthesis
- Audio playback

The entire pipeline runs asynchronously with queue-based communication between stages.
"""

import asyncio
from dotenv import load_dotenv
from sources import RealTimeMicrophoneSource
from asr import CartesiaASR
from llm import AnthropicLLM
from tts import CartesiaTTS
from audio_player import SoundDeviceAudioPlayer

# Load environment variables from .env file
load_dotenv()
if __name__ == "__main__":
    
    async def test_asr():
        """
        Main async function that sets up and runs the complete voice agent pipeline.
        
        Pipeline Flow:
        1. Microphone captures audio chunks (100ms intervals)
        2. ASR converts audio to text transcripts  
        3. LLM processes text and generates responses
        4. TTS synthesizes response text to audio
        5. Audio player outputs synthesized speech
        
        All stages run concurrently using asyncio.gather() for optimal performance.
        """
        # Initialize all pipeline components
        audio_source = RealTimeMicrophoneSource()
        asr = CartesiaASR()
        llm = AnthropicLLM()
        tts = CartesiaTTS()
        player = SoundDeviceAudioPlayer()
        
        # Create async queues as buffers between pipeline stages
        audio_queue = asyncio.Queue()        # Raw audio chunks (bytes)
        text_queue = asyncio.Queue()         # Transcribed text fragments (str)
        response_queue = asyncio.Queue()     # LLM response chunks (str)
        tts_chunk_queue = asyncio.Queue()    # Synthesized audio chunks (bytes)
        
        async def audio_producer():
            """
            Stage 1: Audio Capture
            Continuously captures audio from microphone and feeds it to the ASR queue.
            Sends END signal when audio capture is complete.
            """
            async for chunk in audio_source.stream_audio():
                await audio_queue.put(chunk)

        async def text_producer():
            """
            Stage 2: Speech Recognition  
            Consumes audio chunks and produces text transcriptions.
            Sends transcribed text to the LLM queue for processing.
            """
            async for text in asr.transcribe_stream(audio_queue):
                await text_queue.put(text)

        async def llm_interface():
            """
            Stage 3: Language Model Processing
            Processes transcribed text and generates conversational responses.
            Implements intelligent batching to wait for complete user utterances.
            """
            async for response_text in llm.generate_stream(text_queue):
                print(f"[LLM] Generated response: {response_text}")
                await response_queue.put(response_text)

        async def tts_interface():
            """
            Stage 4: Text-to-Speech Synthesis
            Converts LLM response text into synthesized audio chunks.
            Accumulates text into complete sentences before synthesis.
            """
            async for audio_chunk in tts.synthesize_stream(response_queue):
                print(f"[TTS] Generated audio chunk, {len(audio_chunk)}")
                await tts_chunk_queue.put(audio_chunk)
        
        async def player_interface():
            """
            Stage 5: Audio Playback
            Consumes synthesized audio chunks and plays them through speakers.
            Implements intelligent buffering to prevent audio dropouts.
            """
            await player.play_audio(tts_chunk_queue)

        # 启动生产者和转录器
        # Launch all pipeline stages concurrently
        # Each stage runs independently and communicates via async queues
        await asyncio.gather(
            audio_producer(),    # Stage 1: Audio capture
            text_producer(),     # Stage 2: Speech recognition
            llm_interface(),     # Stage 3: LLM processing
            tts_interface(),     # Stage 4: Text-to-speech
            player_interface()   # Stage 5: Audio playback
        )

    # Run the main async function
    asyncio.run(test_asr())