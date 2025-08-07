import asyncio
import os
from cartesia import AsyncCartesia
from dotenv import load_dotenv

load_dotenv()

async def streaming_stt_example():
    """
    Advanced async STT example for real-time streaming applications.
    This example simulates streaming audio processing with proper error handling
    and demonstrates the new endpointing and word timestamp features.
    """
    client = AsyncCartesia(api_key=os.getenv("CARTESIA_API_KEY"))
    
    try:
        # Create websocket connection with voice activity detection
        ws = await client.stt.websocket(
            model="ink-whisper",             # Model (required)
            language="en",                   # Language of your audio (required)
            encoding="pcm_s16le",            # Audio encoding format (required)
            sample_rate=16000,               # Audio sample rate (required)
            min_volume=0.15,                 # Volume threshold for voice activity detection
            max_silence_duration_secs=0.3,   # Maximum silence duration before endpointing
        )
        
        # Simulate streaming audio data (replace with your audio source)
        async def audio_stream():
            """Simulate real-time audio streaming - replace with actual audio capture"""
            # Load audio file for simulation
            with open("./audio.wav", "rb") as f:
                audio_data = f.read()
            
            # Stream in 100ms chunks (realistic for real-time processing)
            chunk_size = int(16000 * 0.1 * 2)  # 100ms at 16kHz, 16-bit
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if chunk:
                    yield chunk
                    # Simulate real-time streaming delay
                    await asyncio.sleep(0.1)
        
        # Send audio and receive results concurrently
        async def send_audio():
            """Send audio chunks to the STT websocket"""
            try:
                async for chunk in audio_stream():
                    await ws.send(chunk)
                    print(f"Sent audio chunk of {len(chunk)} bytes")
                    # Small delay to simulate realtime applications
                    await asyncio.sleep(0.02)
                
                # Signal end of audio stream
                await ws.send("finalize")
                await ws.send("done")
                print("Audio streaming completed")
                
            except Exception as e:
                print(f"Error sending audio: {e}")
        
        async def receive_transcripts():
            """Receive and process transcription results with word timestamps"""
            full_transcript = ""
            all_word_timestamps = []
            
            try:
                async for result in ws.receive():
                    if result['type'] == 'transcript':
                        text = result['text']
                        is_final = result['is_final']
                        
                        # Handle word-level timestamps
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
                        
                        if is_final:
                            # Final result - this text won't change
                            full_transcript += text + " "
                            print(f"FINAL: {text}")
                        else:
                            # Partial result - may change as more audio is processed
                            print(f"PARTIAL: {text}")
                            
                    elif result['type'] == 'done':
                        print("Transcription completed")
                        break
                        
            except Exception as e:
                print(f"Error receiving transcripts: {e}")
            
            return full_transcript.strip(), all_word_timestamps
        
        print("Starting streaming STT...")
        
        # Use asyncio.gather to run audio sending and transcript receiving concurrently
        _, (final_transcript, word_timestamps) = await asyncio.gather(
            send_audio(),
            receive_transcripts()
        )
        
        print(f"\nComplete transcript: {final_transcript}")
        print(f"Total words with timestamps: {len(word_timestamps)}")
        
        # Clean up
        await ws.close()
        
    except Exception as e:
        print(f"STT streaming error: {e}")
    finally:
        await client.close()

# Run the example
if __name__ == "__main__":
    asyncio.run(streaming_stt_example())