# Voice Agent Backend

A real-time voice conversation agent built with Python, implementing a complete pipeline from speech input to speech output using advanced AI services.

## 🎯 Overview

This voice agent provides a seamless conversational experience by processing real-time audio input, converting it to text, generating intelligent responses, and synthesizing speech output. The entire system is built on asynchronous programming principles with queue-based buffering between each processing stage.

## 🏗️ Architecture

The system follows a modular pipeline architecture with five main components:

```
🎤 Audio Input → 📝 Speech-to-Text → 🧠 LLM Processing → 🔊 Text-to-Speech → 🔈 Audio Output
     (sources)      (asr)           (llm)           (tts)        (audio_player)
```

### Core Components

1. **Audio Source** (`sources.py`)

   - `RealTimeMicrophoneSource`: Captures real-time audio from microphone
   - Provides 16kHz, 16-bit PCM audio in 100ms chunks
   - Uses sounddevice for cross-platform audio capture

2. **Automatic Speech Recognition** (`asr.py`)

   - `CartesiaASR`: Converts audio to text using Cartesia's Whisper-based model
   - Supports real-time streaming transcription
   - Provides word-level timestamps and partial/final results
   - Optimized for Chinese language processing

3. **Large Language Model** (`llm.py`)

   - `AnthropicLLM`: Generates conversational responses using Claude
   - Implements intelligent batching with timeout-based sentence completion
   - Maintains conversation history for context awareness
   - Streams responses for low-latency interaction

4. **Text-to-Speech** (`tts.py`)

   - `CartesiaTTS`: Synthesizes natural speech using Cartesia's Sonic model
   - Accumulates text into complete sentences before synthesis
   - Outputs high-quality 24kHz PCM audio streams
   - Supports real-time streaming synthesis

5. **Audio Playback** (`audio_player.py`)
   - `SoundDeviceAudioPlayer`: Real-time audio playback with buffering
   - Intelligent buffer management to prevent audio dropouts
   - Configurable latency and buffer size settings
   - Supports both float32 and int16 audio formats

### Abstract Interfaces

The system uses well-defined abstract base classes (`components.py`) to ensure modularity and extensibility:

- `AudioSource`: Audio input interface
- `ASRInterface`: Speech recognition interface
- `LLMInterface`: Language model interface
- `TTSInterface`: Text-to-speech interface
- `AudioPlayer`: Audio output interface

## 🔄 Asynchronous Pipeline

The system uses `asyncio.Queue` objects as buffers between each stage, enabling:

- **Concurrent Processing**: All stages run simultaneously
- **Backpressure Handling**: Queues prevent overwhelming downstream components
- **Low Latency**: Streaming processing without waiting for complete inputs
- **Fault Tolerance**: Isolated error handling in each component

### Queue Flow

```python
audio_queue        # Raw audio chunks (bytes)
    ↓
text_queue         # Transcribed text fragments (str)
    ↓
response_queue     # LLM response chunks (str)
    ↓
tts_chunk_queue    # Synthesized audio chunks (bytes)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Audio input/output devices (microphone and speakers)
- API keys for Cartesia and Anthropic services

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd voice-agent-backend
   ```

2. Install dependencies:

   ```bash
   pip install cartesia anthropic sounddevice numpy python-dotenv asyncio
   ```

3. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Configure your `.env` file:
   ```bash
   CARTESIA_API_KEY=your_cartesia_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

### Running the Application

```bash
python main.py
```

The application will:

1. Start capturing audio from your default microphone
2. Begin real-time speech recognition
3. Process your speech with the LLM
4. Synthesize and play back responses
5. Continue the conversation loop

### Stopping the Application

Press `Ctrl+C` to gracefully stop all components.

## ⚙️ Configuration

### Audio Settings

In `sources.py` (`RealTimeMicrophoneSource`):

```python
sample_rate=16000,      # Audio sample rate
channels=1,             # Mono audio
chunk_duration_ms=100,  # 100ms audio chunks
```

In `audio_player.py` (`SoundDeviceAudioPlayer`):

```python
sample_rate=24000,           # Playback sample rate
buffer_size=2048,            # Audio buffer size
min_buffer_samples=4800      # Minimum samples before playback starts
```

### LLM Settings

In `llm.py` (`AnthropicLLM`):

```python
timeout=1.0,                    # Timeout for user speech completion
model="claude-sonnet-4-20250514"  # Anthropic model
```

### TTS Settings

In `tts.py` (`CartesiaTTS`):

```python
sample_rate=24000,                              # Output sample rate
voice_id="a0e99841-438c-4a64-b679-ae501e7d6091"  # Voice selection
```

## 🔧 Technical Details

### Performance Optimizations

1. **Streaming Processing**: No component waits for complete input before starting output
2. **Intelligent Buffering**: Audio player uses adaptive buffering to prevent dropouts
3. **Concurrent Execution**: All pipeline stages run in parallel using asyncio
4. **Memory Efficient**: Processes data in small chunks rather than loading entire conversations

### Error Handling

- Each component includes comprehensive error handling
- Failed operations don't crash the entire pipeline
- Automatic retry mechanisms for transient failures
- Graceful degradation when services are unavailable

### Latency Considerations

- **Audio Chunks**: 100ms for good balance of latency vs. efficiency
- **TTS Buffering**: Waits for complete sentences for natural speech
- **LLM Timeout**: 1-second timeout for detecting end of user speech
- **Audio Playback**: 200ms minimum buffer to prevent dropouts

## 📊 Monitoring and Debugging

The application provides detailed logging for each component:

```
[MicSource] Started recording...
[ASR] First text: 'hello there'
[LLM] Sending to Anthropic: 'hello there'
[TTS] Accumulated text: 'Hello! How can I help you today?'
[AudioPlayer] Added 2048 samples to buffer
```

Enable verbose logging by monitoring console output during operation.

## 🔮 Future Enhancements

- Support for multiple languages
- Voice activity detection (VAD)
- Turn detection
- Interrupt handling (allowing users to interrupt responses)
- Multiple voice options
- WebSocket API for web integration
- Recording and playback of conversations
- Custom wake word detection

## 🤝 Contributing

1. Follow the established abstract interface patterns
2. Maintain asynchronous programming principles
3. Add comprehensive error handling
4. Include detailed logging for debugging
5. Test with various audio configurations

## 📄 License

MIT

## 🙏 Acknowledgments

- [Cartesia](https://cartesia.ai/) for ASR and TTS services
- [Anthropic](https://anthropic.com/) for LLM capabilities
- [sounddevice](https://python-sounddevice.readthedocs.io/) for audio I/O
