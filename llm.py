# llm.py - Large Language Model Interface using Anthropic Claude
"""
This module implements the language model interface using Anthropic's Claude API.

The AnthropicLLM class provides conversational AI capabilities with:
- Streaming response generation for low-latency interaction
- Intelligent text accumulation with timeout-based sentence completion
- Conversation history management for context awareness
- Optimized for voice conversation scenarios
- Support for both partial and complete user utterances

The implementation uses Claude's streaming API to generate responses as they're
being produced, allowing for real-time conversational experiences.
"""

import asyncio
import os
from anthropic import AsyncAnthropic
from components import LLMInterface
from typing import AsyncGenerator, List, Dict

class AnthropicLLM(LLMInterface):
    """
    Anthropic Claude-powered Language Model implementation.
    
    This class provides conversational AI using Claude's streaming API.
    It implements intelligent batching to accumulate user input fragments
    into complete thoughts before generating responses.
    
    Key Features:
    - Streaming response generation for real-time interaction
    - Timeout-based utterance completion detection
    - Conversation history management
    - Optimized system prompts for voice conversation
    - Error handling and graceful degradation
    """
    
    def __init__(self, timeout: float = 1.0, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the Anthropic LLM client.
        
        Args:
            timeout: Seconds to wait for more text before considering utterance complete
            model: Anthropic model name to use for generation
        """
        self.timeout = timeout
        self.model = model
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = AsyncAnthropic(api_key=self.api_key)
        
        # Conversation history for maintaining context
        self.message_history: List[Dict[str, str]] = []
        # System prompt optimized for voice conversation
        self.system_prompt = "You are a helpful AI assistant in a voice conversation. Keep your responses natural and conversational, as if speaking aloud. Avoid using markdown formatting or complex punctuation."
        
        print(f"[LLM] Initialized with model: {model}")

    async def generate_stream(self, text_queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """
        Generate streaming responses from user input text.
        
        This method implements intelligent batching by accumulating text fragments
        from the ASR until a timeout occurs, indicating the user has finished speaking.
        Then it generates a streaming response using Claude's API.
        
        Args:
            text_queue: Queue containing text fragments from ASR
            
        Yields:
            str: Response text chunks for streaming TTS synthesis
        """
        while True:
            # Step 1: Wait for the first text fragment
            try:
                first_text = await text_queue.get()
                # Check for end signal
                if first_text == "END":
                    print("[LLM] Received END signal")
                    break
                buffer = first_text.strip()
                print(f"[LLM] First text: '{buffer}'")
            except asyncio.QueueEmpty:
                continue

            # Step 2: Accumulate additional text until timeout (user finished speaking)
            while True:
                try:
                    # Non-blocking attempt to get more text with timeout
                    more_text = await asyncio.wait_for(text_queue.get(), timeout=self.timeout)
                    if more_text == "END":
                        print(f"[LLM] Got END signal with accumulated text: '{buffer}'")
                        break
                    buffer += " " + more_text.strip()
                    print(f"[LLM] Accumulated: '{buffer}'")
                except asyncio.TimeoutError:
                    # Queue idle beyond timeout → user finished speaking
                    print(f"[LLM] Timeout reached. Final input: '{buffer}'")
                    break

            # Step 3: Generate streaming response if we have valid input
            if buffer.strip():
                try:
                    async for chunk in self._generate_anthropic_stream(buffer):
                        yield chunk
                except Exception as e:
                    print(f"[LLM] Error generating response: {e}")
                    yield "I'm sorry, I encountered an error processing your request. "

    async def _generate_anthropic_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        Generate streaming response using Anthropic's Claude API.
        
        This method handles the actual API call to Claude and manages
        conversation history for context awareness.
        
        Args:
            user_input: The complete user utterance to respond to
            
        Yields:
            str: Response text chunks for real-time synthesis
        """
        try:
            # Add user message to conversation history
            self.message_history.append({"role": "user", "content": user_input})
            
            # Maintain reasonable history length (last 10 conversation turns)
            if len(self.message_history) > 20:  # 10 turns = 20 messages
                self.message_history = self.message_history[-20:]
            
            print(f"[LLM] Sending to Anthropic: '{user_input}'")
            print(f"[LLM] Message history length: {len(self.message_history)}")
            
            # Prepare request with conversation history
            messages = self.message_history.copy()
            
            # Call Anthropic API with streaming
            assistant_response = ""
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=messages
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        assistant_response += text
                        # Stream text chunks for real-time TTS synthesis
                        yield text + " "
                        
            
            # Add assistant response to conversation history
            if assistant_response.strip():
                self.message_history.append({
                    "role": "assistant", 
                    "content": assistant_response.strip()
                })
                print(f"[LLM] Assistant response completed: '{assistant_response[:100]}...'")
        
        except Exception as e:
            print(f"[LLM] Error in Anthropic API call: {e}")
            error_response = "I apologize, but I'm having trouble processing your request right now."
            self.message_history.append({"role": "assistant", "content": error_response})
            yield error_response + " "

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List[Dict[str, str]]: List of message dictionaries with 'role' and 'content'
        """
        return self.message_history.copy()

    def clear_history(self):
        """
        Clear the conversation history.
        
        Useful for starting fresh conversations or managing memory usage.
        """
        self.message_history.clear()
        print("[LLM] Conversation history cleared")

    def set_system_prompt(self, prompt: str):
        """
        Update the system prompt for the conversation.
        
        Args:
            prompt: New system prompt to use for guiding the assistant's behavior
        """
        self.system_prompt = prompt
        print(f"[LLM] System prompt updated: '{prompt[:50]}...'")