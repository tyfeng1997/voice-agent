# llm.py
import asyncio
import os
from anthropic import AsyncAnthropic
from components import LLMInterface
from typing import AsyncGenerator, List, Dict

class AnthropicLLM(LLMInterface):
    def __init__(self, timeout: float = 1.0, model: str = "claude-sonnet-4-20250514"):
        """
        :param timeout: 多久没新文本就认为用户说完了（单位：秒）
        :param model: Anthropic 模型名称
        """
        self.timeout = timeout
        self.model = model
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = AsyncAnthropic(api_key=self.api_key)
        
        # 消息历史管理
        self.message_history: List[Dict[str, str]] = []
        self.system_prompt = "You are a helpful AI assistant in a voice conversation. Keep your responses natural and conversational, as if speaking aloud. Avoid using markdown formatting or complex punctuation."
        
        print(f"[LLM] Initialized with model: {model}")

    async def generate_stream(self, text_queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """从队列接收用户输入，生成流式回复"""
        while True:
            # Step 1: 等待第一个文本块
            try:
                first_text = await text_queue.get()
                # 如果是 "END" 信号，退出
                if first_text == "END":
                    print("[LLM] Received END signal")
                    break
                buffer = first_text.strip()
                print(f"[LLM] First text: '{buffer}'")
            except asyncio.QueueEmpty:
                continue

            # Step 2: 持续消费队列，直到超时（空闲）
            while True:
                try:
                    # 非阻塞地尝试获取更多文本
                    more_text = await asyncio.wait_for(text_queue.get(), timeout=self.timeout)
                    if more_text == "END":
                        print(f"[LLM] Got END signal with accumulated text: '{buffer}'")
                        break
                    buffer += " " + more_text.strip()
                    print(f"[LLM] Accumulated: '{buffer}'")
                except asyncio.TimeoutError:
                    # 队列空闲超过 timeout → 认为用户说完了
                    print(f"[LLM] Timeout reached. Final input: '{buffer}'")
                    break

            # Step 3: 如果有有效输入，调用 Anthropic API 生成流式回复
            if buffer.strip():
                try:
                    async for chunk in self._generate_anthropic_stream(buffer):
                        yield chunk
                except Exception as e:
                    print(f"[LLM] Error generating response: {e}")
                    yield "I'm sorry, I encountered an error processing your request. "

    async def _generate_anthropic_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """调用 Anthropic API 生成流式回复"""
        try:
            # 添加用户消息到历史
            self.message_history.append({"role": "user", "content": user_input})
            
            # 保持历史长度合理（最近10轮对话）
            if len(self.message_history) > 20:  # 10轮对话 = 20条消息
                self.message_history = self.message_history[-20:]
            
            print(f"[LLM] Sending to Anthropic: '{user_input}'")
            print(f"[LLM] Message history length: {len(self.message_history)}")
            
            # 构建请求
            messages = self.message_history.copy()
            
            # 调用 Anthropic API
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
                        # 按词或短语分块发送
                        yield text + " "
                        
                        
            
            # 将助手回复添加到历史
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
        """获取对话历史"""
        return self.message_history.copy()

    def clear_history(self):
        """清空对话历史"""
        self.message_history.clear()
        print("[LLM] Conversation history cleared")

    def set_system_prompt(self, prompt: str):
        """设置系统提示"""
        self.system_prompt = prompt
        print(f"[LLM] System prompt updated: '{prompt[:50]}...'")