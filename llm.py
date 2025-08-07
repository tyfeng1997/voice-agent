# llm.py
import asyncio
from components import LLMInterface
from typing import AsyncGenerator

class StreamingLLM(LLMInterface):
    def __init__(self, timeout: float = 1.0):
        """
        :param timeout: 多久没新文本就认为用户说完了（单位：秒）
        """
        self.timeout = timeout

    async def generate_stream(self, text_queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        while True:
            # Step 1: 等待第一个文本块
            try:
                first_text = await text_queue.get()
                # 如果是 "END" 信号，退出
                if first_text == "END":
                    break
                buffer = first_text
            except asyncio.QueueEmpty:
                continue

            # Step 2: 持续消费队列，直到超时（空闲）
            while True:
                try:
                    # 非阻塞地尝试获取更多文本
                    more_text = await asyncio.wait_for(text_queue.get(), timeout=self.timeout)
                    if more_text == "END":
                        print("LLM got \"END\"  ",buffer)
                        break
                    buffer += " " + more_text
                    print(f"[LLM] Accumulated: '{buffer}'")
                except asyncio.TimeoutError:
                    # 队列空闲超过 timeout → 认为用户说完了
                    print(f"[LLM] Timeout reached. Final input: '{buffer}'")
                    break

            # Step 3: 调用 LLM 生成流式回复
            async for word in self._llm_generate(buffer):
                yield word

            # 可选：重置 buffer，继续监听下一轮
            buffer = ""

    async def _llm_generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """模拟 LLM 流式生成，可替换为真实 API"""
        response = f"I understand you said: '{prompt}'. That's interesting! Let me tell you more about it."
        print("response split  ", response.split())
        for word in response.split():
            # await asyncio.sleep(0.05)
            yield word + " "