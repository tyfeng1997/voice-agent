import asyncio
from enum import Enum
from typing import Dict, Set

class PipelineState(Enum):
    LISTENING = "listening"
    PROCESSING = "processing" 
    RESPONDING = "responding"
    INTERRUPTING = "interrupting"  # 🔥 新增中断清理状态

class ConversationManager:
    """全局对话状态管理器 - 支持同步清理"""
    
    def __init__(self):
        self.state = PipelineState.LISTENING
        self.interrupt_event = asyncio.Event()
        
        # 🔥 组件清理完成信号
        self.cleanup_events: Dict[str, asyncio.Event] = {
            'llm': asyncio.Event(),
            'tts': asyncio.Event(), 
            'audio_player': asyncio.Event()
        }
        
        # 🔥 版本号机制 - 用于丢弃旧数据
        self.current_session_id = 0
        
    def set_state(self, new_state: PipelineState):
        print(f"[ConversationManager] State: {self.state.value} -> {new_state.value}")
        self.state = new_state
        
    async def trigger_interrupt(self):
        """触发中断并等待所有组件清理完成"""
        if self.state != PipelineState.RESPONDING:
            return
            
        print("[ConversationManager] Triggering interrupt...")
        self.set_state(PipelineState.INTERRUPTING)
        
        # 🔥 递增session_id，使旧数据失效
        self.current_session_id += 1
        print(f"[ConversationManager] New session ID: {self.current_session_id}")
        
        # 设置中断信号
        self.interrupt_event.set()
        
        # 🔥 等待所有组件清理完成
        cleanup_tasks = [
            self.cleanup_events['llm'].wait(),
            self.cleanup_events['tts'].wait(), 
            self.cleanup_events['audio_player'].wait()
        ]
        
        try:
            await asyncio.wait_for(asyncio.gather(*cleanup_tasks), timeout=2.0)
            print("[ConversationManager] All components cleaned up successfully")
        except asyncio.TimeoutError:
            print("[ConversationManager] Warning: Cleanup timeout, proceeding anyway")
        
        # 重置所有事件
        self.interrupt_event.clear()
        for event in self.cleanup_events.values():
            event.clear()
            
        # 🔥 只有在清理完成后才切换到LISTENING
        self.set_state(PipelineState.LISTENING)
        
    def signal_cleanup_complete(self, component: str):
        """组件报告清理完成"""
        if component in self.cleanup_events:
            self.cleanup_events[component].set()
            print(f"[ConversationManager] {component} cleanup complete")
            
    def get_current_session_id(self) -> int:
        """获取当前会话ID"""
        return self.current_session_id