import os
from collections import deque
from typing import List, Dict, Optional, Iterator
from openai import OpenAI
import httpx


class StreamResponseWrapper:
    def __init__(self, stream: Iterator, messages: deque):
        self.stream = stream
        self.messages = messages
        self.full_content = ""
        self._consumed = False
    
    def __iter__(self):
        try:
            for chunk in self.stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        self.full_content += delta.content
                yield chunk
        finally:
            if self.full_content and not self._consumed:
                assistant_message = {
                    "role": "assistant",
                    "content": self.full_content,
                }
                self.messages.append(assistant_message)
                self._consumed = True


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3-omni-flash",
        modalities: List[str] = ["text"],
        audio_config: Optional[Dict] = {"voice": "Cherry", "format": "wav"},
        max_history: Optional[int] = 4,
        system_message: Optional[str] = None,
    ):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥，如果不提供则从环境变量DASHSCOPE_API_KEY读取
            base_url: API基础URL
            model: 模型名称
            modalities: 输出模态，默认为["text", "audio"]
            audio_config: 音频配置，默认为{"voice": "Cherry", "format": "wav"}
            max_history: 最大历史消息条数，None表示无限制，默认为None
            system_message: 系统消息（System Message），如果不提供则使用默认的情感识别prompt
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        assert self.api_key is not None, "API key is not set"
        self.base_url = base_url
        self.model = model
        self.modalities = modalities
        self.audio_config = audio_config
        self.max_history = max_history
        
        if system_message is None:
            self.system_message = """你是一个专业的情感识别助手。请严格按照以下规则执行：

                                    **核心任务：**
                                    1. 分析用户文本内容，结合可能的音频特征（语调、语速等），综合判断情感状态
                                    2. 正常回应用户的对话内容
                                    3. **每次回复最后必须单独一行输出情绪编号**

                                    **情感判断规则（严格遵循）：**
                                    - **normal (0)**: 中性、平静、常规对话，无明显情绪色彩
                                    - **happy (1)**: 包含积极词汇、兴奋、愉悦、满足的表达
                                    - **tired (2)**: 提及疲劳、压力、困难、需要休息的内容
                                    - **confident (3)**: 展现自信、决心、肯定、自我鼓励的表达  
                                    - **afraid (4)**: 表现担忧、恐惧、不安、紧张的情绪
                                    - **shy (5)**: 包含害羞、谦虚、不好意思、腼腆的表达

                                    **标签格式要求：**
                                    - 必须放在回复最后，单独一行
                                    - 格式：[EMOTION:编号]
                                    - 示例：[EMOTION:3]

                                    **具体判断示例：**
                                    用户说："这个项目很难，但我相信我一定可以的！"
                                    → 包含"相信""一定可以"等自信表达 → [EMOTION:3]

                                    用户说："谢谢你的夸奖哦，我都有点不好意思了"
                                    → 包含"不好意思"等害羞表达 → [EMOTION:5]
                                    """
        else:
            self.system_message = system_message
        
        http_client = httpx.Client(
            trust_env=False,
            timeout=60.0,
        )
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client,
        )
        
        self.messages: deque = deque(maxlen=max_history)
    
    def chat(
        self,
        content: List[Dict],
        role: str = "user",
        stream: bool = True,
        stream_options: Optional[Dict] = None,
        reset_history: bool = False,
    ) -> Iterator:
        """
        调用大语言模型进行对话
        
        Args:
            content: 消息内容，可以是文本或多媒体内容列表
            role: 消息角色，默认为"user"
            stream: 是否使用流式输出，默认为True
            stream_options: 流式输出选项
            reset_history: 是否重置对话历史，默认为False
        
        Returns:
            流式输出时返回迭代器，非流式输出时返回完整响应
        """
        if reset_history:
            self.messages.clear()
        
        user_message = {
            "role": role,
            "content": content,
        }
        self.messages.append(user_message)
        
        messages_with_system = []
        if self.system_message:
            messages_with_system.append({
                "role": "system",
                "content": self.system_message,
            })
        messages_with_system.extend(list(self.messages))
        call_params = {
            "model": self.model,
            "messages": messages_with_system,
            "modalities": self.modalities,
            "audio": self.audio_config,
            "stream": stream,
        }
        
        if stream_options:
            call_params["stream_options"] = stream_options
        elif stream:
            call_params["stream_options"] = {"include_usage": True}
        
        completion = self.client.chat.completions.create(**call_params)
        
        if stream:
            return StreamResponseWrapper(completion, self.messages)
        else:
            response = completion
            if hasattr(response, 'choices') and response.choices:
                assistant_message = {
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                }
                self.messages.append(assistant_message)
            return response
    
    def set_system_message(self, system_message: str):
        self.system_message = system_message
    
    def get_system_message(self) -> str:
        return self.system_message
    
    def set_max_history(self, max_history: Optional[int]):
        self.max_history = max_history
        current_messages = list(self.messages)
        self.messages = deque(current_messages, maxlen=max_history)
    
    def reset_history(self):
        self.messages.clear()
    
    def get_history(self) -> List[Dict]:
        return list(self.messages)


if __name__ == "__main__":
    # 测试样例：两轮文字对话，测试情感识别功能
    print("=" * 50)
    print("LLMClient 测试：两轮文字对话（情感识别）")
    print("=" * 50)
    
    # 初始化客户端（使用默认的情感识别System Message）
    llm = LLMClient()
    print(f"\nSystem Message已设置: {llm.get_system_message()[:50]}...")
    
    # 第一轮对话
    print("\n【第一轮对话】")
    content1 = [{"type": "text", "text": "我今天刚刚完成了一个重要项目！"}]
    print(f"用户: {content1[0]['text']}")
    print("助手: ", end="", flush=True)
    
    completion1 = llm.chat(content1, stream=True)
    response1 = ""
    for chunk in completion1:
        if hasattr(chunk, 'choices') and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                print(delta.content, end='', flush=True)
                response1 += delta.content
    print()
    
    # 第二轮对话
    print("\n【第二轮对话】")
    content2 = [{"type": "text", "text": "但是我太想休息一下了"}]
    print(f"用户: {content2[0]['text']}")
    print("助手: ", end="", flush=True)
    
    completion2 = llm.chat(content2, stream=True)
    response2 = ""
    for chunk in completion2:
        if hasattr(chunk, 'choices') and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                print(delta.content, end='', flush=True)
                response2 += delta.content
    print()
    
    # 显示对话历史
    print("\n【对话历史】")
    history = llm.get_history()
    print(f"历史消息总数: {len(history)}")
    for i, msg in enumerate(history, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", [])
        # 处理用户消息（content是列表）和助手消息（content是字符串）
        if isinstance(content, list) and content:
            text_content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        elif isinstance(content, str):
            text_content = content
        else:
            text_content = str(content)
        print(f"  {i}. [{role}]: {text_content[:80]}..." if len(text_content) > 80 else f"  {i}. [{role}]: {text_content}")
    
    print("\n测试完成！")