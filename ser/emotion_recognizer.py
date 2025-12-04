import re
from typing import Dict, Optional, List

from ser.llm_client import LLMClient



class TextEmotionRecognizer:
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3-omni-flash",
        modalities: List[str] = ["text"],
        audio_config: Optional[Dict] = None,
        max_history: Optional[int] = 4,
    ):
        """
        初始化文本情感识别器
        
        Args:
            api_key: API密钥，如果不提供则从环境变量DASHSCOPE_API_KEY读取
            base_url: API基础URL
            model: 模型名称
            modalities: 输出模态
            audio_config: 音频配置
            max_history: 最大历史消息条数
        """
        self.llm_client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            modalities=modalities,
            audio_config=audio_config,
            max_history=max_history,
        )
    
    def recognize(
        self,
        text: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, any]:
        """
        识别文本和/或音频的情感
        
        Args:
            text: 用户输入的文本
            audio_url: 音频文件的URL或base64编码的音频数据
            audio_format: 音频格式，默认为"wav"
            stream: 是否使用流式输出，默认为False
        
        Returns:
            包含以下字段的字典：
            - emotion: 情绪标签编号 (0-5)，如果未找到标签则默认为0
            - response: 模型的回复内容（不包含情绪标签）
        """
        content = []
        
        if text:
            content.append({
                "type": "text",
                "text": text,
            })
        else:
            raise ValueError("not text provided")
    
        if stream:
            completion = self.llm_client.chat(content, stream=True)
            full_response = ""
            for chunk in completion:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        full_response += delta.content
        else:
            response = self.llm_client.chat(content, stream=False)
            if hasattr(response, 'choices') and response.choices:
                full_response = response.choices[0].message.content
            else:
                full_response = ""
        
        return self._parse_response(full_response)
    
    def _parse_response(self, raw_response: str) -> Dict[str, any]:
        emotion_pattern = r'\[EMOTION:(\d+)\]'
        match = re.search(emotion_pattern, raw_response)
        
        if match:
            emotion = int(match.group(1))
            if emotion >= 6 or emotion < 0: 
                print(f"error emotion id {emotion}, set to 0")
                emotion = 0 
            response = re.sub(emotion_pattern, '', raw_response).strip()
        else:
            response = raw_response.strip()
            emotion = 0
        
        return {
            "emotion": emotion,
            "response": response,
        } 
    
    def reset_history(self):
        self.llm_client.reset_history()
    
    def get_history(self) -> List[Dict]:
        return self.llm_client.get_history()


if __name__ == "__main__":
    # 测试示例：文本情感识别
    print("=" * 50)
    print("TextEmotionRecognizer 测试")
    print("=" * 50)
    
    # 初始化识别器
    recognizer = TextEmotionRecognizer()
    
    # 测试1: 文本情感识别
    print("\n【测试1: 文本情感识别】")
    text1 = "这个项目很难，但我相信我一定可以的！"
    print(f"输入文本: {text1}")
    
    result1 = recognizer.recognize(text1, stream=False)
    print(f"回复: {result1['response']}")
    print(f"情绪标签编号: {result1['emotion']}")
    
    # 测试2: 第二轮对话
    print("\n【测试2: 第二轮对话】")
    text2 = "谢谢你的夸奖哦，我都有点不好意思了"
    print(f"输入文本: {text2}")
    
    result2 = recognizer.recognize(text2, stream=False)
    print(f"回复: {result2['response']}")
    print(f"情绪标签编号: {result2['emotion']}")
    
    # 显示返回的字典结构
    print("\n【返回字典结构】")
    print(f"result1: {result1}")
    print(f"result2: {result2}")
    
    print("\n测试完成！")

