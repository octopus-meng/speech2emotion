import json
import re
from typing import Dict, Optional, List

from ser.llm_client import LLMClient
from ser.src.prompts import GAIT_PROMPT_CN


class MotionGenerator:
    """机器人运动生成器，根据用户情感和文本输入生成运动参数（方向和速度）"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3-omni-flash",
        modalities: List[str] = ["text"],
        audio_config: Optional[Dict] = None,
        max_history: Optional[int] = 4,
        prompt: Optional[str] = GAIT_PROMPT_CN,
    ):
        """
        初始化运动生成器
        
        Args:
            api_key: API密钥，如果不提供则从环境变量DASHSCOPE_API_KEY读取
            base_url: API基础URL
            model: 模型名称
            modalities: 输出模态
            audio_config: 音频配置
            max_history: 最大历史消息条数
            prompt: 自定义prompt，如果不提供则使用默认的GAIT_PROMPT_CN
        """
        self.llm_client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            modalities=modalities,
            audio_config=audio_config,
            max_history=max_history,
            system_message=prompt
        )
    
    def generate(
        self,
        text: str,
        emotion: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, float]:
        """
        根据文本和情感生成运动参数
        
        Args:
            text: 用户输入的文本
            emotion: 情感标签
            stream: 是否使用流式输出，默认False
        
        Returns:
            包含以下字段的字典：
            - y_vel: 平移速度 (-0.3 ~ 0.3)
            - yaw_vel: 转向速度 (-0.3 ~ 0.3)
            - freq_offset: 步频变化 (-0.1 ~ 0.1)
        """
        if emotion is None:
            emotion = "normal"
        input_text = f"{text} [EMOTION:{emotion}]"
        
        content = [{"type": "text", "text": input_text}]
        
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
    
    def _parse_response(self, raw_response: str) -> Dict[str, float]:
        """
        解析模型响应，提取JSON格式的运动参数
        
        Args:
            raw_response: 原始响应文本
        
        Returns:
            包含y_vel、yaw_vel、freq_offset的字典
        """
        json_pattern = r'\{[^{}]*"y_vel"[^{}]*"yaw_vel"[^{}]*"freq_offset"[^{}]*\}'
        match = re.search(json_pattern, raw_response, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            try:
                gait_params = json.loads(json_str)
                return {
                    "y_vel": float(gait_params.get("y_vel", 0.0)),
                    "yaw_vel": float(gait_params.get("yaw_vel", 0.0)),
                    "freq_offset": float(gait_params.get("freq_offset", 0.0)),
                }
            except json.JSONDecodeError:
                pass
        return {
            "y_vel": 0.0,
            "yaw_vel": 0.0,
            "freq_offset": 0.0,
        }
    
    def reset_history(self):
        self.llm_client.reset_history()
    
    def get_history(self) -> List[Dict]:
        return self.llm_client.get_history()


if __name__ == "__main__":
    # 测试示例：运动参数生成
    print("=" * 50)
    print("MotionGenerator 测试")
    print("=" * 50)
    
    # 初始化生成器
    generator = MotionGenerator()
    
    # 测试1: 快速左转指令（带happy情绪）
    print("\n【测试1: 快速左转指令（happy情绪）】")
    text1 = "快向左转！"
    emotion1 = 1  # happy
    print(f"输入文本: {text1}")
    print(f"情感标签: {emotion1} (happy)")
    
    result1 = generator.generate(text1, emotion=emotion1, stream=False)
    print(f"运动参数: {result1}")
    print(f"  预期: yaw_vel > 0 (左转), freq_offset > 0 (正面情绪)")
    
    # 测试2: 慢速右移指令（无情感标签）
    print("\n【测试2: 慢速右移指令（无情感标签）】")
    text2 = "慢慢向右移动"
    print(f"输入文本: {text2}")
    
    result2 = generator.generate(text2, stream=False)
    print(f"运动参数: {result2}")
    print(f"  预期: y_vel < 0 (右移), yaw_vel = 0, freq_offset ≈ 0")
    
    # 测试3: 疲惫情绪（影响步频）
    print("\n【测试3: 疲惫情绪（影响步频）】")
    text3 = "好想休息一下"
    emotion3 = 2  # tired
    print(f"输入文本: {text3}")
    print(f"情感标签: {emotion3} (tired)")
    
    result3 = generator.generate(text3, emotion=emotion3, stream=False)
    print(f"运动参数: {result3}")
    print(f"  预期: freq_offset < 0 (负面情绪降低步频)")
    
    # 测试4: 自信情绪 + 移动指令
    print("\n【测试4: 自信情绪 + 移动指令】")
    text4 = "向左跨一步"
    emotion4 = 3  # confident
    print(f"输入文本: {text4}")
    print(f"情感标签: {emotion4} (confident)")
    
    result4 = generator.generate(text4, emotion=emotion4, stream=False)
    print(f"运动参数: {result4}")
    print(f"  预期: y_vel > 0 (左移), freq_offset > 0 (正面情绪)")
    
    # 测试5: 害怕情绪
    print("\n【测试5: 害怕情绪】")
    text5 = "有点紧张"
    emotion5 = 4  # afraid
    print(f"输入文本: {text5}")
    print(f"情感标签: {emotion5} (afraid)")
    
    result5 = generator.generate(text5, emotion=emotion5, stream=False)
    print(f"运动参数: {result5}")
    print(f"  预期: freq_offset < 0 (负面情绪)")
    
    # 测试6: 中性对话（无指令）
    print("\n【测试6: 中性对话（无指令）】")
    text6 = "你好"
    emotion6 = 0  # normal
    print(f"输入文本: {text6}")
    print(f"情感标签: {emotion6} (normal)")
    
    result6 = generator.generate(text6, emotion=emotion6, stream=False)
    print(f"运动参数: {result6}")
    print(f"  预期: 所有参数接近0（无明确指令）")
    
    print("\n" + "=" * 50)
    print("测试完成！")
