from typing import Dict, Optional

from ser.emotion_recognizer import TextEmotionRecognizer
from ser.motion_generator import MotionGenerator
X_VEL = 0.8

class GaitGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3-omni-flash",
        modalities: list = ["text"],
        audio_config: Optional[Dict] = None,
        max_history: Optional[int] = 4,
    ):
        """
        初始化步态生成器
        
        Args:
            api_key: API密钥，如果不提供则从环境变量DASHSCOPE_API_KEY读取
            base_url: API基础URL
            model: 模型名称
            modalities: 输出模态
            audio_config: 音频配置
            max_history: 最大历史消息条数
        """
        self.emotion_recognizer = TextEmotionRecognizer(
            api_key=api_key,
            base_url=base_url,
            model=model,
            modalities=modalities,
            audio_config=audio_config,
            max_history=max_history,
        )
        self.motion_generator = MotionGenerator(
            api_key=api_key,
            base_url=base_url,
            model=model,
            modalities=modalities,
            audio_config=audio_config,
            max_history=max_history,
        )
    
    def generate(
        self,
        text: str,
        stream: bool = False,
    ) -> Dict[str, any]:
        """
        根据用户输入生成步态参数
        
        Args:
            text: 用户输入的文本
            stream: 是否使用流式输出，默认False
        
        Returns:
            包含以下字段的字典：
            - y_vel: 平移速度 (-0.3 ~ 0.3)
            - yaw_vel: 转向速度 (-0.3 ~ 0.3)
            - freq_offset: 步频变化 (-0.1 ~ 0.1)
            - emo_label: 情感标签名称 (normal, happy, tired, confident, afraid, shy)
        """
        emotion_result = self.emotion_recognizer.recognize(text, stream=stream)
        emotion_tuple = emotion_result["emotion"]
        emotion_id, emotion_label = emotion_tuple

        motion_result = self.motion_generator.generate(
            text=text,
            emotion=emotion_label,  
            stream=stream,
        )
        
        return {
            "x_vel": X_VEL,
            "y_vel": motion_result["y_vel"],
            "yaw_vel": motion_result["yaw_vel"],
            "freq_offset": motion_result["freq_offset"],
            "emo_label": emotion_label,
        }
    
    def reset_history(self):
        self.emotion_recognizer.reset_history()
        self.motion_generator.reset_history()
    
    def get_history(self):
        return {
            "emotion": self.emotion_recognizer.get_history(),
            "motion": self.motion_generator.get_history(),
        }


if __name__ == "__main__":
    # 测试示例：步态生成
    print("=" * 50)
    print("GaitGenerator 测试")
    print("=" * 50)
    
    # 初始化生成器
    generator = GaitGenerator()
    
    # 测试1: 快速左转指令
    print("\n【测试1: 快速左转指令】")
    text1 = "快向左转！"
    print(f"输入文本: {text1}")
    
    result1 = generator.generate(text1, stream=False)
    print(f"步态参数: {result1}")
    print(f"  情感标签: {result1['emo_label']}")
    print(f"  平移速度: {result1['y_vel']}")
    print(f"  转向速度: {result1['yaw_vel']}")
    print(f"  步频变化: {result1['freq_offset']}")
    
    # 测试2: 慢速右移指令
    print("\n【测试2: 慢速右移指令】")
    text2 = "慢慢向右移动"
    print(f"输入文本: {text2}")
    
    result2 = generator.generate(text2, stream=False)
    print(f"步态参数: {result2}")
    print(f"  情感标签: {result2['emo_label']}")
    
    # 测试3: 疲惫情绪
    print("\n【测试3: 疲惫情绪】")
    text3 = "好想休息一下"
    print(f"输入文本: {text3}")
    
    result3 = generator.generate(text3, stream=False)
    print(f"步态参数: {result3}")
    print(f"  情感标签: {result3['emo_label']}")
    print(f"  预期: freq_offset < 0 (负面情绪降低步频)")
    
    # 测试4: 自信情绪 + 移动指令
    print("\n【测试4: 自信情绪 + 移动指令】")
    text4 = "这个项目很难，但我相信我一定可以的！"
    print(f"输入文本: {text4}")
    
    result4 = generator.generate(text4, stream=False)
    print(f"步态参数: {result4}")
    print(f"  情感标签: {result4['emo_label']}")
    print(f"  预期: freq_offset > 0 (正面情绪)")
    
    print("\n" + "=" * 50)
    print("测试完成！")

