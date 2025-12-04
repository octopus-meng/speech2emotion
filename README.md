# SER - Speech and Emotion Recognition

基于大语言模型的文本情感识别和机器人步态生成工具包。

## 功能特性

- 🎯 **文本情感识别**：根据用户输入的文本识别情感状态
- 🤖 **步态生成**：整合情感识别和运动生成，输出机器人步态参数
- 💬 **多轮对话**：支持上下文对话，自动维护对话历史
- 🏷️ **情感标签**：输出标准化的情感标签（normal, happy, tired, confident, afraid, shy）

## 安装

```bash
pip install -e .
```

## 快速开始

### 基本使用 - 情感识别

```python
from ser import TextEmotionRecognizer

# 初始化识别器
recognizer = TextEmotionRecognizer()

# 识别文本情感
result = recognizer.recognize_text("我今天心情特别好！")
print(result)
# 输出: {'emotion': (1, 'happy'), 'response': '太棒了！...'}

# 多轮对话
result1 = recognizer.recognize_text("我今天刚刚完成了一个重要项目！")
result2 = recognizer.recognize_text("但是我太想休息一下了")
```

### 基本使用 - 步态生成

```python
from ser import GaitGenerator

# 初始化生成器
generator = GaitGenerator()

# 生成步态参数
result = generator.generate("快向左转！")
print(result)
# 输出: {'x_vel': 0.8, 'y_vel': 0.0, 'yaw_vel': 0.2, 'freq_offset': 0.05, 'emo_label': 'normal'}
```

### 配置API密钥

设置环境变量：

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

或在代码中指定：

```python
recognizer = TextEmotionRecognizer(api_key="your_api_key")
generator = GaitGenerator(api_key="your_api_key")
```

## API文档

### TextEmotionRecognizer

文本情感识别器。

#### 初始化参数

- `api_key` (str, optional): API密钥，默认从环境变量 `DASHSCOPE_API_KEY` 读取
- `base_url` (str): API基础URL，默认 `"https://dashscope.aliyuncs.com/compatible-mode/v1"`
- `model` (str): 模型名称，默认 `"qwen3-omni-flash"`
- `modalities` (List[str]): 输出模态，默认 `["text"]`
- `audio_config` (Dict, optional): 音频配置
- `max_history` (int, optional): 最大历史消息条数，默认 `4`

#### 方法

##### recognize_text(text, stream=False)

识别文本情感。

**参数：**
- `text` (str): 用户输入的文本
- `stream` (bool): 是否使用流式输出，默认 `False`

**返回：**
```python
{
    "emotion": tuple,    # 情绪标签元组 (编号, 名称)，如 (1, "happy")
    "response": str      # 模型的回复内容
}
```

**情绪标签：**
- `(0, "normal")`: 正常
- `(1, "happy")`: 开心
- `(2, "tired")`: 疲惫
- `(3, "confident")`: 自信
- `(4, "afraid")`: 害怕
- `(5, "shy")`: 害羞

##### reset_history()

重置对话历史。

##### get_history()

获取当前对话历史。

### GaitGenerator

步态生成器，整合情感识别和运动生成功能。

#### 初始化参数

- `api_key` (str, optional): API密钥，默认从环境变量 `DASHSCOPE_API_KEY` 读取
- `base_url` (str): API基础URL，默认 `"https://dashscope.aliyuncs.com/compatible-mode/v1"`
- `model` (str): 模型名称，默认 `"qwen3-omni-flash"`
- `modalities` (List[str]): 输出模态，默认 `["text"]`
- `audio_config` (Dict, optional): 音频配置
- `max_history` (int, optional): 最大历史消息条数，默认 `4`

#### 方法

##### generate(text, stream=False)

根据用户输入生成步态参数。自动进行情感识别，然后生成运动参数。

**参数：**
- `text` (str): 用户输入的文本
- `stream` (bool): 是否使用流式输出，默认 `False`

**返回：**
```python
{
    "x_vel": float,        # 前进速度，固定为 0.8
    "y_vel": float,       # 平移速度 (-0.3 ~ 0.3)
    "yaw_vel": float,     # 转向速度 (-0.3 ~ 0.3)
    "freq_offset": float, # 步频变化 (-0.1 ~ 0.1)
    "emo_label": str      # 情感标签名称 (normal, happy, tired, confident, afraid, shy)
}
```

##### reset_history()

重置对话历史（包括情感识别和运动生成的历史）。

##### get_history()

获取对话历史，返回包含情感识别和运动生成历史的字典。

### MotionGenerator

运动生成器，根据文本和情感生成运动参数。

#### 方法

##### generate(text, emotion=None, stream=False)

根据文本和情感生成运动参数。

**参数：**
- `text` (str): 用户输入的文本
- `emotion` (int or str, optional): 情感标签编号或名称，如果不提供则默认为 "normal"
- `stream` (bool): 是否使用流式输出，默认 `False`

**返回：**
```python
{
    "y_vel": float,       # 平移速度 (-0.3 ~ 0.3)
    "yaw_vel": float,     # 转向速度 (-0.3 ~ 0.3)
    "freq_offset": float  # 步频变化 (-0.1 ~ 0.1)
}
```

### LLMClient

底层大语言模型客户端，提供更灵活的API调用。

```python
from ser import LLMClient

client = LLMClient()
content = [{"type": "text", "text": "你好"}]
completion = client.chat(content, stream=True)

for chunk in completion:
    # 处理流式输出
    pass
```

## 示例

### 示例1：单次文本情感识别

```python
from ser import TextEmotionRecognizer

recognizer = TextEmotionRecognizer()
result = recognizer.recognize_text("我有点累了")
emotion_id, emotion_name = result['emotion']
print(f"情绪编号: {emotion_id}")
print(f"情绪名称: {emotion_name}")
print(f"回复: {result['response']}")
```

### 示例2：多轮对话

```python
from ser import TextEmotionRecognizer

recognizer = TextEmotionRecognizer(max_history=10)

# 第一轮
result1 = recognizer.recognize_text("我今天心情特别好！")
emotion_id1, emotion_name1 = result1['emotion']
print(f"情绪: {emotion_name1}")  # happy

# 第二轮
result2 = recognizer.recognize_text("但是我太想休息一下了")
emotion_id2, emotion_name2 = result2['emotion']
print(f"情绪: {emotion_name2}")  # tired
```

### 示例3：步态生成

```python
from ser import GaitGenerator

generator = GaitGenerator()

# 生成步态参数
result = generator.generate("快向左转！")
print(f"前进速度: {result['x_vel']}")
print(f"平移速度: {result['y_vel']}")
print(f"转向速度: {result['yaw_vel']}")
print(f"步频变化: {result['freq_offset']}")
print(f"情感标签: {result['emo_label']}")
```

### 示例4：完整流程

```python
from ser import GaitGenerator

generator = GaitGenerator()

# 用户输入
text = "这个项目很难，但我相信我一定可以的！"

# 自动识别情感并生成步态参数
result = generator.generate(text)
print(f"识别的情感: {result['emo_label']}")
print(f"生成的步态参数: {result}")
```

## 依赖项

- `openai`: OpenAI API客户端
- `httpx`: HTTP客户端库


