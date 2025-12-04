# SER - Speech and Emotion Recognition

基于大语言模型的文本情感识别工具包。

## 功能特性

- 🎯 **文本情感识别**：根据用户输入的文本识别情感状态
- 💬 **多轮对话**：支持上下文对话，自动维护对话历史
- 🏷️ **情感标签**：输出标准化的情感标签（normal, happy, tired, confident, afraid, shy）

## 安装

```bash
pip install -e .
```

## 快速开始

### 基本使用

```python
from ser import TextEmotionRecognizer

# 初始化识别器
recognizer = TextEmotionRecognizer()

# 识别文本情感
result = recognizer.recognize_text("我今天心情特别好！")
print(result)
# 输出: {'emotion': 1, 'response': '太棒了！...'}

# 多轮对话
result1 = recognizer.recognize_text("我今天刚刚完成了一个重要项目！")
result2 = recognizer.recognize_text("但是我太想休息一下了")
```

### 配置API密钥

设置环境变量：

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

或在代码中指定：

```python
recognizer = TextEmotionRecognizer(api_key="your_api_key")
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
    "emotion": int,      # 情绪标签编号 (0-5)
    "response": str      # 模型的回复内容
}
```

**情绪标签：**
- `0`: normal (正常)
- `1`: happy (开心)
- `2`: tired (疲惫)
- `3`: confident (自信)
- `4`: afraid (害怕)
- `5`: shy (害羞)


##### reset_history()

重置对话历史。

##### get_history()

获取当前对话历史。

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
print(f"情绪: {result['emotion']}")
print(f"回复: {result['response']}")
```

### 示例2：多轮对话

```python
from ser import TextEmotionRecognizer

recognizer = TextEmotionRecognizer(max_history=10)

# 第一轮
result1 = recognizer.recognize_text("我今天心情特别好！")
print(f"情绪: {result1['emotion']}")  # 1 (happy)

# 第二轮
result2 = recognizer.recognize_text("但是我太想休息一下了")
print(f"情绪: {result2['emotion']}")  # 2 (tired)
```

## 依赖项

- `openai`: OpenAI API客户端
- `httpx`: HTTP客户端库

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
