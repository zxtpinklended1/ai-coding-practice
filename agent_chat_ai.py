import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. 加载环境变量
load_dotenv()
api_key = os.getenv("API_KEY")

# 2. 初始化客户端 (复用周三的代码)
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

def chat_with_ai(prompt, temperature=0.7, max_tokens=100, stream=False):
    """
    封装好的聊天函数
    :param prompt: 用户输入
    :param temperature: 温度 (0.0 - 2.0)，越高越随机，越低越严谨
    :param max_tokens: 回复的最大长度限制
    :param stream: 是否开启流式输出
    """
    messages = [
        {"role": "system", "content": "你是一个专业的 Python 编程助手。"},
        {"role": "user", "content": prompt}
    ]

    try:
        # 3. 发送请求
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream  # 开启流式模式
        )

        # 4. 处理结果
        if stream:
            print("--- 流式输出开始 ---")
            full_content = ""
            for chunk in response:
                # 获取当前片段的文本
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True) # end=""不换行，flush=True强制刷新
                    full_content += content
            print("\n--- 结束 ---")
            return full_content
        else:
            # 普通模式直接返回
            answer= response.choices[0].message.content
            print(f"AI 回复: {answer}")
            return answer

    except Exception as e:
        print(f"出错了: {e}")
        return None

# ================= 实验区域 =================

if __name__ == "__main__":
    test_prompt = "你好，请描述狸花猫的外表。"

    # 实验 A: 正常模式 (Temperature 0.7)
    print("\n>>> 实验 A: 标准回答")
    chat_with_ai(test_prompt)

    # 实验 B: 严谨模式 (Temperature 0.1) - 适合代码生成、数学题
    print("\n>>> 实验 B: 严谨模式 (Temp=0.1)")
    chat_with_ai(test_prompt, temperature=0.1)

    # 实验 C: 疯狂模式 (Temperature 1.5) - 适合创意写作
    print("\n>>> 实验 C: 疯狂模式 (Temp=1.5)")
    chat_with_ai(test_prompt, temperature=1.5)

    # 实验 D: 长度限制 (Max Tokens 10)
    print("\n>>> 实验 D: 截断测试 (Max Tokens=10)")
    chat_with_ai(test_prompt, max_tokens=10)

    # 实验 E: 流式输出体验 (Stream=True)
    print("\n>>> 实验 E: 流式打字机效果")
    chat_with_ai("给我讲一个简短的笑话", stream=True)