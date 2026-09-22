import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com"
)

messages = [{"role": "system", "content": "你是一个专业的 Python 编程助手。"}]

# 历史长度保护：最多保留最近 N 轮（system + 最近若干组问答），防止 Token 无限膨胀
MAX_TURNS = 20

def trim_history():
    """保留 system 首条 + 最近 MAX_TURNS 组(每组 user+assistant 两条)对话。"""
    keep = 1 + MAX_TURNS * 2
    if len(messages) > keep:
        # 删除最旧的一批，但要保证删除后仍以 user 开头成对
        del messages[1: len(messages) - (keep - 1)]

def chat_once_stream(user_input):
    messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            stream=True,          # 开启流式
        )
        print("AI: ", end="", flush=True)
        full = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                piece = chunk.choices[0].delta.content
                print(piece, end="", flush=True)   # 打字机效果
                full += piece
        print()                                    # 流结束后换行
        messages.append({"role": "assistant", "content": full})
        trim_history()                          # 每轮结束后裁剪历史
        return full
    except Exception as e:
        # 出错时把刚才追加的 user 撤回，避免污染历史
        messages.pop()
        print(f"\n[出错了] {e}")
        return None

if __name__ == "__main__":
    print("=== 流式多轮对话已启动，输入 /quit 退出 ===\n")
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("/quit", "/exit", "退出"):
            print("AI: 再见！")
            break
        if not user_input:
            continue
        chat_once_stream(user_input)
        print()