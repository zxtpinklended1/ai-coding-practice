import os,sys
from openai import OpenAI, APIError
from dotenv import load_dotenv

# 1. 加载 .env 文件里的 API_KEY
load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    print("API_KEY 未找到")
    sys.exit(1)

# 2. 创建客户端，base_url 指向 DeepSeek
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"  # ← 关键：换成 DeepSeek 的地址
)
# 初始化上下文
messages=[
    {"role":"system","content":"你现在是一个ai coding资深导师，请你回答简洁高效。"}
]

print("💬 多轮对话已启动（输入 'quit' 退出）")

while True:
    user_input=input("\n我想知道：").strip()
    if user_input.lower()=="quit":
        print("GGGGGOODBYE!")
        break
    if not user_input:
        continue

    messages.append({"role":"user","content":user_input})
    # 3. 发送请求
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=1024,
            temperature=0.7
        )
        ai_reply=response.choices[0].message.content
        messages.append({"role":"assistant","content":ai_reply})
        print(f"\n🤖 AI：{ai_reply}")

    except APIError as e:
        # SDK 统一把各种 API 错误包装成 APIError
        print(f"❌ API错误：{e}")
        messages.pop()
    except Exception as e:
        # 兜底
        print(f"💥 未知错误：{e}")
        messages.pop()