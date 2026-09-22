# 加stream和flush版本

# import os
# from openai import OpenAI
# from dotenv import load_dotenv

# # 1. 加载环境变量
# load_dotenv()
# api_key = os.getenv("API_KEY")

# # 2. 初始化客户端 (复用周三的代码)
# client = OpenAI(
#     api_key=api_key,
#     base_url="https://api.deepseek.com"
# )

# def chat_with_ai(prompt, temperature=0.7, max_tokens=100, stream=False):
#     """
#     封装好的聊天函数
#     :param prompt: 用户输入
#     :param temperature: 温度 (0.0 - 2.0)，越高越随机，越低越严谨
#     :param max_tokens: 回复的最大长度限制
#     :param stream: 是否开启流式输出
#     """
#     messages = [
#         {"role": "system", "content": "你是一个专业的 Python 编程助手。"},
#         {"role": "user", "content": prompt}
#     ]

#     try:
#         # 3. 发送请求
#         response = client.chat.completions.create(
#             model="deepseek-chat",
#             messages=messages,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             stream=stream  # 开启流式模式
#         )

#         # 4. 处理结果
#         if stream:
#             print("--- 流式输出开始 ---")
#             full_content = ""
#             for chunk in response:
#                 # 获取当前片段的文本
#                 if chunk.choices[0].delta.content:
#                     content = chunk.choices[0].delta.content
#                     print(content, end="", flush=True) # end=""不换行，flush=True强制刷新
#                     full_content += content
#             print("\n--- 结束 ---")
#             return full_content
#         else:
#             # 普通模式直接返回
#             answer= response.choices[0].message.content
#             print(f"AI 回复: {answer}")
#             return answer

#     except Exception as e:
#         print(f"出错了: {e}")
#         return None

# # ================= 实验区域 =================

# if __name__ == "__main__":
#     test_prompt = "你好，请描述狸花猫的外表。"

#     # 实验 A: 正常模式 (Temperature 0.7)
#     print("\n>>> 实验 A: 标准回答")
#     chat_with_ai(test_prompt)

#     # 实验 B: 严谨模式 (Temperature 0.1) - 适合代码生成、数学题
#     print("\n>>> 实验 B: 严谨模式 (Temp=0.1)")
#     chat_with_ai(test_prompt, temperature=0.1)

#     # 实验 C: 疯狂模式 (Temperature 1.5) - 适合创意写作
#     print("\n>>> 实验 C: 疯狂模式 (Temp=1.5)")
#     chat_with_ai(test_prompt, temperature=1.5)

#     # 实验 D: 长度限制 (Max Tokens 10)
#     print("\n>>> 实验 D: 截断测试 (Max Tokens=10)")
#     chat_with_ai(test_prompt, max_tokens=10)

#     # 实验 E: 流式输出体验 (Stream=True)
#     print("\n>>> 实验 E: 流式打字机效果")
#     chat_with_ai("给我讲一个简短的笑话", stream=True)



# 多轮对话初版本

# import os,sys
# from openai import OpenAI, APIError
# from dotenv import load_dotenv

# # 1. 加载 .env 文件里的 API_KEY
# load_dotenv()
# api_key = os.getenv("API_KEY")

# if not api_key:
#     print("API_KEY 未找到")
#     sys.exit(1)

# # 2. 创建客户端，base_url 指向 DeepSeek
# client = OpenAI(
#     api_key=api_key,
#     base_url="https://api.deepseek.com"  # ← 关键：换成 DeepSeek 的地址
# )
# # 初始化上下文
# messages=[
#     {"role":"system","content":"你现在是一个ai coding资深导师，请你回答简洁高效。"}
# ]

# print("💬 多轮对话已启动（输入 'quit' 退出）")

# while True:
#     user_input=input("\n我想知道：").strip()
#     if user_input.lower()=="quit":
#         print("GGGGGOODBYE!")
#         break
#     if not user_input:
#         continue

#     messages.append({"role":"user","content":user_input})
#     # 3. 发送请求
#     try:
#         response = client.chat.completions.create(
#             model="deepseek-chat",
#             messages=messages,
#             max_tokens=1024,
#             temperature=0.7
#         )
#         ai_reply=response.choices[0].message.content
#         messages.append({"role":"assistant","content":ai_reply})
#         print(f"\n🤖 AI：{ai_reply}")

#     except APIError as e:
#         # SDK 统一把各种 API 错误包装成 APIError
#         print(f"❌ API错误：{e}")
#         messages.pop()
#     except Exception as e:
#         # 兜底
#         print(f"💥 未知错误：{e}")
#         messages.pop()


# 多轮对话CLI
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