# import os
# from openai import OpenAI
# from datetime import datetime
# from dotenv import load_dotenv
# import yaml
# import json
# import tiktoken


# # 1. 加载环境变量与初始化客户端
# load_dotenv()
# client = OpenAI(
#     api_key=os.getenv("API_KEY"),
#     base_url="https://api.deepseek.com"
# )

# # 定义人设相关函数
# def load_prompts():
#     # 【修改】动态获取当前 .py 文件所在的绝对目录
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     path = os.path.join(base_dir, "prompts.yaml")
    
#     with open(path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)

# # 【修改】直接在这里调用，现在它会自动寻找同目录下的 prompts.yaml
# prompts = load_prompts()

# menu = {
#     "1": ("socrates", "苏格拉底哲学家"),
#     "2": ("architect", "机器人"),
#     "3": ("guide", "二次元新手村向导"),
# }

# # 加入长期记忆
# # 【动态获取路径】找到当前 .py 文件所在的目录，并拼接 memory.json
# MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")

# def load_memory():
#     """加载记忆文件"""
#     if os.path.exists(MEMORY_FILE):
#         try:
#             with open(MEMORY_FILE, "r", encoding="utf-8") as f:
#                 return json.load(f)
#         except (json.JSONDecodeError, ValueError):
#             return {}
#     return {}

# def save_memory(key, value):
#     """将单条记忆保存到文件"""
#     memory = load_memory()
#     memory[key] = value
#     with open(MEMORY_FILE, "w", encoding="utf-8") as f:
#         json.dump(memory, f, ensure_ascii=False, indent=2)

# # MAX_TURNS = 20  # 最多保留的对话轮数

# # def trim_history(messages):
# #     """
# #     智能裁剪历史：
# #     1. 始终保留 System Prompt (index 0)
# #     2. 删除中间过旧的历史，但确保剩余的第一条是 user 消息（保持 user-assistant 成对）
# #     """
# #     keep = 1 + MAX_TURNS * 2
# #     if len(messages) > keep:
# #         del_index = len(messages) - (keep - 1)
# #         if messages[del_index]["role"] == "assistant":
# #             del_index -= 1
# #         del messages[1:del_index]

# # 定义滑动窗口截断函数
# def trim_history_by_token(messages, max_tokens=3000):
#     # 1. 获取分词尺子
#     encoding = tiktoken.encoding_for_model("gpt-4o-mini") #从 tiktoken 里取出一个专门给 gpt-4o-mini 模型用的编码器，并把它存到变量 encoding 里。
#     system_prompt_token = len(encoding.encode(messages[0]["content"]))# encoding.encode(messages[0]["content"])指用这把分词尺子，把这条消息的文本内容切成 token ID 列表。
    
#     # 2. 计算当前总 Token 数（包含人设 + 所有历史对话）
#     total_tokens = system_prompt_token + sum(len(encoding.encode(msg["content"])) for msg in messages[1:])
    
#     # 3. 没超标，直接走人
#     if total_tokens <= max_tokens:
#         return

#     # 4. 超标了，开始从最新一条往前累积
#     safe_tokens = 0 # 安全的额度（最新对话的累加）
    
#     # 从最新一条开始往前遍历 (跳过索引 0)
#     for i in range(len(messages) - 1, 0, -1):
#         current_token = len(encoding.encode(messages[i]["content"]))
        
#         # 如果加上当前这条就超标了，说明当前这条是“第一个必须被丢弃的旧消息”
#         if safe_tokens + current_token + system_prompt_token > max_tokens:
#             # 直接切掉从索引 1 到 当前索引 i (包含 i) 的所有旧历史
#             del messages[1 : i + 1] 
#             print(f"\n[系统提示] 对话历史超限，已自动裁减，当前剩余 Token 约 {safe_tokens + system_prompt_token}")
#             return
            
#         # 没超标，这条消息是安全的，累加到 safe_tokens 里
#         safe_tokens += current_token

# # ============================================================
# # 工具定义区 —— Function Calling 的"能力说明书"
# # ============================================================

# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_current_time",
#             "description": "获取当前的日期和时间",
#             "parameters": {
#                 "type": "object",
#                 "properties": {},
#                 "required": []
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "save_memory",
#             "description": "当用户要求记住某个事实或偏好时调用，例如：姓名、城市、爱好等。",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "key": {
#                         "type": "string",
#                         "description": "记忆的分类，例如：name, city, hobby, favorite_food"
#                     },
#                     "value": {
#                         "type": "string",
#                         "description": "要记住的具体内容，例如：小明, 成都, 吃火锅"
#                     }
#                 },
#                 "required": ["key", "value"]
#             }
#         }
#     }
# ]

# # 实际执行的函数 —— 真正干活的"苦力"
# def get_current_time():
#     """返回当前时间的格式化字符串"""
#     return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


# def chat_once_stream(messages, prompts, key, user_input):
#     """执行单次对话（支持 Function Calling）"""

#     # 1. 每次对话开始前，加载最新记忆
#     memory = load_memory()

#     # 2. 将记忆拼接到 System Prompt (防止无限循环追加，每次都重新赋值)
#     if memory:
#         memory_text = "\n".join(f"- {k}: {v}" for k, v in memory.items())
#         messages[0]["content"] = f"{prompts[key]}\n\n### 📝 用户记忆 ###\n{memory_text}"
#     else:
#         messages[0]["content"] = prompts[key]

#     # 3. 正常发送用户输入
#     messages.append({"role": "user", "content": user_input})

#     try:
#         # ===== 第一次 API 调用：带 tools，让 AI 决定是否需要调用工具 =====
#         response = client.chat.completions.create(
#             model="deepseek-chat",
#             messages=messages,
#             tools=tools,
#             tool_choice="auto",
#             temperature=0.7,
#             max_tokens=1000,
#             stream=False,
#         )

#         msg = response.choices[0].message

#         # 如果模型决定调用工具
#         if msg.tool_calls:
#             # 【核心修复1】把 AI 的工具调用请求整体回塞一次
#             messages.append(msg)

#             for tool_call in msg.tool_calls:
#                 func_name = tool_call.function.name
#                 try:
#                     func_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
#                 except json.JSONDecodeError:
#                     func_args = {}

#                 if func_name == "get_current_time":
#                     result = get_current_time()

#                 elif func_name == "save_memory":
#                     save_memory(func_args.get("key"), func_args.get("value"))
#                     result = f"好的，我已经帮你把 {func_args.get('key')} 记为 {func_args.get('value')} 了。"

#                 else:
#                     result = f"未知工具: {func_name}"

#                 # 【核心修复2】把工具执行结果喂回去
#                 messages.append({
#                     "role": "tool",
#                     "tool_call_id": tool_call.id,
#                     "name": func_name,
#                     "content": result
#                 })

#             # ===== 第二次 API 调用：不带 tools，让 AI 基于工具结果生成最终回复 =====
#             response2 = client.chat.completions.create(
#                 model="deepseek-chat",
#                 messages=messages,
#                 temperature=0.7,
#                 max_tokens=1000,
#             )
#             final_msg = response2.choices[0].message
#             print(f"AI: {final_msg.content}")
#             if final_msg.content:
#                 messages.append({"role": "assistant", "content": final_msg.content})
#                 trim_history_by_token(messages)
#             return final_msg.content

#         # 没有调用工具，直接输出
#         print(f"AI: {msg.content}")
#         if msg.content:
#             messages.append({"role": "assistant", "content": msg.content})
#             trim_history_by_token(messages, max_tokens=3000)
#         return msg.content

#     except Exception as e:
#         messages.pop()
#         print(f"\n[系统错误] {e}")
#         return None


# if __name__ == "__main__":
#     print("=== 🤖 多轮对话 CLI 已启动 ===")
#     print("💡 提示：输入 /quit 或 /exit 退出程序\n")

#     # 选人设
#     for k, (_, name) in menu.items():
#         print(f"{k}. {name}")

#     choice = input("输入编号 (1/2/3): ").strip()
#     key = menu.get(choice, (None, None))[0]

#     if not key or key not in prompts:
#         print("无效的编号，程序退出。")
#     else:
#         messages = [{"role": "system", "content": prompts[key]}]
#         print(f"\n=== 🤖 {menu[choice][1]} 已激活 ===\n")

#         while True:
#             try:
#                 user_input = input("你: ").strip()

#                 if user_input.lower() in ("/quit", "/exit", "退出"):
#                     print("AI: 再见！期待下次交流 👋")
#                     break

#                 if not user_input:
#                     continue

#                 chat_once_stream(messages, prompts, key, user_input)

#             except KeyboardInterrupt:
#                 print("\n\n检测到强制中断，程序已退出。")
#                 break


import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 1. 加载环境变量并初始化客户端
load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com"
)

# 2. 定义工具（用 JSON Schema 告诉 LLM 它可以干什么）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

# 3. 真正干活的 Python 函数
def get_current_time():
    from datetime import datetime
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

# 工具注册表：把工具名字和真正的 Python 函数对应起来
TOOLS_MAP = {
    "get_current_time": get_current_time
}

# 4. 【核心】手写 ReAct 循环
def react_agent(user_prompt):
    print(f"\n🧑‍💻 用户输入: {user_prompt}\n")
    
    # 初始消息列表
    messages = [{"role": "user", "content": user_prompt}]
    
    # 最大循环步数（防止死循环烧钱）
    MAX_ITERATIONS = 5
    
    for i in range(MAX_ITERATIONS):
        print(f"--- ReAct 循环第 {i+1} 轮 ---")
        
        # 第一步：调用 LLM 思考（Think）并决定行动（Act）
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            temperature=0.7
        )
        
        assistant_msg = response.choices[0].message
        messages.append(assistant_msg)
        
        # 第二步：判断 LLM 是否还需要调用工具
        if not assistant_msg.tool_calls:
            # 如果没有调用工具，说明 LLM 已经思考完毕，输出了最终答案
            print(f"\n🤖 AI 最终回复: {assistant_msg.content}")
            return assistant_msg.content
        
        # 第三步：执行工具并观察结果（Observe）
        for tool_call in assistant_msg.tool_calls:
            func_name = tool_call.function.name
            print(f"🛠️ AI 决定调用工具: {func_name}")
            
            if func_name in TOOLS_MAP:
                result = TOOLS_MAP[func_name]()
                print(f"👀 工具执行结果 (Observation): {result}")
                
                # 第四步：把工具结果拼回消息列表，喂给 LLM 进行下一轮思考
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": result
                })
            else:
                print(f"⚠️ 未知的工具: {func_name}")

if __name__ == "__main__":
    # 测试：让它连续调用两次工具
    prompt = input()
    react_agent(prompt)