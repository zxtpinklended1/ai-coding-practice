import os
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import yaml
import json

# 1. 加载环境变量与初始化客户端
load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com"
)

# 定义人设相关函数
def load_prompts():
    # 【修改】动态获取当前 .py 文件所在的绝对目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "prompts.yaml")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 【修改】直接在这里调用，现在它会自动寻找同目录下的 prompts.yaml
prompts = load_prompts()

menu = {
    "1": ("socrates", "苏格拉底哲学家"),
    "2": ("architect", "机器人"),
    "3": ("guide", "二次元新手村向导"),
}

# 加入长期记忆
# 【动态获取路径】找到当前 .py 文件所在的目录，并拼接 memory.json
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")

def load_memory():
    """加载记忆文件"""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}

def save_memory(key, value):
    """将单条记忆保存到文件"""
    memory = load_memory()
    memory[key] = value
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


MAX_TURNS = 20  # 最多保留的对话轮数

def trim_history(messages):
    """
    智能裁剪历史：
    1. 始终保留 System Prompt (index 0)
    2. 删除中间过旧的历史，但确保剩余的第一条是 user 消息（保持 user-assistant 成对）
    """
    keep = 1 + MAX_TURNS * 2
    if len(messages) > keep:
        del_index = len(messages) - (keep - 1)
        if messages[del_index]["role"] == "assistant":
            del_index -= 1
        del messages[1:del_index]


# ============================================================
# 工具定义区 —— Function Calling 的"能力说明书"
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "当用户要求记住某个事实或偏好时调用，例如：姓名、城市、爱好等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "记忆的分类，例如：name, city, hobby, favorite_food"
                    },
                    "value": {
                        "type": "string",
                        "description": "要记住的具体内容，例如：小明, 成都, 吃火锅"
                    }
                },
                "required": ["key", "value"]
            }
        }
    }
]

# 实际执行的函数 —— 真正干活的"苦力"
def get_current_time():
    """返回当前时间的格式化字符串"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


def chat_once_stream(messages, prompts, key, user_input):
    """执行单次对话（支持 Function Calling）"""

    # 1. 每次对话开始前，加载最新记忆
    memory = load_memory()

    # 2. 将记忆拼接到 System Prompt (防止无限循环追加，每次都重新赋值)
    if memory:
        memory_text = "\n".join(f"- {k}: {v}" for k, v in memory.items())
        messages[0]["content"] = f"{prompts[key]}\n\n### 📝 用户记忆 ###\n{memory_text}"
    else:
        messages[0]["content"] = prompts[key]

    # 3. 正常发送用户输入
    messages.append({"role": "user", "content": user_input})

    try:
        # ===== 第一次 API 调用：带 tools，让 AI 决定是否需要调用工具 =====
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=1000,
            stream=False,
        )

        msg = response.choices[0].message

        # 如果模型决定调用工具
        if msg.tool_calls:
            # 【核心修复1】把 AI 的工具调用请求整体回塞一次
            messages.append(msg)

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    func_args = {}

                if func_name == "get_current_time":
                    result = get_current_time()

                elif func_name == "save_memory":
                    save_memory(func_args.get("key"), func_args.get("value"))
                    result = f"好的，我已经帮你把 {func_args.get('key')} 记为 {func_args.get('value')} 了。"

                else:
                    result = f"未知工具: {func_name}"

                # 【核心修复2】把工具执行结果喂回去
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": result
                })

            # ===== 第二次 API 调用：不带 tools，让 AI 基于工具结果生成最终回复 =====
            response2 = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )
            final_msg = response2.choices[0].message
            print(f"AI: {final_msg.content}")
            if final_msg.content:
                messages.append({"role": "assistant", "content": final_msg.content})
                trim_history(messages)
            return final_msg.content

        # 没有调用工具，直接输出
        print(f"AI: {msg.content}")
        if msg.content:
            messages.append({"role": "assistant", "content": msg.content})
            trim_history(messages)
        return msg.content

    except Exception as e:
        messages.pop()
        print(f"\n[系统错误] {e}")
        return None


if __name__ == "__main__":
    print("=== 🤖 多轮对话 CLI 已启动 ===")
    print("💡 提示：输入 /quit 或 /exit 退出程序\n")

    # 选人设
    for k, (_, name) in menu.items():
        print(f"{k}. {name}")

    choice = input("输入编号 (1/2/3): ").strip()
    key = menu.get(choice, (None, None))[0]

    if not key or key not in prompts:
        print("无效的编号，程序退出。")
    else:
        messages = [{"role": "system", "content": prompts[key]}]
        print(f"\n=== 🤖 {menu[choice][1]} 已激活 ===\n")

        while True:
            try:
                user_input = input("你: ").strip()

                if user_input.lower() in ("/quit", "/exit", "退出"):
                    print("AI: 再见！期待下次交流 👋")
                    break

                if not user_input:
                    continue

                chat_once_stream(messages, prompts, key, user_input)

            except KeyboardInterrupt:
                print("\n\n检测到强制中断，程序已退出。")
                break