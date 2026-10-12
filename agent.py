
"""Agent 核心逻辑：长期记忆、Token 滑动窗口、Function Calling 与单次对话。"""
import json
import os
from datetime import datetime

import tiktoken

from config import client, prompts


# ---------- 长期记忆（持久化到 memory.json） ----------
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")


def load_memory():
    """读取记忆文件；文件不存在或内容损坏时返回空字典。"""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def save_memory(key, value):
    """写入或更新单条长期记忆。"""
    memory = load_memory()
    memory[key] = value
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


# ---------- 短期记忆：Token 滑动窗口截断 ----------
# DeepSeek 没有官方 tiktoken 编码器，这里用通用的 cl100k_base 做近似估算。
_ENCODING = tiktoken.get_encoding("cl100k_base")


def trim_history_by_token(messages, max_tokens=3000):
    """保证 System Prompt + 历史对话不超过 max_tokens。

    超标时从最新一条往前累加，删除最早的一批旧消息，始终保留 index 0 的 System Prompt。
    """
    system_tokens = len(_ENCODING.encode(messages[0]["content"]))
    total_tokens = system_tokens + sum(
        len(_ENCODING.encode(m["content"])) for m in messages[1:]
    )
    if total_tokens <= max_tokens:
        return

    safe_tokens = 0
    for i in range(len(messages) - 1, 0, -1):
        current = len(_ENCODING.encode(messages[i]["content"]))
        # 加上当前这条就超标，说明从 index 1 到 i 都是要丢弃的旧历史
        if safe_tokens + current + system_tokens > max_tokens:
            del messages[1 : i + 1]
            print(f"\n[系统提示] 对话历史超限，已自动裁减，当前约 {safe_tokens + system_tokens} token")
            return
        safe_tokens += current


# ---------- Function Calling：工具定义与执行 ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "当用户要求记住某个事实或偏好时调用，例如姓名、城市、爱好等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "记忆分类，如 name, city, hobby"},
                    "value": {"type": "string", "description": "要记住的内容，如 小明, 成都, 吃火锅"},
                },
                "required": ["key", "value"],
            },
        },
    },
]


def get_current_time():
    """返回格式化的当前时间字符串。"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


def _run_tool(name, args):
    """按工具名分发执行，返回结果字符串。"""
    if name == "get_current_time":
        return get_current_time()
    if name == "save_memory":
        save_memory(args.get("key"), args.get("value"))
        return f"好的，我已经帮你把 {args.get('key')} 记为 {args.get('value')} 了。"
    return f"未知工具: {name}"


# ---------- 单次对话 ----------
def chat_once(messages, key, user_input):
    """执行一次对话：注入长期记忆 -> 调用模型（必要时执行工具）-> 截断历史。"""
    # 把长期记忆拼进 System Prompt；每次重新赋值，避免重复叠加
    memory = load_memory()
    if memory:
        memory_text = "\n".join(f"- {k}: {v}" for k, v in memory.items())
        messages[0]["content"] = f"{prompts[key]}\n\n### 用户记忆 ###\n{memory_text}"
    else:
        messages[0]["content"] = prompts[key]

    messages.append({"role": "user", "content": user_input})

    try:
        # 第一次调用：带 tools，由模型决定是否调用工具
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

        if msg.tool_calls:
            # 回塞模型的调用请求
            messages.append(msg)
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                result = _run_tool(name, args)
                # 把工具执行结果喂回模型
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result,
                })
            # 第二次调用：不带 tools，基于工具结果生成最终回复
            response2 = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )
            final = response2.choices[0].message
            print(f"AI: {final.content}")
            if final.content:
                messages.append({"role": "assistant", "content": final.content})
                trim_history_by_token(messages)
            return final.content

        print(f"AI: {msg.content}")
        if msg.content:
            messages.append({"role": "assistant", "content": msg.content})
            trim_history_by_token(messages)
        return msg.content

    except Exception as e:
        messages.pop()
        print(f"\n[系统错误] {e}")
        return None