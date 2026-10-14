"""Agent 核心逻辑：长期记忆、Token 滑动窗口、Function Calling 与 ReAct 循环。"""
import json
import os
from datetime import datetime
import requests
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


def trim_history_by_token(messages, max_tokens=10000):
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
            "name": "calculator",
            "description": "用于执行简单的数学计算，如加减乘除。",
            "parameters": {
                "type": "object", 
                "properties": {
                    "expression": {"type": "string", "description": "要计算的数学表达式，例如 '23 * 4 + 5'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询全球任意城市的当前天气情况。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "要查询的城市名称，如 '北京', 'Shanghai'"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取电脑本地指定路径的文本文件。支持 .txt, .py, .md, .csv, .json 等纯文本格式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要读取的文件的完整路径（支持绝对路径如 'C:/Users/test.txt' 或相对路径 'notes.txt'）。"
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "当用户要求记住某个事实或偏好时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "记忆分类，如 name, hobby"},
                    "value": {"type": "string", "description": "要记住的内容"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function", 
        "function": { 
            "name": "list_files",
            "description": "列出指定目录下的所有文件和文件夹。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string", 
                        "description": "要列出的目录路径，默认为当前目录。"
                    },
                },
                "required": [],
            },
        }, 
    }
]

# 1. 获取当前时间
def get_current_time(args=None):
    """返回格式化的当前时间字符串。"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

# 2. 简单计算器
def calculator(expression):
    """执行数学表达式计算。"""
    try:
        # eval 用于计算字符串形式的数学表达式（学习阶段用，生产环境有安全风险）
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算出错: {e}"

# 3. 查天气 (使用完全免费、免Key的 Open-Meteo 地理编码+天气接口)
def get_weather(city):
    """通过城市名查询当前天气。"""
    try:
        # 第一步：地理编码，将城市名转为经纬度
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_resp = requests.get(geo_url, params={"name": city, "count": 1, "language": "zh", "format": "json"}, timeout=8)
        geo_data = geo_resp.json()
        if "results" not in geo_data:
            return f"未找到城市: {city}"
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        city_name = geo_data["results"][0]["name"]

        # 第二步：根据经纬度获取当前天气
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current_weather": True, "language": "zh"
        }
        weather_resp = requests.get(weather_url, params=params, timeout=8)
        weather_data = weather_resp.json()
        current = weather_data["current_weather"]
        temp = current["temperature"]
        wind_speed = current["windspeed"]
        return f"{city_name} 当前天气：{temp}℃，风速 {wind_speed} km/h"
    except Exception as e:
        return f"天气查询失败: {e}"

# 4. 读取本地文件
import os

def read_file(file_path):
    """读取指定路径的文本文件。自动尝试兼容 utf-8 和 gbk 编码。"""
    try:
        # 处理反斜杠，兼容 Windows 路径
        file_path = file_path.replace("\\", "/")
        
        # 如果传入的不是绝对路径，尝试拼接当前目录
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        if os.path.exists(file_path) and os.path.isfile(file_path):
            # 尝试多种常见编码读取
            encodings = ['utf-8', 'gbk', 'latin1']
            for enc in encodings:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            return f"无法解析该文件：文件存在，但可能不是纯文本格式（如图片或可执行文件）：{file_path}"
        
        return f"未找到文件：{file_path}"
    except Exception as e:
        return f"读取失败：{e}"

# 5. 保存记忆
def _save_memory_tool(key, value):
    """保存记忆的工具包装函数。"""
    save_memory(key, value)
    return "Memory saved successfully"

# 6. 新增：通用的文件列出函数
def list_files(dir_path="."):
    """列出指定目录下的所有文件和文件夹。默认列出当前工作目录。"""
    try:
        dir_path = dir_path.replace("\\", "/")
        if not os.path.isabs(dir_path):
            dir_path = os.path.join(os.getcwd(), dir_path)

        if not os.path.exists(dir_path):
            return f"目录不存在：{dir_path}"

        items = os.listdir(dir_path)
        # 区分文件和文件夹，方便 Agent 判断
        dirs = [i for i in items if os.path.isdir(os.path.join(dir_path, i))]
        files = [i for i in items if os.path.isfile(os.path.join(dir_path, i))]
        return f"📁 文件夹: {dirs}\n📄 文件: {files}"
    except Exception as e:
        return f"列出失败：{e}"

# 工具注册表
TOOLS_MAP = {
    "get_current_time": get_current_time,
    "calculator": calculator,
    "get_weather": get_weather,
    "read_file": read_file,
    "save_memory": _save_memory_tool,
    "list_files": list_files,
}

def _run_tool(name, args):
    if name in TOOLS_MAP:
        result = TOOLS_MAP[name](**args)
        return str(result)
    return f"Unknown tool: {name}"


# ---------- ReAct 循环对话 ----------
def chat_once(messages, key, user_input):
    """执行一次对话：注入长期记忆 -> ReAct 循环（思考->行动->观察）-> 截断历史。

    ReAct 循环本质：
    1. Think（思考）：调用 LLM，让它决定要不要调工具
    2. Act（行动）：如果调了工具，执行工具函数
    3. Observe（观察）：把工具结果塞回 messages
    4. Loop（循环）：带着观察结果重新思考，直到 LLM 决定不调工具为止
    """
    # 把长期记忆拼进 System Prompt
    memory = load_memory()
    if memory:
        memory_text = "\n".join(f"- {k}: {v}" for k, v in memory.items())
        messages[0]["content"] = f"{prompts[key]}\n\n### 用户记忆 ###\n{memory_text}"
    else:
        messages[0]["content"] = prompts[key]

    messages.append({"role": "user", "content": user_input})

    # 最大循环步数（防止死循环烧钱）
    MAX_ITERATIONS = 5

    try:
        for i in range(MAX_ITERATIONS):
            print(f"--- ReAct 循环第 {i + 1} 轮 ---")

            # 第一步：调用 LLM 思考（Think）并决定行动（Act）
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=10000,
                stream=False,
            )

            assistant_msg = response.choices[0].message
            # ✅ 将复杂对象转换为干净的 Python 字典
            assistant_msg_dict = {
                "role": assistant_msg.role,
                "content": assistant_msg.content,
                "tool_calls": assistant_msg.tool_calls,
            }
            messages.append(assistant_msg_dict)


            # 第二步：判断 LLM 是否还需要调用工具
            if not assistant_msg.tool_calls:
                # 没有调用工具，说明 LLM 已经思考完毕，输出最终答案
                print(f"AI: {assistant_msg.content}")
                if assistant_msg.content:
                    messages.append({"role": "assistant", "content": assistant_msg.content})
                    trim_history_by_token(messages)
                return assistant_msg.content

            # 第三步：执行工具并观察结果（Observe）
            for tool_call in assistant_msg.tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                result = _run_tool(func_name, args)
                print(f"[工具执行结果] {result}")

                # 把工具结果拼回消息列表（纯 observation，不说客套话）
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": result,
                })

        # 超过最大迭代次数，强制退出
        print("[系统提示] 达到最大迭代次数，对话结束。")
        return None

    except Exception as e:
        print(f"\n[系统错误] {e}")
        return None