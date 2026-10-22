
"""LangGraph Agent 实现：State、工具、节点、边、图编译、长期记忆、Token 滑动窗口截断。"""
import os
import json
from datetime import datetime
from typing import Annotated, TypedDict

import operator
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

load_dotenv()

# ========== 长期记忆（持久化到 memory.json）==========
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")


def load_memory():
    """读取长期记忆；文件不存在或损坏时返回空字典。"""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


# ========== 工具定义（计算器 / 天气 / 读文件 / 时间 / 记忆）==========

@tool
def get_current_time() -> str:
    """获取当前的日期和时间。"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """执行数学计算，如加减乘除。expression 示例: '23 * 4 + 5'。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return "计算出错: " + str(e)


@tool
def get_weather(city: str) -> str:
    """查询城市当前天气（免 Key，基于 Open-Meteo）。city 示例: '北京', 'Shanghai'。"""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
            timeout=8,
        ).json()
        if "results" not in geo:
            return "未找到城市: " + str(city)
        loc = geo["results"][0]
        cur = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                    "current_weather": True},
            timeout=8,
        ).json()["current_weather"]
        return "%s 当前天气：%s℃，风速 %s km/h" % (loc["name"], cur["temperature"], cur["windspeed"])
    except Exception as e:
        return "天气查询失败: " + str(e)


@tool
def read_file(file_path: str) -> str:
    """读取本地纯文本文件内容，支持绝对/相对路径。"""
    try:
        path = os.path.abspath(file_path)
        if not os.path.isfile(path):
            return "未找到文件：" + file_path
        for enc in ("utf-8", "gbk"):
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        return "无法解析（可能不是纯文本）：" + file_path
    except Exception as e:
        return "读取失败：" + str(e)


@tool
def save_memory(key: str, value: str) -> str:
    """当用户要求记住某个事实或偏好时调用。key 为分类(如 name/hobby)，value 为内容。"""
    memory = load_memory()
    memory[key] = value
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return "已记住: %s" % key


TOOLS = [get_current_time, calculator, get_weather, read_file, save_memory]


# ========== State 定义 ==========

class AgentState(TypedDict):
    """LangGraph 的公共记事本，所有节点共享。"""
    messages: Annotated[list, operator.add]


# ========== 模型 & 人设 ==========

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,
    max_tokens=10000,
).bind_tools(TOOLS)

_system_prompt = "你是一个有用的助手。"


def set_system_prompt(prompt):
    """设置当前人设的系统提示词。"""
    global _system_prompt
    _system_prompt = prompt


# ========== Token 滑动窗口截断（短期记忆，按 token 计数）==========

MAX_CONTEXT_TOKENS = 3000  # 历史消息允许占用的 token 预算（不含 System 与本轮回复）

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None  # 没装 tiktoken 时退化为按字符估算，仍可运行


def _msg_tokens(msg):
    """估算单条消息的 token 数；AIMessage 的 tool_calls 也计入。"""
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    tc = getattr(msg, "tool_calls", None)
    if tc:
        text += str(tc)
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, len(text) // 3)  # 兜底近似：约 3 字符 ~= 1 token


def trim_messages_by_token(messages, budget):
    """从最新往旧累加，保留总 token 不超过 budget 的最近消息，至少保留最后一条。"""
    kept, total = [], 0
    for m in reversed(messages):
        t = _msg_tokens(m)
        if kept and total + t > budget:
            break
        kept.append(m)
        total += t
    kept.reverse()
    # 截断后开头若出现孤立 ToolMessage（其对应的 AIMessage.tool_calls 已被切掉），
    # 直接丢弃，避免部分模型 API 因 tool 消息缺配对而报错。
    while kept and isinstance(kept[0], ToolMessage):
        kept.pop(0)
    return kept


# ========== 节点函数 ==========

def call_model(state):
    """Agent 节点：组装消息发给 LLM，LLM 决定是否调用工具。"""
    # 去掉旧的 SystemMessage，统一在开头重建，保证每次注入的人设+长期记忆都是最新的
    messages = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    # 短期记忆截断：按 token 预算保留最近的上下文（self.history 仍完整保存）
    messages = trim_messages_by_token(messages, MAX_CONTEXT_TOKENS)

    memory = load_memory()
    sys_content = _system_prompt
    if memory:
        memory_text = "\n".join("- %s: %s" % (k, v) for k, v in memory.items())
        sys_content += "\n\n### 用户记忆 ###\n" + memory_text

    response = llm.invoke([SystemMessage(content=sys_content)] + messages)
    return {"messages": [response]}


def call_tools(state):
    """Tools 节点：执行 LLM 请求的工具调用，返回 ToolMessage。"""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        tool_obj = next((t for t in TOOLS if t.name == tc["name"]), None)
        result = tool_obj.invoke(tc["args"]) if tool_obj else "未知工具: " + str(tc["name"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}


def should_continue(state):
    """条件边：有 tool_calls 就去 tools 节点，否则结束。"""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


# ========== 构建图 ==========

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")
app = workflow.compile()

RECURSION_LIMIT = 12  # 防死循环：限制一轮对话最多走多少步


class LangGraphAgent:
    """封装图的调用逻辑，管理多轮对话历史。"""

    def __init__(self, system_prompt):
        set_system_prompt(system_prompt)
        self.history = []

    def chat(self, user_input):
        """发送用户输入，运行图，打印并返回 AI 回复。"""
        current_state = {"messages": self.history + [HumanMessage(content=user_input)]}
        final_state = app.invoke(current_state, config={"recursion_limit": RECURSION_LIMIT})
        new_messages = final_state["messages"][len(self.history) + 1:]
        self.history.extend(new_messages)

        answer = ""
        for msg in new_messages:
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
        if answer:
            print("AI: " + answer)
        return answer