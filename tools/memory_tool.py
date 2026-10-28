# tools/memory_tool.py

import os
import json
from langchain_core.tools import tool

# 注意这里多了一个 dirname，是为了退回上一级目录（即项目根目录）
MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "memory.json"
)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}

@tool
def save_memory(key: str, value: str) -> str:
    """当用户要求记住某个事实或偏好时调用。key 为分类(如 name/hobby)，value 为内容。"""
    memory = load_memory()
    memory[key] = value
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return "已记住: %s" % key