import os
from langchain_core.tools import tool
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