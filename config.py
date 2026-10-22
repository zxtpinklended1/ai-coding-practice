
"""全局配置：API 客户端、人设配置、人设菜单。"""
import os
import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
)


def load_prompts():
    """读取与本文件同目录的 prompts.yaml，返回人设字典。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "prompts.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


prompts = load_prompts()

# 人设菜单：编号 -> (prompts.yaml 中的键, 显示名称)
menu = {
    "1": ("socrates", "苏格拉底哲学家"),
    "2": ("architect", "机器人"),
    "3": ("guide", "二次元新手村向导"),
}