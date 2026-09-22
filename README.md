# 🐍 Python AI Agent 雏形项目

基于 DeepSeek API 和 OpenAI SDK 构建的命令行多轮对话机器人。实现了流式输出、上下文记忆管理及异常容错机制。

## ✨ 核心功能

- **多轮对话记忆**：自动维护 `messages` 历史列表，支持连续追问。
- **流式打字机效果**：实时接收并打印 Token，降低等待焦虑。
- **智能上下文管理**：内置 `trim_history` 算法，自动裁剪过期对话，防止 Token 溢出。
- **安全配置**：通过 `.env` 文件管理敏感 API Key，杜绝密钥泄露。
- **异常兜底**：网络波动或请求失败时自动回滚对话状态，防止逻辑错乱。

## 🚀 快速开始

### 1. 环境准备
确保已安装 Python 3.10+。

```bash
pip install openai python-dotenv