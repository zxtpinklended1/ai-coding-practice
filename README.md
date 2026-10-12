# 🐍 Python AI Agent 雏形项目

基于 DeepSeek API 和 OpenAI SDK 构建的命令行多轮对话机器人。支持多轮对话、长期记忆、Function Calling 以及基于 Token 的上下文窗口管理。

## ✨ 核心功能

- **多轮对话**：维护 `messages` 历史列表，支持连续追问。
- **长期记忆**：通过 `memory.json` 保存用户偏好，重启后仍可读取。
- **Function Calling**：支持调用工具，例如获取当前时间、保存记忆。
- **Token 滑动窗口截断**：使用 `cl100k_base` 编码估算 Token，自动裁剪历史，防止上下文溢出。
- **安全配置**：通过 `.env` 管理 API Key，避免密钥泄露。
- **异常容错**：请求失败时回滚消息状态，避免对话逻辑错乱。

## 🚀 快速开始

### 1. 环境准备

确保已安装 Python 3.10+。

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
API_KEY=your_deepseek_api_key_here
```

### 3. 运行项目

```bash
python main.py
```

运行后先选择人设，再开始对话。输入 `/quit` 或 `/exit` 退出。

## 📁 项目结构

```text
.
├── main.py              # 程序入口，负责交互循环
├── config.py            # 配置模块：客户端初始化、人设加载、人设菜单
├── agent.py             # Agent 核心：记忆管理、Token 截断、工具调用、对话流程
├── prompts.yaml         # 人设提示词配置
├── memory.json          # 长期记忆文件，由程序自动生成
├── requirements.txt     # 依赖清单
└── .env                 # API Key 配置文件，不应提交到 Git
```

## 📝 说明

- `memory.json` 会在对话过程中自动生成。
- `.env` 包含敏感信息，不应提交到 Git。
- 当前版本使用 `tiktoken` 的 `cl100k_base` 编码进行 Token 估算，用于 DeepSeek 对话的上下文长度控制。
