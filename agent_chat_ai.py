import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. 加载环境变量与初始化客户端
load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com"
)

# 2. 初始化对话上下文
messages = [
    {"role": "system", "content": "你是一个专业的 Python 编程助手。"}
]

MAX_TURNS = 20  # 最多保留的对话轮数

def trim_history():
    """
    智能裁剪历史：
    1. 始终保留 System Prompt (index 0)
    2. 删除中间过旧的历史，但确保剩余的第一条是 user 消息（保持 user-assistant 成对）
    """
    keep = 1 + MAX_TURNS * 2
    if len(messages) > keep:
        # 计算要删除到的位置
        del_index = len(messages) - (keep - 1)
        
        # 核心优化：如果删除后第一条是 assistant，就多删一条，让 user 开头
        # 这样可以避免模型收到 "assistant: xxx" 开头却无前文的情况
        if messages[del_index]["role"] == "assistant":
            del_index -= 1
            
        del messages[1:del_index]

def chat_once_stream(user_input):
    """执行单次流式对话"""
    messages.append({"role": "user", "content": user_input})
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            stream=True,
        )
        
        print("AI: ", end="", flush=True)
        full_content = ""
        
        for chunk in response:
            # 安全检查：防止流结束时 chunk.choices 为空导致报错
            if chunk.choices and chunk.choices[0].delta.content:
                piece = chunk.choices[0].delta.content
                print(piece, end="", flush=True)
                full_content += piece
        
        print()  # 换行
        
        # 健壮性优化：只有当 AI 真正输出了内容时，才写入历史
        if full_content.strip():
            messages.append({"role": "assistant", "content": full_content})
            trim_history()
            
        return full_content

    except Exception as e:
        # 异常回滚：如果请求失败，把刚才加入的 user 消息删掉，防止历史错乱
        messages.pop()
        print(f"\n[系统错误] {e}")
        return None

if __name__ == "__main__":
    print("=== 🤖 多轮对话 CLI 已启动 ===")
    print("💡 提示：输入 /quit 或 /exit 退出程序\n")
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            # 退出指令判断（确定性控制流）
            if user_input.lower() in ("/quit", "/exit", "退出"):
                print("AI: 再见！期待下次交流 👋")
                break
                
            # 忽略空输入
            if not user_input:
                continue
                
            chat_once_stream(user_input)
            
        except KeyboardInterrupt:
            print("\n\n检测到强制中断，程序已退出。")
            break