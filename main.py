"""程序入口：选择人设并进入多轮对话主循环。"""
from config import prompts, menu
from langgraph_agent import LangGraphAgent


def main():
    print("=== 多轮对话 CLI 已启动 ===")
    print("提示：输入 /quit 或 /exit 退出程序\n")

    # 选择人设
    for k, (_, name) in menu.items():
        print(f"{k}. {name}")

    choice = input("输入编号 (1/2/3): ").strip()
    key = menu.get(choice, (None, None))[0]

    if not key or key not in prompts:
        print("无效的编号，程序退出。")
        return

    print(f"\n=== {menu[choice][1]} 已激活 ===\n")

    # 初始化 LangGraph Agent
    agent = LangGraphAgent(system_prompt=prompts[key])

    # 主对话循环
    while True:
        try:
            user_input = input("你: ").strip()
            if user_input.lower() in ("/quit", "/exit", "退出"):
                print("AI: 再见！期待下次交流")
                break
            if not user_input:
                continue
            agent.chat(user_input)
        except KeyboardInterrupt:
            print("\n\n检测到强制中断，程序已退出。")
            break


if __name__ == "__main__":
    main()