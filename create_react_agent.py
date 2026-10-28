import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 1. 导入之前写好的工具列表和全局人设
from langgraph_agent import TOOLS, _system_prompt

# 2. 初始化模型 (注意：不要在这里 .bind_tools)
load_dotenv()
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.7
)

# 3. 一行代码创建带记忆的 Agent
memory = MemorySaver()
agent_app = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SystemMessage(content=_system_prompt),
    checkpointer=memory
)

# 4. 运行对话 (注意：必须传 config 和 thread_id 才能启用记忆)
# 4. 启动交互式对话循环
config = {"configurable": {"thread_id": "user_1"}}

print("=== LangGraph 预构建版 Agent 已启动 ===")
print("提示：输入 /quit 退出程序")

while True:
    user_input = input("\n你: ").strip()
    
    if user_input.lower() == '/quit':
        print("AI: 再见！")
        break
    if not user_input:
        continue

    inputs = {"messages": [HumanMessage(content=user_input)]}
    
    try:
        # 每次 invoke 都会自动结合历史记忆
        response = agent_app.invoke(inputs, config=config)
        print(f"AI: {response['messages'][-1].content}")
    except Exception as e:
        print(f"系统出错: {e}")