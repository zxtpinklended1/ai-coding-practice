from langchain_core.tools import tool
@tool
def calculator(expression: str) -> str:
    """执行数学计算，如加减乘除。expression 示例: '23 * 4 + 5'。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return "计算出错: " + str(e)