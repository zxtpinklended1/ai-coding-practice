import os,sys,requests,json
from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv("API_KEY")

if not api_key:
    print("API_KEY 未找到")
    sys.exit(1)

# 初始化上下文
messages=[
    {"role":"system","content":"你现在是一个ai coding资深导师，请你回答简洁高效。"}
]

# 请求头
headers={
    "Authorization":f"Bearer {api_key}",
    "Content-Type":"application/json"
}

while True:
    user_input=input("\n我想知道：").strip()
    if user_input.lower()=="quit":
        print("GGGGGOODBYE!")
        break
    if not user_input:
        continue

    messages.append({"role":"user","content":user_input})

    # 请求体
    payload={
    "model":"deepseek-chat",
    "messages":messages,
    "max_tokens":1024,
    "temperature":0.7
    }
    try:
        response=requests.post(
            "https://api.deepseek.com/chat/completions", 
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()

        result=response.json()
        ai_reply=result["choices"][0]["message"]["content"]
        messages.append({"role":"assistant","content":ai_reply})
        print(f"\n🤖 AI：{ai_reply}")

    except requests.exceptions.Timeout:
        # 只有超时才会进这里
        print("服务器响应太慢")
        messages.pop()

    except requests.exceptions.ConnectionError:
        # 只有网络断了才会进这里
        print("网络连接失败")
        messages.pop()

    except requests.exceptions.HTTPError as e:
        # 只有 4xx/5xx 才会进这里
        print(f"服务器拒绝了：{e.response.status_code}")
        messages.pop()

    except KeyError as e:
        # 只有 JSON 缺少字段才会进这里
        print(f"返回数据格式不对：{e}")
        messages.pop()

    except Exception as e:
        # 上面都没匹配到，才进这个"兜底"
        print(f"未知错误：{e}")
        messages.pop()