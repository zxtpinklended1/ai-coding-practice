import requests
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询城市当前天气（免 Key，基于 Open-Meteo）。city 示例: '北京', 'Shanghai'。"""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
            timeout=8,
        ).json()
        if "results" not in geo:
            return "未找到城市: " + str(city)
        loc = geo["results"][0]
        cur = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                    "current_weather": True},
            timeout=8,
        ).json()["current_weather"]
        return "%s 当前天气：%s℃，风速 %s km/h" % (loc["name"], cur["temperature"], cur["windspeed"])
    except Exception as e:
        return "天气查询失败: " + str(e)