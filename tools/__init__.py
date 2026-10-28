# tools/__init__.py
from .calculator import calculator
from .weather import get_weather
from .file_reader import read_file
from .time_tool import get_current_time
from .memory_tool import save_memory, load_memory

TOOLS = [get_current_time, calculator, get_weather, read_file, save_memory]