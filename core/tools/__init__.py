"""
Tools Registry - T:\\BuddyQ\\core\\tools\\__init__.py
"""

from .datetime_tool import DateTimeTool
from .calculator_tool import CalculatorTool
from .timer_tool import TimerTool
from .notes_tool import NotesTool
from .weather_tool import WeatherTool
from .system_tool import SystemTool
from .web_search import WebSearchTool
from .screen_tool import ScreenTool  # NEW: vision / screen tool


TOOLS = [
    DateTimeTool(),   # instant — no network
    TimerTool(),      # instant — no network (before calculator — "timer" must not match calc)
    CalculatorTool(), # instant — no network
    NotesTool(),      # instant — local file
    WeatherTool(),    # fast — dedicated API, no scraping
    SystemTool(),     # instant — OS calls
    ScreenTool(),     # screenshot + vision model — slower than local tools, faster than full web
    WebSearchTool(),  # slowest — always last
]


def get_all_tools():
    return TOOLS


def find_tool(user_text: str):
    """
    Returns the first matching tool for the user's speech, or None.
    Order matters: faster/more specific tools earlier, slower ones later.
    """
    text_lower = user_text.lower()
    for tool in TOOLS:
        for kw in tool.keywords:
            if kw in text_lower:
                return tool
    return None
