"""
DateTime Tool - T:\BuddyQ\core\tools\datetime_tool.py

What it does:
    Returns current date, time, day of week. No internet needed.

How to replace/upgrade:
    - Add timezone support via config if needed
"""

from datetime import datetime
from .base import BaseTool


class DateTimeTool(BaseTool):

    @property
    def name(self):
        return "datetime"

    @property
    def description(self):
        return "Returns the current date and time."

    @property
    def keywords(self):
        return [
            "what time", "what's the time", "current time",
            "what date", "what's the date", "today's date",
            "what day", "day is it", "what year",
        ]

    def run(self, query: str) -> str:
        now = datetime.now()
        return (
            f"Current date and time: "
            f"{now.strftime('%A, %B %d %Y')} at {now.strftime('%H:%M')}."
        )
