"""
Base Tool Class - T:\BuddyQ\core\tools\base.py

What it does:
    Defines the interface every tool must implement.

How to add a new tool:
    1. Create a new file in tools\ inheriting from BaseTool
    2. Implement name, keywords, and run()
    3. Register it in tools\__init__.py
"""

from abc import ABC, abstractmethod


class BaseTool(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier e.g. 'web_search'"""

    @property
    @abstractmethod
    def description(self) -> str:
        """One sentence description for the LLM prompt."""

    @property
    @abstractmethod
    def keywords(self) -> list:
        """
        List of trigger phrases. If any appear in the user's speech
        the tool is considered a candidate.
        Example: ["search for", "look up", "what is", "who is"]
        """

    @abstractmethod
    def run(self, query: str) -> str:
        """
        Execute the tool with the given query.
        Returns a plain text result string.
        Never raises — catches all exceptions and returns an error string.
        """
