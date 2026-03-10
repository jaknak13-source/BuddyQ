"""
Screen / Vision Tool - core/tools/screen_tool.py

Lets Buddy "see" the current screen (screenshot) and answer questions like:
- Summarise the page on screen.
- Is this website likely legit?
- Explain the code I'm looking at.
- What am I looking at right now?

Requires:
    pip install mss pillow
and a multimodal model (e.g. Qwen3-VL) running behind LM Studio or another
OpenAI-compatible server with vision support.
"""

import base64
import io
import threading
from typing import List, Dict, Optional

import mss
from PIL import Image
from openai import OpenAI

from .base import BaseTool
import config

# Use the same server as the main assistant, but a vision-capable model name.
# Set VISION_MODEL_NAME in config.py; falls back to MODEL_NAME if missing.
VISION_MODEL_NAME = getattr(
    config,
    "VISION_MODEL_NAME",
    getattr(config, "MODEL_NAME", "qwen3-vl-4b-instruct"),
)

# Match main.py: disable reasoning / thinking tokens
_NO_THINKING = {
    "chat_template_kwargs": {"thinking": False},
    "thinking": False,
    "enable_thinking": False,
}

# Single global client is fine
_client_lock = threading.Lock()
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(
                base_url=config.LLM_API_URL,
                api_key="no-key-needed",
            )
        return _client


def _grab_screenshot_base64() -> str:
    """
    Capture a full-screen screenshot and return it as a base64 PNG string.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary display
        img = sct.grab(monitor)

        # Convert raw BGRA to RGB PIL Image
        pil = Image.frombytes("RGB", img.size, img.rgb)

        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        data = buf.getvalue()
        return base64.b64encode(data).decode("utf-8")


class ScreenTool(BaseTool):
    """
    Vision tool that lets Buddy reason about the current screen contents.
    """

    def __init__(self):
        self._history: List[Dict] = []

    def set_history(self, h: List[Dict]):
        self._history = h or []

    @property
    def name(self) -> str:
        return "screen"

    @property
    def description(self) -> str:
        return (
            "Looks at a screenshot of the user's current screen and explains it. "
            "Use for: summarising the page on screen, describing what is visible, "
            "explaining code or diagrams, and giving a quick judgement on whether a "
            "website looks suspicious."
        )

    @property
    def keywords(self) -> list:
        return [
            "on my screen",
            "this page",
            "this website",
            "this site",
            "what am I looking at",
            "summarise the page",
            "summarize the page",
            "explain this code",
            "is this legit",
            "does this website look safe",
        ]

    def run(self, query: str) -> str:
        """
        Capture a screenshot and ask the vision model to answer the user's question
        about what is visible on screen.
        """
        # 1. Capture screen
        try:
            b64_image = _grab_screenshot_base64()
        except Exception as e:
            return f"I tried to capture the screen but it failed with an error: {e}."

        client = _get_client()

        # 2. Build LM Studio / OpenAI-compatible multimodal request
        # Uses content=[input_text, input_image] with data URL; no [img-1] or image_data.
        base_instruction = (
            "You see a screenshot of the user's current screen. "
            "First, briefly describe what is clearly visible. "
            "Then answer the user's question about it. "
            "If you are unsure about something, say that you are not sure instead of guessing. "
            "If the user asks whether a website looks legitimate or safe, comment only on obvious "
            "visual red flags, such as mismatched logos, spelling mistakes, or strange URLs, and "
            "remind them that you cannot guarantee safety.\n\n"
        )

        full_text = base_instruction + f"User question: {query}"

        try:
            resp = client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": full_text,
                            },
                            {
                                "type": "input_image",
                                "image_url": {
                                    "url": "data:image/png;base64," + b64_image,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=320,
                temperature=0.3,
                stream=False,
                extra_body=_NO_THINKING,
            )
            answer = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"I could not analyse the screen because the vision model failed with: {e}."

        if not answer:
            return "I saw the screen but could not extract a clear description or answer."

        return answer
