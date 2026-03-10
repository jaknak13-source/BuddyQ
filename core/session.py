"""
Session Module - Session Management and Logging

What it does:
    Creates a timestamped session folder on startup, writes transcript.json
    as a valid JSON array, and maintains latest.txt for simple LLM consumption.

What it reads:
    - config.py for all paths

What it writes:
    - sessions/YYYY-MM-DD_HH-MM/transcript.json
    - sessions/YYYY-MM-DD_HH-MM/latest.txt
    - sessions/current_session.txt  (path to active session)

How to replace/upgrade:
    - log_user() and log_assistant() are the only functions other modules call
    - get_session_dir() is used by main.py for display
    - transcript.json is a plain list, ready to feed directly to an LLM
"""

import json
import os
import threading
from datetime import datetime

import config

_lock = threading.Lock()
_session_dir: str = ""

# ============================================================================
# INTERNAL HELPERS  (defined before _init_session uses them)
# ============================================================================

def _transcript_path() -> str:
    return os.path.join(_session_dir, "transcript.json")


def _read_transcript() -> list:
    path = _transcript_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_transcript(data: list):
    with open(_transcript_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _append(role: str, text: str):
    with _lock:
        transcript = _read_transcript()
        transcript.append({
            "role":      role,
            "text":      text.strip(),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        _write_transcript(transcript)

        if role == "user":
            with open(os.path.join(_session_dir, "latest.txt"), "w", encoding="utf-8") as f:
                f.write(text.strip())


# ============================================================================
# INIT — called once on import, after helpers are defined
# ============================================================================

def _init_session():
    global _session_dir
    os.makedirs(config.SESSIONS_DIR, exist_ok=True)

    timestamp    = datetime.now().strftime("%Y-%m-%d_%H-%M")
    _session_dir = os.path.join(config.SESSIONS_DIR, timestamp)
    os.makedirs(_session_dir, exist_ok=True)

    # Write pointer file so other tools can find the active session
    with open(config.CURRENT_SESSION_FILE, "w") as f:
        f.write(_session_dir)

    # Initialise empty transcript
    _write_transcript([])


_init_session()


# ============================================================================
# PUBLIC API
# ============================================================================

def log_user(text: str):
    if text and text.strip():
        _append("user", text)


def log_assistant(text: str):
    if text and text.strip():
        _append("assistant", text)


def get_session_dir() -> str:
    return _session_dir


def get_transcript() -> list:
    """Return full transcript as a list, ready to pass to an LLM as history."""
    with _lock:
        return _read_transcript()
