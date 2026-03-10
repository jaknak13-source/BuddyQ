"""
Timer & Alarm Tool - T:\\BuddyQ\\core\\tools\\timer_tool.py

Handles:
  - "set a timer for 20 minutes"
  - "set an alarm for 7:30"
  - "cancel the timer"
  - "how long is left on the timer"
  - "stop the timer"

When a timer fires it puts an alert message into the TTS queue so Buddy
speaks the alert aloud. The TTS queue is wired in via set_tts_queue().
Multiple named timers are supported simultaneously.
"""

import re
import time
import threading
from datetime import datetime, timedelta
from .base import BaseTool

# Wired in by main.py at startup
_tts_queue   = None
_speak_fn    = None   # optional callback: speak_fn(text) → puts text in TTS queue

def set_tts_queue(q):
    global _tts_queue
    _tts_queue = q

def _speak(text: str):
    if _tts_queue:
        _tts_queue.put(text)
    else:
        print(f"[TIMER ALERT] {text}", flush=True)


# ── Timer state ───────────────────────────────────────────────────────────────

_timers: dict[str, threading.Timer] = {}   # name → threading.Timer
_timer_end: dict[str, float]        = {}   # name → end timestamp
_lock = threading.Lock()


def _fire(name: str):
    """Called by threading.Timer when a timer fires."""
    with _lock:
        _timers.pop(name, None)
        _timer_end.pop(name, None)

    label = f"'{name}' timer" if name != "default" else "Your timer"
    _speak(f"{label} is done!")
    print(f"[TIMER] Fired: {name}", flush=True)


# ── Duration parser ───────────────────────────────────────────────────────────

def _parse_duration(text: str) -> int | None:
    """Parse a duration string into total seconds. Returns None if unparseable."""
    text = text.lower()
    total = 0
    found = False

    patterns = [
        (r'(\d+)\s*hour',   3600),
        (r'(\d+)\s*hr',     3600),
        (r'(\d+)\s*h\b',    3600),
        (r'(\d+)\s*minute', 60),
        (r'(\d+)\s*min',    60),
        (r'(\d+)\s*m\b',    60),
        (r'(\d+)\s*second', 1),
        (r'(\d+)\s*sec',    1),
        (r'(\d+)\s*s\b',    1),
    ]
    for pattern, multiplier in patterns:
        m = re.search(pattern, text)
        if m:
            total += int(m.group(1)) * multiplier
            found = True

    # "a minute" / "an hour"
    if re.search(r'\ba\s+minute\b', text):
        total += 60; found = True
    if re.search(r'\ban?\s+hour\b', text):
        total += 3600; found = True
    if re.search(r'\ba\s+second\b', text):
        total += 1; found = True

    return total if found and total > 0 else None


def _parse_clock_time(text: str) -> float | None:
    """
    Parse a clock time like "7:30", "7:30 AM", "19:45" into a future
    Unix timestamp. Returns None if unparseable.
    """
    m = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)?', text, re.IGNORECASE)
    if not m:
        # "7 AM", "7 o'clock"
        m2 = re.search(r'(\d{1,2})\s*(am|pm|o.clock)', text, re.IGNORECASE)
        if m2:
            hour = int(m2.group(1))
            ampm = (m2.group(2) or "").lower()
            if "pm" in ampm and hour < 12: hour += 12
            if "am" in ampm and hour == 12: hour = 0
            now = datetime.now()
            target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target.timestamp()
        return None

    hour   = int(m.group(1))
    minute = int(m.group(2))
    ampm   = (m.group(3) or "").lower()
    if "pm" in ampm and hour < 12: hour += 12
    if "am" in ampm and hour == 12: hour = 0

    now    = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target.timestamp()


# ── Name extractor ────────────────────────────────────────────────────────────

def _extract_name(text: str) -> str:
    """Extract optional timer name: 'set a pasta timer for 10 minutes' → 'pasta'."""
    m = re.search(
        r'(?:set\s+(?:a|an)\s+)(\w+)\s+timer',
        text, re.IGNORECASE
    )
    if m:
        name = m.group(1).lower()
        if name not in ("timer", "alarm", "new", "the"):
            return name
    return "default"


# ── Tool class ────────────────────────────────────────────────────────────────

class TimerTool(BaseTool):

    @property
    def name(self):
        return "timer"

    @property
    def description(self):
        return (
            "Sets timers and alarms. Speaks aloud when the timer fires. "
            "Can run multiple named timers simultaneously."
        )

    @property
    def keywords(self):
        return [
            "set a timer", "set timer", "timer for",
            "set an alarm", "alarm for", "wake me", "remind me in",
            "cancel the timer", "cancel timer", "stop the timer",
            "how long", "time left", "timer status", "timer remaining",
            "what is the timer", "check the timer", "is the timer",
            "countdown", "in ten minutes", "in five minutes",
            "in an hour", "in a minute",
        ]

    def run(self, query: str) -> str:
        text = query.lower().strip()

        # ── Cancel ───────────────────────────────────────────────────────────
        if re.search(r'\b(cancel|stop|clear|delete|remove)\b.*\b(timer|alarm)\b', text, re.I):
            name = _extract_name(text)
            with _lock:
                if name in _timers:
                    _timers[name].cancel()
                    _timers.pop(name)
                    _timer_end.pop(name, None)
                    return f"Timer '{name}' cancelled."
                elif _timers:
                    # Cancel the first active one
                    first = next(iter(_timers))
                    _timers[first].cancel()
                    _timers.pop(first)
                    _timer_end.pop(first, None)
                    return f"Timer cancelled."
            return "No active timer to cancel."

        # ── Status ───────────────────────────────────────────────────────────
        if re.search(r'\b(how\s+long|time\s+left|remaining|status|active)\b', text, re.I):
            with _lock:
                if not _timers:
                    return "No active timers."
                parts = []
                now = time.time()
                for tname, end in _timer_end.items():
                    remaining = int(end - now)
                    if remaining > 0:
                        m, s = divmod(remaining, 60)
                        h, m = divmod(m, 60)
                        if h:
                            parts.append(f"'{tname}': {h}h {m}m {s}s remaining")
                        elif m:
                            parts.append(f"'{tname}': {m}m {s}s remaining")
                        else:
                            parts.append(f"'{tname}': {s}s remaining")
                return ". ".join(parts) if parts else "No active timers."

        # ── Set alarm (clock time) ────────────────────────────────────────────
        if re.search(r'\b(alarm|wake\s+me|alarm\s+at)\b', text, re.I):
            ts = _parse_clock_time(text)
            if ts:
                delay = ts - time.time()
                target = datetime.fromtimestamp(ts).strftime("%I:%M %p")
                name = "alarm"
                with _lock:
                    if name in _timers:
                        _timers[name].cancel()
                    t = threading.Timer(delay, _fire, args=(name,))
                    t.daemon = True
                    t.start()
                    _timers[name]   = t
                    _timer_end[name] = ts
                return f"Alarm set for {target}."
            return "I could not understand the alarm time. Please say something like 'alarm at 7:30 AM'."

        # ── Set countdown timer ───────────────────────────────────────────────
        duration = _parse_duration(text)
        if duration:
            name = _extract_name(text)
            with _lock:
                if name in _timers:
                    _timers[name].cancel()
                end = time.time() + duration
                t = threading.Timer(duration, _fire, args=(name,))
                t.daemon = True
                t.start()
                _timers[name]    = t
                _timer_end[name] = end

            # Human-readable duration
            m, s = divmod(duration, 60)
            h, m = divmod(m, 60)
            if h:
                human = f"{h} hour{'s' if h>1 else ''}" + (f" {m} minute{'s' if m>1 else ''}" if m else "")
            elif m:
                human = f"{m} minute{'s' if m>1 else ''}" + (f" {s} second{'s' if s>1 else ''}" if s else "")
            else:
                human = f"{s} second{'s' if s>1 else ''}"

            label = f" '{name}'" if name != "default" else ""
            return f"Timer{label} set for {human}."

        return (
            "I could not understand the timer. "
            "Try: 'set a timer for 10 minutes' or 'alarm at 7:30 AM'."
        )
