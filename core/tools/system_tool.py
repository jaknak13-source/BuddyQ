"""
System Control Tool - T:\\BuddyQ\\core\\tools\\system_tool.py

Handles hands-free control of Windows:
  - Volume: "turn up the volume", "mute", "set volume to 50"
  - Apps:   "open Chrome", "open Notepad", "open calculator"
  - Screen: "lock the screen", "turn off the screen"
  - Power:  "shut down", "restart", "sleep"
  - Clipboard: "what's in my clipboard", "copy that to clipboard"
  - Browser: "open youtube.com in the browser"

Uses Windows built-in commands and the pycaw library for volume control.
pycaw install: pip install pycaw comtypes --break-system-packages
Falls back to nircmd if pycaw is unavailable (optional third-party tool).
"""

import re
import os
import subprocess
from .base import BaseTool


# ── Volume control via pycaw (preferred) ─────────────────────────────────────

def _set_volume(level: int):
    """Set system volume 0-100 via pycaw or nircmd fallback."""
    level = max(0, min(100, level))
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        # pycaw uses scalar 0.0–1.0
        volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        return f"Volume set to {level}%."
    except ImportError:
        pass
    # nircmd fallback (optional third-party, silent fail if absent)
    try:
        nircmd = os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"), "nircmd", "nircmd.exe")
        if os.path.exists(nircmd):
            # nircmd volume is 0-65535
            subprocess.Popen([nircmd, "setsysvolume", str(int(level / 100 * 65535))])
            return f"Volume set to {level}%."
    except Exception:
        pass
    return "Volume control is not available. Install pycaw: pip install pycaw comtypes"


def _get_volume() -> str:
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        level = int(volume.GetMasterVolumeLevelScalar() * 100)
        muted = volume.GetMute()
        if muted:
            return f"Volume is muted (last level: {level}%)."
        return f"Volume is at {level}%."
    except ImportError:
        return "pycaw not installed — cannot read volume."
    except Exception as e:
        return f"Could not read volume: {e}"


def _mute(mute: bool) -> str:
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMute(1 if mute else 0, None)
        return "Muted." if mute else "Unmuted."
    except ImportError:
        return "pycaw not installed."
    except Exception as e:
        return f"Mute error: {e}"


# ── App launcher ──────────────────────────────────────────────────────────────

_APP_MAP = {
    "chrome":       "chrome.exe",
    "google chrome":"chrome.exe",
    "firefox":      "firefox.exe",
    "edge":         "msedge.exe",
    "microsoft edge":"msedge.exe",
    "notepad":      "notepad.exe",
    "calculator":   "calc.exe",
    "calendar":     "outlookcal:",         # opens calendar in browser/app
    "explorer":     "explorer.exe",
    "file explorer":"explorer.exe",
    "task manager": "taskmgr.exe",
    "word":         "winword.exe",
    "excel":        "excel.exe",
    "powerpoint":   "powerpnt.exe",
    "outlook":      "outlook.exe",
    "vs code":      "code.exe",
    "visual studio code": "code.exe",
    "spotify":      "spotify.exe",
    "discord":      "discord.exe",
    "steam":        "steam.exe",
    "paint":        "mspaint.exe",
    "snipping tool":"snippingtool.exe",
    "settings":     "ms-settings:",
    "control panel":"control.exe",
    "terminal":     "wt.exe",
    "cmd":          "cmd.exe",
    "powershell":   "powershell.exe",
}

def _open_app(name: str) -> str:
    app = _APP_MAP.get(name.lower().strip())
    if not app:
        # Try direct execution as a guess
        app = name.strip()
    try:
        os.startfile(app)
        return f"Opening {name}."
    except Exception:
        try:
            subprocess.Popen(["start", "", app], shell=True)
            return f"Opening {name}."
        except Exception as e:
            return f"Could not open {name}: {e}"


# ── Clipboard ─────────────────────────────────────────────────────────────────

def _get_clipboard() -> str:
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        text = root.clipboard_get()
        root.destroy()
        if text:
            return f"Clipboard contains: {text[:300]}"
        return "Clipboard is empty."
    except Exception as e:
        return f"Could not read clipboard: {e}"

def _set_clipboard(text: str) -> str:
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
        return "Text copied to clipboard."
    except Exception as e:
        return f"Could not write to clipboard: {e}"


# ── Tool class ────────────────────────────────────────────────────────────────

class SystemTool(BaseTool):

    # Stores last assistant response for clipboard operations
    last_response: str = ""

    @property
    def name(self):
        return "system"

    @property
    def description(self):
        return (
            "Controls Windows system functions: volume, app launching, "
            "screen lock, clipboard, and power options."
        )

    @property
    def keywords(self):
        return [
            "open", "launch", "start",
            "volume", "turn up", "turn down", "louder", "quieter",
            "mute", "unmute", "set volume",
            "lock", "lock screen", "lock the screen",
            "shut down", "shutdown", "restart", "reboot", "sleep",
            "clipboard", "copy that", "what's in my clipboard",
            "turn off the screen", "blank screen",
        ]

    def run(self, query: str) -> str:
        text = query.lower().strip()

        # ── Volume ───────────────────────────────────────────────────────────
        if re.search(r'\bmute\b', text) and not re.search(r'\bunmute\b', text):
            return _mute(True)
        if re.search(r'\bunmute\b|\bun-mute\b', text):
            return _mute(False)

        m = re.search(r'(?:set\s+)?volume\s+(?:to\s+)?(\d+)', text)
        if m:
            return _set_volume(int(m.group(1)))

        if re.search(r'\b(louder|turn\s+up|volume\s+up|increase\s+volume|raise\s+volume)\b', text):
            return _set_volume(70)   # sensible bump

        if re.search(r'\b(quieter|turn\s+down|volume\s+down|lower\s+volume|decrease\s+volume)\b', text):
            return _set_volume(30)

        if re.search(r'\b(what.?s\s+the\s+volume|current\s+volume|volume\s+level)\b', text):
            return _get_volume()

        # ── Power / screen ────────────────────────────────────────────────────
        if re.search(r'\bshut\s*down\b|\bpower\s+off\b', text):
            subprocess.Popen("shutdown /s /t 30", shell=True)
            return "Shutting down in 30 seconds. Say 'cancel shutdown' to abort."

        if re.search(r'\bcancel\s+shutdown\b', text):
            subprocess.Popen("shutdown /a", shell=True)
            return "Shutdown cancelled."

        if re.search(r'\brestart\b|\breboot\b', text):
            subprocess.Popen("shutdown /r /t 30", shell=True)
            return "Restarting in 30 seconds."

        if re.search(r'\bsleep\b|\bhibernate\b', text):
            subprocess.Popen("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True)
            return "Going to sleep."

        if re.search(r'\block\b.*\bscreen\b|\block\s+(?:the\s+)?(?:pc|computer|screen)\b', text):
            subprocess.Popen("rundll32.exe user32.dll,LockWorkStation", shell=True)
            return "Screen locked."

        if re.search(r'\b(turn\s+off|blank)\s+(?:the\s+)?screen\b', text):
            subprocess.Popen('powershell -command "(Add-Type -MemberDefinition \'[DllImport(\\"user32.dll\\")] public static extern int SendMessage(int hWnd, int hMsg, int wParam, int lParam);\' -Name Win32 -PassThru)::SendMessage(-1, 0x0112, 0xF170, 2)"', shell=True)
            return "Screen turned off."

        # ── Clipboard ─────────────────────────────────────────────────────────
        if re.search(r'\bclipboard\b|\bwhat.?s\s+copied\b', text):
            return _get_clipboard()

        if re.search(r'\bcopy\s+that\b|\bsave\s+that\s+to\s+clipboard\b', text):
            if self.last_response:
                return _set_clipboard(self.last_response)
            return "There is nothing recent to copy."

        # ── App launcher ──────────────────────────────────────────────────────
        m = re.search(
            r'\b(?:open|launch|start)\s+(.+?)(?:\s+(?:please|now|app|for\s+me))?\s*$',
            text, re.IGNORECASE
        )
        if m:
            app_name = m.group(1).strip()
            # If it looks like a URL, open in browser
            if re.search(r'\.\w{2,}', app_name) or 'www.' in app_name:
                url = app_name if app_name.startswith('http') else 'https://' + app_name
                try:
                    import webbrowser
                    webbrowser.open(url)
                    return f"Opening {url} in your browser."
                except Exception as e:
                    return f"Could not open URL: {e}"
            return _open_app(app_name)

        return "I did not understand that system command."
