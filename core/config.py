"""
Config Module - Centralized Settings and Configuration

What it does:
Stores all configurable values for BuddyQ.
Auto-detects voice file at runtime so no hardcoded filenames needed.

What it reads:
- Filesystem: tts/voices/ for .onnx files

What it writes:
- Nothing. Read-only after init.

How to replace/upgrade:
- Edit the constants section below for tuning.
- Drop a new .onnx voice into tts/voices/ and it will be picked up automatically.
"""

import os

# ============================================================================
# PATHS (all relative to T:/BuddyQ)
# ============================================================================

BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR             = os.path.join(BASE_DIR, "core", "temp")
HF_HOME              = os.path.join(BASE_DIR, "stt", "model_cache")
TTS_PIPER            = os.path.join(BASE_DIR, "tts", "piper.exe")
VOICE_DIR            = os.path.join(BASE_DIR, "tts", "voices")
SESSIONS_DIR         = os.path.join(BASE_DIR, "sessions")
CURRENT_SESSION_FILE = os.path.join(BASE_DIR, "sessions", "current_session.txt")

# ============================================================================
# AUDIO SETTINGS
# ============================================================================

SAMPLE_RATE       = 16000  # Hz — do not change (Whisper expects 16k)
SILENCE_THRESHOLD = 0.015  # RMS level below which = silence
SILENCE_DURATION  = 2.1    # seconds of silence before recording ends
MAX_RECORD_SECONDS = 90    # hard cap on one recording

# ============================================================================
# WHISPER SETTINGS
# ============================================================================

WHISPER_MODEL   = "tiny"  # tiny / base / small
WHISPER_COMPUTE = "int8"  # int8 = fastest on CPU
WHISPER_THREADS = 4       # match physical core count (i7-6700HQ = 4)

# ============================================================================
# LLM
# ============================================================================

LLM_API_URL = "http://127.0.0.1:1234/v1"

# Main chat/text model — must match the exact id shown in LM Studio server tab
# or at http://127.0.0.1:1234/v1/models
# Example for llama.cpp: "qwen"  |  Example for LM Studio: "qwen3.5-4b-instruct"
MODEL_NAME = "qwen3.5-4b-instruct"  # <-- check exact id at /v1/models and paste here

# Vision model for the screen tool — set to the same value if your main model
# is already a VL model (e.g. Qwen3-VL), otherwise point to a separate VL model id
VISION_MODEL_NAME = "qwen3.5-4b-instruct"  # <-- change to VL model id when you load one

# ============================================================================
# AUTO-DETECT VOICE FILE
# ============================================================================

def _auto_detect_voice_file() -> str:
    # Return path to first .onnx file found in voices/. Prefers 'lessac'.
    if not os.path.isdir(VOICE_DIR):
        return ""
    onnx = [f for f in os.listdir(VOICE_DIR) if f.endswith(".onnx")]
    if not onnx:
        return ""
    for f in onnx:
        if "lessac" in f.lower():
            return os.path.join(VOICE_DIR, f)
    return os.path.join(VOICE_DIR, sorted(onnx)[0])

VOICE_FILE = _auto_detect_voice_file()

# ============================================================================
# STARTUP VALIDATION
# ============================================================================

def validate():
    ok = True
    if not os.path.isfile(TTS_PIPER):
        print(f"[CONFIG] WARNING: piper.exe not found at {TTS_PIPER}", flush=True)
        ok = False
    if not VOICE_FILE:
        print(f"[CONFIG] WARNING: no .onnx voice file found in {VOICE_DIR}", flush=True)
        ok = False
    else:
        print(f"[CONFIG] Voice file  : {VOICE_FILE}", flush=True)
    print(f"[CONFIG] Base dir    : {BASE_DIR}", flush=True)
    print(f"[CONFIG] HF cache    : {HF_HOME}", flush=True)
    print(f"[CONFIG] LLM model   : {MODEL_NAME}", flush=True)
    print(f"[CONFIG] Vision model: {VISION_MODEL_NAME}", flush=True)
    return ok
