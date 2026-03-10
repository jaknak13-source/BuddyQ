"""
TTS Worker Module - BuddyQ Voice Assistant
T:\\BuddyQ\\core\\tts_worker.py

What it does:
    Takes text sentences from a Queue, synthesizes each via Piper TTS,
    and plays the resulting WAV through the system default output device.
    Each sentence is played in a single non-chunked sd.play() call to
    avoid glitches and gaps.

How to replace/upgrade:
    Replace _synthesize_and_play() to swap TTS engine.
    The Queue contract stays the same — main.py needs no changes.
"""

import os
import sys
import subprocess
import threading
import time
import wave
import numpy as np
import sounddevice as sd
from queue import Queue, Empty
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# ============================================================================
# GLOBAL STATE
# ============================================================================

_stopped      = False
_input_queue: Optional[Queue]               = None
_interrupt_event: Optional[threading.Event] = None   # wired in by main.py

IS_SPEAKING = threading.Event()   # read by stt_worker to pause mic during playback


# ============================================================================
# PUBLIC WIRING
# ============================================================================

def set_interrupt_event(event: threading.Event):
    """Called by main.py once at startup to wire in the interrupt event."""
    global _interrupt_event
    _interrupt_event = event


# ============================================================================
# AUDIO PLAYBACK
# ============================================================================

def _play_wav(wav_path: str):
    """
    Read a WAV file and play it through the default output device.
    Plays the entire sentence in ONE sd.play() call — no chunking, no gaps.
    Interrupt is checked every 50ms via a polling loop while audio plays.
    """
    # ── Read WAV ─────────────────────────────────────────────────
    try:
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            rate       = wf.getframerate()
            sampwidth  = wf.getsampwidth()
            n_frames   = wf.getnframes()
            frames     = wf.readframes(n_frames)
    except Exception as e:
        print(f"[TTS] WAV read error: {e}", flush=True)
        return

    if not frames or n_frames == 0:
        print("[TTS] WAV is empty — skipping.", flush=True)
        return

    # ── Decode to float32 ────────────────────────────────────────
    if sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    # Stereo -> mono
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    audio = np.clip(audio, -1.0, 1.0)

    device = sd.default.device[1]

    # ── Play ─────────────────────────────────────────────────────
    try:
        sd.play(audio, samplerate=rate, device=device)

        # Poll every 50ms — allows interrupt without blocking
        duration = len(audio) / rate
        deadline = time.time() + duration + 0.5
        while time.time() < deadline:
            if not sd.get_stream().active:
                break
            if _interrupt_event and _interrupt_event.is_set():
                sd.stop()
                return
            time.sleep(0.05)

        sd.wait()

    except Exception as e:
        print(f"[TTS] sd.play error: {e}", flush=True)
        try:
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        except Exception as e2:
            print(f"[TTS] winsound fallback failed: {e2}", flush=True)


# ============================================================================
# SYNTHESIS
# ============================================================================

def _synthesize_and_play(text: str):
    """Synthesize text via Piper and play the result. Blocks until done."""
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    out_wav = os.path.join(config.TEMP_DIR, f"tts_{threading.get_ident()}.wav")

    try:
        proc = subprocess.Popen(
            [config.TTS_PIPER, "--model", config.VOICE_FILE,
             "--output_file", out_wav],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _stdout, stderr = proc.communicate(input=text.encode("utf-8"), timeout=20)
        if proc.returncode != 0:
            err = stderr.decode("utf-8", "ignore")[:200]
            print(f"[TTS] Piper error (code {proc.returncode}): {err}", flush=True)
            return
    except subprocess.TimeoutExpired:
        proc.kill()
        print("[TTS] Piper timed out.", flush=True)
        return
    except Exception as e:
        print(f"[TTS] Piper launch error: {e}", flush=True)
        return

    if not os.path.exists(out_wav) or os.path.getsize(out_wav) == 0:
        print("[TTS] Piper produced no output.", flush=True)
        return

    print(f"[TTS] {text[:80]}", flush=True)
    _play_wav(out_wav)

    try:
        os.remove(out_wav)
    except Exception:
        pass


# ============================================================================
# WORKER LOOP
# ============================================================================

def _worker_loop():
    global _stopped
    while not _stopped:
        try:
            text = _input_queue.get(timeout=0.5)
        except Empty:
            continue

        if not text or not text.strip():
            continue

        IS_SPEAKING.set()
        try:
            _synthesize_and_play(text.strip())
        finally:
            IS_SPEAKING.clear()


# ============================================================================
# PUBLIC API
# ============================================================================

def start():
    """Warm up Piper TTS and start the worker thread."""
    global _input_queue, _stopped
    _stopped     = False
    _input_queue = Queue()

    os.makedirs(config.TEMP_DIR, exist_ok=True)

    try:
        out_idx = sd.default.device[1]
        dev     = sd.query_devices(out_idx)
        print(f"      Audio output: [{out_idx}] {dev['name']} "
              f"({dev['max_output_channels']}ch @ {int(dev['default_samplerate'])}Hz)",
              flush=True)
    except Exception as e:
        print(f"      Audio device query failed: {e}", flush=True)

    print("      Warming up Piper TTS...", flush=True)
    try:
        _synthesize_and_play("Ready.")
        print("      Piper warmed up.", flush=True)
    except Exception as e:
        print(f"      Warmup failed (non-fatal): {e}", flush=True)

    t = threading.Thread(target=_worker_loop, daemon=True, name="TTSWorker")
    t.start()


def stop():
    global _stopped
    _stopped = True
    sd.stop()


def get_input_queue() -> Queue:
    return _input_queue
