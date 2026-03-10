"""
STT Worker Module - Speech-to-Text Processing

What it does:
    Captures audio from microphone, detects speech using VAD,
    records until the speaker has been silent for SILENCE_DURATION,
    then transcribes the full utterance as one message.

What it reads:
    - Audio from the default microphone
    - Configuration from config.py

What it writes:
    - Transcribed text to session latest.txt
    - Transcript entries to transcript.json
    - Full utterance as one string to output Queue

How to replace/upgrade:
    - Adjust SILENCE_DURATION in config.py to tune when recording stops
    - Adjust SILENCE_THRESHOLD in config.py to tune mic sensitivity
    - Swap faster-whisper by replacing _transcribe() only
"""

import os
import sys
import threading
import numpy as np
import sounddevice as sd
from queue import Queue
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import session

# ============================================================================
# GLOBAL STATE
# ============================================================================

_stopped = False
_output_queue: Optional[Queue] = None
_model = None
_is_speaking_event: Optional[threading.Event] = None

# ============================================================================
# MODEL LOADING
# ============================================================================

def _preload_model():
    from faster_whisper import WhisperModel

    os.environ["HF_HOME"] = config.HF_HOME
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    print(f"      Loading Whisper '{config.WHISPER_MODEL}' ({config.WHISPER_COMPUTE}) ...", flush=True)
    model = WhisperModel(
        config.WHISPER_MODEL,
        device="cpu",
        compute_type=config.WHISPER_COMPUTE,
        cpu_threads=config.WHISPER_THREADS,
    )

    # Warm-up pass
    dummy = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
    list(model.transcribe(dummy, beam_size=1)[0])
    print("      Whisper model warmed up.", flush=True)
    return model


# ============================================================================
# AUDIO HELPER
# ============================================================================

def _rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(data ** 2)))


# ============================================================================
# LISTENING LOOP
# ============================================================================

def _listen_loop():
    """
    VAD loop that records one complete utterance at a time.

    State machine:
        WAITING  — listening for speech to begin
        RECORDING — speech detected, accumulating audio
        SILENCE  — speech stopped, counting silence chunks
                   if silence exceeds SILENCE_DURATION → transcribe
                   if speech resumes → back to RECORDING

    This means Buddy waits for a full natural pause before transcribing,
    so long sentences and mid-sentence pauses are captured as one message.
    """
    global _stopped

    chunk_ms      = 100                                        # ms per chunk
    chunk_samples = int(config.SAMPLE_RATE * chunk_ms / 1000) # samples per chunk

    # How many silent chunks before we consider the utterance done
    # Uses config.SILENCE_DURATION (seconds) — default 1.5s recommended
    silence_chunks_needed = int(config.SILENCE_DURATION * 1000 / chunk_ms)

    # Minimum speech chunks before we bother transcribing (avoids noise blips)
    min_speech_chunks = 3   # 300ms minimum utterance

    print("      Microphone listening started.", flush=True)

    STATE_WAITING   = 0
    STATE_RECORDING = 1
    STATE_SILENCE   = 2

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
        device=None,
    ) as stream:

        state         = STATE_WAITING
        buffer        = []       # accumulated speech audio
        silence_count = 0
        speech_count  = 0

        while not _stopped:
            chunk, _ = stream.read(chunk_samples)
            chunk    = chunk.copy()
            is_loud  = _rms(chunk) > config.SILENCE_THRESHOLD

            # ── WAITING: look for speech to start ───────────────
            if state == STATE_WAITING:
                if is_loud:
                    buffer       = [chunk]
                    speech_count = 1
                    silence_count = 0
                    state        = STATE_RECORDING

            # ── RECORDING: accumulate audio ──────────────────────
            elif state == STATE_RECORDING:
                buffer.append(chunk)

                if is_loud:
                    speech_count  += 1
                    silence_count  = 0
                else:
                    silence_count += 1
                    if silence_count >= silence_chunks_needed:
                        # Pause detected — move to silence state
                        state = STATE_SILENCE

                # Hard cap to avoid endless recording
                if len(buffer) > int(config.MAX_RECORD_SECONDS * 1000 / chunk_ms):
                    state = STATE_SILENCE

            # ── SILENCE: decide whether to transcribe or resume ──
            elif state == STATE_SILENCE:
                if is_loud:
                    # Speaker resumed — keep recording
                    buffer.append(chunk)
                    speech_count  += 1
                    silence_count  = 0
                    state         = STATE_RECORDING
                else:
                    # Still silent — transcribe now
                    if speech_count >= min_speech_chunks:
                        # Skip if TTS is currently playing
                        if not (_is_speaking_event and _is_speaking_event.is_set()):
                            audio = np.concatenate(buffer, axis=0).flatten()
                            _transcribe(audio)

                    # Reset for next utterance
                    buffer        = []
                    speech_count  = 0
                    silence_count = 0
                    state         = STATE_WAITING


def _transcribe(audio: np.ndarray):
    """Transcribe audio and push result to output queue."""
    try:
        segments, _ = _model.transcribe(
            audio,
            beam_size=1,
            language="en",
            vad_filter=True,
        )
        text = " ".join(s.text for s in segments).strip()
    except Exception as e:
        print(f"[STT] Transcription error: {e}", flush=True)
        return

    # Ignore very short results — likely noise or a single filler sound
    if not text or len(text.split()) < 2:
        return

    session.log_user(text)
    _output_queue.put(text)


# ============================================================================
# PUBLIC API
# ============================================================================

def set_is_speaking_event(event: threading.Event):
    global _is_speaking_event
    _is_speaking_event = event


def start():
    global _output_queue, _stopped, _model

    _stopped      = False
    _output_queue = Queue()
    _model        = _preload_model()

    t = threading.Thread(target=_listen_loop, daemon=True, name="STTWorker")
    t.start()


def stop():
    global _stopped
    _stopped = True


def get_output_queue() -> Queue:
    return _output_queue
