"""
Memory Module - BuddyQ Voice Assistant
T:\\BuddyQ\\core\\memory.py

DESIGN
======
Single flat memory file: memory/memory.json

The file contains one key: "memory" — a plain text string written in
natural language. The LLM maintains this text directly. It reads like
a personal briefing note, not a database.

On every session shutdown, the LLM receives:
  - The full current memory file (as text)
  - The full session transcript (cleaned of tool noise)
  - A strict instruction to rewrite the entire memory block

The LLM decides:
  - What new information from the session to add
  - What existing information is now outdated and should be updated
  - What is least important and should be dropped to stay under 5000 words
  - What is critical and must always be kept (name, location, key facts)

The result replaces the entire memory file. This means the memory is
always a coherent, curated document — never a growing append-only log.

HARD LIMIT
==========
5000 words maximum. The LLM is instructed to enforce this itself.
If the rewritten memory would exceed 5000 words, it must drop the
least important information until it fits.

INJECTION
=========
build_memory_block() returns the memory text for injection into the
system prompt. It is NEVER placed in the TTS queue.

WORD COUNT SAFETY
=================
After the LLM returns the new memory, _count_words() checks the length.
If it somehow exceeds 5500 words (LLM failed to comply), it is hard-
truncated at the sentence level to stay safe.
"""

import os
import sys
import re
import json
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# CONFIG
# ============================================================================

MEMORY_DIR   = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory"
)
MEMORY_FILE  = os.path.join(MEMORY_DIR, "memory.json")

MAX_WORDS          = 5000   # hard cap enforced by LLM and verified after
SAFETY_WORD_CAP    = 5500   # if LLM ignores the cap, hard truncate here
MIN_SESSION_TURNS  = 2      # minimum user turns before compression runs

_lock = threading.Lock()

# ============================================================================
# WORD COUNT
# ============================================================================

def _count_words(text: str) -> int:
    return len(text.split())


def _hard_truncate_to_words(text: str, max_words: int) -> str:
    """Truncate text to max_words at sentence boundary where possible."""
    words = text.split()
    if len(words) <= max_words:
        return text
    # Truncate and try to end at a sentence boundary
    truncated = " ".join(words[:max_words])
    last_sentence_end = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )
    if last_sentence_end > max_words * 0.8:  # only if we're not losing too much
        return truncated[:last_sentence_end + 1].strip()
    return truncated.strip()

# ============================================================================
# STORAGE
# ============================================================================

def _load() -> str:
    """
    Load memory text from disk.
    Returns empty string if file doesn't exist or is corrupt.
    """
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("memory", "").strip()
    except Exception as e:
        print(f"[MEM] Load error (non-fatal): {e}", flush=True)
    return ""


def _save(memory_text: str):
    """Save memory text to disk atomically via a temp file."""
    try:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        data = {
            "memory":       memory_text,
            "word_count":   _count_words(memory_text),
            "last_updated": time.strftime("%Y-%m-%d %H:%M"),
        }
        tmp = MEMORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, MEMORY_FILE)
    except Exception as e:
        print(f"[MEM] Save error: {e}", flush=True)


# ============================================================================
# PUBLIC READ API
# ============================================================================

def build_memory_block() -> str:
    """
    Return memory text for injection into the LLM system prompt.
    FOR SYSTEM PROMPT INJECTION ONLY — must never reach the TTS queue.
    Returns empty string if no memory exists yet.
    """
    with _lock:
        text = _load()

    if not text:
        return ""

    return (
        "--- PERSISTENT MEMORY ---\n"
        "The following is a summary of what you know about the user from previous sessions.\n"
        "Use this silently as background context. Never read it aloud or acknowledge it directly.\n\n"
        + text +
        "\n--- END MEMORY ---"
    )


def get_memory_text() -> str:
    """Return raw memory text. Empty string if none."""
    with _lock:
        return _load()


def get_word_count() -> int:
    """Return current memory word count."""
    with _lock:
        return _count_words(_load())


def clear_memory():
    """Wipe all memory. Irreversible."""
    with _lock:
        _save("")
    print("[MEM] Memory cleared.", flush=True)


# ============================================================================
# TRANSCRIPT CLEANER
# ============================================================================

_re_user_question = re.compile(
    r'(?:User asked|User question|Transcribed text):\s*(.+?)(?:\n|$)',
    re.IGNORECASE
)

def _clean_transcript(conversation: list) -> str:
    """
    Convert raw conversation list to a clean readable transcript.
    Strips tool injection prefixes, STT notes, and system noise.
    Returns a plain text string suitable for sending to the LLM.
    """
    lines = []
    for m in conversation:
        role    = "User" if m["role"] == "user" else "Buddy"
        content = m.get("content", "")

        # Strip tool-injected prompts — extract only the user's actual question
        noisy_prefixes = (
            "LIVE DATA FROM WEB SEARCH",
            "LIVE WEB DATA",
            "[STT note",
            "IMPORTANT: The web data",
        )
        if any(content.startswith(p) for p in noisy_prefixes):
            match = _re_user_question.search(content)
            if match:
                content = match.group(1).strip()
            else:
                continue

        content = content.strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


# ============================================================================
# SHUTDOWN MEMORY UPDATE (the core of this module)
# ============================================================================

def update_memory_on_shutdown(
    conversation: list,
    llm_client,
    model_name: str,
    no_thinking_body: dict,
):
    """
    Called once at shutdown with the full session conversation.

    What happens:
      1. Clean the session transcript (strip tool noise)
      2. Load the current full memory file
      3. Send both to the LLM with a rewrite instruction
      4. LLM rewrites the entire memory, staying under 5000 words
      5. Safety-check the word count
      6. Save the new memory, replacing the old one entirely

    This is a BLOCKING call — it waits up to 30 seconds for the LLM.
    Call update_memory_on_shutdown_async() to run it on a background thread.
    """
    # Skip if session had almost no content
    user_turns = [m for m in conversation if m["role"] == "user"]
    if len(user_turns) < MIN_SESSION_TURNS:
        print("[MEM] Session too short — skipping memory update.", flush=True)
        return

    transcript = _clean_transcript(conversation)
    if not transcript.strip():
        print("[MEM] No usable transcript — skipping memory update.", flush=True)
        return

    with _lock:
        current_memory = _load()

    word_count    = _count_words(current_memory)
    today         = time.strftime("%Y-%m-%d")

    if current_memory:
        memory_section = (
            f"CURRENT MEMORY FILE ({word_count} words):\n"
            f"---\n{current_memory}\n---\n\n"
        )
    else:
        memory_section = "CURRENT MEMORY FILE: (empty — no memory yet)\n\n"

    prompt = (
        f"You are the memory manager for a personal voice assistant called Buddy.\n"
        f"Today's date: {today}\n\n"
        f"{memory_section}"
        f"SESSION TRANSCRIPT (today's conversation):\n"
        f"---\n{transcript}\n---\n\n"
        f"TASK:\n"
        f"Rewrite the memory file to incorporate any important new information from "
        f"today's session. Return ONLY the new memory text — nothing else.\n\n"
        f"STRICT RULES:\n"
        f"1. MAXIMUM {MAX_WORDS} WORDS. This is a hard limit. If needed, remove the "
        f"least important information to stay under it. Never exceed it.\n"
        f"2. ALWAYS KEEP: user's name, location, occupation, age, family members, "
        f"important relationships, stated preferences, recurring topics, long-term goals.\n"
        f"3. DROP IF NEEDED (least important first): one-off questions that won't recur, "
        f"general knowledge queries, anything already outdated, redundant phrasings.\n"
        f"4. Write in clear natural prose — short paragraphs, no bullet points, no headers, "
        f"no JSON, no markdown. Write it like a concise personal briefing note.\n"
        f"5. Only include information about the USER. Do not include facts about the "
        f"world, current events, or anything that isn't personal to this user.\n"
        f"6. If the session contained nothing worth remembering, return the existing "
        f"memory unchanged (still respecting the word limit).\n"
        f"7. Do not add today's date or session number. Do not include meta-commentary "
        f"like 'the user asked about X today'. Write it as standing facts.\n"
        f"8. Output ONLY the memory text. No preamble, no explanation, no formatting."
    )

    print("[MEM] Sending memory update request to LLM...", flush=True)

    try:
        resp = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,      # enough for 5000 words with room
            temperature=0.2,      # low temperature = consistent, factual output
            stream=False,
            extra_body=no_thinking_body,
        )

        new_memory = (resp.choices[0].message.content or "").strip()

        # Strip accidental markdown fences if model adds them
        new_memory = re.sub(r"^```[a-z]*\n?", "", new_memory, flags=re.MULTILINE)
        new_memory = re.sub(r"\n?```$",        "", new_memory, flags=re.MULTILINE)
        new_memory = new_memory.strip()

        if not new_memory:
            print("[MEM] LLM returned empty memory — keeping existing.", flush=True)
            return

        # Safety check — hard truncate if LLM ignored the word limit
        actual_words = _count_words(new_memory)
        if actual_words > SAFETY_WORD_CAP:
            print(
                f"[MEM] WARNING: LLM returned {actual_words} words (limit {MAX_WORDS}). "
                f"Hard-truncating to {MAX_WORDS}.", flush=True
            )
            new_memory = _hard_truncate_to_words(new_memory, MAX_WORDS)
            actual_words = _count_words(new_memory)

        with _lock:
            _save(new_memory)

        print(
            f"[MEM] Memory updated — {actual_words} words saved to {MEMORY_FILE}",
            flush=True
        )

    except Exception as e:
        print(f"[MEM] Update failed: {e} — existing memory preserved.", flush=True)


def update_memory_on_shutdown_async(
    conversation: list,
    llm_client,
    model_name: str,
    no_thinking_body: dict,
):
    """
    Non-blocking wrapper — runs update_memory_on_shutdown on a background thread.
    Waits up to 30 seconds for it to finish before allowing process to exit.
    Call this from main.py during shutdown.
    """
    t = threading.Thread(
        target=update_memory_on_shutdown,
        args=(conversation, llm_client, model_name, no_thinking_body),
        daemon=True,
        name="MemoryUpdater",
    )
    t.start()
    t.join(timeout=30)
    if t.is_alive():
        print("[MEM] Memory update timed out — partial save may have occurred.", flush=True)
