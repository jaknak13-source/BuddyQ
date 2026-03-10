"""
Pipeline - T:\\BuddyQ\\core\\pipeline.py

════════════════════════════════════════════════════════════════
  This file owns all processing logic. main.py wires the
  hardware (STT, TTS, LLM) and calls run_turn() — that is all.

  To add a feature:
    - New tool behaviour   → tools\\ folder
    - Routing tweaks       → buddy_config.py  (Tier-1 regex)
    - Prompt/tone changes  → buddy_config.py  (SYSTEM_PROMPT)
    - Core logic changes   → this file (pipeline.py)
    - Hardware/wiring      → main.py
════════════════════════════════════════════════════════════════

Public API used by main.py:
    init(llm, model_name, no_thinking, history, history_lock,
         session_turns, session_lock, interrupt_event, stopped_event)
    run_turn(user_text, tts_queue) -> str
    is_interrupted() -> bool
"""

import re
import time
import threading
from queue import Queue

# Filled by init()
_llm            = None
_model_name     = None
_no_thinking    = None
_history        = None
_history_lock   = None
_session_turns  = None
_session_lock   = None
_interrupt_event  = None
_stopped_event    = None

# Imported after init() because they live in the same package
_mem        = None
_cfg        = None
_timer_mod  = None
_notes_mod  = None
_system_mod = None


def init(llm, model_name, no_thinking,
         history, history_lock,
         session_turns, session_lock,
         interrupt_event, stopped_event,
         mem, cfg, timer_mod, notes_mod, system_mod):
    """Called once from main() after all workers are started."""
    global _llm, _model_name, _no_thinking
    global _history, _history_lock
    global _session_turns, _session_lock
    global _interrupt_event, _stopped_event
    global _mem, _cfg, _timer_mod, _notes_mod, _system_mod

    _llm           = llm
    _model_name    = model_name
    _no_thinking   = no_thinking
    _history       = history
    _history_lock  = history_lock
    _session_turns = session_turns
    _session_lock  = session_lock
    _interrupt_event = interrupt_event
    _stopped_event   = stopped_event
    _mem        = mem
    _cfg        = cfg
    _timer_mod  = timer_mod
    _notes_mod  = notes_mod
    _system_mod = system_mod


# ============================================================================
# TEXT HELPERS
# ============================================================================

def _split_sentences(text: str) -> list:
    """
    Split streamed LLM text into TTS-ready sentences.
    Requires at least 30 chars before splitting to avoid false splits on
    abbreviations (U.S.), decimals ($67,000.50), or short fragments.
    """
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    result = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) >= 30:
            result.append(p)
        elif result:
            result[-1] = result[-1] + ' ' + p
        else:
            result.append(p)
    return result


def _number_to_words(n: float) -> str:
    """Convert a number to spoken English words."""
    try:
        if n != int(n):
            int_part = int(n)
            dec_part = str(n).split('.')[1]
            return (f"{_number_to_words(int_part)} point "
                    f"{' '.join(_number_to_words(int(d)) for d in dec_part)}")
        n = int(n)
        if n < 0:  return f"minus {_number_to_words(-n)}"
        if n == 0: return 'zero'
        ones = ['','one','two','three','four','five','six','seven','eight','nine',
                'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen',
                'seventeen','eighteen','nineteen']
        if n < 20: return ones[n]
        tens = ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
        if n < 100:
            t, u = divmod(n, 10)
            return tens[t] + (f'-{ones[u]}' if u else '')
        if n < 1_000:
            h, r = divmod(n, 100)
            return f"{ones[h]} hundred{' ' + _number_to_words(r) if r else ''}"
        if n < 1_000_000:
            t, r = divmod(n, 1000)
            return f"{_number_to_words(t)} thousand{' ' + _number_to_words(r) if r else ''}"
        if n < 1_000_000_000:
            m, r = divmod(n, 1_000_000)
            return f"{_number_to_words(m)} million{' ' + _number_to_words(r) if r else ''}"
        b, r = divmod(n, 1_000_000_000)
        return f"{_number_to_words(b)} billion{' ' + _number_to_words(r) if r else ''}"
    except Exception:
        return str(n)


def normalize_for_speech(text: str) -> str:
    """
    Convert written text into natural spoken English for TTS.
    Handles currencies, units, symbols, abbreviations, HTML entities.
    """
    # HTML entities
    text = (text.replace('&amp;', 'and').replace('&lt;', 'less than')
                .replace('&gt;', 'greater than').replace('&nbsp;', ' ')
                .replace('&#x27;', "'").replace('&quot;', '"')
                .replace('&#39;', "'"))

    # Strip markdown
    text = re.sub(r'[*#`_~\[\]|\\]', '', text)
    text = re.sub(r'\bhttps?://\S+', 'a link', text)

    # Deduplicate currency codes already written out
    text = re.sub(r'\bthe\s+(USD|EUR|GBP|BTC|ETH|JPY)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(USD|EUR|GBP|BTC|ETH|JPY)\s+(US dollars|euros|pounds|bitcoin|yen)\b',
                  lambda m: m.group(2), text, flags=re.IGNORECASE)
    text = re.sub(r'\b(dollars?|euros?|pounds?|yen|bitcoin)\s+(USD|EUR|GBP|JPY|BTC)\b',
                  lambda m: m.group(1), text, flags=re.IGNORECASE)

    # Currency symbols
    def _currency(m):
        symbol = m.group(1)
        number = m.group(2).replace(',', '')
        suffix = m.group(3) or ''
        names  = {'$': 'US dollars', '€': 'euros', '£': 'pounds',
                  '¥': 'yen', '₿': 'bitcoin', 'CHF': 'Swiss francs'}
        unit   = names.get(symbol, symbol)
        try:    spoken = _number_to_words(float(number))
        except: spoken = number
        sfx = {'k':'thousand','K':'thousand','m':'million','M':'million',
               'b':'billion','B':'billion','t':'trillion','T':'trillion'}
        return f"{spoken}{' ' + sfx[suffix] if suffix in sfx else ''} {unit}"
    text = re.sub(r'([\$€£¥₿]|CHF)\s*([\d,]+(?:\.\d+)?)([kKmMbBtT]?)', _currency, text)

    # Number + currency code
    _cmap = {'USD':'US dollars','EUR':'euros','GBP':'pounds','JPY':'yen',
             'BTC':'bitcoin','ETH':'ethereum','CHF':'Swiss francs'}
    def _num_code(m):
        try:    spoken = _number_to_words(float(m.group(1).replace(',','')))
        except: spoken = m.group(1)
        return f"{spoken} {_cmap.get(m.group(2).upper(), m.group(2))}"
    text = re.sub(r'\b([\d,]+(?:\.\d+)?)\s*(USD|EUR|GBP|JPY|BTC|ETH|CHF)\b', _num_code, text)

    # Percentages
    text = re.sub(r'([\d,]+(?:\.\d+)?)\s*%',
        lambda m: f"{_number_to_words(float(m.group(1).replace(',','')))} percent", text)

    # Large comma-separated numbers
    text = re.sub(r'\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\b',
        lambda m: _number_to_words(float(m.group(1).replace(',', ''))), text)

    # Units
    units = {
        r'\bkm/h\b':'kilometres per hour',  r'\bmph\b':'miles per hour',
        r'\bm/s\b':'metres per second',     r'\bkm\b':'kilometres',
        r'\bcm\b':'centimetres',            r'\bmm\b':'millimetres',
        r'\bkg\b':'kilograms',              r'\bg\b(?=\s)':'grams',
        r'\blbs?\b':'pounds',               r'\boz\b':'ounces',
        r'\bml\b':'millilitres',            r'\bkWh\b':'kilowatt hours',
        r'\bkW\b':'kilowatts',              r'\bGHz\b':'gigahertz',
        r'\bMHz\b':'megahertz',             r'\bGB\b':'gigabytes',
        r'\bTB\b':'terabytes',              r'\bMB\b':'megabytes',
        r'\bKB\b':'kilobytes',              r'\b°C\b':'degrees Celsius',
        r'\b°F\b':'degrees Fahrenheit',     r'\b°\b':'degrees',
        r'\bm²\b':'square metres',          r'\bft²\b':'square feet',
        r'\bsq\s*ft\b':'square feet',       r'\bsq\s*m\b':'square metres',
    }
    for pat, rep in units.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    # Abbreviations
    abbrevs = {
        r'\bUSA\b':'the United States', r'\bUS\b':'the US',
        r'\bUK\b':'the UK',             r'\bEU\b':'the European Union',
        r'\bUN\b':'the United Nations', r'\bGDP\b':'GDP',
        r'\bCEO\b':'CEO',               r'\bAI\b':'AI',
        r'\bURL\b':'link',              r'\be\.g\.\b':'for example',
        r'\bi\.e\.\b':'that is',        r'\betc\.\b':'and so on',
        r'\bvs\.\b':'versus',           r'\bapprox\.?\b':'approximately',
        r'\bDr\.\b':'Doctor',           r'\bMr\.\b':'Mister',
        r'\bMrs\.\b':'Missus',          r'\bMs\.\b':'Miss',
        r'\bProf\.\b':'Professor',      r'\bSt\.\b':'Saint',
        r'\bno\.\b':'number',
    }
    for pat, rep in abbrevs.items():
        text = re.sub(pat, rep, text)

    # 24h → 12h time
    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        period = 'AM' if h < 12 else 'PM'
        h12 = h % 12 or 12
        return f"{h12}:{mi:02d} {period}"
    text = re.sub(r'\b([01]?\d|2[0-3]):([0-5]\d)\b', _time, text)

    # Ordinals
    ordinals = {
        '1st':'first','2nd':'second','3rd':'third','4th':'fourth',
        '5th':'fifth','6th':'sixth','7th':'seventh','8th':'eighth',
        '9th':'ninth','10th':'tenth','11th':'eleventh','12th':'twelfth',
        '20th':'twentieth','21st':'twenty-first','30th':'thirtieth',
        '31st':'thirty-first','100th':'hundredth',
    }
    for num, word in ordinals.items():
        text = re.sub(rf'\b{num}\b', word, text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s([.,!?])', r'\1', text)
    return text.strip()

# alias
_normalize_for_speech = normalize_for_speech


# ============================================================================
# LLM CALL  (streaming → sentence chunks → TTS queue)
# ============================================================================

def _call_llm(prompt: str, tts_queue: Queue) -> str:
    """
    Stream the LLM response, pushing complete sentences to TTS as they arrive.
    Checks _interrupt_event between every sentence.
    Returns the full response text (normalised for speech).
    """
    _interrupt_event.clear()

    # Build messages with combined system block (date + memory + prompt)
    now = time.strftime("%A, %d %B %Y, %H:%M")
    system_parts = [_cfg.SYSTEM_PROMPT, f"\nCurrent date and time: {now}."]
    mem_block = _mem.build_memory_block()
    if mem_block:
        system_parts.append(f"\n{mem_block}")

    with _history_lock:
        history_turns = [m for m in _history if m["role"] != "system"]

    messages = (
        [{"role": "system", "content": "".join(system_parts)}]
        + history_turns
        + [{"role": "user", "content": prompt}]
    )

    full_response   = ""
    buffer          = ""
    was_interrupted = False
    in_think_block  = False

    try:
        stream = _llm.chat.completions.create(
            model       = _model_name,
            messages    = messages,
            max_tokens  = _cfg.MAX_TOKENS,
            temperature = _cfg.TEMPERATURE,
            stream      = True,
            extra_body  = _no_thinking,
        )

        for chunk in stream:
            if _stopped_event.is_set() or _interrupt_event.is_set():
                was_interrupted = True
                break

            delta = chunk.choices[0].delta.content
            if not delta:
                continue

            # Strip <think> blocks (safety net for non-compliant models)
            if '<think>' in delta:
                in_think_block = True
            if in_think_block:
                if '</think>' in delta:
                    in_think_block = False
                    delta = delta.split('</think>', 1)[-1]
                else:
                    continue
            if not delta:
                continue

            buffer        += delta
            full_response += delta

            sentences = _split_sentences(buffer)
            if len(sentences) > 1:
                for sentence in sentences[:-1]:
                    if _interrupt_event.is_set():
                        was_interrupted = True
                        break
                    cleaned = normalize_for_speech(sentence)
                    if cleaned:
                        tts_queue.put(cleaned)
                buffer = sentences[-1]

            if was_interrupted:
                break

        if not was_interrupted:
            remainder = normalize_for_speech(buffer)
            if remainder:
                tts_queue.put(remainder)
        else:
            # Drain TTS queue so nothing queued before interrupt plays
            while not tts_queue.empty():
                try:    tts_queue.get_nowait()
                except: break
            print("\n[INTERRUPT] Stopped.", flush=True)

    except Exception as e:
        print(f"[LLM] Error: {e}", flush=True)
        full_response = "Sorry, I could not reach the language model."
        if not was_interrupted:
            tts_queue.put(full_response)

    full_response = normalize_for_speech(full_response)

    # Update in-session history
    with _history_lock:
        if full_response.strip():
            _history.append({"role": "user",      "content": prompt})
            _history.append({"role": "assistant",  "content": full_response})
        non_system = [m for m in _history if m["role"] != "system"]
        if len(non_system) > _cfg.MAX_HISTORY_TURNS * 2:
            excess = len(non_system) - _cfg.MAX_HISTORY_TURNS * 2
            _history[:] = ([m for m in _history if m["role"] == "system"]
                           + non_system[excess:])

    # Log real turns for memory compression at shutdown
    if full_response.strip() and not was_interrupted:
        with _session_lock:
            _session_turns.append({"role": "user",      "content": prompt})
            _session_turns.append({"role": "assistant",  "content": full_response})

    return full_response


# ============================================================================
# HYBRID TOOL ROUTER
# ============================================================================

# Tool descriptions for the LLM classifier (Tier 2)
_TOOL_DESCRIPTIONS = """
Available tools (reply with exactly one name, or "none"):
- datetime     : current date or time
- calculator   : math, percentages, unit conversions (e.g. miles to km)
- timer        : set/cancel timers or alarms, check time remaining
- notes        : save notes, manage shopping or to-do lists
- weather      : current weather or forecast for any city
- system       : volume control, open apps, lock screen, clipboard, shutdown
- web_search   : anything requiring current real-world data — news, prices,
                 recent events, people in current roles, sports scores,
                 or any fact that may have changed since 2023
- none         : answer from training — conversation, stable facts, history,
                 science, definitions, math explanations, creative tasks
""".strip()


def _llm_classify_tool(user_text: str) -> str:
    """
    Non-streaming LLM call that returns the tool name for this utterance.
    max_tokens=6, temperature=0.0 — fast and deterministic (~200-300ms).
    Includes last 2 conversation turns so follow-ups are routed correctly.
    """
    with _history_lock:
        recent = [m for m in _history if m["role"] in ("user","assistant")][-4:]
    ctx = "\n".join(
        f"{'User' if m['role']=='user' else 'Buddy'}: {m['content'][:120].replace(chr(10),' ')}"
        for m in recent
    )
    context_block = f"Recent conversation:\n{ctx}\n\n" if ctx.strip() else ""

    prompt = (
        f"{_TOOL_DESCRIPTIONS}\n\n"
        f"{context_block}"
        f'User said: "{user_text}"\n\n'
        f"Which tool should handle this? Reply with ONLY the tool name or 'none'."
    )

    try:
        resp = _llm.chat.completions.create(
            model      = _model_name,
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 6,
            temperature= 0.0,
            stream     = False,
            extra_body = _no_thinking,
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        answer = answer.replace(" ", "_").split("\n")[0].split(".")[0].strip()
        valid  = {"datetime","calculator","timer","notes",
                  "weather","system","web_search","none"}
        result = answer if answer in valid else "none"
        print(f"[ROUTER] LLM → '{user_text[:50]}' → {result}", flush=True)
        return result
    except Exception as e:
        print(f"[ROUTER] Classifier error: {e} — fallback to none", flush=True)
        return "none"


def _route(user_text: str) -> str:
    """
    Hybrid two-tier router.
    Tier 1: instant regex from buddy_config.py  (0ms)
    Tier 2: LLM classifier for everything else  (~200-300ms)
    Returns a tool name or 'none'.
    """
    t = user_text.lower()
    cfg = _cfg

    if cfg.ROUTE_DIRECT.search(t):
        print("[ROUTER] Regex → none", flush=True);  return "none"
    if cfg.ROUTE_TIMER.search(t):
        print("[ROUTER] Regex → timer", flush=True); return "timer"
    if cfg.ROUTE_CALC.search(t):
        print("[ROUTER] Regex → calculator", flush=True); return "calculator"
    if cfg.ROUTE_NOTES.search(t):
        print("[ROUTER] Regex → notes", flush=True); return "notes"
    if cfg.ROUTE_SYSTEM.search(t):
        print("[ROUTER] Regex → system", flush=True); return "system"
    if cfg.ROUTE_SEARCH.search(t):
        print("[ROUTER] Regex → web_search", flush=True); return "web_search"

    return _llm_classify_tool(user_text)


def _get_tool(name: str):
    from tools import get_all_tools
    for t in get_all_tools():
        if t.name == name:
            return t
    return None


# ============================================================================
# MAIN ENTRY POINT  (called by main.py for every user utterance)
# ============================================================================

def is_interrupted() -> bool:
    return _interrupt_event.is_set()


def run_turn(user_text: str, tts_queue: Queue) -> str:
    """
    Full pipeline for one user utterance:
      route → run tool → LLM speaks result (or LLM direct)

    Interrupt is checked:
      - Before starting
      - After routing
      - After tool execution
      - Per-token during LLM streaming
      - Every 50ms during TTS playback (in tts_worker)
    """
    def _update_last(resp: str):
        _notes_mod.last_response = resp
        _system_mod.SystemTool.last_response = resp

    if is_interrupted():
        return ""

    # ── Route ─────────────────────────────────────────────────────────────────
    tool_name = _route(user_text)

    if is_interrupted():
        return ""

    # ── No tool — LLM direct ──────────────────────────────────────────────────
    if tool_name == "none":
        resp = _call_llm(user_text, tts_queue)
        _update_last(resp)
        return resp

    # ── Get tool instance ─────────────────────────────────────────────────────
    tool = _get_tool(tool_name)
    if not tool:
        print(f"[ROUTER] Tool '{tool_name}' not found — LLM fallback", flush=True)
        resp = _call_llm(user_text, tts_queue)
        _update_last(resp)
        return resp

    print(f"[TOOL] Running: {tool_name}", flush=True)

    # ── Wire context into stateful tools ──────────────────────────────────────
    if tool_name == "timer":
        _timer_mod.set_tts_queue(tts_queue)

    if tool_name == "system":
        with _history_lock:
            last_a = next(
                (m["content"] for m in reversed(_history) if m["role"] == "assistant"), ""
            )
        tool.last_response = last_a

    if hasattr(tool, "set_history"):
        with _history_lock:
            tool.set_history(list(_history))

    # ── Run tool ──────────────────────────────────────────────────────────────
    try:
        result = tool.run(user_text)
        print(f"[TOOL] Result: {result[:120].replace(chr(10),' ')}", flush=True)
    except Exception as e:
        result = f"Tool error: {e}"
        print(f"[TOOL] Error: {e}", flush=True)

    if is_interrupted():
        return ""

    # ── Instant tools: speak directly (no extra LLM call) ────────────────────
    if _cfg.TOOLS_SPEAK_DIRECT and tool_name in {
        "datetime", "calculator", "timer", "notes", "system", "weather"
    }:
        spoken = normalize_for_speech(result)
        tts_queue.put(spoken)
        _update_last(spoken)
        return spoken

    # ── Web search: pass to LLM for natural spoken delivery ──────────────────
    broad = bool(re.search(
        r'\b(summarize|summarise|summary|overview|headlines|roundup|'
        r'news\s*(today|this\s*week)|what.*happening|catch\s*me\s*up)\b',
        user_text, re.IGNORECASE
    ))
    length_hint = (
        "Give a thorough spoken summary using all key points. Finish every sentence."
        if broad else
        "Answer in one to three complete spoken sentences. Finish every sentence."
    )
    prompt = (
        f"SEARCH RESULT:\n{result}\n\n"
        f"User asked: {user_text}\n\n"
        f"Speak this naturally. Do not mention searching or data sources. {length_hint}"
    )
    resp = _call_llm(prompt, tts_queue)
    _update_last(resp)
    return resp
