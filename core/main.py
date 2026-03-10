"""
Main Orchestrator - BuddyQ Voice Assistant

What it does:
    Wires STT -> tool detection -> LLM (streaming) -> TTS.
    Aggressively searches the web for anything factual or time-sensitive.
    The LLM is explicitly blocked from answering factual questions from memory.

How to replace/upgrade:
    - Add tools in core/tools/ — nothing here needs changing
    - Replace _call_llm() body to swap LLM backend
    - Edit SYSTEM_PROMPT directly or import from system_prompt.py

Required packages:
    pip install faster-whisper sounddevice numpy openai
"""

import os
import sys
import re
import threading
import time
from queue import Queue, Empty

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import session
import stt_worker
import tts_worker
import memory as mem
from tools import find_tool, get_all_tools
from tools.web_search import WebSearchTool, set_llm as _ws_set_llm
from tools import timer_tool as _timer_mod
from tools import notes_tool as _notes_mod
from tools import system_tool as _system_mod

from openai import OpenAI

_stopped_event = threading.Event()
_web_search    = WebSearchTool()

# ============================================================================
# LLM CONFIG
# ============================================================================

MODEL_NAME = config.MODEL_NAME
MAX_TOKENS  = 1500  # high cap — conciseness is enforced by prompt, not token limit
TEMPERATURE = 0.35

# Disable chain-of-thought / reasoning tokens (Qwen3, DeepSeek-R1, etc.)
# Passed as extra_body to llm — ignored silently by models that don't support it
_NO_THINKING = {
    "chat_template_kwargs": {"thinking": False},  # used by llama.cpp-style Qwen
    "thinking": False,                            # some servers (incl. LM Studio)
    "enable_thinking": False,                     # alternate flag name
}

# Interrupt flag — set when user says stop/cancel while LLM or TTS is active
_interrupt_event = threading.Event()

SYSTEM_PROMPT = (

    "/no_think\n"

    # ── IDENTITY ──────────────────────────────────────────────────────────────
    "You are Buddy, a smart, proactive, and highly capable personal voice assistant. "
    "You run locally on the user's machine and speak through a text-to-speech engine. "
    "You are not a chatbot. You are an intelligent agent that thinks, researches, "
    "reasons, and acts on behalf of the user.\n\n"

    # ── CRITICAL KNOWLEDGE RULE ───────────────────────────────────────────────
    "CRITICAL KNOWLEDGE RULE — READ THIS FIRST:\n"
    "Your training data ends in early 2024. You must NEVER answer the following "
    "from memory alone — always use web search data when it is provided:\n"
    "- Any price, value, rate, or cost (crypto, stocks, currency, goods)\n"
    "- Current weather or forecasts\n"
    "- News, recent events, or anything that happened after 2023\n"
    "- Sports scores, standings, or results\n"
    "- Who currently holds any position, title, or record\n"
    "- Software versions, product specs, or release dates\n"
    "- Population figures, statistics, or rankings\n"
    "When web search results are provided, use them as your ONLY source for these topics. "
    "If no search results are provided and the question is time-sensitive, say you do not "
    "have current data rather than guessing from memory.\n\n"

    # ── STT CONTEXT AWARENESS ─────────────────────────────────────────────────
    "STT CONTEXT AWARENESS AND CLARIFICATION:\n"
    "- The user's messages are transcribed by speech recognition which is imperfect.\n"
    "- Words may be misheard, merged, or slightly wrong. Use conversation history "
    "to infer the most likely intended meaning.\n"
    "- If a message seems garbled but you can guess the intent, answer what was likely "
    "meant without commenting on the transcription.\n"
    "- If the meaning is completely unclear and context does not help, ask ONE short "
    "specific question to clarify — for example: 'Did you mean X or Y?' or 'Could you "
    "repeat that? I didn't quite catch it.'\n"
    "- If you need more information to give a useful answer, ask for it proactively. "
    "For example if someone asks to compare products without specifying which ones, "
    "ask which ones they mean.\n"
    "- Never pretend to understand something you didn't. A short clarifying question "
    "is always better than a wrong answer.\n\n"


    "CORE BEHAVIOUR:\n"
    "- You are proactive. If you notice something important the user did not ask about "
    "but would clearly want to know, mention it briefly.\n"
    "- You are honest. If you are not sure about something, say so clearly. "
    "Never guess or present uncertain information as fact.\n"
    "- You are adaptive. Match the user's tone — casual in conversation, "
    "precise when they need technical or factual answers.\n"
    "- You have initiative. If a question is ambiguous, make a reasonable assumption, "
    "answer based on that, and briefly state your assumption so the user can correct you.\n"
    "- You ask follow-up questions when they would genuinely improve your answer, "
    "but never ask more than one follow-up at a time.\n\n"

    # ── REASONING ─────────────────────────────────────────────────────────────
    "REASONING:\n"
    "- Think step by step internally before answering complex questions, "
    "but only speak the conclusion, not your reasoning process.\n"
    "- For multi-part questions, address each part in order.\n"
    "- If a question contains a false premise, gently correct it before answering.\n"
    "- When comparing options, give a clear recommendation rather than just listing pros and cons.\n\n"

    # ── RESPONSE LENGTH ───────────────────────────────────────────────────────
    "RESPONSE LENGTH:\n"
    "- Match length to what the question genuinely needs.\n"
    "- Simple facts: one sentence only. Example: 'Mount Everest is eight thousand eight hundred forty-nine metres tall.'\n"
    "- Normal questions: two to three sentences.\n"
    "- Summaries, overviews, news roundups, how-to questions, explanations: as many sentences as needed to be complete and useful. Do not artificially shorten these.\n"
    "- Always finish every sentence completely. Never trail off or stop mid-sentence.\n"
    "- Never pad short answers with filler. Never truncate long answers just to be brief.\n"
    "- If an answer would take more than about thirty seconds to speak, give the most important points first, then offer to go deeper.\n\n"

    # ── PROACTIVE BEHAVIOUR ───────────────────────────────────────────────────
    "PROACTIVE BEHAVIOUR:\n"
    "- If the user asks about something with important dimensions they did not mention, "
    "briefly cover the most important one they missed.\n"
    "- If the user's question implies a decision, help them make it rather than presenting "
    "information neutrally.\n"
    "- If you notice an error or misunderstanding, correct it kindly and briefly before answering.\n"
    "- If a topic is time-sensitive and your information may be outdated, say so proactively.\n"
    "- If the user seems to be working toward a goal across multiple messages, "
    "remember the context and connect your answers to that goal.\n\n"

    # ── FOLLOW-UP QUESTIONS ───────────────────────────────────────────────────
    "FOLLOW-UP QUESTIONS:\n"
    "- Ask a follow-up question when the request is ambiguous or more detail would "
    "significantly improve your answer.\n"
    "- Never ask a follow-up for simple factual queries.\n"
    "- Never ask more than one question at a time.\n"
    "- Keep follow-up questions short and specific.\n\n"

    # ── VOICE AND FORMAT RULES ────────────────────────────────────────────────
    "VOICE AND FORMAT RULES — these are absolute and must never be broken:\n"
    "- Write in plain spoken English only. Your output goes directly to a speech engine.\n"
    "- Never use markdown: no asterisks, hashes, underscores, backticks, or tildes.\n"
    "- Never use bullet points, numbered lists, or dashes as list markers.\n"
    "- Never use tables, code blocks, or section headers.\n"
    "- Never write URLs, file paths, or raw code.\n"
    "- If you need to list things, speak them naturally: 'The options are X, Y, and Z.'\n"
    "- If asked for code, explain it verbally instead.\n"
    "- Write 'for example' instead of e.g. and 'that is' instead of i.e.\n"
    "- Use commas and full stops naturally so the speech engine pauses correctly.\n"
    "- Use natural contractions: I'm, you're, it's, don't, can't, won't.\n\n"

    # ── TONE ──────────────────────────────────────────────────────────────────
    "TONE:\n"
    "- Be direct and confident. Do not hedge unnecessarily.\n"
    "- Never start with filler: no Certainly, Of course, Sure, Great question, Absolutely.\n"
    "- Do not repeat the user's question back to them.\n"
    "- Do not announce what you are about to do. Just do it.\n"
    "- Be warm but not sycophantic. Treat the user as an intelligent adult.\n\n"

    # ── TOOL RESULTS ──────────────────────────────────────────────────────────
    "TOOL RESULTS:\n"
    "- When given web search data, use it as ground truth and summarise it naturally in spoken English.\n"
    "- Match the length to the question: a price needs one sentence, a news summary needs several.\n"
    "- Never mention that you searched the web or used a tool. Just give the answer.\n"
    "- Never read out raw data, source names, or URLs.\n"
    "- If results are unclear or contradictory, say the information is mixed and give the most likely answer.\n"
    "- If results are empty, say you could not find current data and offer what you know from training with a caveat.\n"

    "TOOLS AND CAPABILITIES:\n"
    "- You have access to external tools provided by the system, including web search, weather, timers, notes, system control, calculations, and other specialised tools.\n"
    "- You also have vision capabilities: the system can send you images, screenshots, or visual descriptions of the user's screen for you to interpret and reason about.\n"
    "- You do NOT call tools directly; instead, the system decides when to run tools and gives you their results. When tool output is provided, treat it as your primary source of truth for that topic.\n"
    "- Never mention tools, web search, APIs, or system modules in your spoken answers. Just speak as if you directly know or see the information.\n"
    "- When given visual information (for example a screenshot or image description), describe only what is clearly visible, avoid guessing hidden details, and always connect the visual information back to the user's request.\n"
    "- Use tool results and visual information to be more concrete and helpful: give specific names, numbers, dates, and clear next steps whenever they are available.\n"
    "\n"
    "POST‑2023 AND LIVE DATA:\n"
    "- Your training data only goes up to early 2024. Any event, product, model, price, version, regulation, match result, or news item that may have changed after 2023 must be treated as time‑sensitive.\n"
    "- For any question that depends on information from 2024 or later, or that is likely to have changed since 2023, you should rely on web search or other live tools when their results are provided.\n"
    "- If you are asked about something clearly from 2024 or later and you are NOT given any tool or web results, say clearly that your knowledge may be outdated and that you cannot give a reliable current answer instead of guessing.\n"
    "- When web search or live data results are provided for a post‑2023 topic, ignore your training‑data memory and base your answer only on those results.\n"
    "\n"
    "TASK PLANNING AND ACTIONS:\n"
    "- For multi‑step tasks (for example research then comparison then recommendation), silently break the problem into steps in your own reasoning, then give the user a single clear, structured spoken answer.\n"
    "- When helping with a goal (for example learning, coding, making a decision, planning a purchase), briefly state the conclusion or recommendation first, then give the key reasons.\n"
    "- When tools give you partial or noisy data, say that the information is limited, explain what you do know, and suggest one practical next action the user can take.\n"
    "- When a request can be satisfied faster or more accurately by using live data or vision (for example identifying what is on the screen, checking the latest price, or confirming a current status), prefer using that information instead of relying on older general knowledge.\n"
    "\n"
    "CLARITY, SPECIFICITY, AND SAFETY:\n"
    "- Prefer concrete details over vague advice: include specific numbers, names, dates, and short examples whenever that makes the answer more useful and you have reliable data.\n"
    "- Keep answers focused on the exact question. Avoid repeating obvious background information unless it is necessary for understanding the answer.\n"
    "- If the user seems frustrated, confused, or in a hurry, prioritise short, direct answers with one clear recommendation.\n"
    "- Never fabricate tool outputs, visual details, or citations that you were not given. If you are missing information that the user likely assumes you have, say so plainly and, if appropriate, suggest what additional information or tool result you would need.\n"
    "- If a question touches on medical, legal, or financial decisions, give careful, conservative guidance, emphasise that you are not a licensed professional, and suggest consulting a qualified expert for critical decisions.\n"
    "\n"
    "PROACTIVE ASSISTANCE:\n"
    "- When the user asks an open question where several interpretations are possible, pick the most likely useful one, answer it clearly, and briefly mention the assumption you made so they can correct you.\n"
    "- If you notice that the user is working on an ongoing project or topic across multiple turns, bring in relevant context from earlier in the conversation without being repetitive, and remind them of previous decisions or constraints when useful.\n"
    "- When the user asks about something that clearly benefits from follow‑up (for example choosing between options, planning steps, or configuring software), end your answer with one short, optional follow‑up question that would most improve your help.\n"
    "\n"

)

_history      = [{"role": "system", "content": SYSTEM_PROMPT}]
_history_lock = threading.Lock()
MAX_HISTORY_TURNS = 50   # system + 10 user/assistant pairs = safe for 2048 ctx

# Session conversation log for memory compression at shutdown
# Contains only real user/assistant turns (no system, no tool prompts)
_session_turns: list = []
_session_lock  = threading.Lock()

llm = OpenAI(base_url=config.LLM_API_URL, api_key="no-key-needed")
# Wire LLM into web search tool for query optimisation and summarisation
_ws_set_llm(llm, MODEL_NAME, _NO_THINKING)

# Sidebar: wire TTS queue into timer tool so alarms can speak
# (done after tts_worker.start() in main() — see _wire_tools_post_start)


def _build_messages(user_prompt: str) -> list:
    """
    Build the message list for the LLM call.
    All system content (prompt + date + memory) is merged into ONE system message
    at position 0 — required by Qwen and most other model chat templates.
    Memory block is context-only and never reaches TTS.
    """
    now = time.strftime("%A, %d %B %Y, %H:%M")

    # Build combined system content
    system_parts = [SYSTEM_PROMPT]
    system_parts.append(f"\nCurrent date and time: {now}.")

    mem_block = mem.build_memory_block()
    if mem_block:
        system_parts.append(f"\n{mem_block}")

    combined_system = "".join(system_parts)

    with _history_lock:
        history_turns = [m for m in _history if m["role"] != "system"]

    messages = (
        [{"role": "system", "content": combined_system}]
        + history_turns
        + [{"role": "user", "content": user_prompt}]
    )
    return messages

# ============================================================================
# SEARCH DECISION ENGINE
#
# Tiered approach — fastest first:
#   Tier 1: NEVER_SEARCH  — conversation, stable timeless facts → LLM only
#   Tier 2: ALWAYS_SEARCH — prices, weather, news, current roles → always search
#   Tier 3: LLM_CLASSIFIER — ambiguous cases → ask LLM to decide (fast, no stream)
#
# The LLM classifier is a tiny non-streaming call used only when the regex
# tiers don't give a clear answer. It returns "search" or "skip" in one token.
# ============================================================================

# Stable, timeless facts — LLM training data is reliable, no search needed
_NEVER_SEARCH = re.compile(r"""
    # Pure conversation
    \b(thank\s*you|thanks|hello|hi\b|hey\b|bye\b|goodbye|
       how\s*are\s*you|what\s*can\s*you\s*do|
       tell\s*a\s*joke|what\s*is\s*your\s*name|
       who\s*made\s*you|who\s*are\s*you|
       set\s*a\s*timer|remind\s*me)\b |

    # Geography — mountain heights, river lengths, ocean depths don't change
    \b(tallest\s*mountain|highest\s*mountain|mount\s*everest|
       longest\s*river|deepest\s*(ocean|point|lake|trench)|mariana\s*trench|
       largest\s*country|smallest\s*country|capital\s*of|capital\s*city|
       how\s*far\s*is|distance\s*between)\b |

    # Language and culture — stable
    \b(what\s*language|which\s*language|do\s*(people|they)\s*(speak|talk)|
       official\s*language|spoken\s*in|language\s*of|
       what\s*religion|main\s*religion|what\s*alphabet|writing\s*system)\b |

    # Physics and science — constants don't change
    \b(speed\s*of\s*light|speed\s*of\s*sound|boiling\s*point|melting\s*point|
       freezing\s*point|atomic\s*number|periodic\s*table|
       how\s*does\s*(gravity|photosynthesis|evolution|dna|rna|fusion|fission)\s*work|
       what\s*is\s*(gravity|photosynthesis|mitosis|osmosis|relativity))\b |

    # History — completed unchanging events
    \b(world\s*war\s*(one|two|1|2|i\b|ii\b)|
       who\s*invented|who\s*discovered|who\s*wrote|who\s*painted|who\s*composed|
       when\s*(was|did|were)\s*(the\s*)?(moon\s*landing|berlin\s*wall|titanic|
       french\s*revolution|american\s*revolution|renaissance|industrial\s*revolution))\b |

    # Math and conversions
    \b(how\s*many\s*(centimetres|inches|feet|metres|miles|kilometres|
       seconds|minutes|hours|days|weeks|months)\s*in|
       what\s*is\s*\d+\s*(plus|minus|times|divided|percent\s*of)|
       square\s*root\s*of|convert\s*\d+)\b |

    # Definitions
    \b(what\s*does\s*\w+\s*mean|definition\s*of|meaning\s*of|
       translate\s*\w+\s*to|how\s*do\s*you\s*say)\b
""", re.IGNORECASE | re.VERBOSE)

# Live data — always needs a fresh web search, no exceptions
_ALWAYS_SEARCH = re.compile(r"""
    # Financial — prices change by the second
    \b(price\s*of|current\s*price|how\s*much\s*(is|does|cost)|
       stock\s*price|share\s*price|market\s*cap|
       crypto|cryptocurrency|bitcoin|ethereum|btc|eth|
       exchange\s*rate|forex|usd|eur|gbp|
       inflation\s*rate|interest\s*rate|mortgage\s*rate)\b |

    # Weather — always live
    \b(weather|forecast|will\s*it\s*rain|is\s*it\s*(raining|snowing|sunny|cloudy)|
       temperature\s*(today|tomorrow|outside)|
       degrees\s*(celsius|fahrenheit))\b |

    # Breaking news and current events
    \b(latest\s*news|breaking\s*news|what.*happening\s*(now|today)|
       news\s*today|headlines\s*today|current\s*events|
       what\s*happened\s*(today|yesterday|this\s*week)|
       top\s*stories|news\s*about)\b |

    # Current officeholders — changes with elections
    \b(current\s*president|current\s*prime\s*minister|current\s*chancellor|
       current\s*ceo|current\s*king|current\s*queen|current\s*pope|
       who\s*is\s*(currently|now)\s*(the\s*)?(president|pm|chancellor|ceo|leader)|
       who\s*leads|who\s*runs|who\s*is\s*in\s*charge\s*of)\b |

    # Live sports
    \b(live\s*score|final\s*score|match\s*result|game\s*result|
       league\s*table|standings\s*(today|now)|who\s*won\s*(last|the)\s*(game|match|race)|
       latest\s*(game|match|result))\b |

    # Explicit recency signals
    \b(right\s*now|at\s*this\s*moment|as\s*of\s*today|
       this\s*(morning|afternoon|evening)|
       today['s]?\s*(price|rate|value|score|news|result))\b
""", re.IGNORECASE | re.VERBOSE)

# Ambiguous queries that MIGHT need search — send to LLM classifier
_MAYBE_SEARCH = re.compile(r"""
    \b(who\s*is|what\s*is\s*the\s*(latest|current|new)|
       is\s*(it\s*still|he\s*still|she\s*still|they\s*still)|
       has\s*(anything|something)\s*changed|
       what\s*(happened|changed)|
       when\s*did|did\s*they|are\s*they\s*still|
       how\s*many\s*people|population\s*of|gdp\s*of|
       what\s*version|latest\s*version|new\s*model|
       is\s*\w+\s*still\s*(alive|active|president|ceo|running))\b
""", re.IGNORECASE | re.VERBOSE)


def _llm_should_search(user_text: str) -> bool:
    """
    Ask the LLM whether this query needs a web search.
    Used only for ambiguous cases that the regex tiers can't classify.
    Non-streaming, single-token response for minimal latency (~200ms).
    """
    try:
        resp = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": (
                    f"Does answering this question require current real-world data "
                    f"(news, live prices, recent events, current officeholders, "
                    f"recent sports results, or anything that may have changed since 2023)?\n"
                    f"Question: {user_text}\n"
                    f"Reply with exactly one word: search or skip"
                )
            }],
            max_tokens=3,
            temperature=0.0,
            stream=False,
            extra_body=_NO_THINKING,
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        result = answer.startswith("search")
        print(f"[SEARCH CLASSIFIER] '{user_text[:50]}' → {answer} → {'search' if result else 'skip'}", flush=True)
        return result
    except Exception as e:
        print(f"[SEARCH CLASSIFIER] Error: {e} — defaulting to skip", flush=True)
        return False


def _needs_search(text: str) -> bool:
    """
    Three-tier decision:
    1. NEVER_SEARCH match → False (no search)
    2. ALWAYS_SEARCH match → True (always search)
    3. MAYBE_SEARCH match → ask LLM classifier
    4. Default → False (LLM answers from training)
    """
    if _NEVER_SEARCH.search(text):
        return False
    if _ALWAYS_SEARCH.search(text):
        return True
    if _MAYBE_SEARCH.search(text):
        return _llm_should_search(text)
    return False


# ============================================================================
# TEXT HELPERS
# ============================================================================

def _split_sentences(text: str) -> list:
    """
    Split streamed LLM text into sentences for TTS.
    Only splits on sentence-ending punctuation followed by a capital letter.
    Requires completed sentence to be at least 30 chars to avoid premature
    splits on abbreviations (U.S.), decimals ($67,000.50), or short fragments.
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
            result[-1] = result[-1] + ' ' + p  # merge short fragment into previous
        else:
            result.append(p)
    return result


def _normalize_for_speech(text: str) -> str:
    """
    Convert written text into natural spoken English for TTS.
    Handles currencies, units, symbols, abbreviations, HTML entities,
    and anything that would sound wrong when read aloud.
    """

    # ── HTML entities (from web scraping) ────────────────────────
    text = text.replace('&amp;', 'and').replace('&lt;', 'less than') \
               .replace('&gt;', 'greater than').replace('&nbsp;', ' ') \
               .replace('&#x27;', "'").replace('&quot;', '"') \
               .replace('&#39;', "'")

    # ── Strip markdown and formatting ────────────────────────────
    text = re.sub(r'[*#`_~\[\]|\\]', '', text)
    text = re.sub(r'\bhttps?://\S+', 'a link', text)

    # ── Remove redundant "the USD" / "the US dollars USD" patterns ─
    # These appear when LLM already wrote "dollars" and normalizer adds more
    text = re.sub(r'\bthe\s+(USD|EUR|GBP|BTC|ETH|JPY)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(USD|EUR|GBP|BTC|ETH|JPY)\s+(US dollars|euros|pounds|bitcoin|yen)\b',
                  lambda m: m.group(2), text, flags=re.IGNORECASE)
    # Remove trailing currency codes after already-written currency words
    text = re.sub(r'\b(dollars?|euros?|pounds?|yen|bitcoin)\s+(USD|EUR|GBP|JPY|BTC)\b',
                  lambda m: m.group(1), text, flags=re.IGNORECASE)

    # ── Currencies — symbol+number → words (only when symbol present) ──
    def _currency(m):
        symbol  = m.group(1)
        number  = m.group(2).replace(',', '')
        suffix  = m.group(3) or ''
        names   = {'$': 'US dollars', '€': 'euros', '£': 'pounds',
                   '¥': 'yen', '₿': 'bitcoin', 'CHF': 'Swiss francs'}
        unit    = names.get(symbol, symbol)
        try:
            spoken = _number_to_words(float(number))
        except Exception:
            spoken = number
        suffix_map = {'k': ' thousand', 'K': ' thousand',
                      'm': ' million',  'M': ' million',
                      'b': ' billion',  'B': ' billion',
                      'T': ' trillion', 't': ' trillion'}
        return f"{spoken}{suffix_map.get(suffix, '')} {unit}"

    text = re.sub(r'([\$€£¥₿]|CHF)\s*([\d,]+(?:\.\d+)?)([kKmMbBtT]?)', _currency, text)

    # Number then currency code: 95000 USD → ninety-five thousand US dollars
    _code_map = {'USD': 'US dollars', 'EUR': 'euros', 'GBP': 'pounds',
                 'JPY': 'yen', 'BTC': 'bitcoin', 'ETH': 'ethereum', 'CHF': 'Swiss francs'}
    def _num_code(m):
        try:
            spoken = _number_to_words(float(m.group(1).replace(',', '')))
        except Exception:
            spoken = m.group(1)
        return f"{spoken} {_code_map.get(m.group(2).upper(), m.group(2))}"
    text = re.sub(r'\b([\d,]+(?:\.\d+)?)\s*(USD|EUR|GBP|JPY|BTC|ETH|CHF)\b', _num_code, text)

    # ── Percentages ───────────────────────────────────────────────
    text = re.sub(r'([\d,]+(?:\.\d+)?)\s*%',
        lambda m: f"{_number_to_words(float(m.group(1).replace(',','')))  } percent", text)

    # ── Large plain numbers with commas (e.g. 1,234,567) ─────────
    text = re.sub(r'\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\b',
        lambda m: _number_to_words(float(m.group(1).replace(',', ''))), text)

    # ── Units ─────────────────────────────────────────────────────
    units = {
        r'\bkm/h\b': 'kilometres per hour',   r'\bmph\b':  'miles per hour',
        r'\bm/s\b':  'metres per second',      r'\bkm\b':   'kilometres',
        r'\bcm\b':   'centimetres',            r'\bmm\b':   'millimetres',
        r'\bkg\b':   'kilograms',              r'\bg\b(?=\s)': 'grams',
        r'\blbs?\b': 'pounds',                 r'\boz\b':   'ounces',
        r'\bml\b':   'millilitres',            r'\bkWh\b':  'kilowatt hours',
        r'\bkW\b':   'kilowatts',              r'\bGHz\b':  'gigahertz',
        r'\bMHz\b':  'megahertz',              r'\bGB\b':   'gigabytes',
        r'\bTB\b':   'terabytes',              r'\bMB\b':   'megabytes',
        r'\bKB\b':   'kilobytes',              r'\b°C\b':   'degrees Celsius',
        r'\b°F\b':   'degrees Fahrenheit',     r'\b°\b':    'degrees',
        r'\bm²\b':   'square metres',          r'\bft²\b':  'square feet',
        r'\bsq\s*ft\b': 'square feet',         r'\bsq\s*m\b':  'square metres',
    }
    for pattern, replacement in units.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # ── Common abbreviations ──────────────────────────────────────
    abbrevs = {
        r'\bUSA\b': 'the United States', r'\bUS\b': 'the US',
        r'\bUK\b': 'the UK',             r'\bEU\b': 'the European Union',
        r'\bUN\b': 'the United Nations', r'\bNATO\b': 'NATO',
        r'\bGDP\b': 'GDP',               r'\bCEO\b': 'CEO',
        r'\bAI\b': 'AI',                 r'\bAPI\b': 'API',
        r'\bURL\b': 'link',              r'\be\.g\.\b': 'for example',
        r'\bi\.e\.\b': 'that is',        r'\betc\.\b': 'and so on',
        r'\bvs\.\b': 'versus',           r'\bapprox\.?\b': 'approximately',
        r'\bDr\.\b': 'Doctor',           r'\bMr\.\b': 'Mister',
        r'\bMrs\.\b': 'Missus',          r'\bMs\.\b': 'Miss',
        r'\bProf\.\b': 'Professor',      r'\bSt\.\b': 'Saint',
        r'\bno\.\b': 'number',           r'\bvol\.\b': 'volume',
    }
    for pattern, replacement in abbrevs.items():
        text = re.sub(pattern, replacement, text)

    # ── 24h time → 12h spoken ─────────────────────────────────────
    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        period = 'AM' if h < 12 else 'PM'
        h12    = h % 12 or 12
        return f"{h12}:{mi:02d} {period}"
    text = re.sub(r'\b([01]?\d|2[0-3]):([0-5]\d)\b', _time, text)

    # ── Ordinals ──────────────────────────────────────────────────
    ordinals = {
        '1st': 'first',    '2nd': 'second',   '3rd': 'third',
        '4th': 'fourth',   '5th': 'fifth',     '6th': 'sixth',
        '7th': 'seventh',  '8th': 'eighth',    '9th': 'ninth',
        '10th': 'tenth',   '11th': 'eleventh', '12th': 'twelfth',
        '13th': 'thirteenth', '20th': 'twentieth', '21st': 'twenty-first',
        '22nd': 'twenty-second', '23rd': 'twenty-third', '30th': 'thirtieth',
        '31st': 'thirty-first', '100th': 'hundredth',
    }
    for num, word in ordinals.items():
        text = re.sub(rf'\b{num}\b', word, text, flags=re.IGNORECASE)

    # ── Clean up whitespace ───────────────────────────────────────
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s([.,!?])', r'\1', text)
    return text.strip()


def _number_to_words(n: float) -> str:
    """Convert a number to spoken English words."""
    try:
        if n != int(n):
            # Decimal: say the integer part + "point" + digits
            int_part  = int(n)
            dec_part  = str(n).split('.')[1]
            return f"{_number_to_words(int_part)} point {' '.join(_number_to_words(int(d)) for d in dec_part)}"
        n = int(n)
        if n < 0:
            return f"minus {_number_to_words(-n)}"
        if n == 0:  return 'zero'
        if n == 1:  return 'one'
        if n == 2:  return 'two'
        if n == 3:  return 'three'
        if n == 4:  return 'four'
        if n == 5:  return 'five'
        if n == 6:  return 'six'
        if n == 7:  return 'seven'
        if n == 8:  return 'eight'
        if n == 9:  return 'nine'
        if n == 10: return 'ten'
        if n == 11: return 'eleven'
        if n == 12: return 'twelve'
        if n == 13: return 'thirteen'
        if n == 14: return 'fourteen'
        if n == 15: return 'fifteen'
        if n == 16: return 'sixteen'
        if n == 17: return 'seventeen'
        if n == 18: return 'eighteen'
        if n == 19: return 'nineteen'
        tens = ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
        if n < 100:
            t, u = divmod(n, 10)
            return tens[t] + (f'-{_number_to_words(u)}' if u else '')
        if n < 1000:
            h, r = divmod(n, 100)
            return f"{_number_to_words(h)} hundred{' ' + _number_to_words(r) if r else ''}"
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


# keep old name as alias so nothing else breaks
_clean_for_speech = _normalize_for_speech


# ============================================================================
# LLM CALL  (streaming, sentence by sentence to TTS)
# ============================================================================

def _call_llm(prompt: str, tts_queue: Queue) -> str:
    """Send prompt to LLM, stream tokens, push complete sentences to TTS.
    Stops immediately if _interrupt_event is set."""
    _interrupt_event.clear()

    messages = _build_messages(prompt)

    full_response   = ""
    buffer          = ""
    was_interrupted = False
    _in_think_block = False   # track if we're inside a <think>...</think> block

    try:
        stream = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
            extra_body=_NO_THINKING,
        )

        for chunk in stream:
            if _stopped_event.is_set() or _interrupt_event.is_set():
                was_interrupted = True
                break
            delta = chunk.choices[0].delta.content
            if not delta:
                continue

            # Strip <think>...</think> blocks that some models emit even when
            # thinking=False is set (safety net for non-compliant models)
            if '<think>' in delta:
                _in_think_block = True
            if _in_think_block:
                if '</think>' in delta:
                    _in_think_block = False
                    # Keep only what comes after </think>
                    delta = delta.split('</think>', 1)[-1]
                else:
                    continue  # discard token entirely

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
                    cleaned = _normalize_for_speech(sentence)
                    if cleaned:
                        tts_queue.put(cleaned)
                buffer = sentences[-1]

            if was_interrupted:
                break

        if not was_interrupted:
            remainder = _normalize_for_speech(buffer)
            if remainder:
                tts_queue.put(remainder)
        else:
            while not tts_queue.empty():
                try:
                    tts_queue.get_nowait()
                except Exception:
                    break
            print("\n[INTERRUPT] Stopped.", flush=True)

    except Exception as e:
        print(f"[LLM] Error: {e}", flush=True)
        full_response = "Sorry, I could not reach the language model."
        if not was_interrupted:
            tts_queue.put(full_response)

    full_response = _normalize_for_speech(full_response)

    # Store in in-session history for context window
    with _history_lock:
        if full_response.strip():
            _history.append({"role": "user",      "content": prompt})
            _history.append({"role": "assistant",  "content": full_response})
        # Trim: keep only last MAX_HISTORY_TURNS pairs (not system messages)
        non_system = [m for m in _history if m["role"] != "system"]
        if len(non_system) > MAX_HISTORY_TURNS * 2:
            excess = len(non_system) - MAX_HISTORY_TURNS * 2
            _history[:] = [m for m in _history if m["role"] == "system"] + non_system[excess:]

    # Log to session turns for memory compression at shutdown
    if full_response.strip() and not was_interrupted:
        with _session_lock:
            _session_turns.append({"role": "user",      "content": prompt})
            _session_turns.append({"role": "assistant",  "content": full_response})

    return full_response


# ============================================================================
# PIPELINE — decision logic
# ============================================================================

def _interrupted() -> bool:
    """Quick check: has user said stop since the last LLM call started?"""
    return _interrupt_event.is_set()


def _process(user_text: str, tts_queue: Queue) -> str:
    """
    Full tool dispatch pipeline with interrupt checks at every stage.

    Priority order:
    1. Interrupt already set → bail out immediately
    2. Pure conversation     → LLM directly
    3. Instant tools (datetime, calculator, timer, notes, system)
    4. Weather (dedicated API — faster + more accurate than web scraping)
    5. Time-sensitive queries → forced web search
    6. Keyword-matched tools
    7. LLM classifier for ambiguous queries
    8. LLM directly (fallback)

    Interrupt is checked:
    - Before starting
    - After each tool run
    - Inside _call_llm (per-token during streaming)
    - Inside tts_worker (every 50ms during playback)
    """

    # ── Pre-check: already interrupted? ──────────────────────────────────────
    if _interrupted():
        return ""

    # ── Update last_response in side tools (for "save that" / "copy that") ──
    def _update_last(resp: str):
        _notes_mod.last_response   = resp
        _system_mod.SystemTool.last_response = resp

    # ── 1. Pure conversation shortcut ─────────────────────────────────────────
    if _NEVER_SEARCH.search(user_text):
        resp = _call_llm(user_text, tts_queue)
        _update_last(resp)
        return resp

    # ── 2. Find best matching tool ────────────────────────────────────────────
    tool = find_tool(user_text)

    # ── 3. Instant local tools — no network, no latency ──────────────────────
    INSTANT_TOOLS = ("datetime", "calculator", "timer", "notes", "system")

    if tool and tool.name in INSTANT_TOOLS:
        print(f"[TOOL] Using: {tool.name}", flush=True)

        # Wire TTS queue into timer so alarms speak aloud
        if tool.name == "timer":
            _timer_mod.set_tts_queue(tts_queue)

        # Wire last response into system tool for "copy that"
        if tool.name == "system":
            with _history_lock:
                last_assistant = next(
                    (m["content"] for m in reversed(_history) if m["role"] == "assistant"),
                    ""
                )
            tool.last_response = last_assistant

        result = tool.run(user_text)

        if _interrupted():
            return ""

        # Instant tools speak their result directly — no LLM reformulation needed
        # (keeps latency near zero for timer confirmations, calc answers, etc.)
        spoken = _normalize_for_speech(result)
        tts_queue.put(spoken)
        _update_last(spoken)
        return spoken

    # ── 4. Weather — dedicated free API ──────────────────────────────────────
    if tool and tool.name == "weather":
        print("[TOOL] Using: weather (Open-Meteo API)", flush=True)
        with _history_lock:
            tool.set_history(list(_history))
        result = tool.run(user_text)

        if _interrupted():
            return ""

        spoken = _normalize_for_speech(result)
        tts_queue.put(spoken)
        _update_last(spoken)
        return spoken
        
    # ── 4b. Screen / vision tool ────────────────────────────────────────────
    if tool and tool.name == "screen":
        print("[TOOL] Using: screen (vision)", flush=True)
        with _history_lock:
            tool.set_history(list(_history))
        result = tool.run(user_text)

        if _interrupted():
            return ""

        spoken = _normalize_for_speech(result)
        tts_queue.put(spoken)
        _update_last(spoken)
        return spoken


    # ── 5. Force web search for live data ────────────────────────────────────
    if _needs_search(user_text):
        print("[TOOL] Force web search — live data query", flush=True)
        tool = _web_search

    # ── 6. Keyword-matched web search ────────────────────────────────────────
    elif tool and tool.name == "web_search":
        print("[TOOL] Using: web_search (keyword match)", flush=True)

    # ── 7. LLM decides for ambiguous queries ─────────────────────────────────
    else:
        resp = _call_llm(user_text, tts_queue)
        _update_last(resp)
        return resp

    # ── Web search pipeline ───────────────────────────────────────────────────
    if _interrupted():
        return ""

    # Pass current history for context-aware query optimisation
    if hasattr(tool, "set_history"):
        with _history_lock:
            tool.set_history(list(_history))

    print("[TOOL] Searching...", flush=True)
    try:
        tool_result = tool.run(user_text)
        print(f"[TOOL] Result: {tool_result[:150].replace(chr(10), ' ')}", flush=True)
    except Exception as e:
        tool_result = f"Search failed: {e}"
        print(f"[TOOL] Error: {e}", flush=True)

    if _interrupted():
        return ""

    broad_request = bool(re.search(
        r'\b(summarize|summarise|summary|overview|roundup|round.?up|'
        r'top\s*\d+|list\s*(of|the)|all\s*the|everything\s*about|'
        r'news\s*(today|tonight|this\s*week)|global\s*news|world\s*news|'
        r'headline|headlines|what.*happening|catch\s*me\s*up|brief\s*me)\b',
        user_text, re.IGNORECASE
    ))

    length_instruction = (
        "Give a thorough spoken summary covering all the key topics from the data above. "
        "Speak naturally, use as many sentences as needed, and finish every sentence completely."
        if broad_request else
        "Give a complete, direct answer in one to three spoken sentences using the data above. "
        "Always finish every sentence completely."
    )

    prompt = (
        f"SEARCH RESULT (already summarised — speak this naturally):\n"
        f"{tool_result}\n\n"
        f"Original question: {user_text}\n\n"
        f"Deliver this answer in natural spoken English. "
        f"Do NOT re-summarise or change the facts. "
        f"Do NOT mention searching, tools, or data sources. "
        f"{length_instruction}"
    )

    resp = _call_llm(prompt, tts_queue)
    _update_last(resp)
    return resp


# Interrupt words — saying any of these stops TTS and LLM immediately
_INTERRUPT_RE = re.compile(
    r'^\s*(stop|cancel|shut up|enough|quiet|pause|nevermind|never mind|'
    r'forget it|stop talking|be quiet|silence)\s*[.!?]?\s*$',
    re.IGNORECASE
)

# Words that are likely STT noise / false triggers — ignore these
_NOISE_RE = re.compile(
    r'^\s*(uh|um|ah|oh|hm|hmm|mm|yeah|okay|ok|right|sure|yep|nope|'
    r'the|a|and|or|but|so|like|you know)\s*[.!?]?\s*$',
    re.IGNORECASE
)


def _pipeline_loop(stt_queue: Queue, tts_queue: Queue):
    while not _stopped_event.is_set():
        try:
            text = stt_queue.get(timeout=0.5)
        except Empty:
            continue

        if not text or not text.strip():
            continue

        # Ignore STT noise
        if _NOISE_RE.match(text):
            print(f"[STT] Ignored noise: '{text}'", flush=True)
            continue

        # ── Interrupt detection ───────────────────────────────────
        if _INTERRUPT_RE.match(text):
            print(f"\n[INTERRUPT] User said: '{text}' — stopping.", flush=True)
            _interrupt_event.set()
            tts_worker.IS_SPEAKING.clear()
            # Drain TTS queue
            while not tts_queue.empty():
                try:
                    tts_queue.get_nowait()
                except Exception:
                    break
            continue

        ts = time.strftime("%H:%M:%S")
        print(f"\n[{ts}] User  : {text}", flush=True)
        session.log_user(text)

        # ── Context-aware STT correction ─────────────────────────
        words = text.split()
        # Heuristics for likely garbled input:
        # - Very short (≤2 words) with history available
        # - Contains phonetically weird combos that suggest mishearing
        # - Has filler-like fragments
        garble_hints = [
            len(words) <= 2 and len(_history) > 2,
            any(c in text for c in ['...', '??', '  ']),
            # Likely mis-transcribed: random short words that don't form a question
            len(words) <= 4 and not re.search(
                r'\b(what|who|where|when|why|how|is|are|can|do|does|tell|show|give)\b',
                text, re.IGNORECASE
            ) and len(_history) > 2,
        ]
        likely_garbled = any(garble_hints)

        if likely_garbled:
            augmented = (
                f"[STT note: this voice transcription may be imperfect. "
                f"Use conversation history to infer what the user most likely meant.]\n"
                f"Transcribed text: {text}"
            )
        else:
            augmented = text

        # ── Process ───────────────────────────────────────────────
        # _process runs on the pipeline thread. The STT thread continues
        # polling — if the user says a stop word while Buddy is thinking,
        # searching, or speaking, _interrupt_event fires immediately and
        # _process checks it between every major stage.
        print(f"[{ts}] Buddy : ", end="", flush=True)
        _interrupt_event.clear()   # fresh slate for this turn

        response = _process(augmented, tts_queue)

        if response:
            print(response, flush=True)
            session.log_assistant(response)
        else:
            print("[interrupted]", flush=True)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 60)
    print("  BuddyQ Voice Assistant")
    print("=" * 60, flush=True)

    print("\n[1/4] Validating config...", flush=True)
    config.validate()

    print("\n[2/4] Session started.", flush=True)
    print(f"      Folder: {session.get_session_dir()}", flush=True)
    mem_words = mem.get_word_count()
    if mem_words:
        print(f"      Memory: {mem_words}/{mem.MAX_WORDS} words loaded.", flush=True)
    else:
        print("      Memory: empty (no previous sessions).", flush=True)

    print("\n[3/4] Starting STT worker...", flush=True)
    stt_worker.start()
    stt_queue = stt_worker.get_output_queue()

    print("\n[4/4] Starting TTS worker...", flush=True)
    tts_worker.start()
    tts_queue = tts_worker.get_input_queue()
    stt_worker.set_is_speaking_event(tts_worker.IS_SPEAKING)
    tts_worker.set_interrupt_event(_interrupt_event)  # wire interrupt into TTS
    _timer_mod.set_tts_queue(tts_queue)               # wire TTS into timer alerts

    print("\n[LLM] Checking LLM server...", flush=True)
    try:
        llm.models.list()
        print("[LLM] Server reachable.", flush=True)
    except Exception:
        print(f"[LLM] WARNING: Could not reach {config.LLM_API_URL}", flush=True)
        print("      Start LLM-server before speaking.", flush=True)

    _stopped_event.clear()
    pipeline = threading.Thread(
        target=_pipeline_loop,
        args=(stt_queue, tts_queue),
        daemon=True,
        name="Pipeline",
    )
    pipeline.start()

    print("\n" + "=" * 60)
    print("  Ready — speak into your microphone")
    print("  Ctrl+C to quit")
    print("=" * 60 + "\n", flush=True)

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[SHUTDOWN] Stopping...", flush=True)
        _stopped_event.set()
        stt_worker.stop()
        tts_worker.stop()
        try:
            pipeline.join(timeout=3)
        except KeyboardInterrupt:
            pass

        # Send full session to LLM for memory rewrite (waits up to 30s)
        with _session_lock:
            turns = list(_session_turns)
        if turns:
            print("[MEM] Updating memory from session (please wait)...", flush=True)
            mem.update_memory_on_shutdown_async(turns, llm, MODEL_NAME, _NO_THINKING)
            words = mem.get_word_count()
            print(f"[MEM] Memory file: {words}/{mem.MAX_WORDS} words.", flush=True)
        else:
            print("[MEM] No session turns — memory unchanged.", flush=True)

        print("[SHUTDOWN] Done.", flush=True)


if __name__ == "__main__":
    main()
