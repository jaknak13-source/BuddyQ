"""
Web Search Tool - core/tools/web_search.py

Pipeline (fastest first — skips later steps when earlier ones suffice):

Step 1  Query optimiser
        LLM rewrites garbled STT query (~200ms) when available.
        Skipped for short, clear queries to save latency.

Step 2  DuckDuckGo search
        Gets up to 10 result snippets (~1–3s).
        Retries with a slightly widened query if snippets are thin.

Step 3  Fast-path extraction (no LLM)
        Regex extracts direct facts from snippets (<1ms) for:
        prices, names, officeholders, population, GDP, basic stats.
        Returns immediately if a clean answer is found.

Step 4  Page fetch (only if snippets are thin)
        Fetches the best page and strips boilerplate, yielding
        up to 2000 chars of plain text.

Step 5  LLM summariser
        For complex, multi-fact questions. Uses short prompt
        and low max_tokens to stay quick, but asks for concrete,
        specific answers.

Hard wall‑clock timeout: MAX_TOTAL_SECS (default 30s).
"""

import re
import time
import urllib.request
import urllib.parse
from typing import List, Dict, Optional, Tuple

from .base import BaseTool

# ── HTTP settings ─────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,de;q=0.7",
}

SEARCH_TIMEOUT = 8   # per HTTP request
FETCH_TIMEOUT = 10   # full page fetch
MAX_TOTAL_SECS = 30  # hard limit for entire pipeline
MAX_RETRIES = 3      # max retry attempts
MIN_GOOD_SNIPPETS = 5

# ── Credible domains ──────────────────────────────────────────────────────────

CREDIBLE_DOMAINS = {
    "wikipedia.org", "britannica.com", "wolframalpha.com",
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com",
    "bloomberg.com", "ft.com", "economist.com",
    "dw.com", "euronews.com", "aljazeera.com",
    "coinmarketcap.com", "coingecko.com", "coindesk.com",
    "investing.com", "marketwatch.com", "cnbc.com",
    "finance.yahoo.com", "tradingeconomics.com",
    "weather.com", "metoffice.gov.uk", "weather.gov", "accuweather.com",
    "nature.com", "nih.gov", "who.int", "nasa.gov",
    "techcrunch.com", "theverge.com", "arstechnica.com",
    "espn.com", "skysports.com", "formula1.com",
    "gov.uk", "europa.eu", "un.org", "worldbank.org", "imf.org",
    "statista.com", "ourworldindata.org",
    "zillow.com", "realtor.com", "numbeo.com",
}

# ── LLM (injected at startup) ─────────────────────────────────────────────────

_llm_client = None
_model_name: Optional[str] = None
_no_thinking: Dict = {}


def set_llm(client, model: str, no_thinking: dict):
    """Preferred name (internal)."""
    global _llm_client, _model_name, _no_thinking
    _llm_client = client
    _model_name = model
    _no_thinking = no_thinking


def setllm(client, model: str, no_thinking: dict):
    """
    Backwards‑compatible alias used by main.py:
    from tools.web_search import WebSearchTool, setllm as wssetllm
    """
    set_llm(client, model, no_thinking)

# ============================================================================ #
# FAST‑PATH EXTRACTOR (Step 3 — no LLM)
# ============================================================================ #

# Patterns that indicate a snippet likely contains the direct answer
_PRICE_RE = re.compile(
    r"\$[\d,]+(?:\.\d{1,2})?"
    r"|\bUSD\s*[\d,]+"
    r"|[\d,]+\s*(?:US\s*)?dollars?",
    re.I,
)
_PERSON_RE = re.compile(
    r"\b(is|was|remains?|became?|elected|appointed|serving as)\b"
    r".{2,80}\b(president|pm|prime minister|chancellor|ceo|king|queen|pope|leader)\b",
    re.I,
)
_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)
_PERCENT_RE = re.compile(r"[\d.]+\s*%", re.I)
_NUMBER_RE = re.compile(
    r"\b[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand|trillion)?\b",
    re.I,
)

# Query types that get fast‑path treatment
_FASTPATH_QUERY = re.compile(
    r"""
    \b(
        price|cost|worth|value|trading|
        bitcoin|btc|ethereum|eth|crypto|stock|
        president|prime\s*minister|chancellor|ceo|leader|king|queen|pope|
        who\s*is|who\s*was|who\s*leads|who\s*runs|
        population|gdp|temperature|how\s*much|how\s*many|
        born|died|founded|invented|discovered|released|launched
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _fast_extract(query: str, results: List[Dict]) -> Optional[str]:
    """
    Try to extract a direct factual answer from snippets without the LLM.
    Returns a clean plain‑English answer string, or None if we can't.
    """
    if not results or not _FASTPATH_QUERY.search(query):
        return None

    # Prefer credible sources, then longer snippets
    ordered = sorted(
        results,
        key=lambda r: (not r["credible"], -len(r["snippet"])),
    )
    q_lower = query.lower()

    for r in ordered[:6]:
        snippet = r["snippet"].strip()
        if len(snippet) < 20:
            continue

        # ── Price queries ───────────────────────────────────────────────
        if any(
            w in q_lower
            for w in (
                "price",
                "cost",
                "worth",
                "trading",
                "bitcoin",
                "btc",
                "ethereum",
                "eth",
                "crypto",
                "stock",
            )
        ):
            if _PRICE_RE.search(snippet):
                for sent in re.split(r"[.!?]", snippet):
                    if _PRICE_RE.search(sent) and len(sent.split()) >= 4:
                        return sent.strip() + "."
                return snippet[:200]

        # ── Current officeholder queries ───────────────────────────────
        if any(
            w in q_lower
            for w in (
                "president",
                "prime minister",
                "chancellor",
                "ceo",
                "king",
                "queen",
                "pope",
                "who is",
                "who leads",
                "who runs",
            )
        ):
            if _PERSON_RE.search(snippet):
                for sent in re.split(r"[.!?]", snippet):
                    if len(sent.split()) >= 4 and any(
                        role in sent.lower()
                        for role in (
                            "president",
                            "prime minister",
                            "chancellor",
                            "ceo",
                            "leader",
                        )
                    ):
                        return sent.strip() + "."
                return snippet[:200]

        # ── Population / GDP / statistics ─────────────────────────────
        if any(
            w in q_lower
            for w in ("population", "gdp", "how many", "how much")
        ):
            if _NUMBER_RE.search(snippet) or _PERCENT_RE.search(snippet):
                for sent in re.split(r"[.!?]", snippet):
                    if (
                        _NUMBER_RE.search(sent)
                        or _PERCENT_RE.search(sent)
                    ) and len(sent.split()) >= 5:
                        return sent.strip() + "."

    return None

# ============================================================================ #
# QUERY OPTIMISER (Step 1)
# ============================================================================ #

_SKIP_OPTIMISE = re.compile(
    r"""
    ^(
        what\s+is\s+the\s+(current\s+)?(price|temperature|population)|
        who\s+is\s+the\s+(current\s+)?(president|pm|ceo|chancellor|leader)|
        bitcoin\s+price|btc\s+price|ethereum\s+price|
        weather\s+in\s+\w+|temperature\s+in\s+\w+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Generic filler to strip when no LLM is available
_FILLER_RE = re.compile(
    r"\b(can you|could you|please|tell me|show me|i want to know|"
    r"what is|what's|whats)\b",
    re.IGNORECASE,
)


def _optimise_query(raw: str, history: List[Dict]) -> str:
    """
    Rewrite STT‑garbled query into clean search terms.
    Uses LLM when available; otherwise falls back to heuristic cleaning.
    """
    raw = (raw or "").strip()
    if not raw:
        return raw

    # Short, clear queries: skip optimiser for speed
    if _SKIP_OPTIMISE.search(raw):
        return raw

    # No LLM available → basic cleanup only
    if not _llm_client or not _model_name:
        base = _FILLER_RE.sub("", raw)
        base = re.sub(r"\s+", " ", base).strip(" ?.!").lower()
        year = time.strftime("%Y")
        if any(
            w in raw.lower()
            for w in ("price", "current", "today", "latest", "now", "president", "ceo")
        ):
            return f"{base} {year}"
        return base

    # With LLM: use recent context to repair STT errors
    context_lines: List[str] = []
    for m in [m for m in history if m.get("role") in ("user", "assistant")][-4:]:
        role = "User" if m["role"] == "user" else "Buddy"
        c = (m.get("content", "") or "")[:150]
        if c.strip():
            context_lines.append(f"{role}: {c.strip()}")
    ctx = "\n".join(context_lines) or "(no prior context)"

    now = time.strftime("%Y-%m-%d")
    year = time.strftime("%Y")

    prompt = (
        f"Today: {now}\n"
        f"Context:\n{ctx}\n\n"
        f"User said (may be garbled): \"{raw}\"\n\n"
        "Task: Rewrite this into the best 3–8 keyword web search query.\n"
        "Fix transcription errors using the context above.\n"
        f"If it asks for current or live data, append {year}.\n"
        "Output ONLY the final search query, no quotes, no extra words."
    )

    try:
        resp = _llm_client.chat.completions.create(
            model=_model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
            temperature=0.1,
            stream=False,
            extra_body=_no_thinking,
        )
        q = (resp.choices[0].message.content or "").strip().strip("\"'")
        q = re.sub(r"\s+", " ", q)
        if q and len(q) < 160:
            print(f"[SEARCH] Query: \"{raw}\" → \"{q}\"", flush=True)
            return q
    except Exception as e:
        print(f"[SEARCH] Optimiser error: {e}", flush=True)

    # Fallback: light cleanup
    base = _FILLER_RE.sub("", raw)
    base = re.sub(r"\s+", " ", base).strip(" ?.!").lower()
    return base


def _widen_query(query: str, attempt: int) -> str:
    """Widen a query for retry — no LLM call, just trim words."""
    words = query.split()
    if attempt == 1 and len(words) > 3:
        # Drop last word (often too specific)
        return " ".join(words[:-1])
    if attempt == 2 and len(words) > 4:
        # Keep first 3–4 strong tokens
        return " ".join(words[:4])
    return query

# ============================================================================ #
# DDG SEARCH (Step 2)
# ============================================================================ #


def _is_credible(url: str) -> bool:
    try:
        domain = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
        return any(domain == d or domain.endswith("." + d) for d in CREDIBLE_DOMAINS)
    except Exception:
        return False


def _ddg_search(query: str) -> List[Dict]:
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[SEARCH] DDG failed: {e}", flush=True)
        return []

    snippets = re.findall(
        r'class="result__snippet"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )
    titles = re.findall(
        r'class="result__a"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )
    hrefs = re.findall(
        r'class="result__a"[^>]*href="(.*?)"',
        html,
        re.DOTALL,
    )

    results: List[Dict] = []
    for i, raw_snip in enumerate(snippets[:10]):
        clean_s = re.sub(r"<[^>]+>", " ", raw_snip)
        clean_s = re.sub(r"\s+", " ", clean_s).strip()
        if not clean_s:
            continue

        raw_title = titles[i] if i < len(titles) else ""
        clean_t = re.sub(r"<[^>]+>", " ", raw_title)
        clean_t = re.sub(r"\s+", " ", clean_t).strip()

        raw_url = hrefs[i] if i < len(hrefs) else ""
        raw_url = raw_url.strip()
        if raw_url and not raw_url.startswith("http"):
            raw_url = "https://" + raw_url.lstrip("/")

        results.append(
            {
                "title": clean_t,
                "snippet": clean_s,
                "url": raw_url,
                "credible": _is_credible(raw_url),
            }
        )
    return results


def _snippets_are_useful(results: List[Dict]) -> bool:
    """
    A result set is considered useful if we have at least MIN_GOOD_SNIPPETS
    with 8+ words, or at least 2 credible domains with 6+ words each.
    """
    if not results:
        return False
    long_count = sum(1 for r in results if len(r["snippet"].split()) >= 8)
    cred_count = sum(
        1
        for r in results
        if r["credible"] and len(r["snippet"].split()) >= 6
    )
    return long_count >= MIN_GOOD_SNIPPETS or cred_count >= 2

# ============================================================================ #
# PAGE FETCHER (Step 4)
# ============================================================================ #


def _fetch_page_text(url: str, max_chars: int = 2000) -> str:
    if not url or not url.startswith("http"):
        return ""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            ct = resp.headers.get("Content-Type", "")
            if "text/html" not in ct and "text/plain" not in ct:
                return ""
            html = resp.read(49152).decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[SEARCH] Page fetch failed: {e}", flush=True)
        return ""

    # Strip heavy boilerplate
    html = re.sub(
        r"<(script|style|head|nav|footer|header)[^>]*>.*?</\1>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    # Basic entity cleanup (avoid speaking raw entities)
    for ent, ch in [
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
    ]:
        text = text.replace(ent, ch)
    return text[:max_chars]

# ============================================================================ #
# LLM SUMMARISER (Step 5)
# ============================================================================ #


def _summarise_results(
    original: str,
    optimised: str,
    results: List[Dict],
    page_text: str,
    history: List[Dict],
) -> str:
    if not _llm_client or not _model_name:
        # Fallback: just return a couple of snippets
        return " ".join(r["snippet"] for r in results[:3])

    ctx_lines: List[str] = []
    for m in [m for m in history if m.get("role") in ("user", "assistant")][-4:]:
        role = "User" if m["role"] == "user" else "Buddy"
        c = (m.get("content", "") or "")[:200]
        if c.strip() and not c.startswith("SEARCH RESULT"):
            ctx_lines.append(f"{role}: {c.strip()}")
    ctx = "\n".join(ctx_lines)

    material = ""
    for i, r in enumerate(results[:5], 1):
        lbl = " [trusted]" if r["credible"] else ""
        material += f"\nResult {i}{lbl}: {r['title']}\n{r['snippet']}\n"
    if page_text:
        material += f"\nPage content:\n{page_text}\n"

    if not material.strip():
        return "No usable search results were found."

    ctx_block = f"Conversation context:\n{ctx}\n\n" if ctx else ""
    prompt = (
        f"/no_think\n"
        f"{ctx_block}"
        f"User asked: \"{original}\"\n"
        f"Optimised search query: \"{optimised}\"\n\n"
        f"Search results:\n{material}\n\n"
        "Task: Answer the user's question using ONLY the information above.\n"
        "Be concrete and specific: include exact numbers, names, and dates"
        " when present. If the information conflicts, say that it is mixed"
        " and give the most likely answer.\n"
        "Length: 2–4 full sentences, plain spoken English for text‑to‑speech."
        " No bullet points, no URLs, no source names, no lists of sites."
    )

    try:
        resp = _llm_client.chat.completions.create(
            model=_model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=140,  # short, but enough for 3–4 sentences
            temperature=0.2,
            stream=False,
            extra_body=_no_thinking,
        )
        summary = (resp.choices[0].message.content or "").strip()
        summary = re.sub(r"[*#`]+", "", summary).strip()
        return summary or "The search results did not contain a clear answer."
    except Exception as e:
        print(f"[SEARCH] Summariser error: {e}", flush=True)
        return " ".join(r["snippet"] for r in results[:3])

# ============================================================================ #
# TOOL CLASS
# ============================================================================ #


class WebSearchTool(BaseTool):
    def __init__(self):
        self._history: List[Dict] = []

    def set_history(self, h: List[Dict]):
        self._history = h or []

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Searches the web for current information: news, prices, weather, "
            "sports, people in current roles, and recent events."
        )

    @property
    def keywords(self) -> list:
        return [
            "search for",
            "look up",
            "find out",
            "search the web",
            "latest news",
            "what happened",
            "current events",
            "who is",
            "when did",
            "where is",
            "how many",
            "how much",
            "price of",
            "cost of",
            "tell me about",
        ]

    def run(self, raw_utterance: str) -> str:
        """
        Full pipeline. Exits early at the fastest step that gives a good answer.
        Hard MAX_TOTAL_SECS timeout.
        """
        deadline = time.time() + MAX_TOTAL_SECS
        history = list(self._history)

        # Step 1: Optimise query
        if time.time() >= deadline:
            return "Search timed out."
        optimised = _optimise_query(raw_utterance, history) or raw_utterance

        # Step 2: Search with retry
        all_results: List[Dict] = []
        current_query = optimised

        for attempt in range(MAX_RETRIES + 1):
            if time.time() >= deadline:
                print("[SEARCH] Deadline during search.", flush=True)
                break

            print(f"[SEARCH] Attempt {attempt + 1}: \"{current_query}\"", flush=True)
            results = _ddg_search(current_query)

            if _snippets_are_useful(results):
                all_results = results
                print(f"[SEARCH] Got {len(results)} results.", flush=True)
                break

            print(
                f"[SEARCH] Thin results on attempt {attempt + 1} "
                f"({len(results)} results).",
                flush=True,
            )
            all_results = results  # keep whatever we have

            if attempt < MAX_RETRIES:
                widened = _widen_query(optimised, attempt + 1)
                if widened == current_query:
                    break
                current_query = widened

        if not all_results:
            return (
                f"I searched for '{optimised}' but found no results. "
                "Please try rephrasing."
            )

        # Sort: credible first, then longer snippets
        best = sorted(
            all_results,
            key=lambda r: (not r["credible"], -len(r["snippet"])),
        )[:6]

        # Step 3: Fast‑path extraction — no LLM
        fast = _fast_extract(optimised or raw_utterance, best)
        if fast:
            print(
                "[SEARCH] Fast‑path answered — skipping page fetch and summariser.",
                flush=True,
            )
            return fast

        # Step 4: Page fetch if snippets too thin
        page_text = ""
        total_words = sum(len(r["snippet"].split()) for r in best)
        if total_words < 60 and best and time.time() < deadline - 10:
            print(
                f"[SEARCH] Fetching page ({total_words} snippet words)...",
                flush=True,
            )
            page_text = _fetch_page_text(best[0]["url"])

        # Step 5: LLM summariser
        if time.time() >= deadline:
            return " ".join(r["snippet"] for r in best[:3])

        return _summarise_results(raw_utterance, current_query, best, page_text, history)
