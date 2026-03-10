<div align="center">

<br/>

██████╗ ██╗ ██╗██████╗ ██████╗ ██╗ ██╗ ██████╗
██╔══██╗██║ ██║██╔══██╗██╔══██╗╚██╗ ██╔╝██╔═══██╗
██████╔╝██║ ██║██║ ██║██║ ██║ ╚████╔╝ ██║ ██║
██╔══██╗██║ ██║██║ ██║██║ ██║ ╚██╔╝ ██║▄▄ ██║
██████╔╝╚██████╔╝██████╔╝██████╔╝ ██║ ╚██████╔╝
╚═════╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝ ╚══▀▀═╝

text

**A fully local, privacy-first voice assistant — no cloud, no subscriptions.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?style=flat-square&logo=windows)](.)
[![Backend](https://img.shields.io/badge/Backend-LM%20Studio%20%2F%20llama.cpp-green?style=flat-square)](https://lmstudio.ai)
[![STT](https://img.shields.io/badge/STT-faster--whisper-orange?style=flat-square)](https://github.com/SYSTRAN/faster-whisper)
[![TTS](https://img.shields.io/badge/TTS-Piper-purple?style=flat-square)](https://github.com/rhasspy/piper)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<br/>

> *Speak. See. Search. Respond. Everything stays on your machine.*

<br/>

</div>

---

## What is BuddyQ?

BuddyQ is a **fully offline, privacy-first voice assistant** that runs entirely on your local hardware. No OpenAI API key required — it talks to a local LM Studio (or llama.cpp-compatible) server over HTTP.

It listens continuously, understands natural speech, can **see your screen via local vision models**, thinks with a local language model, searches the web for current information when needed, remembers things across sessions, and speaks back in a natural voice — all in real time on a regular laptop.

Every component is modular and independently replaceable. Swap the STT engine, change the TTS voice, swap the LLM backend (LM Studio, pure llama.cpp server, or any OpenAI-compatible endpoint), or add new tools without touching the core pipeline.

> **Status:** Early alpha — the core pipeline, web search, and vision tooling work, but APIs, prompts, and configs may still change.

<br/>

## ✨ Features

| Feature | Details |
|---|---|
| 🎙️ **Always-on voice input** | VAD-based continuous microphone listening — records full utterances, triggers on speech, stops on natural silence |
| 🧠 **Local LLM via LM Studio / llama.cpp** | Uses the OpenAI SDK pointed at `LLM_API_URL` — works with Qwen3.5, Qwen3-VL, Gemma, Llama, Mistral and other GGUF / LM Studio models |
| 👀 **Screen / vision tool** | Takes a full-screen screenshot and sends it to a local vision-capable model (e.g. Qwen3-VL-4B) to describe what is on your screen, explain code, or judge whether a website looks suspicious (visually only) |
| 🔊 **Natural TTS** | Piper TTS — fast, clear, fully offline, real-time on CPU |
| 🌐 **Smart web search** | Five-step DuckDuckGo pipeline: query optimiser → search with retries → fast-path regex extraction → optional page fetch → LLM summariser, with a hard 30s wall-clock limit for the entire search |
| 🔍 **Credible source filtering** | Search results from Wikipedia, Reuters, BBC, AP, CoinGecko, and 40+ trusted domains are prioritised over random pages |
| 🧠 **Persistent memory** | Remembers facts about you and past sessions across restarts — compressed and stored locally as JSON |
| ⏹️ **Voice interrupt** | Say "stop" or "cancel" at any time to immediately halt TTS playback and LLM generation |
| 📅 **Always date-aware** | Current date and time injected into every LLM prompt — never claims it doesn't know what day it is |
| 💬 **STT error recovery** | Detects garbled transcriptions and uses conversation history to infer intent — asks for clarification when genuinely unclear |
| 🚫 **No reasoning tokens** | Thinking/chain-of-thought is disabled via `extra_body={"chat_template_kwargs": {"thinking": False}}` for both normal and web-search LLM calls — faster, more concise responses on models that support it |
| 📁 **Session logging** | Every conversation saved as structured JSON with timestamps |
| 🔒 **100% private** | Nothing leaves your machine (web search uses DuckDuckGo HTML interface directly; LLM/TTS/STT are local) |
| 🧩 **Modular tools** | Each tool is its own file implementing a small `BaseTool` interface (web search, weather, system control, notes, timers, screen/vision, etc.) |

<br/>

## 🏗️ Architecture

```text
  🎤 Microphone
       │
       ▼
  ┌──────────┐   VAD + silence detection
  │  STT     │──────────────────────────────▶  transcribed text
  │ Whisper  │
  └──────────┘
       │
       ▼
  ┌──────────────────┐
  │  Interrupt check  │──── "stop" / "cancel" ──▶  halt pipeline
  └──────────────────┘
       │
       ▼
  ┌──────────────────┐   Tool + search router
  │  Tool Router     │──── "on my screen" ─────▶  Screen / vision tool
  │                  │──── live data ─────────▶  Web search tool
  │                  │──── instant tools ─────▶  timer / notes / system
  │                  │──── pure chat ─────────▶  LLM directly
  └──────────────────┘
       │
       ▼ (tool results injected when used)
  ┌──────────┐   OpenAI-compatible client   ┌─────────────────────────┐
  │   LLM    │─────────────────────────────▶│ system prompt + date +  │
  │ LM Studio│                              │ memory + history        │
  │ / llama  │                              └─────────────────────────┘
  └──────────┘
       │
       │ sentence by sentence
       ▼
  ┌──────────┐   speech normalizer   ┌────────────────┐
  │   TTS    │──────────────────────▶│  🔊 Speaker     │
  │  Piper   │  ($95k → "95 thousand │                │
  └──────────┘   US dollars", etc.)  └────────────────┘
       │
       ▼
  ┌──────────┐
  │  Memory  │──── session end ──────▶  facts + episodes → memory.json
  │ (JSON)   │
  └──────────┘
       │
       ▼
  ┌──────────┐
  │ Session  │──────────────────────▶  📄 sessions/YYYY-MM-DD/transcript.json
  │  Logger  │
  └──────────┘

📁 Project Structure
text
BuddyQ/
│
├── start.bat                    ← Single launch file — runs everything
├── requirements.txt             ← Python dependencies
├── README.md
│
├── core/                        ← Main application logic
│   ├── main.py                  ← Orchestrator, pipeline, search engine, speech normalizer, tool router
│   ├── config.py                ← All settings in one place (LLM_API_URL, model names, audio config)
│   ├── memory.py                ← Persistent memory — facts + episode storage
│   ├── session.py               ← Session management + JSON logging
│   ├── stt_worker.py            ← Speech-to-text (faster-whisper)
│   ├── tts_worker.py            ← Text-to-speech (Piper)
│   ├── temp/                    ← Temporary audio files (auto-managed)
│   │
│   └── tools/                   ← Tool modules — add new tools here
│       ├── __init__.py          ← Tool registry + helper functions
│       ├── base.py              ← BaseTool interface
│       ├── web_search.py        ← DuckDuckGo search with credible source filtering + LLM summariser
│       ├── weather_tool.py      ← Weather via free Open-Meteo API (no scraping, no API key)
│       ├── system_tool.py       ← System actions ("copy that", "save that", etc.)
│       ├── notes_tool.py        ← Local note storage and recall
│       ├── timer_tool.py        ← Timers and alarms wired into TTS
│       └── screen_tool.py       ← Screen / vision tool using local LM Studio vision model
│
├── stt/                         ← STT environment
│   ├── venv/                    ← Python virtual environment
│   └── model_cache/             ← Whisper model files (auto-downloaded on first run)
│
├── tts/                         ← TTS binaries and voices
│   ├── piper.exe                ← Piper TTS executable
│   └── voices/                  ← Voice model files (.onnx + .onnx.json)
│
├── models/                      ← LLM files (if you run llama.cpp directly)
│   ├── llama-server.exe         ← (optional) llama.cpp server binary
│   └── *.gguf                   ← Your GGUF model file
│
├── memory/                      ← Persistent memory (auto-created)
│   └── memory.json              ← Facts and session episodes
│
└── sessions/                    ← Conversation history (auto-created)
    └── YYYY-MM-DD_HH-MM/
        ├── transcript.json      ← Full conversation as JSON
        └── latest.txt           ← Most recent user utterance

🖥️ Hardware Requirements
BuddyQ is designed to run on everyday laptops. A discrete GPU (e.g. GTX 1660) helps a lot for larger vision models in LM Studio, but CPU-only still works with small models.

Component	Minimum	Recommended
CPU	4-core with AVX2	Intel i5/i7 6th gen+ or AMD Ryzen 5+
RAM	8 GB	16 GB
Storage	5 GB free	10 GB free
GPU	Not required	6 GB+ VRAM if you want fast vision (Qwen3-VL-4B, etc.)
OS	Windows 10/11	Windows 10/11
Microphone	Any USB or built-in	Headset for best STT accuracy
Tested on: Intel i7-6700HQ · 16 GB RAM · GTX 1660 · Windows 11 · LM Studio with Qwen3.5-4B and Qwen3-VL-4B.


⚡ Installation
Step 1 — Clone the repository
powershell
git clone https://github.com/yourusername/BuddyQ.git
cd BuddyQ
Step 2 — Create the Python virtual environment
powershell
python -m venv stt\venv
stt\venv\Scripts\python.exe -m pip install --upgrade pip
stt\venv\Scripts\python.exe -m pip install -r requirements.txt
Step 3 — Download Piper TTS
(Use the same steps as before: download piper.exe and a voice .onnx + .onnx.json into tts\voices\.)

Step 4 — Choose your LLM backend
BuddyQ expects an OpenAI-compatible HTTP endpoint configured via core/config.py:

python
# core/config.py
LLM_API_URL        = "http://127.0.0.1:1234/v1"  # LM Studio default
MODEL_NAME         = "qwen3.5-4b"                # main text model
VISION_MODEL_NAME  = "qwen3-vl-4b-instruct"      # vision / screen model
You have two options:

Option A — LM Studio (recommended)
Install LM Studio.

Download a text model (e.g. qwen3.5-4b) and a vision model (e.g. qwen3-vl-4b-instruct).

Start the LM Studio local server on port 1234 with both models available.

Copy the model IDs from LM Studio into MODEL_NAME and VISION_MODEL_NAME.

Option B — Raw llama.cpp server
If you prefer the old llama.cpp server, set LLM_API_URL to http://127.0.0.1:8080/v1 and start llama-server.exe with OpenAI-compatible endpoints enabled.

Step 5 — Start your backend
LM Studio: click “Start Server” and wait until /v1/chat/completions is available.

llama.cpp: run e.g.:

powershell
models\llama-server.exe --model models\your-model.gguf --port 8080 --ctx-size 4096 --threads 4
Step 6 — Launch BuddyQ
powershell
start.bat

⚙️ Configuration (updated)
python
# core/config.py (excerpt)

# ── LLM backend ───────────────────────────────────────────────
LLM_API_URL        = "http://127.0.0.1:1234/v1"
MODEL_NAME         = "qwen3.5-4b"
VISION_MODEL_NAME  = "qwen3-vl-4b-instruct"

# ── Audio / STT ───────────────────────────────────────────────
SAMPLE_RATE        = 16000
SILENCE_THRESHOLD  = 0.015
SILENCE_DURATION   = 1.5
MAX_RECORD_SECONDS = 8

WHISPER_MODEL      = "tiny"
WHISPER_COMPUTE    = "int8"
WHISPER_THREADS    = 4
python
# core/main.py (top)

MODEL_NAME  = config.MODEL_NAME
MAX_TOKENS  = 20000
TEMPERATURE = 0.45

_NO_THINKING = {"chat_template_kwargs": {"thinking": False}}

llm = OpenAI(base_url=config.LLM_API_URL, api_key="no-key-needed")
_ws_set_llm(llm, MODEL_NAME, _NO_THINKING)

👀 Screen / Vision Tool
The screen tool lets the assistant see your current desktop:

Summarise the page on screen.

Describe UI elements and diagrams.

Explain code shown in your editor.

Comment on obvious visual red flags on websites (typos, mismatched logos, weird URLs).

Implementation highlights (core/tools/screen_tool.py):

Uses mss to capture the primary monitor.

Encodes the screenshot as base64 PNG.

Calls the vision model via OpenAI-compatible LM Studio API:

python
resp = client.chat.completions.create(
    model=VISION_MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": full_text},
                {"type": "input_image",
                 "image_url": {"url": "data:image/png;base64," + b64_image}},
            ],
        }
    ],
    max_tokens=320,
    temperature=0.3,
    stream=False,
    extra_body=_NO_THINKING,
)
Use phrases like:

“What’s on my screen?”

“Summarise this page.”

“Explain this code on screen.”

“Does this website look legit visually?”


🌐 Web Search (updated pipeline)
core/tools/web_search.py implements a five-step DuckDuckGo pipeline:

Optional query optimiser using the main LLM (skipped for simple queries).

DuckDuckGo HTML search (up to 10 snippets, 8s timeout, 3 retries).

Regex fast-path extraction for prices, officeholders, and basic stats.

Page fetch only when snippets are thin (< 60 words).

LLM summariser with max_tokens=120 and thinking disabled.

A credible-domain list prioritises sites like:

wikipedia.org, britannica.com

reuters.com, apnews.com, bbc.com

coinmarketcap.com, coingecko.com, coindesk.com

nature.com, nih.gov, nasa.gov

many others


🧠 Memory System
Memory is stored in memory/memory.json as:

facts: stable personal info (name, location, preferences).

episodes: short summaries of notable conversations.

Loaded on startup and injected into a single combined system message so Qwen-style chat templates are respected.


🗣️ Voice Commands
Examples:

Any question → normal answer.

stop, cancel, shut up, forget it, stop talking → immediately abort TTS and LLM.


🧩 Adding a New Tool
Create core/tools/my_tool.py implementing BaseTool, then register it in core/tools/__init__.py. The orchestrator discovers tools by name and keyword list; you don’t need to change the main pipeline.


🔧 Troubleshooting
No audio output → check Windows default audio device and Piper logs.

Model not found / 404 → verify LLM_API_URL and MODEL_NAME against LM Studio server logs.

Vision very slow → downscale screenshots in screen_tool.py and keep LM Studio context length reasonable (e.g. 8k–12k).

Search too slow → DDG and summariser have hard timeouts; ensure network is OK and reduce MAX_TOTAL_SECS if desired.


🗺️ Roadmap
 Continuous VAD-based listening

 Local LLM via LM Studio / llama.cpp

 Web search with credible source filtering

 Persistent cross-session memory

 Voice interrupt (stop/cancel)

 Speech normalizer (currencies, units, abbreviations)

 Thinking / reasoning token suppression

 Screen / vision tool using local models

 Wake word detection

 Calendar and reminders tool

 Smart home / Home Assistant integration

 Multi-language STT and TTS

 Linux and macOS support

 Simple GUI / tray icon
 Possible robitics application/mobile assistent 


🤝 Contributing
Fork, create a branch, commit, and open a PR. Good first issues:

New tools (calendar, browser automation, IDE helpers).

Performance improvements for screen / vision and search.

Extended credible-domain list.

Cross-platform audio support.


📄 License
MIT License — see LICENSE for full text.

<div align="center">
Built to run anywhere. Designed to stay private.

Star the repo if you find it useful.

Issues and PRs welcome.

</div