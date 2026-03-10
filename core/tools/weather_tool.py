"""
Weather Tool - core/tools/weather_tool.py

Uses Open-Meteo API — completely free, no API key, no account required.
Also uses the Open-Meteo geocoding API to resolve city names to coordinates.

Supports:
- Current weather: "what's the weather in Berlin"
- Forecast: "weather tomorrow in Paris", "weekend weather"
- Specific data: "will it rain today in London", "temperature in Tokyo"

All temperatures in Celsius (configurable via UNIT_CELSIUS below).
"""

import re
import json
import urllib.request
import urllib.parse
from typing import List, Dict, Tuple, Optional

from .base import BaseTool

UNIT_CELSIUS = True  # False → Fahrenheit

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 8

# WMO weather code descriptions
# (no manual "thunderstorm" added unless API returns the matching code)
WMO_CODES: Dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "icy fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "heavy thunderstorm with hail",
}


def _wmo(code: int) -> str:
    try:
        code_int = int(code)
    except Exception:
        return "unknown"
    return WMO_CODES.get(code_int, f"weather code {code_int}")


def _clean_city_string(raw: str) -> str:
    """
    Clean noisy location strings, including helper phrases from the LLM like:
    "the user most likely meant Berlin, Germany".
    Also helps when STT slightly mangles words by cutting to the first clear token.
    """
    if not raw:
        return ""

    text = raw.strip().strip("'\"")

    # Remove common helper prefixes
    prefixes = [
        "the user most likely meant",
        "user most likely meant",
        "i think the user meant",
        "i think you meant",
        "probably",
    ]
    lower = text.lower()
    for p in prefixes:
        if lower.startswith(p):
            text = text[len(p):].strip(" ,-'\"")
            lower = text.lower()
            break

    # If it still contains that phrase, just bail
    if "the user most likely meant" in lower:
        return ""

    # Keep up to first comma ("Berlin, Germany" -> "Berlin")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if parts:
        text = parts[0]

    # Strip non-letter garbage from ends (helps a bit with small STT glitches)
    text = re.sub(r"^[^A-Za-z]+", "", text)
    text = re.sub(r"[^A-Za-z\s]+$", "", text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _geocode(city: str) -> Optional[Tuple[float, float, str]]:
    """Return (lat, lon, display_name) for a city name, or None."""
    city = _clean_city_string(city)
    if not city:
        return None

    params = urllib.parse.urlencode(
        {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json",
        }
    )
    try:
        with urllib.request.urlopen(
            f"{GEOCODE_URL}?{params}", timeout=TIMEOUT
        ) as r:
            data = json.loads(r.read())
        results = data.get("results", [])
        if not results:
            return None
        r0 = results[0]
        name = r0.get("name", city)
        country = r0.get("country", "")
        display = f"{name}, {country}" if country else name
        return float(r0["latitude"]), float(r0["longitude"]), display
    except Exception:
        return None


_STOPWORDS = {
    "what",
    "how",
    "will",
    "is",
    "are",
    "the",
    "weather",
    "today",
    "tomorrow",
    "forecast",
    "temperature",
    "it",
    "like",
}


def _extract_city(text: str, history: List[Dict]) -> str:
    """
    Extract city name from query, or fall back to last city mentioned in history.
    More robust to STT / mispronunciation by:
    - cleaning helper phrases,
    - preferring explicit "in X" patterns,
    - then capitalised tokens in the whole query,
    - then history.
    """
    original = text or ""
    cleaned = _clean_city_string(original)
    text_for_regex = cleaned if cleaned else original

    # Explicit: "weather in Berlin", "temperature in New York"
    m = re.search(
        r"(?:in|for|at|near)\s+([A-Z][a-zA-Z\s]{1,40})(?:\s*[,.!?]|$)",
        text_for_regex,
    )
    if m:
        return _clean_city_string(m.group(1))

    # Try to find a capitalised proper noun anywhere
    m = re.search(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b", text_for_regex)
    if m:
        candidate = m.group(1).strip()
        if candidate.lower() not in _STOPWORDS:
            return _clean_city_string(candidate)

    # Fall back to last plausible city in conversation history
    for msg in reversed(history or []):
        content = msg.get("content", "")
        city_m = re.search(
            r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b", content
        )
        if city_m:
            candidate = city_m.group(1).strip()
            if candidate.lower() not in {
                "what",
                "the",
                "who",
                "where",
                "when",
                "how",
                "buddy",
            }:
                return _clean_city_string(candidate)

    return ""


def _day_offset(text: str) -> int:
    """Return forecast day offset: 0=today, 1=tomorrow, 7=this week."""
    text = text.lower()
    if "tomorrow" in text:
        return 1
    if "day after" in text:
        return 2
    if "weekend" in text:
        return _days_until_weekend()
    if "next week" in text:
        return 7
    if "in two days" in text:
        return 2
    if "in 3 days" in text or "in three days" in text:
        return 3
    return 0


def _days_until_weekend() -> int:
    from datetime import datetime

    dow = datetime.now().weekday()  # Mon=0, Sat=5, Sun=6
    if dow >= 5:
        return 0
    return 5 - dow


class WeatherTool(BaseTool):
    def __init__(self):
        self._history: List[Dict] = []

    def set_history(self, h: List[Dict]):
        self._history = h or []

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return (
            "Gets current weather and forecasts for any city worldwide. "
            "Free, no API key. Use for: 'weather in X', 'will it rain', "
            "'temperature tomorrow'."
        )

    @property
    def keywords(self) -> list:
        return [
            "weather",
            "temperature",
            "forecast",
            "will it rain",
            "is it raining",
            "is it snowing",
            "will it snow",
            "sunny",
            "cloudy",
            "wind",
            "humidity",
            "weather today",
            "weather tomorrow",
            "weekend weather",
            "degrees",
            "how cold",
            "how hot",
            "what's it like outside",
        ]

    def run(self, query: str) -> str:
        text = (query or "").strip()
        city = _extract_city(text, self._history)

        if not city:
            return (
                "I need a city name to check the weather. "
                "Try: 'what is the weather in Berlin?'."
            )

        coords = _geocode(city)
        if not coords:
            return (
                f"I could not find a location called '{city}'. "
                "Please try again with a clearer city name."
            )

        lat, lon, display_name = coords
        day = _day_offset(text.lower())

        temp_unit = "celsius" if UNIT_CELSIUS else "fahrenheit"
        wind_unit = "kmh"
        params = urllib.parse.urlencode(
            {
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,"
                    "apparent_temperature,weather_code,"
                    "wind_speed_10m,precipitation"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,"
                    "temperature_2m_min,precipitation_sum,"
                    "wind_speed_10m_max"
                ),
                "temperature_unit": temp_unit,
                "wind_speed_unit": wind_unit,
                "precipitation_unit": "mm",
                "timezone": "auto",
                "forecast_days": max(day + 1, 3),
            }
        )

        try:
            with urllib.request.urlopen(
                f"{WEATHER_URL}?{params}", timeout=TIMEOUT
            ) as r:
                data = json.loads(r.read())
        except Exception as e:
            return f"Weather API request failed: {e}"

        unit = "°C" if UNIT_CELSIUS else "°F"

        if day == 0:
            # Current conditions
            cur = data.get("current", {}) or {}
            temp = cur.get("temperature_2m", "?")
            feels = cur.get("apparent_temperature", "?")
            humidity = cur.get("relative_humidity_2m", "?")
            wind = cur.get("wind_speed_10m", "?")
            wcode = cur.get("weather_code", 0)
            precip = cur.get("precipitation", 0)
            condition = _wmo(wcode)

            parts = [
                f"In {display_name} it's currently {temp}{unit} ({condition}),",
                f"feels like {feels}{unit}.",
                f"Humidity is {humidity}% and wind is {wind} km/h.",
            ]

            try:
                if precip and float(precip) > 0:
                    parts.append(
                        f"There has been {precip} millimetres of precipitation recently."
                    )
            except Exception:
                pass

            return " ".join(parts)

        # Forecast for specific day
        daily = data.get("daily", {}) or {}
        dates = daily.get("time", []) or []
        highs = daily.get("temperature_2m_max", []) or []
        lows = daily.get("temperature_2m_min", []) or []
        codes = daily.get("weather_code", []) or []
        precips = daily.get("precipitation_sum", []) or []
        winds = daily.get("wind_speed_10m_max", []) or []

        if not dates or not highs or not lows:
            return "Forecast data is not available for that location."

        idx = min(day, len(dates) - 1)
        if idx >= len(highs):
            return "Forecast data not available for that day."

        date_str = dates[idx] if idx < len(dates) else "that day"
        condition = (
            _wmo(codes[idx])
            if idx < len(codes)
            else "unknown"
        )
        high = highs[idx]
        low = lows[idx]
        rain = precips[idx] if idx < len(precips) else 0
        wind = winds[idx] if idx < len(winds) else "?"

        label = "Tomorrow" if day == 1 else f"On {date_str}"
        parts = [
            f"{label} in {display_name}: {condition},",
            f"high of {high}{unit}, low of {low}{unit}.",
            f"Wind up to {wind} km/h.",
        ]

        try:
            if rain and float(rain) > 0:
                parts.append(f"Expected rainfall: {rain} millimetres.")
        except Exception:
            pass

        return " ".join(parts)
