"""
Calculator Tool - T:\\BuddyQ\\core\\tools\\calculator_tool.py

Handles: arithmetic, percentages, unit conversions, currency conversions.
Runs entirely locally — no LLM, no network. Returns exact answers in < 1ms.

Supported:
  - Arithmetic:    "what is 347 times 19", "12 divided by 4", "square root of 144"
  - Percentages:   "what is 15 percent of 80", "20 is what percent of 250"
  - Unit converts: "convert 5 miles to kilometres", "100 fahrenheit in celsius"
  - Currency:      always deferred to web_search (rates change daily)
"""

import re
import math
from .base import BaseTool


# ── Safe math evaluator ───────────────────────────────────────────────────────

_SAFE_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "pow": math.pow,
    "floor": math.floor, "ceil": math.ceil,
}

def _safe_eval(expr: str) -> float:
    """Evaluate a math expression safely — no builtins, no imports."""
    try:
        result = eval(expr, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
        return float(result)
    except Exception as e:
        raise ValueError(f"Cannot evaluate: {expr} ({e})")


# ── Text → expression normaliser ─────────────────────────────────────────────

_WORD_OPS = [
    (r'\bplus\b',              '+'),
    (r'\bminus\b',             '-'),
    (r'\btimes\b',             '*'),
    (r'\bmultiplied\s*by\b',   '*'),
    (r'\bdivided\s*by\b',      '/'),
    (r'\bover\b',              '/'),
    (r'\bto\s*the\s*power\s*of\b', '**'),
    (r'\bsquared\b',           '**2'),
    (r'\bcubed\b',             '**3'),
]

def _normalise_expr(text: str) -> str:
    text = text.lower()
    for pattern, replacement in _WORD_OPS:
        text = re.sub(pattern, replacement, text)
    # "square root of X" → "sqrt(X)"
    text = re.sub(r'square\s*root\s*of\s*([\d.]+)', r'sqrt(\1)', text)
    # Remove commas in numbers: 1,000 → 1000
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # Strip non-math chars
    text = re.sub(r'[^0-9+\-*/().%sqrt\s]', '', text)
    return text.strip()


def _fmt(n: float) -> str:
    """Format number: integer if whole, otherwise up to 6 significant figures."""
    if n == int(n) and abs(n) < 1e15:
        return f"{int(n):,}"
    return f"{n:,.6g}"


# ── Unit conversion tables ────────────────────────────────────────────────────

# All conversions are TO a base unit, then FROM base unit.
# Base units: metres (length), kilograms (weight), litres (volume),
#             celsius (temp), square-metres (area), km/h (speed)

_LENGTH = {   # to metres
    "mm": 0.001, "millimetre": 0.001, "millimeter": 0.001,
    "cm": 0.01,  "centimetre": 0.01,  "centimeter": 0.01,
    "m":  1.0,   "metre": 1.0,        "meter": 1.0,
    "km": 1000,  "kilometre": 1000,   "kilometer": 1000,
    "in": 0.0254,"inch": 0.0254,      "inches": 0.0254,
    "ft": 0.3048,"foot": 0.3048,      "feet": 0.3048,
    "yd": 0.9144,"yard": 0.9144,      "yards": 0.9144,
    "mi": 1609.344,"mile": 1609.344,  "miles": 1609.344,
    "nm": 1852,  "nautical mile": 1852,
}

_WEIGHT = {   # to grams
    "mg": 0.001, "milligram": 0.001,
    "g":  1.0,   "gram": 1.0,
    "kg": 1000,  "kilogram": 1000,
    "t":  1e6,   "tonne": 1e6, "metric ton": 1e6,
    "oz": 28.3495,"ounce": 28.3495, "ounces": 28.3495,
    "lb": 453.592,"pound": 453.592, "pounds": 453.592,
    "st": 6350.29,"stone": 6350.29,
}

_VOLUME = {   # to millilitres
    "ml": 1.0,   "millilitre": 1.0, "milliliter": 1.0,
    "cl": 10.0,  "centilitre": 10.0,
    "l":  1000,  "litre": 1000,     "liter": 1000,
    "tsp": 4.929,"teaspoon": 4.929,
    "tbsp": 14.787,"tablespoon": 14.787,
    "fl oz": 29.574,"fluid ounce": 29.574,
    "cup": 236.588,
    "pt": 473.176,"pint": 473.176,
    "qt": 946.353,"quart": 946.353,
    "gal": 3785.41,"gallon": 3785.41,
}

_SPEED = {    # to km/h
    "km/h": 1.0, "kph": 1.0,
    "mph": 1.60934, "miles per hour": 1.60934,
    "m/s": 3.6,  "metres per second": 3.6, "meters per second": 3.6,
    "knot": 1.852, "knots": 1.852,
}

_AREA = {     # to square metres
    "m2": 1.0, "sq m": 1.0, "square metre": 1.0, "square meter": 1.0,
    "km2": 1e6, "sq km": 1e6,
    "ft2": 0.0929, "sq ft": 0.0929, "square foot": 0.0929, "square feet": 0.0929,
    "mi2": 2.59e6, "sq mi": 2.59e6, "square mile": 2.59e6,
    "acre": 4046.86, "acres": 4046.86,
    "ha": 10000, "hectare": 10000,
}

_UNIT_GROUPS = [_LENGTH, _WEIGHT, _VOLUME, _SPEED, _AREA]


def _convert_units(value: float, from_u: str, to_u: str) -> str:
    from_u = from_u.lower().strip()
    to_u   = to_u.lower().strip()

    # Temperature (special case — not multiplicative)
    def _temp(v, f, t):
        if f in ("c","celsius","°c") and t in ("f","fahrenheit","°f"):
            return v * 9/5 + 32, "°F"
        if f in ("f","fahrenheit","°f") and t in ("c","celsius","°c"):
            return (v - 32) * 5/9, "°C"
        if f in ("c","celsius","°c") and t in ("k","kelvin"):
            return v + 273.15, "K"
        if f in ("k","kelvin") and t in ("c","celsius","°c"):
            return v - 273.15, "°C"
        if f in ("f","fahrenheit","°f") and t in ("k","kelvin"):
            return (v - 32) * 5/9 + 273.15, "K"
        return None, None

    r, unit = _temp(value, from_u, to_u)
    if r is not None:
        return f"{_fmt(value)} {from_u} = {_fmt(r)} {unit}"

    # Other unit groups
    for group in _UNIT_GROUPS:
        if from_u in group and to_u in group:
            base = value * group[from_u]
            result = base / group[to_u]
            return f"{_fmt(value)} {from_u} = {_fmt(result)} {to_u}"

    return f"Unit conversion from {from_u} to {to_u} is not supported."


# ── Percentage helpers ────────────────────────────────────────────────────────

def _handle_percentage(text: str) -> str | None:
    # "what is X percent of Y"
    m = re.search(r'([\d.]+)\s*%?\s*percent\s+of\s+([\d,.]+)', text, re.I)
    if m:
        pct = float(m.group(1))
        total = float(m.group(2).replace(',', ''))
        result = pct / 100 * total
        return f"{_fmt(pct)}% of {_fmt(total)} = {_fmt(result)}"

    # "X is what percent of Y"
    m = re.search(r'([\d,.]+)\s+is\s+what\s+percent\s+of\s+([\d,.]+)', text, re.I)
    if m:
        part  = float(m.group(1).replace(',', ''))
        whole = float(m.group(2).replace(',', ''))
        if whole == 0:
            return "Cannot divide by zero."
        pct = part / whole * 100
        return f"{_fmt(part)} is {_fmt(pct)}% of {_fmt(whole)}"

    # "X percent increase/decrease from Y"
    m = re.search(r'([\d.]+)\s*%\s*(increase|decrease)\s+(?:from|of)\s+([\d,.]+)', text, re.I)
    if m:
        pct    = float(m.group(1))
        direction = m.group(2).lower()
        base   = float(m.group(3).replace(',', ''))
        delta  = base * pct / 100
        result = base + delta if direction == "increase" else base - delta
        return f"{_fmt(pct)}% {direction} from {_fmt(base)} = {_fmt(result)}"

    return None


# ── Unit conversion pattern ───────────────────────────────────────────────────

_UNIT_RE = re.compile(
    r'([\d,.]+)\s*([a-z°/²]+(?:\s+[a-z]+)?)\s+'
    r'(?:in|to|into|as|converted?\s+to)\s+'
    r'([a-z°/²]+(?:\s+[a-z]+)?)',
    re.IGNORECASE
)


class CalculatorTool(BaseTool):

    @property
    def name(self):
        return "calculator"

    @property
    def description(self):
        return (
            "Performs exact arithmetic, percentage calculations, and unit conversions locally. "
            "Use for any math question, 'X percent of Y', or 'convert X to Y'."
        )

    @property
    def keywords(self):
        return [
            "calculate", "compute",
            "what is 1", "what is 2", "what is 3", "what is 4", "what is 5",
            "what is 6", "what is 7", "what is 8", "what is 9", "what is 0",
            "how much is",
            "plus", "minus", "times", "divided by", "multiplied by",
            "square root", "percent of", "what percent", "percentage",
            "convert", "in kilometres", "in miles", "in celsius", "in fahrenheit",
            "in kilograms", "in pounds", "in litres", "in gallons",
        ]

    def run(self, query: str) -> str:
        text = query.lower().strip()

        # 1. Percentage
        pct = _handle_percentage(text)
        if pct:
            return pct

        # 2. Unit conversion
        m = _UNIT_RE.search(text)
        if m:
            try:
                value  = float(m.group(1).replace(',', ''))
                from_u = m.group(2).strip()
                to_u   = m.group(3).strip()
                return _convert_units(value, from_u, to_u)
            except Exception:
                pass

        # 3. General arithmetic
        # Extract a numeric expression from the query
        expr_text = re.sub(
            r'\b(what\s+is|calculate|compute|equals?|tell\s+me|result\s+of|value\s+of)\b',
            '', text, flags=re.IGNORECASE
        ).strip()
        expr = _normalise_expr(expr_text)

        if not expr:
            return "I could not parse that calculation. Please rephrase it."

        try:
            result = _safe_eval(expr)
            # Clean up the display expression
            display = re.sub(r'\s+', ' ', expr_text).strip()
            return f"{display} = {_fmt(result)}"
        except Exception as e:
            return f"Calculation error: {e}"
