"""
Notes & Lists Tool - T:\\BuddyQ\\core\\tools\\notes_tool.py

Persistent local storage for notes, lists, and reminders.
All data stored in memory/notes.json — survives restarts.

Supports:
  - Notes:       "add a note: buy milk", "read my notes", "delete note 2"
  - Lists:       "add eggs to my shopping list", "read shopping list", "clear shopping list"
  - Quick save:  "save that" (saves last assistant response as a note)
"""

import re
import json
import os
import time
from .base import BaseTool

_NOTES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "memory", "notes.json"
)

# Injected by main.py — last assistant response for "save that"
last_response: str = ""


# ── Storage ───────────────────────────────────────────────────────────────────

def _load() -> dict:
    try:
        if os.path.exists(_NOTES_FILE):
            with open(_NOTES_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"notes": [], "lists": {}}


def _save(data: dict):
    os.makedirs(os.path.dirname(_NOTES_FILE), exist_ok=True)
    tmp = _NOTES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, _NOTES_FILE)


# ── Tool ─────────────────────────────────────────────────────────────────────

class NotesTool(BaseTool):

    @property
    def name(self):
        return "notes"

    @property
    def description(self):
        return (
            "Saves, reads, and manages personal notes and lists. "
            "Use for shopping lists, to-do items, and saving information for later."
        )

    @property
    def keywords(self):
        return [
            "add a note", "make a note", "note that", "write down",
            "add to my", "add to the", "put on my list",
            "shopping list", "to-do", "todo", "grocery list",
            "read my notes", "show my notes", "what are my notes",
            "read the list", "what's on my list", "show my list",
            "delete note", "remove note", "clear the list", "clear my",
            "save that", "save this", "remember this",
        ]

    def run(self, query: str) -> str:
        text  = query.strip()
        lower = text.lower()
        data  = _load()

        # ── LIST operations ───────────────────────────────────────────────────

        # Extract list name if present: "add eggs to my shopping list"
        list_match = re.search(
            r'(?:add|put|remove|delete|clear|read|show|what.?s\s+on)\s+'
            r'(?:.*?\s+(?:to|from|on)\s+(?:my\s+)?)?'
            r'(\w+(?:\s+\w+)?)\s+list',
            lower
        )
        list_name = list_match.group(1).strip() if list_match else None

        # Add item to list
        add_list = re.search(
            r'add\s+(.+?)\s+(?:to|on)\s+(?:my\s+|the\s+)?(?:(\w+(?:\s+\w+)?)\s+)?list',
            lower
        )
        if add_list:
            item = add_list.group(1).strip()
            lname = (add_list.group(2) or "shopping").strip()
            if lname not in data["lists"]:
                data["lists"][lname] = []
            # Avoid duplicates
            if item not in [i.lower() for i in data["lists"][lname]]:
                data["lists"][lname].append(item)
                _save(data)
                return f"Added '{item}' to your {lname} list."
            return f"'{item}' is already on your {lname} list."

        # Remove from list
        remove_list = re.search(
            r'remove\s+(.+?)\s+from\s+(?:my\s+|the\s+)?(?:(\w+(?:\s+\w+)?)\s+)?list',
            lower
        )
        if remove_list:
            item  = remove_list.group(1).strip()
            lname = (remove_list.group(2) or "shopping").strip()
            lst   = data["lists"].get(lname, [])
            new   = [i for i in lst if i.lower() != item]
            if len(new) < len(lst):
                data["lists"][lname] = new
                _save(data)
                return f"Removed '{item}' from your {lname} list."
            return f"'{item}' was not on your {lname} list."

        # Read list
        if re.search(r'\b(read|show|what.?s\s+on|tell\s+me|get)\b', lower) and list_name:
            lst = data["lists"].get(list_name, [])
            if not lst:
                return f"Your {list_name} list is empty."
            items = ", ".join(lst)
            return f"Your {list_name} list: {items}."

        # Clear list
        if re.search(r'\bclear\b', lower) and list_name:
            data["lists"][list_name] = []
            _save(data)
            return f"Your {list_name} list has been cleared."

        # ── NOTE operations ───────────────────────────────────────────────────

        # Save last response
        if re.search(r'\b(save\s+that|save\s+this|remember\s+this)\b', lower):
            content = last_response or text
            ts = time.strftime("%Y-%m-%d %H:%M")
            data["notes"].append({"text": content, "date": ts})
            _save(data)
            return "Saved as a note."

        # Add note
        add_note = re.search(
            r'(?:add\s+a?\s*note|make\s+a?\s*note|note\s+that|write\s+down|remember\s+that)\s*[:\-]?\s*(.+)',
            lower
        )
        if add_note:
            content = add_note.group(1).strip()
            # Use original case from query
            orig_match = re.search(re.escape(add_note.group(1)), text, re.IGNORECASE)
            if orig_match:
                content = orig_match.group(0)
            ts = time.strftime("%Y-%m-%d %H:%M")
            data["notes"].append({"text": content, "date": ts})
            _save(data)
            return f"Note saved: '{content}'"

        # Read notes
        if re.search(r'\b(read|show|list|what\s+are|get)\b.*\bnotes?\b', lower):
            notes = data["notes"]
            if not notes:
                return "You have no saved notes."
            if len(notes) == 1:
                return f"You have 1 note: {notes[0]['text']}"
            lines = [f"You have {len(notes)} notes."]
            for i, n in enumerate(notes[-5:], 1):   # show last 5
                lines.append(f"Note {i}: {n['text']}")
            return " ".join(lines)

        # Delete note by number
        del_match = re.search(r'delete\s+note\s+(\d+)', lower)
        if del_match:
            idx = int(del_match.group(1)) - 1
            if 0 <= idx < len(data["notes"]):
                removed = data["notes"].pop(idx)
                _save(data)
                return f"Deleted note: '{removed['text']}'"
            return f"Note {idx + 1} does not exist."

        # Clear all notes
        if re.search(r'clear\s+(?:all\s+)?(?:my\s+)?notes?', lower):
            data["notes"] = []
            _save(data)
            return "All notes cleared."

        # Fallback: treat as a new note
        ts = time.strftime("%Y-%m-%d %H:%M")
        data["notes"].append({"text": text, "date": ts})
        _save(data)
        return f"Saved as a note: '{text}'"
