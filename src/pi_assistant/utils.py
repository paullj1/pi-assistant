import json
import re
from dataclasses import dataclass

from . import config


@dataclass
class Turn:
    role: str
    content: str


def serialize_messages(history):
    messages = []
    for item in history:
        if isinstance(item, Turn):
            messages.append(item.__dict__)
        else:
            messages.append(item)
    return messages


def headers_json():
    h = {"Content-Type": "application/json"}
    if config.API_KEY:
        h["Authorization"] = f"Bearer {config.API_KEY}"
    return h


def headers_auth():
    h = {}
    if config.API_KEY:
        h["Authorization"] = f"Bearer {config.API_KEY}"
    return h


def headers_ws():
    headers = []
    if config.API_KEY:
        headers.append(f"Authorization: Bearer {config.API_KEY}")
    return headers


def api_base_ws() -> str:
    if config.API_BASE.startswith("https://"):
        return "wss://" + config.API_BASE[len("https://") :]
    if config.API_BASE.startswith("http://"):
        return "ws://" + config.API_BASE[len("http://") :]
    return config.API_BASE


def debug(msg: str):
    if config.DEBUG:
        print(f"[debug] {msg}")


def debug_payload(payload: dict):
    if not config.DEBUG:
        return
    safe = dict(payload)
    if "messages" in safe:
        safe["messages"] = [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in safe["messages"]
        ]
    if "tools" in safe:
        safe["tools"] = safe.get("tools", [])
    print(f"[debug] chat payload: {json.dumps(safe, ensure_ascii=True)}")


def normalize_text(text: str) -> str:
    cleaned = []
    for ch in text.lower():
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def extract_complete_sentences(text: str) -> tuple[list[str], str]:
    sentences = []
    start = 0
    for match in re.finditer(r"[.!?]+(?:\s+|$)", text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    return sentences, text[start:]
