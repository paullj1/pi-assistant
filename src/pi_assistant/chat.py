import json

import requests

from . import config
from .utils import debug, debug_payload, headers_json


def openai_chat_stream(messages, tools=None, on_text_delta=None, on_tool_call=None):
    """
    POST {API_BASE}/chat/completions with stream=True.
    Calls on_text_delta for each content delta.
    Returns the assembled assistant message.
    """
    url = f"{config.API_BASE}/chat/completions"
    payload = {
        "model": config.CHAT_MODEL,
        "messages": messages,
        "temperature": 0.4,
        "stream": True,
    }
    if config.REASONING_EFFORT:
        payload["reasoning"] = {"effort": config.REASONING_EFFORT}
    if tools:
        payload["tools"] = tools
    debug(
        f"chat stream request tools={len(payload.get('tools', []))} messages={len(messages)}"
    )
    debug_payload(payload)
    r = requests.post(
        url, headers=headers_json(), json=payload, stream=True, timeout=180
    )
    r.raise_for_status()

    content_parts = []
    tool_calls = {}
    role = "assistant"
    saw_tool_calls = False

    for raw_line in r.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("data:"):
            data = line[5:].strip()
        else:
            continue
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {}) or {}
        if "role" in delta:
            role = delta.get("role") or role
        if "content" in delta:
            text_delta = delta.get("content") or ""
            if text_delta:
                content_parts.append(text_delta)
                if on_text_delta:
                    on_text_delta(text_delta)
        if "tool_calls" in delta:
            if not saw_tool_calls and on_tool_call:
                on_tool_call()
            saw_tool_calls = True
            for call in delta.get("tool_calls", []):
                idx = call.get("index", 0)
                entry = tool_calls.get(idx)
                if entry is None:
                    entry = {
                        "id": call.get("id"),
                        "type": call.get("type"),
                        "function": {"name": "", "arguments": ""},
                    }
                    tool_calls[idx] = entry
                if call.get("id"):
                    entry["id"] = call.get("id")
                if call.get("type"):
                    entry["type"] = call.get("type")
                fn = call.get("function") or {}
                if fn.get("name"):
                    entry["function"]["name"] = fn.get("name")
                if fn.get("arguments"):
                    entry["function"]["arguments"] += fn.get("arguments")

    message = {"role": role, "content": "".join(content_parts)}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return message
