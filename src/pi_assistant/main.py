"""
Raspberry Pi voice assistant (wake word + barge-in)

Pipeline:
  Mic -> wake word -> record until silence (RMS VAD) -> (optional) noise gate -> STT (whisper)
      -> Chat (OpenAI-compatible /v1/chat/completions via your Caddy/OpenWebUI)
      -> TTS (OpenAI-compatible /v1/audio/speech via your Caddy/TTS)
      -> Speaker (aplay)

Key additions vs earlier version:
  - Python-side mic gain (ASSISTANT_MIC_SOFT_GAIN)
  - Optional simple noise gate (ASSISTANT_NOISE_GATE)
  - Optional input device selection (ASSISTANT_INPUT_DEVICE)
  - Optional explicit ALSA playback device for aplay (ASSISTANT_ALSA_PLAYBACK)
"""

import argparse
import json
import os
import queue
import re
import subprocess
import time
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from threading import Condition, Event, Lock, Thread
from typing import Any

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

# ----------------------------
# Config (env overrides)
# ----------------------------
XDG_CONFIG_HOME = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
CONFIG_PATH = os.path.join(XDG_CONFIG_HOME, "pi-assistant", "config.toml")
DEFAULT_MCP_CONFIG = os.path.join(XDG_CONFIG_HOME, "pi-assistant", "mcp.json")

try:
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return {}
    if isinstance(data, dict) and isinstance(data.get("assistant"), dict):
        return data["assistant"]
    return data if isinstance(data, dict) else {}


_CONFIG = _load_config(CONFIG_PATH)


def _get_config_value(key: str, default: Any):
    if key in os.environ:
        return os.environ[key]
    return _CONFIG.get(key, default)


def _as_str(key: str, default: str) -> str:
    val = _get_config_value(key, default)
    return str(val) if val is not None else ""


def _as_int(key: str, default: int) -> int:
    val = _get_config_value(key, default)
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except Exception:
        return default


def _as_float(key: str, default: float) -> float:
    val = _get_config_value(key, default)
    if isinstance(val, float):
        return val
    try:
        return float(val)
    except Exception:
        return default


def _as_bool(key: str, default: bool) -> bool:
    val = _get_config_value(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return default


API_BASE = _as_str("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
API_KEY = _as_str("OPENAI_API_KEY", "")

CHAT_MODEL = _as_str("OPENAI_MODEL", "gpt-oss:120b")
REASONING_EFFORT = _as_str("ASSISTANT_REASONING_EFFORT", "").strip()

TTS_MODEL = _as_str("ASSISTANT_TTS_MODEL", "tts-1")
TTS_VOICE = _as_str("ASSISTANT_TTS_VOICE", "alloy")
TTS_FORMAT = _as_str("ASSISTANT_TTS_FORMAT", "wav")
TTS_CHUNK_CHARS = _as_int("ASSISTANT_TTS_CHUNK_CHARS", 220)
MCP_CONFIG = _as_str("ASSISTANT_MCP_CONFIG", "").strip()

# Audio
SAMPLE_RATE = _as_int("ASSISTANT_SAMPLE_RATE", 16000)
CHANNELS = _as_int("ASSISTANT_CHANNELS", 1)

# Record/VAD tuning
MAX_SECONDS = _as_float("ASSISTANT_MAX_SECONDS", 12.0)
SILENCE_SECONDS = _as_float("ASSISTANT_SILENCE_SECONDS", 0.8)
RMS_THRESHOLD = _as_float("ASSISTANT_RMS_THRESHOLD", 0.010)

# Mic gain + noise control
MIC_SOFT_GAIN = _as_float("ASSISTANT_MIC_SOFT_GAIN", 1.0)  # 1.0 = no boost
NOISE_GATE = _as_float("ASSISTANT_NOISE_GATE", 0.00)  # try 0.005–0.02; 0 disables
CLIP = _as_bool("ASSISTANT_CLIP", True)

# Device selection
INPUT_DEVICE_ENV = _as_str("ASSISTANT_INPUT_DEVICE", "").strip()
INPUT_DEVICE = int(INPUT_DEVICE_ENV) if INPUT_DEVICE_ENV else None

ALSA_PLAYBACK = _as_str("ASSISTANT_ALSA_PLAYBACK", "default")

# Wake cue
WAKE_CUE = _as_str("ASSISTANT_WAKE_CUE", "tts").strip().lower()
WAKE_CUE_TTS = _as_str("ASSISTANT_WAKE_CUE_TTS", "How can I help you?")
WAKE_BEEP_HZ = _as_float("ASSISTANT_WAKE_BEEP_HZ", 880.0)
WAKE_BEEP_MS = _as_int("ASSISTANT_WAKE_BEEP_MS", 120)
WAKE_BEEP_GAIN = _as_float("ASSISTANT_WAKE_BEEP_GAIN", 0.2)
WAKE_LISTEN_SECONDS = _as_float("ASSISTANT_WAKE_LISTEN_SECONDS", 10.0)
WAKE_LISTEN_FULL_WINDOW = _as_bool("ASSISTANT_WAKE_LISTEN_FULL_WINDOW", True)

# Stop cue
STOP_CUE = _as_str("ASSISTANT_STOP_CUE", "beep").strip().lower()
STOP_CUE_TTS = _as_str("ASSISTANT_STOP_CUE_TTS", "Okay.")
STOP_BEEP_HZ = _as_float("ASSISTANT_STOP_BEEP_HZ", 520.0)
STOP_BEEP_MS = _as_int("ASSISTANT_STOP_BEEP_MS", 120)
STOP_BEEP_GAIN = _as_float("ASSISTANT_STOP_BEEP_GAIN", 0.2)

# Working cue
WORKING_CUE = _as_str("ASSISTANT_WORKING_CUE", "beep").strip().lower()
WORKING_CUE_TTS = _as_str("ASSISTANT_WORKING_CUE_TTS", "One moment.")
WORKING_BEEP_HZ = _as_float("ASSISTANT_WORKING_BEEP_HZ", 320.0)
WORKING_BEEP_MS = _as_int("ASSISTANT_WORKING_BEEP_MS", 900)
WORKING_BEEP_GAIN = _as_float("ASSISTANT_WORKING_BEEP_GAIN", 0.18)
WORKING_BEEP_PAUSE_MS = _as_int("ASSISTANT_WORKING_BEEP_PAUSE_MS", 450)
WORKING_BEEP_STYLE = _as_str("ASSISTANT_WORKING_BEEP_STYLE", "sequence").strip().lower()
WORKING_BEEP_HZ2 = _as_float("ASSISTANT_WORKING_BEEP_HZ2", 380.0)
WORKING_BEEP_HZ3 = _as_float("ASSISTANT_WORKING_BEEP_HZ3", 450.0)

# Conversation end detection
END_PROMPT = "Anything else?"
END_USER_RESPONSES = {
    "no",
    "nope",
    "nah",
    "no thanks",
    "no thank you",
    "that's all",
    "that is all",
    "that's it",
    "that is it",
    "all done",
    "done",
}

# Debug logging
DEBUG = _as_bool("ASSISTANT_DEBUG", False)

# Wake word (Whisper)
WAKE_WORD = _as_str("ASSISTANT_WAKE_WORD", "atlas").strip().lower()
WAKE_PHRASES_ENV = _as_str("ASSISTANT_WAKE_PHRASES", "").strip()
WAKE_PHRASES = [p.strip().lower() for p in WAKE_PHRASES_ENV.split(",") if p.strip()]
WAKE_PHRASES = WAKE_PHRASES or [WAKE_WORD]
WAKE_MAX_SECONDS = _as_float("ASSISTANT_WAKE_MAX_SECONDS", 1.97)
WAKE_SILENCE_SECONDS = _as_float("ASSISTANT_WAKE_SILENCE_SECONDS", 0.6)
WAKE_RMS_THRESHOLD = _as_float("ASSISTANT_WAKE_RMS_THRESHOLD", 0.0005)
WAKE_FUZZY_THRESHOLD = _as_float("ASSISTANT_WAKE_FUZZY_THRESHOLD", 0.6)
WAKE_FULL_FUZZY_THRESHOLD = _as_float("ASSISTANT_WAKE_FULL_FUZZY_THRESHOLD", 0.75)
WAKE_MIN_TOKEN_LEN = _as_int("ASSISTANT_WAKE_MIN_TOKEN_LEN", 3)
WAKE_WHISPER_MODEL = _as_str("ASSISTANT_WAKE_WHISPER_MODEL", "tiny.en")
WAKE_MIN_RMS_FOR_STT = _as_float("ASSISTANT_WAKE_MIN_RMS_FOR_STT", 0.0010)


@dataclass
class Turn:
    role: str
    content: str


def _serialize_messages(history):
    messages = []
    for item in history:
        if isinstance(item, Turn):
            messages.append(item.__dict__)
        else:
            messages.append(item)
    return messages


def _headers_json():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _debug(msg: str):
    if DEBUG:
        print(f"[debug] {msg}")


def _debug_payload(payload: dict):
    if not DEBUG:
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


# ----------------------------
# MCP tools (optional)
# ----------------------------
class MCPServer:
    def __init__(self, name: str, config: dict):
        self._name = name
        self._config = config
        self._proc = None
        self._lock = Lock()
        self._cond = Condition(self._lock)
        self._responses = {}
        self._next_id = 1

    def start(self):
        cmd = [self._config["command"]] + self._config.get("args", [])
        env = os.environ.copy()
        env.update(self._config.get("env", {}))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        Thread(target=self._read_stdout, daemon=True).start()
        Thread(target=self._read_stderr, daemon=True).start()
        self._initialize()

    def stop(self):
        if not self._proc:
            return
        try:
            self._proc.terminate()
        except Exception:
            pass

    def _read_stderr(self):
        if not self._proc or not self._proc.stderr:
            return
        for line in self._proc.stderr:
            _debug(f"mcp[{self._name}] stderr: {line.decode(errors='ignore').strip()}")

    def _read_stdout(self):
        if not self._proc or not self._proc.stdout:
            return
        for raw in self._proc.stdout:
            try:
                line = raw.decode("utf-8").strip()
            except Exception:
                continue
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception as e:
                _debug(f"mcp[{self._name}] json decode failed: {e}")
                continue
            self._handle_message(msg)

    def _handle_message(self, msg: dict):
        if "id" in msg:
            with self._cond:
                self._responses[msg["id"]] = msg
                self._cond.notify_all()

    def _send(self, msg: dict):
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("MCP server not running")
        data = (json.dumps(msg) + "\n").encode("utf-8")
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict | None = None, timeout=30):
        with self._cond:
            req_id = self._next_id
            self._next_id += 1
        req = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            req["params"] = params
        self._send(req)
        with self._cond:
            deadline = time.time() + timeout
            while req_id not in self._responses:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"MCP request timeout: {method}")
                self._cond.wait(timeout=remaining)
            return self._responses.pop(req_id)

    def _notify(self, method: str, params: dict | None = None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def _initialize(self):
        self._request(
            "initialize",
            {
                "clientInfo": {"name": "assistant", "version": "1.0"},
                "protocolVersion": "2024-11-05",
                "capabilities": {},
            },
        )
        self._notify("notifications/initialized", {})

    def list_tools(self):
        resp = self._request("tools/list")
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict):
        resp = self._request("tools/call", {"name": name, "arguments": arguments})
        return resp.get("result", {})


class MCPManager:
    def __init__(self, config_path: str):
        self._config_path = config_path
        self._servers = {}
        self._tool_map = {}
        self._tools = []

    def start(self):
        if not self._config_path:
            return
        try:
            with open(self._config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            _debug(f"failed to read MCP config: {e}")
            return

        if not isinstance(config, dict) or "mcpServers" not in config:
            _debug("invalid MCP config: expected top-level mcpServers")
            return

        servers = config.get("mcpServers", {})

        for name, srv in servers.items():
            try:
                transport = srv.get("transport", "stdio")
                if transport != "stdio":
                    _debug(f"mcp server {name} unsupported transport: {transport}")
                    continue
                server = MCPServer(name, srv)
                server.start()
                self._servers[name] = server
            except Exception as e:
                _debug(f"mcp server {name} failed to start: {e}")

        self._load_tools()

    def _load_tools(self):
        for name, server in self._servers.items():
            try:
                tools = server.list_tools()
            except Exception as e:
                _debug(f"mcp server {name} list tools failed: {e}")
                continue

            for tool in tools:
                tool_name = f"{name}.{tool['name']}"
                self._tool_map[tool_name] = (name, tool["name"])
                self._tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool.get("description", ""),
                            "parameters": tool.get("inputSchema", {}),
                        },
                    }
                )

    def tools(self):
        return self._tools

    def call(self, tool_name: str, arguments: dict) -> str:
        mapping = self._tool_map.get(tool_name)
        if not mapping:
            raise KeyError(f"Unknown MCP tool: {tool_name}")
        server_name, raw_name = mapping
        result = self._servers[server_name].call_tool(raw_name, arguments)
        content = result.get("content", [])
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(json.dumps(item))
        return "\n".join(p for p in parts if p)


# ----------------------------
# Shared audio stream
# ----------------------------
class AudioStream:
    def __init__(self):
        self._queue = queue.Queue()
        self._status_count = 0

        def cb(indata, frames, time_info, status):
            if status:
                self._status_count += 1
                _debug(f"audio callback status={status} count={self._status_count}")
            self._queue.put(indata.copy())

        self._stream = sd.InputStream(
            device=INPUT_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=cb,
            blocksize=0,
        )

    def start(self):
        self._stream.start()
        _debug("audio stream started")

    def pause(self):
        if self._stream.active:
            self._stream.stop()
            _debug("audio stream paused")

    def resume(self):
        if not self._stream.active:
            self._stream.start()
            _debug("audio stream resumed")

    def stop(self):
        self._stream.stop()
        self._stream.close()
        _debug("audio stream stopped")

    def get(self, timeout=0.1):
        return self._queue.get(timeout=timeout)

    def drain(self):
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            _debug(f"audio queue drained {drained} chunks")


# ----------------------------
# OpenAI-compatible chat
# ----------------------------
def openai_chat(messages, tools=None):
    """
    POST {API_BASE}/chat/completions
    Returns assistant text.
    """
    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.4,
        "stream": False,
    }
    if REASONING_EFFORT:
        payload["reasoning"] = {"effort": REASONING_EFFORT}
    if tools:
        payload["tools"] = tools
    _debug(
        f"chat request tools={len(payload.get('tools', []))} messages={len(messages)}"
    )
    _debug_payload(payload)
    r = requests.post(url, headers=_headers_json(), json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]


# ----------------------------
# OpenAI-compatible TTS
# ----------------------------
def _synthesize_tts(text: str) -> str:
    """
    POST {API_BASE}/audio/speech and write audio to /tmp.
    Returns the output path.
    """
    url = f"{API_BASE}/audio/speech"
    payload = {
        "model": TTS_MODEL,
        "voice": TTS_VOICE,
        "input": text,
        "response_format": TTS_FORMAT,
    }
    r = requests.post(url, headers=_headers_json(), json=payload, timeout=180)
    r.raise_for_status()

    fmt = TTS_FORMAT.lower()
    out_path = f"/tmp/assistant_tts_{int(time.time() * 1000)}.{fmt}"
    with open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


def _play_audio(path: str) -> subprocess.Popen:
    fmt = TTS_FORMAT.lower()
    if fmt == "wav":
        return subprocess.Popen(["aplay", "-q", "-D", ALSA_PLAYBACK, path])
    else:
        return subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]
        )


def speak_async(text: str, working_cue: bool = False) -> subprocess.Popen:
    """
    Synthesize and play TTS without blocking.
    """
    stop_event = None
    thread = None
    if working_cue:
        stop_event, thread = _start_working_cue_loop()
    try:
        out_path = _synthesize_tts(text)
    finally:
        _stop_working_cue_loop(stop_event, thread)
    return _play_audio(out_path)


def _split_tts_chunks(text: str, max_chars: int) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    parts = cleaned.split(". ")
    sentences = []
    for i, part in enumerate(parts):
        if not part:
            continue
        if i < len(parts) - 1:
            sentences.append(part.rstrip() + ".")
        else:
            sentences.append(part)

    chunks = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                if len(current) + len(word) + 1 > max_chars and current:
                    chunks.append(current)
                    current = word
                else:
                    current = f"{current} {word}".strip()
            continue

        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks


def stop_playback(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except Exception:
            proc.kill()


_CUE_PATHS = {}


def _ensure_beep(path_key: str, hz: float, ms: int, gain: float) -> str:
    if path_key in _CUE_PATHS:
        return _CUE_PATHS[path_key]

    duration_s = max(0.02, ms / 1000.0)
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    wave_data = (np.sin(2 * np.pi * hz * t) * gain).astype(np.float32)
    pcm16 = (wave_data * 32767.0).astype(np.int16)

    out_path = f"/tmp/assistant_{path_key}.wav"
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

    _CUE_PATHS[path_key] = out_path
    return out_path


def _ensure_working_beep() -> str:
    path_key = "working_cue"
    if path_key in _CUE_PATHS:
        return _CUE_PATHS[path_key]

    duration_s = max(0.02, WORKING_BEEP_MS / 1000.0)
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)

    if WORKING_BEEP_STYLE == "sequence":
        thirds = max(1, len(t) // 3)
        wave_data = np.empty_like(t)
        wave_data[:thirds] = np.sin(2 * np.pi * WORKING_BEEP_HZ * t[:thirds])
        wave_data[thirds : 2 * thirds] = np.sin(
            2 * np.pi * WORKING_BEEP_HZ2 * t[thirds : 2 * thirds]
        )
        wave_data[2 * thirds :] = np.sin(2 * np.pi * WORKING_BEEP_HZ3 * t[2 * thirds :])
    elif WORKING_BEEP_STYLE == "swoop":
        f0 = max(20.0, WORKING_BEEP_HZ * 0.8)
        f1 = max(20.0, WORKING_BEEP_HZ2)
        freqs = f0 + (f1 - f0) * (t / duration_s)
        phase = 2 * np.pi * np.cumsum(freqs) / SAMPLE_RATE
        wave_data = np.sin(phase)
    elif WORKING_BEEP_STYLE == "chime":
        split = max(1, len(t) // 2)
        wave_data = np.empty_like(t)
        wave_data[:split] = np.sin(2 * np.pi * WORKING_BEEP_HZ * t[:split])
        wave_data[split:] = np.sin(2 * np.pi * WORKING_BEEP_HZ2 * t[split:])
    else:
        wave_data = np.sin(2 * np.pi * WORKING_BEEP_HZ * t)

    fade = np.linspace(1.0, 0.0, num=wave_data.size, endpoint=True)
    wave_data = (wave_data * fade * WORKING_BEEP_GAIN).astype(np.float32)
    pcm16 = (wave_data * 32767.0).astype(np.int16)

    out_path = "/tmp/assistant_working_cue.wav"
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

    _CUE_PATHS[path_key] = out_path
    return out_path


def _play_cue(mode: str, tts_text: str, beep_key: str, hz: float, ms: int, gain: float):
    if mode in ("off", "none", "0", "false"):
        _debug(f"{beep_key} cue disabled")
        return

    if mode == "tts":
        _debug(f"{beep_key} cue tts")
        proc = speak_async(tts_text)
        proc.wait()
        return

    if mode == "beep":
        _debug(f"{beep_key} cue beep")
        if beep_key == "working_cue":
            path = _ensure_working_beep()
        else:
            path = _ensure_beep(beep_key, hz, ms, gain)
        subprocess.run(["aplay", "-q", "-D", ALSA_PLAYBACK, path], check=False)
        return


def play_wake_cue():
    _play_cue(
        WAKE_CUE, WAKE_CUE_TTS, "wake_cue", WAKE_BEEP_HZ, WAKE_BEEP_MS, WAKE_BEEP_GAIN
    )


def play_stop_cue():
    _play_cue(
        STOP_CUE, STOP_CUE_TTS, "stop_cue", STOP_BEEP_HZ, STOP_BEEP_MS, STOP_BEEP_GAIN
    )


def play_working_cue():
    _play_cue(
        WORKING_CUE,
        WORKING_CUE_TTS,
        "working_cue",
        WORKING_BEEP_HZ,
        WORKING_BEEP_MS,
        WORKING_BEEP_GAIN,
    )


def _start_working_cue_loop():
    if WORKING_CUE in ("off", "none", "0", "false"):
        return None, None

    if WORKING_CUE == "tts":
        try:
            play_working_cue()
        except Exception as e:
            _debug(f"working cue failed: {e}")
        return None, None

    stop_event = Event()

    def _loop():
        pause_s = max(0.02, WORKING_BEEP_PAUSE_MS / 1000.0)
        while not stop_event.is_set():
            _play_cue(
                WORKING_CUE,
                WORKING_CUE_TTS,
                "working_cue",
                WORKING_BEEP_HZ,
                WORKING_BEEP_MS,
                WORKING_BEEP_GAIN,
            )
            time.sleep(pause_s)

    thread = Thread(target=_loop, daemon=True)
    thread.start()
    return stop_event, thread


def _stop_working_cue_loop(stop_event, thread):
    if stop_event is None:
        return
    stop_event.set()
    if thread is not None:
        thread.join(timeout=1.0)


def _listen_for_user(audio_stream: AudioStream, play_cue: bool) -> str:
    if play_cue:
        try:
            play_wake_cue()
        except Exception as e:
            _debug(f"wake cue failed: {e}")

    if WAKE_LISTEN_FULL_WINDOW:
        pcm = record_after_speech_start_from_stream(
            audio_stream,
            max_wait_seconds=WAKE_LISTEN_SECONDS,
            max_seconds=WAKE_LISTEN_SECONDS,
            silence_seconds=SILENCE_SECONDS,
            threshold=RMS_THRESHOLD,
            drain=False,
        )
    else:
        pcm = record_until_silence_from_stream(
            audio_stream,
            max_seconds=WAKE_LISTEN_SECONDS,
            silence_seconds=SILENCE_SECONDS,
            drain=False,
        )
    if not pcm:
        return ""

    return stt_whisper(pcm)


# ----------------------------
# Audio record + VAD
# ----------------------------
def record_until_silence(
    max_seconds=MAX_SECONDS,
    silence_seconds=SILENCE_SECONDS,
    threshold=RMS_THRESHOLD,
):
    """
    Records float32 audio until silence for `silence_seconds` (simple RMS VAD),
    or until `max_seconds` is reached.
    Returns raw PCM int16 bytes (mono @ SAMPLE_RATE).
    """
    q = queue.Queue()

    def cb(indata, frames, time_info, status):
        q.put(indata.copy())

    chunks = []
    silence_run = 0.0
    start = time.time()

    with sd.InputStream(
        device=INPUT_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=cb,
        blocksize=0,
    ):
        while True:
            chunk = q.get()
            chunks.append(chunk)

            rms = float(np.sqrt(np.mean(chunk**2)))
            dt = len(chunk) / float(SAMPLE_RATE)

            silence_run = silence_run + dt if rms < threshold else 0.0

            elapsed = time.time() - start
            if silence_run >= silence_seconds and elapsed > 0.6:
                break
            if elapsed >= max_seconds:
                break

    audio = np.concatenate(chunks, axis=0)

    # Downmix if needed
    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1, keepdims=True)

    audio = audio.reshape(-1)

    # --- gain + noise control ---
    if MIC_SOFT_GAIN != 1.0:
        audio = audio * MIC_SOFT_GAIN

    if CLIP:
        audio = np.clip(audio, -1.0, 1.0)

    # Simple noise gate (kills hiss in quiet parts)
    if NOISE_GATE and NOISE_GATE > 0.0:
        audio[np.abs(audio) < NOISE_GATE] = 0.0

    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def record_until_silence_from_stream(
    audio_stream: AudioStream,
    max_seconds=MAX_SECONDS,
    silence_seconds=SILENCE_SECONDS,
    threshold=RMS_THRESHOLD,
    drain=True,
):
    chunks = []
    silence_run = 0.0
    start = time.time()

    if drain:
        audio_stream.drain()

    while True:
        try:
            chunk = audio_stream.get(timeout=0.1)
        except queue.Empty:
            if time.time() - start >= max_seconds:
                break
            continue

        chunks.append(chunk)
        rms = float(np.sqrt(np.mean(chunk**2)))
        dt = len(chunk) / float(SAMPLE_RATE)

        silence_run = silence_run + dt if rms < threshold else 0.0

        elapsed = time.time() - start
        if silence_run >= silence_seconds and elapsed > 0.6:
            break
        if elapsed >= max_seconds:
            break

    if not chunks:
        return b""

    audio = np.concatenate(chunks, axis=0)

    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1, keepdims=True)

    audio = audio.reshape(-1)

    if MIC_SOFT_GAIN != 1.0:
        audio = audio * MIC_SOFT_GAIN

    if CLIP:
        audio = np.clip(audio, -1.0, 1.0)

    if NOISE_GATE and NOISE_GATE > 0.0:
        audio[np.abs(audio) < NOISE_GATE] = 0.0

    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def record_after_speech_start_from_stream(
    audio_stream: AudioStream,
    max_wait_seconds: float,
    max_seconds: float,
    silence_seconds: float,
    threshold: float,
    drain: bool = True,
):
    if drain:
        audio_stream.drain()

    started = False
    chunks = []
    silence_run = 0.0
    start = time.time()

    while True:
        try:
            chunk = audio_stream.get(timeout=0.1)
        except queue.Empty:
            elapsed = time.time() - start
            if not started and elapsed >= max_wait_seconds:
                break
            if started and elapsed >= max_seconds:
                break
            continue

        rms = float(np.sqrt(np.mean(chunk**2)))
        dt = len(chunk) / float(SAMPLE_RATE)

        if not started:
            if rms >= threshold:
                started = True
            else:
                if time.time() - start >= max_wait_seconds:
                    break
                continue

        chunks.append(chunk)
        silence_run = silence_run + dt if rms < threshold else 0.0

        elapsed = time.time() - start
        if silence_run >= silence_seconds and elapsed > 0.6:
            break
        if elapsed >= max_seconds:
            break

    if not chunks:
        return b""

    audio = np.concatenate(chunks, axis=0)

    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1, keepdims=True)

    audio = audio.reshape(-1)

    if MIC_SOFT_GAIN != 1.0:
        audio = audio * MIC_SOFT_GAIN

    if CLIP:
        audio = np.clip(audio, -1.0, 1.0)

    if NOISE_GATE and NOISE_GATE > 0.0:
        audio[np.abs(audio) < NOISE_GATE] = 0.0

    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16.tobytes()


# ----------------------------
# Offline STT via Whisper
# ----------------------------
_WHISPER = None


def stt_whisper(pcm16_bytes: bytes) -> str:
    global _WHISPER
    if _WHISPER is None:
        _WHISPER = WhisperModel(
            "tiny.en",  # try tiny.en first if you want
            device="cpu",
            compute_type="int8",
        )

    audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    segments, _ = _WHISPER.transcribe(
        audio,
        language="en",
        vad_filter=True,  # important
    )

    return " ".join(seg.text.strip() for seg in segments).strip()


_WHISPER_WAKE = None


def stt_whisper_wake(pcm16_bytes: bytes) -> str:
    global _WHISPER_WAKE
    if _WHISPER_WAKE is None:
        _debug(f"loading wake whisper model: {WAKE_WHISPER_MODEL}")
        _WHISPER_WAKE = WhisperModel(
            WAKE_WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
        )

    audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    segments, _ = _WHISPER_WAKE.transcribe(
        audio,
        language="en",
        vad_filter=True,
        beam_size=1,
    )

    return " ".join(seg.text.strip() for seg in segments).strip()


def print_audio_devices():
    print("\nPortAudio devices (sounddevice):")
    try:
        print(sd.query_devices())
    except Exception as e:
        print("Could not query devices:", e)
    print("")


def _normalize_text(text: str) -> str:
    cleaned = []
    for ch in text.lower():
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def _contains_wake_word(text: str) -> bool:
    normalized = _normalize_text(text)
    for phrase in WAKE_PHRASES:
        phrase_norm = _normalize_text(phrase)
        if phrase_norm in normalized:
            return True

        ratio = SequenceMatcher(None, normalized, phrase_norm).ratio()
        if DEBUG:
            _debug(f"wake fuzzy full '{phrase_norm}' vs '{normalized}' -> {ratio:.2f}")
        if ratio >= WAKE_FULL_FUZZY_THRESHOLD:
            return True

        normalized_joined = normalized.replace(" ", "")
        phrase_joined = phrase_norm.replace(" ", "")
        ratio = SequenceMatcher(None, normalized_joined, phrase_joined).ratio()
        if DEBUG:
            _debug(
                f"wake fuzzy join '{phrase_joined}' vs '{normalized_joined}' -> {ratio:.2f}"
            )
        if ratio >= WAKE_FULL_FUZZY_THRESHOLD:
            return True

        phrase_tokens = [t for t in phrase_norm.split() if len(t) >= WAKE_MIN_TOKEN_LEN]
        text_tokens = [t for t in normalized.split() if len(t) >= WAKE_MIN_TOKEN_LEN]

        for p_tok in phrase_tokens:
            for t_tok in text_tokens:
                ratio = SequenceMatcher(None, p_tok, t_tok).ratio()
                if DEBUG:
                    _debug(f"wake fuzzy '{p_tok}' vs '{t_tok}' -> {ratio:.2f}")
                if ratio >= WAKE_FUZZY_THRESHOLD:
                    return True

    return False


def _should_end_conversation(reply: str) -> bool:
    return _normalize_text(reply).endswith(_normalize_text(END_PROMPT))


def _is_user_done(text: str) -> bool:
    normalized = _normalize_text(text)
    return normalized in END_USER_RESPONSES


def wait_for_wake_word(
    wake_event: Event,
    stop_event: Event,
    audio_stream: AudioStream,
):
    while not stop_event.is_set():
        _debug(
            "listening for wake word "
            f"(max={WAKE_MAX_SECONDS}s silence={WAKE_SILENCE_SECONDS}s "
            f"rms<{WAKE_RMS_THRESHOLD})"
        )
        pcm = record_until_silence_from_stream(
            audio_stream,
            max_seconds=WAKE_MAX_SECONDS,
            silence_seconds=WAKE_SILENCE_SECONDS,
            threshold=WAKE_RMS_THRESHOLD,
        )
        if not pcm:
            _debug("wake chunk empty")
            continue

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
        if rms < WAKE_MIN_RMS_FOR_STT:
            _debug(f"wake chunk rms={rms:.4f} below stt threshold, skipping")
            continue

        audio_stream.pause()
        try:
            text = stt_whisper_wake(pcm)
        finally:
            audio_stream.resume()
        if DEBUG:
            _debug(f"wake chunk rms={rms:.4f} text={text!r}")
        if text and _contains_wake_word(text):
            _debug("wake word detected")
            wake_event.set()
            return


def main():
    parser = argparse.ArgumentParser(description="Pi Assistant")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (overrides ASSISTANT_DEBUG).",
    )
    args = parser.parse_args()

    global DEBUG
    if args.debug:
        DEBUG = True

    if args.list_devices or _as_bool("ASSISTANT_LIST_DEVICES", False):
        print_audio_devices()
        return

    mcp_config_path = MCP_CONFIG or DEFAULT_MCP_CONFIG
    mcp_manager = MCPManager(mcp_config_path)
    mcp_manager.start()
    mcp_tools = mcp_manager.tools()
    if DEBUG:
        _debug(f"mcp tools loaded: {len(mcp_tools)}")
        if mcp_tools:
            _debug(f"mcp tool names: {[t['function']['name'] for t in mcp_tools]}")

    print("Pi Voice Assistant (wake word + barge-in)\n")
    print(f"API base:      {API_BASE}")
    print(f"Chat model:    {CHAT_MODEL}")
    if mcp_config_path:
        print(f"MCP config:    {mcp_config_path}")
        print(f"MCP tools:     {len(mcp_tools)}")
    if REASONING_EFFORT:
        print(f"Reasoning:     {REASONING_EFFORT}")
    print(f"TTS:           model={TTS_MODEL} voice={TTS_VOICE} format={TTS_FORMAT}")
    print(f"Playback:      ALSA={ALSA_PLAYBACK}")
    print(f"Wake word:     {', '.join(WAKE_PHRASES)}")
    print(f"Wake model:    {WAKE_WHISPER_MODEL}")
    print(f"Wake cue:      {WAKE_CUE}")
    print(f"Working cue:   {WORKING_CUE}")
    print(f"Working style:{'':1} {WORKING_BEEP_STYLE}")
    print(f"Listen window: {WAKE_LISTEN_SECONDS:.1f}s")
    print(f"Listen full:   {WAKE_LISTEN_FULL_WINDOW}")
    print(f"Mic gain:      {MIC_SOFT_GAIN}x   Noise gate: {NOISE_GATE}   Clip: {CLIP}")
    print(f"Input device:  {INPUT_DEVICE if INPUT_DEVICE is not None else '(default)'}")
    print("\nSay the wake word to speak. Say it again to interrupt.\n")

    history = [
        Turn(
            "system",
            "You are a helpful voice assistant. Keep replies concise and conversational. "
            "Normalize your text for TTS.  Expand abbreviations like 'Ave.' to 'Avenue', "
            "'Dr.' to Doctor, 'e.g.' to 'for example', 'etc.' to 'and so on'. Spell out numbers "
            "and symbols (e.g., 1st to first, $100 to one hundred dollars, 90ºF to ninety "
            " degrees fahrenheit).  Read text naturally, ensuring clarity for the user. "
            f"When you are done with a user's request, end by asking exactly: '{END_PROMPT}' "
            "and wait for a response. If the user responds negatively (e.g., no, no thanks), "
            "consider the conversation ended and do not ask further questions.",
        )
    ]

    wake_event = Event()
    audio_stream = AudioStream()

    try:
        audio_stream.start()
        pending_wake = False

        while True:
            if not pending_wake:
                stop_event = Event()
                listener = Thread(
                    target=wait_for_wake_word,
                    args=(wake_event, stop_event, audio_stream),
                    daemon=True,
                )
                listener.start()
                wake_event.wait()
                stop_event.set()
                listener.join()
            wake_event.clear()
            pending_wake = False

            pending_user_text = None
            prompt_cue = True
            while True:
                if pending_user_text is None:
                    text = _listen_for_user(audio_stream, play_cue=prompt_cue)
                else:
                    text = pending_user_text
                    pending_user_text = None

                prompt_cue = False

                if not text:
                    print("(no speech recognized)")
                    try:
                        play_stop_cue()
                    except Exception as e:
                        _debug(f"stop cue failed: {e}")
                    break

                print(f"You: {text}")
                history.append(Turn("user", text))

                stop_cue_event, cue_thread = _start_working_cue_loop()
                try:
                    reply_msg = openai_chat(_serialize_messages(history), mcp_tools)
                    while reply_msg.get("tool_calls"):
                        if not mcp_tools:
                            _debug("model requested tools but none are configured")
                            reply_msg = {
                                "role": "assistant",
                                "content": "Tools are unavailable.",
                            }
                            break
                        history.append(reply_msg)
                        for call in reply_msg.get("tool_calls", []):
                            tool_name = call.get("function", {}).get("name", "")
                            args_raw = call.get("function", {}).get("arguments", "{}")
                            try:
                                args = json.loads(args_raw) if args_raw else {}
                            except Exception:
                                args = {}
                            try:
                                result = mcp_manager.call(tool_name, args)
                            except Exception as e:
                                result = f"Tool error: {e}"
                            history.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.get("id"),
                                    "content": result,
                                }
                            )
                        reply_msg = openai_chat(_serialize_messages(history), mcp_tools)
                except requests.RequestException as e:
                    print("LLM request failed:", e)
                    try:
                        speak("I had trouble reaching the language model.")
                    except Exception:
                        pass
                    continue
                except Exception as e:
                    print("LLM error:", e)
                    try:
                        speak("Something went wrong.")
                    except Exception:
                        pass
                    continue
                finally:
                    _stop_working_cue_loop(stop_cue_event, cue_thread)

                reply = (reply_msg.get("content") or "").strip()
                print(f"Assistant: {reply}\n")
                history.append({"role": "assistant", "content": reply})

                if reply:
                    chunks = _split_tts_chunks(reply, TTS_CHUNK_CHARS)
                    if not chunks:
                        chunks = [reply]

                    stop_event = Event()
                    interrupt_listener = Thread(
                        target=wait_for_wake_word,
                        args=(wake_event, stop_event, audio_stream),
                        daemon=True,
                    )
                    interrupt_listener.start()

                    interrupted = False
                    audio_queue = queue.Queue()
                    synth_done = Event()

                    def _prefetch():
                        for chunk in chunks:
                            try:
                                path = _synthesize_tts(chunk)
                            except Exception as e:
                                audio_queue.put(e)
                                break
                            audio_queue.put(path)
                        synth_done.set()

                    synth_thread = Thread(target=_prefetch, daemon=True)
                    synth_thread.start()

                    stop_cue_event, cue_thread = _start_working_cue_loop()
                    first_audio = True

                    while not synth_done.is_set() or not audio_queue.empty():
                        try:
                            item = audio_queue.get(timeout=0.1)
                        except queue.Empty:
                            if wake_event.is_set():
                                wake_event.clear()
                                interrupted = True
                                break
                            continue

                        if first_audio:
                            _stop_working_cue_loop(stop_cue_event, cue_thread)
                            first_audio = False

                        if isinstance(item, Exception):
                            print("TTS failed:", item)
                            interrupted = True
                            break

                        proc = _play_audio(item)
                        while proc.poll() is None:
                            if wake_event.is_set():
                                wake_event.clear()
                                stop_playback(proc)
                                interrupted = True
                                break
                            time.sleep(0.05)
                        if interrupted:
                            break

                    _stop_working_cue_loop(stop_cue_event, cue_thread)

                    stop_event.set()
                    interrupt_listener.join()

                    if interrupted:
                        prompt_cue = True
                        continue

                if _should_end_conversation(reply):
                    follow_text = _listen_for_user(audio_stream, play_cue=False)
                    if not follow_text:
                        try:
                            play_stop_cue()
                        except Exception as e:
                            _debug(f"stop cue failed: {e}")
                        break
                    if _is_user_done(follow_text):
                        try:
                            play_stop_cue()
                        except Exception as e:
                            _debug(f"stop cue failed: {e}")
                        break

                    pending_user_text = follow_text
                    prompt_cue = False

    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        audio_stream.stop()


if __name__ == "__main__":
    main()
