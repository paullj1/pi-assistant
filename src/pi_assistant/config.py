import os
from typing import Any

try:
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# ----------------------------
# Config (env overrides)
# ----------------------------
XDG_CONFIG_HOME = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
CONFIG_PATH = os.path.join(XDG_CONFIG_HOME, "pi-assistant", "config.toml")
DEFAULT_MCP_CONFIG = os.path.join(XDG_CONFIG_HOME, "pi-assistant", "mcp.json")


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

CHAT_MODEL = _as_str("OPENAI_MODEL", "gpt-5.2")
REASONING_EFFORT = _as_str("ASSISTANT_REASONING_EFFORT", "").strip()

TTS_MODEL = _as_str("ASSISTANT_TTS_MODEL", "tts-1")
TTS_VOICE = _as_str("ASSISTANT_TTS_VOICE", "alloy")
TTS_FORMAT = _as_str("ASSISTANT_TTS_FORMAT", "wav")
TTS_CHUNK_CHARS = _as_int("ASSISTANT_TTS_CHUNK_CHARS", 220)
MCP_CONFIG = _as_str("ASSISTANT_MCP_CONFIG", "").strip()

STT_MODEL = _as_str("ASSISTANT_STT_MODEL", "Systran/faster-whisper-small")
STT_STREAM = _as_bool("ASSISTANT_STT_STREAM", False)
STT_STREAM_FINAL_TIMEOUT = _as_float("ASSISTANT_STT_STREAM_FINAL_TIMEOUT", 3.0)
STT_STREAM_PRE_ROLL = _as_float("ASSISTANT_STT_STREAM_PRE_ROLL", 1.5)
STT_STREAM_KEEPALIVE_SECONDS = _as_float("ASSISTANT_STT_STREAM_KEEPALIVE_SECONDS", 0.5)

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

# Wake word (STT)
WAKE_WORD = _as_str("ASSISTANT_WAKE_WORD", "atlas").strip().lower()
WAKE_PHRASES_ENV = _as_str("ASSISTANT_WAKE_PHRASES", "").strip()
WAKE_PHRASES = [p.strip().lower() for p in WAKE_PHRASES_ENV.split(",") if p.strip()]
WAKE_PHRASES = WAKE_PHRASES or ([WAKE_WORD] if WAKE_WORD else [])
WAKE_MAX_SECONDS = _as_float("ASSISTANT_WAKE_MAX_SECONDS", 1.97)
WAKE_SILENCE_SECONDS = _as_float("ASSISTANT_WAKE_SILENCE_SECONDS", 0.6)
WAKE_RMS_THRESHOLD = _as_float("ASSISTANT_WAKE_RMS_THRESHOLD", 0.0005)
WAKE_FUZZY_THRESHOLD = _as_float("ASSISTANT_WAKE_FUZZY_THRESHOLD", 0.6)
WAKE_FULL_FUZZY_THRESHOLD = _as_float("ASSISTANT_WAKE_FULL_FUZZY_THRESHOLD", 0.75)
WAKE_MIN_TOKEN_LEN = _as_int("ASSISTANT_WAKE_MIN_TOKEN_LEN", 3)
WAKE_MIN_RMS_FOR_STT = _as_float("ASSISTANT_WAKE_MIN_RMS_FOR_STT", 0.0010)
