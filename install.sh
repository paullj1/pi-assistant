#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
CONFIG_DIR="$XDG_CONFIG_HOME/pi-assistant"
SERVICE_DIR="$XDG_CONFIG_HOME/systemd/user"
SERVICE_PATH="$SERVICE_DIR/pi-assistant.service"
BIN_PATH="$(command -v pi-assistant || true)"
if [[ -z "$BIN_PATH" ]]; then
  BIN_PATH="$HOME/.local/bin/pi-assistant"
fi

mkdir -p "$CONFIG_DIR"
mkdir -p "$SERVICE_DIR"

DEFAULT_CONFIG_CONTENT=$(
  cat <<'EOF'
[assistant]
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-5.2"
ASSISTANT_REASONING_EFFORT = ""

ASSISTANT_TTS_MODEL = "tts-1"
ASSISTANT_TTS_VOICE = "alloy"
ASSISTANT_TTS_FORMAT = "wav"
ASSISTANT_TTS_CHUNK_CHARS = 220
ASSISTANT_STT_MODEL = "Systran/faster-whisper-small"
ASSISTANT_STT_STREAM = false
ASSISTANT_STT_STREAM_FINAL_TIMEOUT = 3.0
ASSISTANT_STT_STREAM_PRE_ROLL = 1.5

ASSISTANT_SAMPLE_RATE = 16000
ASSISTANT_CHANNELS = 1

ASSISTANT_MAX_SECONDS = 12.0
ASSISTANT_SILENCE_SECONDS = 0.8
ASSISTANT_RMS_THRESHOLD = 0.010

ASSISTANT_MIC_SOFT_GAIN = 1.0
ASSISTANT_NOISE_GATE = 0.0
ASSISTANT_CLIP = true

ASSISTANT_INPUT_DEVICE = ""
ASSISTANT_ALSA_PLAYBACK = "default"

ASSISTANT_WAKE_CUE = "tts"
ASSISTANT_WAKE_CUE_TTS = "How can I help you?"
ASSISTANT_WAKE_BEEP_HZ = 880.0
ASSISTANT_WAKE_BEEP_MS = 120
ASSISTANT_WAKE_BEEP_GAIN = 0.2
ASSISTANT_WAKE_LISTEN_SECONDS = 10.0
ASSISTANT_WAKE_LISTEN_FULL_WINDOW = true

ASSISTANT_STOP_CUE = "beep"
ASSISTANT_STOP_CUE_TTS = "Okay."
ASSISTANT_STOP_BEEP_HZ = 520.0
ASSISTANT_STOP_BEEP_MS = 120
ASSISTANT_STOP_BEEP_GAIN = 0.2

ASSISTANT_WORKING_CUE = "beep"
ASSISTANT_WORKING_CUE_TTS = "One moment."
ASSISTANT_WORKING_BEEP_HZ = 320.0
ASSISTANT_WORKING_BEEP_MS = 900
ASSISTANT_WORKING_BEEP_GAIN = 0.18
ASSISTANT_WORKING_BEEP_PAUSE_MS = 450
ASSISTANT_WORKING_BEEP_STYLE = "sequence"
ASSISTANT_WORKING_BEEP_HZ2 = 380.0
ASSISTANT_WORKING_BEEP_HZ3 = 450.0

ASSISTANT_DEBUG = false

ASSISTANT_WAKE_WORD = "atlas"
ASSISTANT_WAKE_PHRASES = ""
ASSISTANT_WAKE_MAX_SECONDS = 1.97
ASSISTANT_WAKE_SILENCE_SECONDS = 0.6
ASSISTANT_WAKE_RMS_THRESHOLD = 0.0005
ASSISTANT_WAKE_FUZZY_THRESHOLD = 0.6
ASSISTANT_WAKE_FULL_FUZZY_THRESHOLD = 0.75
ASSISTANT_WAKE_MIN_TOKEN_LEN = 3
ASSISTANT_WAKE_MIN_RMS_FOR_STT = 0.001

ASSISTANT_MCP_CONFIG = ""
ASSISTANT_LIST_DEVICES = false
EOF
)

if [[ ! -f "$CONFIG_DIR/mcp.json" ]]; then
  cat > "$CONFIG_DIR/mcp.json" <<'EOF'
{
  "mcpServers": {
    "ddg-search": {
      "command": "uvx",
      "args": ["duckduckgo-mcp-server"]
    },
    "mcp-datetime": {
      "command": "uvx",
      "args": ["mcp-datetime"]
    }
  }
}
EOF
fi

if [[ ! -f "$CONFIG_DIR/config.toml" ]]; then
  printf "%s\n" "$DEFAULT_CONFIG_CONTENT" > "$CONFIG_DIR/config.toml"
else
  PI_ASSISTANT_DEFAULT_CONFIG="$DEFAULT_CONFIG_CONTENT" python3 - "$CONFIG_DIR/config.toml" <<'PY'
import os
import sys
import tomllib

path = sys.argv[1]
defaults_raw = os.environ.get("PI_ASSISTANT_DEFAULT_CONFIG", "")
defaults = tomllib.loads(defaults_raw) if defaults_raw else {}

with open(path, "rb") as f:
    existing = tomllib.load(f)


def _serialize_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_serialize_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported value type: {type(value)}")


def _write_section(lines, section_name, section_data, ordered_keys=None):
    lines.append(f"[{section_name}]")
    keys = ordered_keys or list(section_data.keys())
    for key in keys:
        if key not in section_data:
            continue
        lines.append(f"{key} = {_serialize_value(section_data[key])}")
    lines.append("")


merged = dict(existing)

defaults_assistant = defaults.get("assistant", {})
existing_assistant = existing.get("assistant", {})
assistant_keys = list(defaults_assistant.keys())

merged_assistant = {}
for key in assistant_keys:
    if key in existing_assistant:
        merged_assistant[key] = existing_assistant[key]
    else:
        merged_assistant[key] = defaults_assistant[key]

if defaults_assistant:
    merged["assistant"] = merged_assistant

for section, value in defaults.items():
    if section == "assistant":
        continue
    if section not in merged:
        merged[section] = value

lines = []
if "assistant" in merged:
    _write_section(lines, "assistant", merged["assistant"], assistant_keys)

for section, value in merged.items():
    if section == "assistant":
        continue
    if isinstance(value, dict):
        _write_section(lines, section, value)
    else:
        lines.append(f"{section} = {_serialize_value(value)}")

content = "\n".join(lines).rstrip() + "\n"
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
PY
fi

if command -v uv >/dev/null 2>&1; then
  if compgen -G "$ROOT_DIR"/*.whl > /dev/null; then
    WHEEL_PATH="$(ls -1 "$ROOT_DIR"/*.whl | head -n 1)"
    uv tool install --force "$WHEEL_PATH"
  elif [[ -f "$ROOT_DIR/pyproject.toml" ]]; then
    uv pip install "$ROOT_DIR"
  else
    uv tool install --force "git+https://github.com/paullj1/pi-assistant"
  fi
else
  if [[ -f "$ROOT_DIR/pyproject.toml" ]]; then
    python3 -m pip install "$ROOT_DIR"
  else
    python3 -m pip install "git+https://github.com/paullj1/pi-assistant"
  fi
fi

cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=Pi Assistant
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=$BIN_PATH
Restart=on-failure
RestartSec=2
Environment=PYTHONUNBUFFERED=1
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now pi-assistant.service

echo "Installed. Config: $CONFIG_DIR/mcp.json"
