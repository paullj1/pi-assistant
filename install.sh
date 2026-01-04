#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
CONFIG_DIR="$XDG_CONFIG_HOME/pi-assistant"
SERVICE_DIR="$XDG_CONFIG_HOME/systemd/user"

mkdir -p "$CONFIG_DIR"
mkdir -p "$SERVICE_DIR"

if [[ -f "$ROOT_DIR/mcp.json" ]]; then
  cp -f "$ROOT_DIR/mcp.json" "$CONFIG_DIR/mcp.json"
fi

if [[ -f "$ROOT_DIR/config.toml" ]]; then
  cp -n "$ROOT_DIR/config.toml" "$CONFIG_DIR/config.toml"
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install "$ROOT_DIR"
else
  python3 -m pip install "$ROOT_DIR"
fi

cp -f "$ROOT_DIR/pi-assistant.service" "$SERVICE_DIR/pi-assistant.service"

systemctl --user daemon-reload
systemctl --user enable --now pi-assistant.service

echo "Installed. Config: $CONFIG_DIR/mcp.json"
