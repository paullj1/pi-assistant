import json
import os
import subprocess
import time
from threading import Condition, Lock, Thread

import requests

from .utils import debug


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
            debug(f"mcp[{self._name}] stderr: {line.decode(errors='ignore').strip()}")

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
                debug(f"mcp[{self._name}] json decode failed: {e}")
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


class MCPHttpServer:
    def __init__(self, name: str, config: dict):
        self._name = name
        self._config = config
        self._next_id = 1

    def list_tools(self):
        resp = self._request("tools/list")
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict):
        resp = self._request("tools/call", {"name": name, "arguments": arguments})
        return resp.get("result", {})

    def _request(self, method: str, params: dict | None = None, timeout=30):
        req_id = self._next_id
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params

        url = self._config.get("url", "").strip()
        if not url:
            raise RuntimeError(f"MCP http server {self._name} missing url")
        headers = dict(self._config.get("headers", {}))
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        timeout_s = float(self._config.get("timeout", timeout))
        r = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout_s,
            stream=True,
        )
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            last_msg = None
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    msg = json.loads(data)
                except Exception:
                    continue
                last_msg = msg
                if msg.get("id") == req_id and ("result" in msg or "error" in msg):
                    break
            if last_msg is None:
                raise RuntimeError("empty MCP stream response")
            return last_msg

        return r.json()


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
            with open(self._config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            debug(f"failed to read MCP config: {e}")
            return

        if not isinstance(cfg, dict) or "mcpServers" not in cfg:
            debug("invalid MCP config: expected top-level mcpServers")
            return

        servers = cfg.get("mcpServers", {})

        for name, srv in servers.items():
            try:
                transport = srv.get("transport", "stdio")
                if transport == "stdio":
                    server = MCPServer(name, srv)
                    server.start()
                    self._servers[name] = server
                elif transport == "streamable-http":
                    server = MCPHttpServer(name, srv)
                    self._servers[name] = server
                else:
                    debug(f"mcp server {name} unsupported transport: {transport}")
                    continue
            except Exception as e:
                debug(f"mcp server {name} failed to start: {e}")

        self._load_tools()

    def _load_tools(self):
        for name, server in self._servers.items():
            try:
                tools = server.list_tools()
            except Exception as e:
                debug(f"mcp server {name} list tools failed: {e}")
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
