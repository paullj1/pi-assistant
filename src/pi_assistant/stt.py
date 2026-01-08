import io
import json
import queue
import time
import wave
from collections import deque
from difflib import SequenceMatcher
from threading import Event, Lock, Thread
from urllib.parse import urlencode

import numpy as np
import requests
import websocket

from . import config
from .audio import (
    AudioStream,
    _process_audio_float,
    record_after_speech_start_from_stream,
    record_until_silence_from_stream,
)
from .utils import debug, headers_auth, headers_ws, normalize_text, api_base_ws


class RMSGate:
    def __init__(self, window_seconds: float, floor: float):
        self._window_frames = max(1, int(window_seconds / 0.02))
        self._values = deque(maxlen=self._window_frames)
        self._floor = floor

    def update(self, rms: float) -> float:
        self._values.append(rms)
        return self.threshold()

    def threshold(self) -> float:
        if not self._values:
            return self._floor
        avg = sum(self._values) / len(self._values)
        return max(self._floor, avg)

    def should_send(self, rms: float) -> bool:
        threshold = self.update(rms)
        return rms >= threshold


def _pcm_rms(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio**2)))


class KeepAliveGate:
    def __init__(self, interval_seconds: float):
        self._interval = interval_seconds
        self._last_sent = 0.0

    def should_send(self) -> bool:
        now = time.time()
        if now - self._last_sent >= self._interval:
            self._last_sent = now
            return True
        return False


def _contains_wake_word(text: str) -> bool:
    if not config.WAKE_PHRASES:
        return False
    normalized = normalize_text(text)
    for phrase in config.WAKE_PHRASES:
        phrase_norm = normalize_text(phrase)
        if phrase_norm in normalized:
            return True

        ratio = SequenceMatcher(None, normalized, phrase_norm).ratio()
        if config.DEBUG:
            debug(f"wake fuzzy full '{phrase_norm}' vs '{normalized}' -> {ratio:.2f}")
        if ratio >= config.WAKE_FULL_FUZZY_THRESHOLD:
            return True

        normalized_joined = normalized.replace(" ", "")
        phrase_joined = phrase_norm.replace(" ", "")
        ratio = SequenceMatcher(None, normalized_joined, phrase_joined).ratio()
        if config.DEBUG:
            debug(
                f"wake fuzzy join '{phrase_joined}' vs '{normalized_joined}' -> {ratio:.2f}"
            )
        if ratio >= config.WAKE_FULL_FUZZY_THRESHOLD:
            return True

        phrase_tokens = [
            t for t in phrase_norm.split() if len(t) >= config.WAKE_MIN_TOKEN_LEN
        ]
        text_tokens = [
            t for t in normalized.split() if len(t) >= config.WAKE_MIN_TOKEN_LEN
        ]

        for p_tok in phrase_tokens:
            for t_tok in text_tokens:
                ratio = SequenceMatcher(None, p_tok, t_tok).ratio()
                if config.DEBUG:
                    debug(f"wake fuzzy '{p_tok}' vs '{t_tok}' -> {ratio:.2f}")
                if ratio >= config.WAKE_FUZZY_THRESHOLD:
                    return True

    return False


def _pcm16_to_wav_bytes(pcm16_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(pcm16_bytes)
    return buf.getvalue()


def _stt_openai(pcm16_bytes: bytes) -> str:
    """
    POST {API_BASE}/audio/transcriptions
    Returns the transcript text.
    """
    url = f"{config.API_BASE}/audio/transcriptions"
    wav_bytes = _pcm16_to_wav_bytes(pcm16_bytes)
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {
        "model": config.STT_MODEL,
        "response_format": "json",
    }
    r = requests.post(
        url,
        headers=headers_auth(),
        data=data,
        files=files,
        timeout=180,
    )
    r.raise_for_status()
    try:
        payload = r.json()
        return (
            payload.get("text")
            or payload.get("transcript")
            or payload.get("output")
            or ""
        ).strip()
    except ValueError:
        return r.text.strip()


def stt_whisper(pcm16_bytes: bytes) -> str:
    return _stt_openai(pcm16_bytes)


def stt_whisper_wake(pcm16_bytes: bytes) -> str:
    return _stt_openai(pcm16_bytes)


def _stt_ws_url() -> str:
    base = api_base_ws().rstrip("/")
    params = {
        "model": config.STT_MODEL,
        "response_format": "json",
    }
    return f"{base}/audio/transcriptions?{urlencode(params)}"


def _update_transcript(assembled: str, new_text: str, is_delta: bool) -> str:
    if not new_text:
        return assembled
    if is_delta:
        return assembled + new_text
    if new_text.startswith(assembled):
        return new_text
    if assembled.startswith(new_text):
        return assembled
    return assembled + new_text


def _apply_transcript_message(
    assembled: str, payload: dict, last_full: str
) -> tuple[str, str]:
    if not payload:
        return assembled, last_full
    choices = payload.get("choices") or []
    if choices:
        delta = choices[0].get("delta", {}) or {}
        content = delta.get("content") or ""
        if content:
            if config.DEBUG:
                debug(f"stt chunk: {content!r}")
            assembled = _update_transcript(assembled, content, True)
        text = choices[0].get("text") or ""
        if text:
            if config.DEBUG:
                debug(f"stt chunk: {text!r}")
            assembled = _update_transcript(assembled, text, True)
        return assembled, last_full

    if isinstance(payload.get("segment"), dict):
        text = payload["segment"].get("text") or ""
        if text:
            if config.DEBUG:
                debug(f"stt chunk: {text!r}")
            assembled = _update_transcript(assembled, text, True)
        return assembled, last_full

    segments = payload.get("segments")
    if isinstance(segments, list) and segments:
        joined = " ".join(seg.get("text", "").strip() for seg in segments).strip()
        if joined:
            if config.DEBUG:
                debug(f"stt chunk: {joined!r}")
            assembled = _update_transcript(assembled, joined, False)
            last_full = joined
        return assembled, last_full

    text = (
        payload.get("text")
        or payload.get("transcript")
        or payload.get("output")
        or ""
    )
    if text:
        is_delta = bool(payload.get("delta"))
        if config.DEBUG:
            debug(f"stt chunk: {text!r}")
        if not is_delta and text == last_full:
            return assembled, last_full
        assembled = _update_transcript(assembled, text, is_delta)
        last_full = text if not is_delta else last_full
    return assembled, last_full


def _stt_openai_streaming_from_stream(
    audio_stream: AudioStream,
    full_window: bool,
    max_wait_seconds: float,
    max_seconds: float,
    silence_seconds: float,
    threshold: float,
    pre_roll: bytes = b"",
) -> tuple[str, bytes]:
    if config.SAMPLE_RATE != 16000:
        debug("streaming stt expects 16k sample rate")

    url = _stt_ws_url()
    ws = websocket.create_connection(url, header=headers_ws())
    ws.settimeout(0.5)

    transcript = ""
    last_full = ""
    recv_error = None
    sending_done = Event()
    last_message_time = time.time()
    pcm = b""
    send_buffer = bytearray()
    frame_bytes = int(config.SAMPLE_RATE * 0.02) * 2
    rms_gate = RMSGate(window_seconds=1.0, floor=config.RMS_THRESHOLD)
    keepalive = KeepAliveGate(config.STT_STREAM_KEEPALIVE_SECONDS)

    def _flush_buffer():
        while len(send_buffer) >= frame_bytes:
            frame = bytes(send_buffer[:frame_bytes])
            del send_buffer[:frame_bytes]
            try:
                ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception:
                if config.DEBUG:
                    debug("stt stream send failed")
                return

    def _send_chunk(pcm_chunk: bytes, rms: float):
        if not pcm_chunk:
            return
        rms_value = _pcm_rms(pcm_chunk)
        if rms_gate.should_send(rms_value):
            send_buffer.extend(pcm_chunk)
            _flush_buffer()
            return
        if keepalive.should_send():
            send_buffer.extend(pcm_chunk)
            _flush_buffer()

    def _recv_loop():
        nonlocal transcript, last_full, recv_error, last_message_time
        while True:
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                if sending_done.is_set():
                    if time.time() - last_message_time >= config.STT_STREAM_FINAL_TIMEOUT:
                        break
                continue
            except Exception as e:
                recv_error = e
                break
            if msg is None:
                break
            last_message_time = time.time()
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                payload = json.loads(msg)
            except Exception:
                continue
            transcript, last_full = _apply_transcript_message(
                transcript, payload, last_full
            )

    recv_thread = Thread(target=_recv_loop, daemon=True)
    recv_thread.start()

    try:
        if pre_roll:
            offset = 0
            while offset < len(pre_roll):
                frame = pre_roll[offset : offset + frame_bytes]
                offset += frame_bytes
                _send_chunk(frame, _pcm_rms(frame))
        if full_window:
            pcm = record_after_speech_start_from_stream(
                audio_stream,
                max_wait_seconds=max_wait_seconds,
                max_seconds=max_seconds,
                silence_seconds=silence_seconds,
                threshold=threshold,
                drain=False,
                on_chunk=_send_chunk,
                apply_noise_gate=False,
            )
        else:
            pcm = record_until_silence_from_stream(
                audio_stream,
                max_seconds=max_seconds,
                silence_seconds=silence_seconds,
                threshold=threshold,
                drain=False,
                on_chunk=_send_chunk,
                apply_noise_gate=False,
            )
        if not pcm:
            sending_done.set()
            recv_thread.join(timeout=1.0)
            ws.close()
            return "", b""
    finally:
        sending_done.set()
        if send_buffer:
            try:
                ws.send(bytes(send_buffer), opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception:
                pass

    recv_thread.join(timeout=config.STT_STREAM_FINAL_TIMEOUT + 1.0)
    ws.close()
    if recv_error:
        debug(f"stt stream recv error: {recv_error}")
    if pre_roll:
        pcm = pre_roll + pcm
    return transcript.strip(), pcm


def _listen_for_user_streaming(audio_stream: AudioStream, pre_roll: bytes) -> str:
    try:
        transcript, pcm = _stt_openai_streaming_from_stream(
            audio_stream,
            config.WAKE_LISTEN_FULL_WINDOW,
            config.WAKE_LISTEN_SECONDS,
            config.WAKE_LISTEN_SECONDS,
            config.SILENCE_SECONDS,
            config.RMS_THRESHOLD,
            pre_roll=pre_roll,
        )
        if transcript:
            return transcript
        if pcm:
            return stt_whisper(pcm)
        return ""
    except Exception as e:
        debug(f"stt streaming failed, falling back: {e}")
    if config.WAKE_LISTEN_FULL_WINDOW:
        pcm = record_after_speech_start_from_stream(
            audio_stream,
            max_wait_seconds=config.WAKE_LISTEN_SECONDS,
            max_seconds=config.WAKE_LISTEN_SECONDS,
            silence_seconds=config.SILENCE_SECONDS,
            threshold=config.RMS_THRESHOLD,
            drain=False,
        )
    else:
        pcm = record_until_silence_from_stream(
            audio_stream,
            max_seconds=config.WAKE_LISTEN_SECONDS,
            silence_seconds=config.SILENCE_SECONDS,
            drain=False,
        )
    if not pcm:
        return ""
    return stt_whisper(pcm)


class LiveWakeStreamer:
    def __init__(self, audio_stream: AudioStream):
        self._audio_stream = audio_stream
        self._stop_event = Event()
        self._thread = None
        self._recv_thread = None
        self._wake_event = None
        self._pre_roll = deque()
        self._pre_roll_samples = 0
        self._pre_roll_limit = max(0.0, config.STT_STREAM_PRE_ROLL)
        self._transcript = ""
        self._lock = Lock()

    def start(self, wake_event: Event):
        self._wake_event = wake_event
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=1.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_pre_roll(self) -> bytes:
        with self._lock:
            data = b"".join(self._pre_roll)
        return data

    def _append_pre_roll(self, pcm: bytes):
        if self._pre_roll_limit <= 0:
            return
        samples = len(pcm) // 2
        with self._lock:
            self._pre_roll.append(pcm)
            self._pre_roll_samples += samples
            max_samples = int(self._pre_roll_limit * config.SAMPLE_RATE)
            while self._pre_roll_samples > max_samples and self._pre_roll:
                popped = self._pre_roll.popleft()
                self._pre_roll_samples -= len(popped) // 2

    def _apply_live_transcript(self, payload: dict) -> bool:
        with self._lock:
            self._transcript, _ = _apply_transcript_message(
                self._transcript, payload, ""
            )
            if len(self._transcript) > 400:
                self._transcript = self._transcript[-400:]
            text = self._transcript
        return _contains_wake_word(text)

    def _run(self):
        url = _stt_ws_url()
        try:
            ws = websocket.create_connection(url, header=headers_ws())
        except Exception as e:
            debug(f"wake ws connect failed: {e}")
            return
        ws.settimeout(0.5)
        send_buffer = bytearray()
        frame_bytes = int(config.SAMPLE_RATE * 0.02) * 2
        last_rms_log = 0.0
        rms_gate = RMSGate(window_seconds=1.0, floor=config.WAKE_RMS_THRESHOLD)
        keepalive = KeepAliveGate(config.STT_STREAM_KEEPALIVE_SECONDS)

        def _recv_loop():
            while not self._stop_event.is_set():
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception:
                    break
                if msg is None or isinstance(msg, (bytes, bytearray)):
                    continue
                try:
                    payload = json.loads(msg)
                except Exception:
                    continue
                if self._apply_live_transcript(payload):
                    if self._wake_event:
                        self._wake_event.set()
                    self._stop_event.set()
                    break

        self._recv_thread = Thread(target=_recv_loop, daemon=True)
        self._recv_thread.start()

        try:
            while not self._stop_event.is_set():
                try:
                    chunk = self._audio_stream.get(timeout=0.1)
                except queue.Empty:
                    continue
                pcm_chunk = _process_audio_float(chunk, apply_noise_gate=False)
                rms_value = _pcm_rms(pcm_chunk)
                if config.DEBUG:
                    now = time.time()
                    if now - last_rms_log >= 1.0:
                        last_rms_log = now
                        debug(f"stt stream rms={rms_value:.4f}")
                self._append_pre_roll(pcm_chunk)
                should_send = rms_gate.should_send(rms_value) or keepalive.should_send()
                if pcm_chunk and should_send:
                    send_buffer.extend(pcm_chunk)
                    while len(send_buffer) >= frame_bytes:
                        frame = bytes(send_buffer[:frame_bytes])
                        del send_buffer[:frame_bytes]
                        try:
                            ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
                        except Exception:
                            if config.DEBUG:
                                debug("stt stream send failed")
                            self._stop_event.set()
                            break
        finally:
            try:
                if send_buffer:
                    ws.send(bytes(send_buffer), opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception:
                pass
            try:
                ws.close()
            except Exception:
                pass


def wait_for_wake_word(
    wake_event: Event,
    stop_event: Event,
    audio_stream: AudioStream,
):
    while not stop_event.is_set():
        debug(
            "listening for wake word "
            f"(max={config.WAKE_MAX_SECONDS}s silence={config.WAKE_SILENCE_SECONDS}s "
            f"rms<{config.WAKE_RMS_THRESHOLD})"
        )
        pcm = record_until_silence_from_stream(
            audio_stream,
            max_seconds=config.WAKE_MAX_SECONDS,
            silence_seconds=config.WAKE_SILENCE_SECONDS,
            threshold=config.WAKE_RMS_THRESHOLD,
        )
        if not pcm:
            debug("wake chunk empty")
            continue

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
        if rms < config.WAKE_MIN_RMS_FOR_STT:
            debug(f"wake chunk rms={rms:.4f} below stt threshold, skipping")
            continue

        audio_stream.pause()
        try:
            text = stt_whisper_wake(pcm)
        finally:
            audio_stream.resume()
        if config.DEBUG:
            debug(f"wake chunk rms={rms:.4f} text={text!r}")
        if text and _contains_wake_word(text):
            debug("wake word detected")
            wake_event.set()
            return
