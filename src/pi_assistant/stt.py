import io
import json
import time
import wave
from threading import Event, Thread
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
from .utils import debug, headers_auth, headers_ws, api_base_ws


from .gating import RMSGate, KeepAliveGate, pcm_rms


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
    rms_gate = RMSGate(window_seconds=1.0, floor=config.RMS_THRESHOLD, frame_seconds=0.02)
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
        rms_value = pcm_rms(pcm_chunk)
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
                _send_chunk(frame, pcm_rms(frame))
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


def _listen_for_user_streaming(
    audio_stream: AudioStream, pre_roll: bytes, max_wait_seconds: float | None
) -> str:
    try:
        full_window = config.WAKE_LISTEN_FULL_WINDOW if max_wait_seconds is None else True
        wait_seconds = (
            config.WAKE_LISTEN_SECONDS if max_wait_seconds is None else max_wait_seconds
        )
        transcript, pcm = _stt_openai_streaming_from_stream(
            audio_stream,
            full_window,
            wait_seconds,
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


__all__ = [
    "stt_whisper",
    "_listen_for_user_streaming",
]
