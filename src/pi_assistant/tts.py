import queue
import subprocess
import time
from threading import Event, Thread

import requests

from . import config
from .utils import extract_assistant_meta, extract_complete_sentences, headers_json


def synthesize_tts(text: str) -> str:
    """
    POST {API_BASE}/audio/speech and write audio to /tmp.
    Returns the output path.
    """
    url = f"{config.API_BASE}/audio/speech"
    payload = {
        "model": config.TTS_MODEL,
        "voice": config.TTS_VOICE,
        "input": text,
        "response_format": config.TTS_FORMAT,
    }
    r = requests.post(url, headers=headers_json(), json=payload, timeout=180)
    r.raise_for_status()

    fmt = config.TTS_FORMAT.lower()
    out_path = f"/tmp/assistant_tts_{int(time.time() * 1000)}.{fmt}"
    with open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


def play_audio(path: str) -> subprocess.Popen:
    fmt = config.TTS_FORMAT.lower()
    if fmt == "wav":
        return subprocess.Popen(["aplay", "-q", "-D", config.ALSA_PLAYBACK, path])
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
        from .audio import start_working_cue_loop, stop_working_cue_loop

        stop_event, thread = start_working_cue_loop()
    try:
        out_path = synthesize_tts(text)
    finally:
        if working_cue:
            from .audio import stop_working_cue_loop

            stop_working_cue_loop(stop_event, thread)
    return play_audio(out_path)


def split_tts_chunks(text: str, max_chars: int) -> list[str]:
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


class StreamingTTS:
    def __init__(self, wake_event: Event, on_first_audio):
        self._wake_event = wake_event
        self._on_first_audio = on_first_audio
        self._sentence_queue = queue.Queue()
        self._audio_queue = queue.Queue()
        self._buffer = ""
        self._aborted = Event()
        self._interrupted = Event()
        self._disabled = False
        self._started_audio = Event()
        self._synth_thread = Thread(target=self._synth_loop, daemon=True)
        self._play_thread = Thread(target=self._play_loop, daemon=True)

    def start(self):
        self._synth_thread.start()
        self._play_thread.start()

    def on_text_delta(self, delta: str):
        if self._disabled or self._aborted.is_set():
            return
        self._buffer += delta
        sentences, self._buffer = extract_complete_sentences(self._buffer)
        for sentence in sentences:
            self._sentence_queue.put(sentence)

    def on_tool_call(self):
        self.disable()

    def disable(self):
        if self._disabled:
            return
        self._disabled = True
        self._aborted.set()
        self._sentence_queue.put(None)

    def finish(self) -> bool:
        if not self._disabled and not self._aborted.is_set():
            tail = self._buffer.strip()
            tail, _ = extract_assistant_meta(tail)
            if tail:
                self._sentence_queue.put(tail)
        self._sentence_queue.put(None)
        self._synth_thread.join()
        self._play_thread.join()
        return self._interrupted.is_set()

    def used(self) -> bool:
        return self._started_audio.is_set()

    def _synth_loop(self):
        while True:
            item = self._sentence_queue.get()
            if item is None:
                break
            if self._aborted.is_set():
                break
            try:
                path = synthesize_tts(item)
            except Exception as e:
                self._audio_queue.put(e)
                break
            self._audio_queue.put(path)
        self._audio_queue.put(None)

    def _play_loop(self):
        first_audio = True
        while True:
            if self._aborted.is_set():
                break
            if self._wake_event.is_set():
                self._wake_event.clear()
                self._interrupted.set()
                break
            try:
                item = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                self._interrupted.set()
                break
            if first_audio:
                self._started_audio.set()
                if self._on_first_audio:
                    self._on_first_audio()
                first_audio = False
            proc = play_audio(item)
            while proc.poll() is None:
                if self._wake_event.is_set() or self._aborted.is_set():
                    self._wake_event.clear()
                    stop_playback(proc)
                    self._interrupted.set()
                    return
                time.sleep(0.05)
