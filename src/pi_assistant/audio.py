import array
import math
import queue
import subprocess
import time
import wave
from threading import Event, Thread

import numpy as np
import sounddevice as sd

from . import config
from .utils import debug


def _generate_beep(path: str, hz: float, ms: int, gain: float):
    sample_rate = 44100
    length = max(0.05, ms / 1000.0)
    count = int(sample_rate * length)
    data = array.array("h")

    for i in range(count):
        t = i / sample_rate
        sample = math.sin(2.0 * math.pi * hz * t) * gain
        data.append(int(sample * 32767))

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def _generate_waiting_tone_cycle(
    sample_rate: int = 44100,
    freq: float = 220.0,
    tone_seconds: float = 0.43,
    echo_count: int = 6,
    echo_delay: float = 0.26,
    echo_decay: float = 0.85,
    rest_seconds: float = 0.5,
) -> tuple[array.array, int]:
    total_seconds = tone_seconds + echo_delay * echo_count + rest_seconds
    total_samples = int(total_seconds * sample_rate)
    data = array.array("h", [0] * total_samples)

    def add_tone(start_time: float, amplitude: float) -> None:
        start_idx = int(start_time * sample_rate)
        tone_samples = int(tone_seconds * sample_rate)
        for i in range(tone_samples):
            t = i / sample_rate
            fade = max(0.0, 1.0 - (i / tone_samples))
            sample = math.sin(2.0 * math.pi * freq * t) * amplitude * fade
            idx = start_idx + i
            if idx < total_samples:
                data[idx] += int(sample * 32767)

    add_tone(0.0, 0.35)
    for n in range(1, echo_count + 1):
        add_tone(n * echo_delay, 0.35 * (echo_decay**n))

    return data, sample_rate


_CUE_PATHS = {}


def _ensure_beep(path_key: str, hz: float, ms: int, gain: float) -> str:
    if path_key in _CUE_PATHS:
        return _CUE_PATHS[path_key]

    out_path = f"/tmp/assistant_{path_key}.wav"
    _generate_beep(out_path, hz, ms, gain)
    _CUE_PATHS[path_key] = out_path
    return out_path


def _ensure_working_beep() -> str:
    path_key = "working_cue"
    if path_key in _CUE_PATHS:
        return _CUE_PATHS[path_key]

    out_path = f"/tmp/assistant_{path_key}.wav"
    data, sample_rate = _generate_waiting_tone_cycle()
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())

    _CUE_PATHS[path_key] = out_path
    return out_path


def _play_cue(mode: str, tts_text: str, beep_key: str, hz: float, ms: int, gain: float):
    if mode in ("off", "none", "0", "false"):
        debug(f"{beep_key} cue disabled")
        return

    if mode == "tts":
        from .tts import speak_async

        debug(f"{beep_key} cue tts")
        proc = speak_async(tts_text)
        proc.wait()
        return

    if mode == "beep":
        debug(f"{beep_key} cue beep")
        if beep_key == "working_cue":
            path = _ensure_working_beep()
        else:
            path = _ensure_beep(beep_key, hz, ms, gain)
        subprocess.run(["aplay", "-q", "-D", config.ALSA_PLAYBACK, path], check=False)
        return


def play_wake_cue():
    _play_cue(
        config.WAKE_CUE,
        config.WAKE_CUE_TTS,
        "wake_cue",
        config.WAKE_BEEP_HZ,
        config.WAKE_BEEP_MS,
        config.WAKE_BEEP_GAIN,
    )


def play_stop_cue():
    _play_cue(
        config.STOP_CUE,
        config.STOP_CUE_TTS,
        "stop_cue",
        config.STOP_BEEP_HZ,
        config.STOP_BEEP_MS,
        config.STOP_BEEP_GAIN,
    )


def play_working_cue():
    _play_cue(
        config.WORKING_CUE,
        config.WORKING_CUE_TTS,
        "working_cue",
        config.WORKING_BEEP_HZ,
        config.WORKING_BEEP_MS,
        config.WORKING_BEEP_GAIN,
    )


def start_working_cue_loop():
    if config.WORKING_CUE in ("off", "none", "0", "false"):
        return None, None

    if config.WORKING_CUE == "tts":
        try:
            play_working_cue()
        except Exception as e:
            debug(f"working cue failed: {e}")
        return None, None

    stop_event = Event()

    def _loop():
        if config.WORKING_CUE == "beep":
            pause_s = 0.0
        else:
            pause_s = max(0.02, config.WORKING_BEEP_PAUSE_MS / 1000.0)
        while not stop_event.is_set():
            _play_cue(
                config.WORKING_CUE,
                config.WORKING_CUE_TTS,
                "working_cue",
                config.WORKING_BEEP_HZ,
                config.WORKING_BEEP_MS,
                config.WORKING_BEEP_GAIN,
            )
            time.sleep(pause_s)

    thread = Thread(target=_loop, daemon=True)
    thread.start()
    return stop_event, thread


def stop_working_cue_loop(stop_event, thread):
    if stop_event is None:
        return
    stop_event.set()
    if thread is not None:
        thread.join(timeout=1.0)


class AudioStream:
    def __init__(self):
        self._queue = queue.Queue()
        self._queues = [self._queue]
        self._status_count = 0

        def cb(indata, frames, time_info, status):
            if status:
                self._status_count += 1
                debug(f"audio callback status={status} count={self._status_count}")
            data = indata.copy()
            for q in list(self._queues):
                q.put(data)

        self._stream = sd.InputStream(
            device=config.INPUT_DEVICE,
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype="float32",
            callback=cb,
            blocksize=0,
        )

    def start(self):
        self._stream.start()
        debug("audio stream started")

    def pause(self):
        if self._stream.active:
            self._stream.stop()
            debug("audio stream paused")

    def resume(self):
        if not self._stream.active:
            self._stream.start()
            debug("audio stream resumed")

    def stop(self):
        self._stream.stop()
        self._stream.close()
        debug("audio stream stopped")

    def get(self, timeout=0.1):
        return self._queue.get(timeout=timeout)

    def subscribe(self) -> queue.Queue:
        q = queue.Queue()
        self._queues.append(q)
        return q

    def drain(self, q: queue.Queue | None = None):
        target = q or self._queue
        drained = 0
        while True:
            try:
                target.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            debug(f"audio queue drained {drained} chunks")


def _process_audio_float(audio: np.ndarray, apply_noise_gate: bool = True) -> bytes:
    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1, keepdims=True)

    audio = audio.reshape(-1)

    if config.MIC_SOFT_GAIN != 1.0:
        audio = audio * config.MIC_SOFT_GAIN

    if config.CLIP:
        audio = np.clip(audio, -1.0, 1.0)

    if apply_noise_gate and config.NOISE_GATE and config.NOISE_GATE > 0.0:
        audio[np.abs(audio) < config.NOISE_GATE] = 0.0

    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def record_until_silence_from_stream(
    audio_stream: AudioStream,
    max_seconds=config.MAX_SECONDS,
    silence_seconds=config.SILENCE_SECONDS,
    threshold=config.RMS_THRESHOLD,
    drain=True,
    on_chunk=None,
    apply_noise_gate: bool = True,
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
        if on_chunk:
            on_chunk(
                _process_audio_float(chunk, apply_noise_gate=apply_noise_gate), rms
            )
        rms = float(np.sqrt(np.mean(chunk**2)))
        dt = len(chunk) / float(config.SAMPLE_RATE)

        silence_run = silence_run + dt if rms < threshold else 0.0

        elapsed = time.time() - start
        if silence_run >= silence_seconds and elapsed > 0.6:
            break
        if elapsed >= max_seconds:
            break

    if not chunks:
        return b""

    audio = np.concatenate(chunks, axis=0)
    return _process_audio_float(audio, apply_noise_gate=apply_noise_gate)


def record_after_speech_start_from_stream(
    audio_stream: AudioStream,
    max_wait_seconds: float,
    max_seconds: float,
    silence_seconds: float,
    threshold: float,
    drain: bool = True,
    on_chunk=None,
    apply_noise_gate: bool = True,
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
        dt = len(chunk) / float(config.SAMPLE_RATE)

        if not started:
            if rms >= threshold:
                started = True
            else:
                if time.time() - start >= max_wait_seconds:
                    break
                continue

        chunks.append(chunk)
        if on_chunk:
            on_chunk(
                _process_audio_float(chunk, apply_noise_gate=apply_noise_gate), rms
            )
        silence_run = silence_run + dt if rms < threshold else 0.0

        elapsed = time.time() - start
        if silence_run >= silence_seconds and elapsed > 0.6:
            break
        if elapsed >= max_seconds:
            break

    if not chunks:
        return b""

    audio = np.concatenate(chunks, axis=0)
    return _process_audio_float(audio, apply_noise_gate=apply_noise_gate)


def print_audio_devices():
    print("\nPortAudio devices (sounddevice):")
    try:
        print(sd.query_devices())
    except Exception as e:
        print("Could not query devices:", e)
    print("")
