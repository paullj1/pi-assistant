import inspect
import os
import time
from collections import deque
from threading import Event, Thread

import numpy as np

try:
    from openwakeword.model import Model
except Exception:  # pragma: no cover
    Model = None

from . import config
from .audio import AudioStream, _process_audio_float
from .gating import RMSGate, pcm_rms
from .utils import debug


class WakeWordDetector:
    def __init__(self, audio_stream: AudioStream):
        self._audio_stream = audio_stream
        self._stop_event = Event()
        self._thread = None
        self._wake_event = None
        self._queue = None
        self._pre_roll = deque()
        self._pre_roll_samples = 0
        self._last_trigger = 0.0
        self._model = None
        self._rms_gate = RMSGate(
            window_seconds=config.WAKE_RMS_WINDOW_SECONDS,
            floor=config.WAKE_RMS_FLOOR,
            frame_seconds=0.025,
        )

    def start(self, wake_event: Event):
        self._wake_event = wake_event
        self._stop_event.clear()
        if self._queue is None:
            self._queue = self._audio_stream.subscribe()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_pre_roll(self) -> bytes:
        data = b"".join(self._pre_roll)
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        return data

    def _append_pre_roll(self, pcm: bytes):
        if config.WAKE_PRE_ROLL_SECONDS <= 0:
            return
        samples = len(pcm) // 2
        self._pre_roll.append(pcm)
        self._pre_roll_samples += samples
        max_samples = int(config.WAKE_PRE_ROLL_SECONDS * config.SAMPLE_RATE)
        while self._pre_roll_samples > max_samples and self._pre_roll:
            popped = self._pre_roll.popleft()
            self._pre_roll_samples -= len(popped) // 2

    def _load_model(self):
        model_dir = os.path.expanduser(config.WAKE_MODEL_DIR)
        os.makedirs(model_dir, exist_ok=True)
        if Model is None:
            raise RuntimeError("openwakeword is not available")

        if config.DEBUG:
            debug(f"wake model dir: {model_dir}")

        self._model = self._create_model(model_dir)

    def _create_model(self, model_dir: str):
        # Try to adapt to openwakeword versions with different constructor APIs.
        sig = inspect.signature(Model)
        params = sig.parameters
        candidate = self._find_model_path(model_dir)
        if candidate is None:
            self._download_models(model_dir)
            candidate = self._find_model_path(model_dir)
        if config.DEBUG:
            debug(f"wake model path: {candidate or '(not found)'}")
        model_dir_kw = None
        for name in ("model_path", "models_dir", "model_dir"):
            if name in params:
                model_dir_kw = name
                break
        device_kw = None
        for name in ("device", "inference_device"):
            if name in params:
                device_kw = name
                break
        if device_kw is None:
            for name in ("inference_framework",):
                if name in params:
                    device_kw = name
                    break
        device_value = config.WAKE_DEVICE
        if device_value in ("cpuexecutionprovider", "cpu_execution_provider"):
            device_value = "cpu"
        if device_kw == "inference_framework":
            device_value = "onnx" if device_value == "cpu" else device_value
        allow_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        candidates = []
        if "wakeword_models" in params:
            base = {"wakeword_models": [config.WAKE_MODEL]}
            if model_dir_kw:
                base[model_dir_kw] = model_dir
            if device_kw:
                base[device_kw] = device_value
            elif allow_kwargs:
                base["device"] = device_value
            candidates.append(base)
        if "wakeword_model_paths" in params and candidate:
            base = {"wakeword_model_paths": [candidate]}
            if model_dir_kw:
                base[model_dir_kw] = model_dir
            if device_kw:
                base[device_kw] = device_value
            elif allow_kwargs:
                base["device"] = device_value
            candidates.append(base)
        if model_dir_kw or device_kw:
            base = {}
            if model_dir_kw:
                base[model_dir_kw] = model_dir
            if device_kw:
                base[device_kw] = device_value
            elif allow_kwargs:
                base["device"] = device_value
            candidates.append(base)
        candidates.append({})

        for kwargs in candidates:
            try:
                model = Model(**kwargs)
                self._ensure_model_loaded(model, model_dir)
                return model
            except TypeError:
                continue
            except Exception as e:
                debug(f"wake model init failed: {e}")

        # Last attempt with the best-known kwargs after download.
        for kwargs in candidates:
            try:
                model = Model(**kwargs)
                self._ensure_model_loaded(model, model_dir)
                return model
            except Exception:
                continue

        raise RuntimeError("Unable to initialize openwakeword model")

    def _download_models(self, model_dir: str):
        try:
            import openwakeword
        except Exception as e:
            debug(f"wake model download failed: {e}")
            return

        try:
            from openwakeword import utils  # type: ignore

            if hasattr(utils, "download_models"):
                debug(f"wake model download: {config.WAKE_MODEL} -> {model_dir}")
                utils.download_models([config.WAKE_MODEL], model_dir)
                return
        except Exception:
            pass

        def _download_file(url: str):
            try:
                from openwakeword import utils  # type: ignore

                if hasattr(utils, "download_file"):
                    utils.download_file(url, model_dir)
                    return True
            except Exception:
                pass
            try:
                import requests

                local_name = url.split("/")[-1]
                out_path = os.path.join(model_dir, local_name)
                with requests.get(url, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                return True
            except Exception as e:
                debug(f"wake model download failed: {e}")
                return False

        os.makedirs(model_dir, exist_ok=True)
        for model_map in ("FEATURE_MODELS", "VAD_MODELS", "MODELS"):
            values = getattr(openwakeword, model_map, {})
            for item in values.values():
                url = item.get("download_url")
                if not url:
                    continue
                local_name = url.split("/")[-1]
                target = os.path.join(model_dir, local_name)
                if not os.path.exists(target):
                    debug(f"wake model download: {url} -> {model_dir}")
                    _download_file(url)
                if url.endswith(".tflite"):
                    onnx_url = url.replace(".tflite", ".onnx")
                    onnx_name = onnx_url.split("/")[-1]
                    onnx_target = os.path.join(model_dir, onnx_name)
                    if not os.path.exists(onnx_target):
                        debug(f"wake model download: {onnx_url} -> {model_dir}")
                        _download_file(onnx_url)

    def _find_model_path(self, model_dir: str) -> str | None:
        candidates = []
        dirs = self._model_search_dirs(model_dir)
        if config.DEBUG:
            debug(f"wake model search dirs: {dirs}")
        for path in dirs:
            if not os.path.isdir(path):
                continue
            target = f"{config.WAKE_MODEL}".lower()
            for name in os.listdir(path):
                lower = name.lower()
                if not lower.endswith(".onnx"):
                    continue
                if target in lower:
                    candidates.append(os.path.join(path, name))
        return sorted(candidates)[-1] if candidates else None

    def _model_search_dirs(self, model_dir: str) -> list[str]:
        dirs = [model_dir]
        try:
            import openwakeword

            pkg_dir = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
            dirs.append(pkg_dir)
        except Exception:
            pass
        return dirs

    def _ensure_model_loaded(self, model, model_dir: str):
        if hasattr(model, "load_wakeword_models"):
            try:
                model.load_wakeword_models([config.WAKE_MODEL], model_dir=model_dir)
            except Exception:
                pass
        if hasattr(model, "set_wakeword_models"):
            try:
                model.set_wakeword_models([config.WAKE_MODEL])
            except Exception:
                pass

    def _should_trigger(self, scores: dict) -> bool:
        score = scores.get(config.WAKE_MODEL, 0.0)
        if config.DEBUG:
            debug(f"wake score {score:.3f}")
        return score >= config.WAKE_THRESHOLD

    def _cooldown_elapsed(self) -> bool:
        return time.time() - self._last_trigger >= config.WAKE_COOLDOWN_SECONDS

    def _run(self):
        if self._model is None:
            try:
                self._load_model()
            except Exception as e:
                debug(f"wake model load failed: {e}")
                return

        last_rms_log = 0.0
        frame_samples = 400
        frame_bytes = frame_samples * 2
        pcm_buffer = bytearray()

        while not self._stop_event.is_set():
            if self._queue is None:
                time.sleep(0.05)
                continue
            try:
                chunk = self._queue.get(timeout=0.1)
            except Exception:
                continue
            pcm = _process_audio_float(chunk, apply_noise_gate=False)
            self._append_pre_roll(pcm)
            pcm_buffer.extend(pcm)
            while len(pcm_buffer) >= frame_bytes:
                frame = bytes(pcm_buffer[:frame_bytes])
                del pcm_buffer[:frame_bytes]
                rms_value = pcm_rms(frame)
                if config.DEBUG:
                    now = time.time()
                    if now - last_rms_log >= 1.0:
                        last_rms_log = now
                        debug(f"wake rms={rms_value:.4f}")
                if not self._rms_gate.should_send(rms_value):
                    continue
                audio_samples = np.frombuffer(frame, dtype=np.int16)
                scores = self._model.predict(audio_samples)
                if self._should_trigger(scores) and self._cooldown_elapsed():
                    self._last_trigger = time.time()
                    if self._wake_event:
                        self._wake_event.set()
