import argparse
import json
import queue
import time
from threading import Event, Thread

import requests

from . import config
from .audio import (
    AudioStream,
    play_stop_cue,
    play_wake_cue,
    print_audio_devices,
    start_working_cue_loop,
    stop_working_cue_loop,
)
from .chat import openai_chat_stream
from .mcp import MCPManager
from .stt import stt_whisper
from .wake import WakeWordDetector
from .tts import (
    StreamingTTS,
    play_audio,
    speak_async,
    split_tts_chunks,
    stop_playback,
    synthesize_tts,
)
from .utils import Turn, debug, extract_assistant_meta, serialize_messages


def chat_with_streaming_tts(messages, tools, wake_event: Event):
    stop_cue_event, cue_thread = start_working_cue_loop()
    stream_tts = StreamingTTS(
        wake_event, lambda: stop_working_cue_loop(stop_cue_event, cue_thread)
    )
    stream_tts.start()
    try:
        reply_msg = openai_chat_stream(
            messages,
            tools,
            on_text_delta=stream_tts.on_text_delta,
            on_tool_call=stream_tts.on_tool_call,
        )
    finally:
        stop_working_cue_loop(stop_cue_event, cue_thread)
    interrupted = stream_tts.finish()
    used_stream_tts = stream_tts.used()
    return reply_msg, interrupted if used_stream_tts else False, used_stream_tts


def _listen_for_user(
    audio_stream: AudioStream,
    play_cue: bool,
    pre_roll: bytes = b"",
    max_wait_seconds: float | None = None,
) -> str:
    if play_cue:
        try:
            play_wake_cue()
        except Exception as e:
            debug(f"wake cue failed: {e}")

    if config.STT_STREAM:
        from .stt import _listen_for_user_streaming

        return _listen_for_user_streaming(audio_stream, pre_roll, max_wait_seconds)

    if max_wait_seconds is not None:
        from .audio import record_after_speech_start_from_stream

        pcm = record_after_speech_start_from_stream(
            audio_stream,
            max_wait_seconds=max_wait_seconds,
            max_seconds=config.WAKE_LISTEN_SECONDS,
            silence_seconds=config.SILENCE_SECONDS,
            threshold=config.RMS_THRESHOLD,
            drain=False,
        )
    elif config.WAKE_LISTEN_FULL_WINDOW:
        from .audio import record_after_speech_start_from_stream

        pcm = record_after_speech_start_from_stream(
            audio_stream,
            max_wait_seconds=config.WAKE_LISTEN_SECONDS,
            max_seconds=config.WAKE_LISTEN_SECONDS,
            silence_seconds=config.SILENCE_SECONDS,
            threshold=config.RMS_THRESHOLD,
            drain=False,
        )
    else:
        from .audio import record_until_silence_from_stream

        pcm = record_until_silence_from_stream(
            audio_stream,
            max_seconds=config.WAKE_LISTEN_SECONDS,
            silence_seconds=config.SILENCE_SECONDS,
            drain=False,
        )
    if not pcm:
        return ""

    if pre_roll:
        pcm = pre_roll + pcm
    return stt_whisper(pcm)


def _should_end_conversation(reply: str) -> bool:
    from .utils import normalize_text

    return normalize_text(reply).endswith(normalize_text(config.END_PROMPT))


def _is_user_done(text: str) -> bool:
    from .utils import normalize_text

    normalized = normalize_text(text)
    return normalized in config.END_USER_RESPONSES


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

    if args.debug:
        config.DEBUG = True

    if args.list_devices or config._as_bool("ASSISTANT_LIST_DEVICES", False):
        print_audio_devices()
        return

    mcp_config_path = config.MCP_CONFIG or config.DEFAULT_MCP_CONFIG
    mcp_manager = MCPManager(mcp_config_path)
    mcp_manager.start()
    mcp_tools = mcp_manager.tools()
    if config.DEBUG:
        debug(f"mcp tools loaded: {len(mcp_tools)}")
        if mcp_tools:
            debug(f"mcp tool names: {[t['function']['name'] for t in mcp_tools]}")

    print("Pi Voice Assistant (wake word + barge-in)\n")
    print(f"API base:      {config.API_BASE}")
    print(f"Chat model:    {config.CHAT_MODEL}")
    if mcp_config_path:
        print(f"MCP config:    {mcp_config_path}")
        print(f"MCP tools:     {len(mcp_tools)}")
    if config.REASONING_EFFORT:
        print(f"Reasoning:     {config.REASONING_EFFORT}")
    print(
        f"TTS:           model={config.TTS_MODEL} voice={config.TTS_VOICE} format={config.TTS_FORMAT}"
    )
    print(f"Playback:      ALSA={config.ALSA_PLAYBACK}")
    print(f"STT:           model={config.STT_MODEL}")
    print(f"STT streaming: {config.STT_STREAM}")
    print(f"Wake model:    {config.WAKE_MODEL}")
    print(f"Wake thresh:   {config.WAKE_THRESHOLD:.2f}")
    print(f"Wake pre-roll: {config.WAKE_PRE_ROLL_SECONDS:.1f}s")
    print(f"Wake cue:      {config.WAKE_CUE}")
    print(f"Working cue:   {config.WORKING_CUE}")
    print(f"Working style:{'':1} {config.WORKING_BEEP_STYLE}")
    print(f"Listen window: {config.WAKE_LISTEN_SECONDS:.1f}s")
    print(f"Listen full:   {config.WAKE_LISTEN_FULL_WINDOW}")
    print(
        f"Mic gain:      {config.MIC_SOFT_GAIN}x   Noise gate: {config.NOISE_GATE}   Clip: {config.CLIP}"
    )
    print(
        f"Input device:  {config.INPUT_DEVICE if config.INPUT_DEVICE is not None else '(default)'}"
    )
    print("\nSay the wake word to speak. Say it again to interrupt.\n")

    history = [
        Turn(
            "system",
            "You are a helpful voice assistant. Keep replies concise and conversational. "
            "Normalize your text for TTS.  Expand abbreviations like 'Ave.' to 'Avenue', "
            "'Dr.' to Doctor, 'e.g.' to 'for example', 'etc.' to 'and so on'. Spell out numbers "
            "and symbols (e.g., 1st to first, $100 to one hundred dollars, 90ºF to ninety "
            " degrees fahrenheit).  Read text naturally, ensuring clarity for the user. "
            "Decide whether the conversation should continue. If it should, end by asking exactly: "
            f"'{config.END_PROMPT}'. If it should not, do not ask a follow-up question. "
            "In all cases, append a final line with assistant-only metadata in JSON format like this: "
            '<assistant_meta>{"done": true}</assistant_meta> where done=true means end the conversation.',
        )
    ]

    wake_event = Event()
    audio_stream = AudioStream()
    wake_detector = WakeWordDetector(audio_stream)
    wake_pre_roll = b""

    try:
        audio_stream.start()
        wake_detector.start(wake_event)
        pending_wake = False

        while True:
            if not pending_wake:
                wake_event.clear()
                wake_pre_roll = b""
                wake_event.wait()
                wake_pre_roll = wake_detector.get_pre_roll()
            wake_event.clear()
            pending_wake = False

            pending_user_text = None
            prompt_cue = True
            while True:
                if pending_user_text is None:
                    if prompt_cue:
                        one_shot = _listen_for_user(
                            audio_stream,
                            play_cue=False,
                            pre_roll=wake_pre_roll,
                            max_wait_seconds=config.WAKE_ONE_SHOT_GRACE_SECONDS,
                        )
                        wake_pre_roll = b""
                        if one_shot:
                            text = one_shot
                            prompt_cue = False
                        else:
                            text = _listen_for_user(
                                audio_stream, play_cue=True, pre_roll=b""
                            )
                    else:
                        text = _listen_for_user(
                            audio_stream, play_cue=prompt_cue, pre_roll=wake_pre_roll
                        )
                        wake_pre_roll = b""
                else:
                    text = pending_user_text
                    pending_user_text = None

                prompt_cue = False

                if not text:
                    print("(no speech recognized)")
                    try:
                        play_stop_cue()
                    except Exception as e:
                        debug(f"stop cue failed: {e}")
                    break

                print(f"You: {text}")
                history.append(Turn("user", text))

                try:
                    reply_msg, interrupted, used_stream_tts = chat_with_streaming_tts(
                        serialize_messages(history), mcp_tools, wake_event
                    )
                    while reply_msg.get("tool_calls"):
                        if not mcp_tools:
                            debug("model requested tools but none are configured")
                            reply_msg = {
                                "role": "assistant",
                                "content": "Tools are unavailable.",
                            }
                            used_stream_tts = False
                            interrupted = False
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
                        reply_msg, interrupted, used_stream_tts = chat_with_streaming_tts(
                            serialize_messages(history),
                            mcp_tools,
                            wake_event,
                        )
                except requests.RequestException as e:
                    print("LLM request failed:", e)
                    try:
                        speak_async("I had trouble reaching the language model.")
                    except Exception:
                        pass
                    continue
                except Exception as e:
                    print("LLM error:", e)
                    try:
                        speak_async("Something went wrong.")
                    except Exception:
                        pass
                    continue

                reply = (reply_msg.get("content") or "").strip()
                reply, meta = extract_assistant_meta(reply)
                print(f"Assistant: {reply}\n")
                history.append({"role": "assistant", "content": reply})
                done = bool(meta.get("done")) if isinstance(meta, dict) else False

                if interrupted:
                    prompt_cue = True
                    continue

                if reply:
                    if not used_stream_tts:
                        chunks = split_tts_chunks(reply, config.TTS_CHUNK_CHARS)
                        if not chunks:
                            chunks = [reply]

                        interrupted = False
                        audio_queue = queue.Queue()
                        synth_done = Event()

                        def _prefetch():
                            for chunk in chunks:
                                try:
                                    path = synthesize_tts(chunk)
                                except Exception as e:
                                    audio_queue.put(e)
                                    break
                                audio_queue.put(path)
                            synth_done.set()

                        synth_thread = Thread(target=_prefetch, daemon=True)
                        synth_thread.start()

                        stop_cue_event, cue_thread = start_working_cue_loop()
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
                                stop_working_cue_loop(stop_cue_event, cue_thread)
                                first_audio = False

                            if isinstance(item, Exception):
                                print("TTS failed:", item)
                                interrupted = True
                                break

                            proc = play_audio(item)
                            while proc.poll() is None:
                                if wake_event.is_set():
                                    wake_event.clear()
                                    stop_playback(proc)
                                    interrupted = True
                                    break
                                time.sleep(0.05)
                            if interrupted:
                                break

                        stop_working_cue_loop(stop_cue_event, cue_thread)

                        if interrupted:
                            prompt_cue = True
                            continue

                if done:
                    try:
                        play_stop_cue()
                    except Exception as e:
                        debug(f"stop cue failed: {e}")
                    break

                if _should_end_conversation(reply):
                    follow_text = _listen_for_user(
                        audio_stream,
                        play_cue=False,
                        max_wait_seconds=config.FOLLOWUP_WAIT_SECONDS,
                    )
                    if not follow_text:
                        try:
                            proc = speak_async(config.FOLLOWUP_NO_RESPONSE_TTS)
                            proc.wait()
                        except Exception as e:
                            debug(f"stop cue failed: {e}")
                        break
                    if _is_user_done(follow_text):
                        try:
                            play_stop_cue()
                        except Exception as e:
                            debug(f"stop cue failed: {e}")
                        break

                    pending_user_text = follow_text
                    prompt_cue = False

    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        wake_detector.stop()
        audio_stream.stop()
