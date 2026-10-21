"""Model worker using stdin/stdout messages; this process serves no HTTP UI."""

import json
import os
from pathlib import Path
import queue
import runpy
import secrets
import sys
import threading
import traceback
from types import SimpleNamespace

import numpy as np

from .pauses import segments_with_boundaries


def _activity_bounds(audio, sample_rate):
    """Locate sustained speech using 10 ms RMS frames, ignoring quiet model noise."""
    frame = max(1, round(sample_rate * 0.01))
    frame_count = len(audio) // frame
    if frame_count < 12:
        return None
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    frames = np.asarray(mono[:frame_count * frame], dtype=np.float32).reshape(frame_count, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    threshold = max(100.0, float(rms.max()) * 0.035)
    active = np.flatnonzero(rms > threshold)
    if not active.size:
        return None
    return int(active[0] * frame), int(min(len(audio), (active[-1] + 1) * frame))


def trim_leading_quiet(audio, sample_rate):
    bounds = _activity_bounds(audio, sample_rate)
    if bounds is None:
        return audio
    start = max(0, bounds[0] - round(sample_rate * 0.06))
    return audio[start:] if start >= round(sample_rate * 0.08) else audio


def trim_trailing_quiet(audio, sample_rate):
    bounds = _activity_bounds(audio, sample_rate)
    if bounds is None:
        return audio
    end = min(len(audio), bounds[1] + round(sample_rate * 0.06))
    return audio[:end] if len(audio) - end >= round(sample_rate * 0.08) else audio


def main():
    channel = sys.stdout
    sys.stdout = sys.stderr  # Model diagnostic output must not enter the protocol.
    commands = queue.Queue()
    stopped = threading.Event()
    engine = {}
    last_cache_key = None

    def send(kind, **values):
        channel.write(json.dumps({"kind": kind, **values}, ensure_ascii=False) + "\n")
        channel.flush()

    def read_commands():
        for line in sys.stdin:
            try:
                message = json.loads(line)
                if message.get("command") == "cancel":
                    stopped.set()
                    pipeline = engine.get("tts_pipeline")
                    if pipeline is not None:
                        pipeline.stop()
                else:
                    commands.put(message)
            except Exception:
                traceback.print_exc()
        # Closing the owner's stdin also releases weights if it exits mid-generation.
        os._exit(0)

    threading.Thread(target=read_commands, daemon=True).start()
    try:
        config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        fast = config["parallel"]
        filename = "inference_webui_fast.py" if fast else "inference_webui.py"
        sys.argv = [filename, config["language"]]
        engine.update(
            runpy.run_path(
                str(Path("GPT_SoVITS") / filename), run_name="tts_workspace_engine"
            )
        )
        version = engine["model_version"]
        if fast:
            path = config["sovits_path"]
            path = engine["name2sovits_path"].get(path, path)
            version = engine["get_sovits_version_from_path_fast"](path)[1]
        send(
            "loaded",
            model_version=version,
            parallel=fast,
            languages=list(engine["dict_language"]),
            graph_available=bool(engine["graph_available"]),
            flash_available=bool(engine["flash_attn_supported"]),
        )
    except Exception as error:
        traceback.print_exc()
        send("error", message=str(error))
        return

    while True:
        message = commands.get()
        if message.get("command") != "generate":
            continue
        stopped.clear()
        try:
            data = message["inputs"]
            parts = segments_with_boundaries(data["text"], data["cut"])
            pause_seconds = float(data["pause"])
            if not 0 <= pause_seconds <= 2 or not np.isfinite(pause_seconds):
                raise ValueError("分段停顿须为 0–2 秒")
            punctuation_seconds = float(data.get("punctuation_pause", 0.15))
            if not 0 <= punctuation_seconds <= 1 or not np.isfinite(punctuation_seconds):
                raise ValueError("标点停顿须为 0–1 秒")
            refs = [
                SimpleNamespace(name=path) for path in data.get("aux_refs", [])
            ] or None
            cache_key = (data["text"], data["text_language"], data["cut"],
                         data["reference"], data["prompt_text"], data["prompt_language"],
                         bool(data["ref_free"]), tuple(data.get("aux_refs", [])))
            reuse_semantics = bool(data["freeze"] and len(parts) == 1 and cache_key == last_cache_key)
            request_seed = int(data["seed"])
            if fast and len(parts) > 1 and (data["random_seed"] or request_seed < 0):
                request_seed = secrets.randbelow(2**32)
            common = dict(
                prompt_text=data["prompt_text"],
                top_k=data["top_k"],
                top_p=data["top_p"],
                temperature=data["temperature"],
                sample_steps=data["sample_steps"],
                cfg_rate=data["cfg_rate"],
            )
            samples, sample_rate, result_seed = [], None, request_seed
            for index, (part, boundary) in enumerate(parts):
                if stopped.is_set():
                    break
                if fast:
                    stream = engine["inference"](
                        text=part,
                        text_lang=data["text_language"],
                        ref_audio_path=data["reference"],
                        aux_ref_audio_paths=refs,
                        prompt_lang=data["prompt_language"],
                        text_split_method="不切",
                        batch_size=data["batch_size"],
                        speed_factor=data["speed"],
                        ref_text_free=data["ref_free"],
                        split_bucket=data["split_bucket"],
                        fragment_interval=0,
                        seed=(request_seed + index) % 2**32 if len(parts) > 1 else data["seed"],
                        keep_random=data["random_seed"] if len(parts) == 1 else False,
                        use_cuda_graph=data["cuda_graph"],
                        use_flash_attention=data["flash_attention"],
                        repetition_penalty=data["repetition_penalty"],
                        super_sampling=data["super_sampling"],
                        **common,
                    )
                else:
                    stream = engine["get_tts_wav"](
                        ref_wav_path=data["reference"],
                        prompt_language=data["prompt_language"],
                        text=part,
                        text_language=data["text_language"],
                        how_to_cut="不切",
                        ref_free=data["ref_free"],
                        speed=data["speed"],
                        if_freeze=reuse_semantics,
                        inp_refs=refs,
                        if_sr=data["super_sampling"],
                        pause_second=0,
                        use_cuda_graph=data["cuda_graph"],
                        use_flash_attention=data["flash_attention"],
                        **common,
                    )
                piece_audio = None
                for item in stream:
                    if stopped.is_set():
                        break
                    audio, result_seed = item if fast else (item, request_seed)
                    rate, chunk = int(audio[0]), np.asarray(audio[1], dtype=np.int16)
                    if sample_rate is not None and rate != sample_rate:
                        raise ValueError("分段采样率不一致，无法拼接")
                    sample_rate = rate
                    piece_audio = chunk if piece_audio is None else np.concatenate((piece_audio, chunk))
                if stopped.is_set():
                    break
                if piece_audio is None or piece_audio.size == 0:
                    raise ValueError("有文本片段未生成音频，请调整切分方式")
                piece_audio = trim_leading_quiet(piece_audio, sample_rate)
                if samples:
                    samples[-1] = trim_trailing_quiet(samples[-1], sample_rate)
                    between = punctuation_seconds if boundary == "punctuation" else pause_seconds
                    if between:
                        samples.append(np.zeros(round(sample_rate * between), dtype=np.int16))
                samples.append(piece_audio)
                send("progress", id=message["id"], current=index + 1, total=len(parts))
            if samples and not stopped.is_set():
                import soundfile as sf

                target = Path(message["output_dir"]) / f"{message['id']}.wav"
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, np.concatenate(samples), sample_rate, subtype="PCM_16")
                shown_seed = request_seed if fast and len(parts) > 1 else result_seed
                send("audio", id=message["id"], path=str(target), seed=int(shown_seed))
                last_cache_key = cache_key if not fast and len(parts) == 1 else None
            elif not stopped.is_set():
                raise ValueError("没有生成音频，请检查参考音频、标注和模型设置")
            send("done", id=message["id"], cancelled=stopped.is_set())
        except Exception as error:
            last_cache_key = None
            traceback.print_exc()
            send("error", id=message["id"], message=str(error))


if __name__ == "__main__":
    main()
