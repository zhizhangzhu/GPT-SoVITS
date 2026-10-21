"""Verified decode/separation stages reused by conversion retries."""
from pathlib import Path

import numpy as np
import soundfile as sf

from . import storage as db
from .audio import decode
from .separation import signature
from tools.workspace_resources import require_space

VERSION = 3
SEPARATION_VERSION = 2
DENOISE_VERSION = 1


def inherit(worker):
    """Copy prior task intermediates; prepare() verifies every reused stage afterwards."""
    previous = worker.job.get("retried_from")
    marker = worker.path / "inherited-stages.json"
    if not previous or marker.exists() or previous == worker.job.get("id"):
        return
    directory = db.job_dir(previous)
    previous_options = db.read(directory / "job.json", {}).get("options", {})
    old_vocal = signature(previous_options.get("separation_model", "hp2"))
    new_vocal = signature(worker.options.get("separation_model", "hp2"))
    old_cleanup = previous_options.get("cleanup", "separate" if previous_options.get("mixed") else "none")
    new_cleanup = worker.options.get("cleanup", "separate" if worker.options.get("mixed") else "none")
    same_vocals = old_cleanup == new_cleanup and (old_vocal == new_vocal or new_cleanup != "separate")
    names = ["decoded.wav", "convert-stages.json"]
    if same_vocals:
        names += ["vocals.wav", "accompaniment.wav"]
    same_karaoke = signature(previous_options.get("vocal_split_model", "mel_karaoke")) == signature(worker.options.get("vocal_split_model", "mel_karaoke"))
    if same_vocals and same_karaoke and worker.options.get("vocal_mode", "off") != "off":
        names += ["lead_vocals.wav", "backing_vocals.wav", "vocal-split.json"]
    # These pairs are verified against the exact source hash and pinned model
    # by dereverb.prepare(). Never publish inherited files before validation.
    for track in ("lead", "backing", "vocals"):
        enabled = worker.options.get("dereverb_backing" if track == "backing" else "dereverb_lead", False)
        compatible = same_vocals and (track == "vocals" or same_karaoke)
        if enabled and compatible:
            names += ["dry_" + track + ".wav", track + "_reverb.wav", "dereverb-" + track + ".json"]
    candidates = [name for name in names if (directory / name).is_file() and not (directory / name).is_symlink()]
    require_space(worker.path, sum((directory / name).stat().st_size for name in candidates), "复制可复用分离缓存")
    for name in candidates:
        worker.check()
        target = worker.path / name
        if target.exists():
            continue
        temporary = worker.path / (".inherit-" + name)
        try:
            worker.update("复制可复用分离缓存：" + name, 3)
            with (directory / name).open("rb") as src, temporary.open("wb") as dst:
                while True:
                    worker.check()
                    block = src.read(1024 * 1024)
                    if not block:
                        break
                    dst.write(block)
            temporary.replace(target)
        except FileNotFoundError:
            # A deleted prior task simply leaves stages to be recomputed.
            continue
        finally:
            temporary.unlink(missing_ok=True)
    db.write(marker, {"from": previous})


def align_backing(source, target, frames, check):
    """Only trim or pad a short codec tail; never shift the backing in time."""
    info = sf.info(source)
    if info.samplerate != 44100 or info.channels != 2 or abs(info.frames - frames) > 4410:
        raise ValueError("人声与伴奏时长相差超过 0.1 秒，请上传同一起点的配对音轨")
    with sf.SoundFile(source) as input_audio, sf.SoundFile(target, "w", samplerate=44100, channels=2,
                                                           subtype="PCM_16") as output_audio:
        written = 0
        while written < frames:
            check()
            block = input_audio.read(min(65536, frames - written), dtype="float32", always_2d=True)
            if not len(block):
                block = np.zeros((min(65536, frames - written), 2), dtype=np.float32)
            output_audio.write(block)
            written += len(block)


def fingerprint(path, frames=None, digest=None):
    path = Path(path)
    info = sf.info(path)
    if info.samplerate != 44100 or info.channels != 2 or info.frames <= 0:
        raise ValueError("转换缓存音轨格式不正确，需要重新处理")
    if frames is not None and info.frames != frames:
        raise ValueError("转换缓存音轨时长不一致，需要重新处理")
    return {"sha256": (digest or db.digest)(path), "bytes": path.stat().st_size, "frames": info.frames}


def matches(path, saved, frames=None, digest=None):
    if not saved or not Path(path).is_file():
        return False
    try:
        current = fingerprint(path, frames, digest)
    except (OSError, ValueError, RuntimeError):
        return False
    return current == saved


def prepare(worker):
    inherit(worker)
    original = Path(worker.job["input"])
    input_sha = worker.hash_file(original, "校验转换源文件", 3, 4)
    provided = bool(worker.job.get("accompaniment_input"))
    backing_sha = worker.hash_file(worker.job["accompaniment_input"], "校验伴奏源文件", 4, 5) if provided else None
    cleanup = worker.options.get("cleanup", "separate" if worker.options.get("mixed") else "none")
    mode = "provided" if provided else cleanup if cleanup in {"denoise", "separate"} else "vocals"
    state_path = worker.path / "convert-stages.json"
    try:
        state = db.read(state_path, {})
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    if (state.get("version") != VERSION or state.get("input_sha256") != input_sha
            or state.get("mode") != mode or state.get("accompaniment_sha256") != backing_sha):
        state = {"version": VERSION, "input_sha256": input_sha, "mode": mode,
                 "accompaniment_sha256": backing_sha}
        db.write(state_path, state)
    decoded = worker.path / "decoded.wav"
    decoded_digest = lambda path: worker.hash_file(path, "校验解码音轨", 4, 6)
    if matches(decoded, state.get("decoded"), digest=decoded_digest):
        worker.update("复用已校验的解码音轨", 6)
    else:
        worker.update("解码转换音频", 4)
        temporary = worker.path / "decoded.part.wav"
        temporary.unlink(missing_ok=True)
        decode(original, temporary, check_cancel=worker.check)
        details = fingerprint(temporary, digest=decoded_digest)
        space_factor = 5 if provided else 7 if worker.options["mixed"] else 2
        require_space(worker.path, details["bytes"] * space_factor
                      + 64 * 1024 ** 2, "SVC 人声转换与临时音轨")
        temporary.replace(decoded)
        state["decoded"] = details
        state.pop("separated", None)
        db.write(state_path, state)
    worker.check()
    if mode == "vocals":
        return decoded
    if mode == "denoise":
        vocals = worker.path / "vocals.wav"
        processed = state.get("processed", {})
        if (processed.get("version") == DENOISE_VERSION
                and processed.get("decoded_sha256") == state["decoded"]["sha256"]
                and matches(vocals, processed.get("vocals"), frames=state["decoded"]["frames"],
                            digest=lambda path: worker.hash_file(path, "校验降噪人声", 6, 7))):
            worker.update("复用已校验的语音降噪结果", 9)
            return vocals
        worker.update("语音降噪", 9)
        temporary = worker.path / "vocals.part.wav"
        temporary.unlink(missing_ok=True)
        worker.denoise(decoded, temporary)
        details = fingerprint(temporary, state["decoded"]["frames"],
                             digest=lambda path: worker.hash_file(path, "校验降噪结果", 9, 9.5))
        temporary.replace(vocals)
        state["processed"] = {"version": DENOISE_VERSION, "decoded_sha256": state["decoded"]["sha256"],
                               "vocals": details}
        state.pop("separated", None)
        db.write(state_path, state)
        return vocals
    vocals = worker.path / "vocals.wav"
    accompaniment = worker.path / "accompaniment.wav"
    separated = state.get("separated", {})
    if not isinstance(separated, dict):
        separated = {}
    frames = state["decoded"]["frames"]
    if mode == "provided":
        if (separated.get("accompaniment_sha256") == backing_sha
                and matches(accompaniment, separated.get("backing"), frames,
                            digest=lambda path: worker.hash_file(path, "校验伴奏缓存", 6, 8))):
            worker.update("复用已校验的独立伴奏", 9)
            return decoded
        worker.update("解码独立伴奏并核对时长", 7)
        temporary = worker.path / "accompaniment.part.wav"
        aligned = worker.path / "accompaniment.aligned.wav"
        temporary.unlink(missing_ok=True)
        aligned.unlink(missing_ok=True)
        decode(worker.job["accompaniment_input"], temporary, check_cancel=worker.check)
        try:
            align_backing(temporary, aligned, frames, worker.check)
            details = fingerprint(aligned, frames, digest=lambda path: worker.hash_file(path, "校验配对伴奏", 8, 9))
            aligned.replace(accompaniment)
            state["separated"] = {"accompaniment_sha256": backing_sha, "backing": details}
            db.write(state_path, state)
        finally:
            temporary.unlink(missing_ok=True)
            aligned.unlink(missing_ok=True)
        return decoded
    if (separated.get("version") == SEPARATION_VERSION
            and separated.get("model", {"id": "hp2", "version": 2}) == signature(worker.options.get("separation_model", "hp2"))
            and separated.get("decoded_sha256") == state["decoded"]["sha256"]
            and matches(vocals, separated.get("vocals"), frames,
                        digest=lambda path: worker.hash_file(path, "校验人声缓存", 6, 7))
            and matches(accompaniment, separated.get("accompaniment"), frames,
                        digest=lambda path: worker.hash_file(path, "校验伴奏缓存", 7, 8))):
        worker.update("复用已校验的人声与伴奏分离结果", 9)
        return vocals
    worker.update("分离人声与伴奏", 9)
    # An inherited old-model manifest must not advertise old stems while the
    # replacement model is running or after it fails.
    state.pop("separated", None)
    (worker.path / "vocal-split.json").unlink(missing_ok=True)
    db.write(state_path, state)
    partial_voice = worker.path / "vocals.part.wav"
    partial_inst = worker.path / "accompaniment.part.wav"
    partial_voice.unlink(missing_ok=True)
    partial_inst.unlink(missing_ok=True)
    require_space(worker.path, state["decoded"]["bytes"] * 7 + 64 * 1024 ** 2,
                  "SVC 人声分离与转换")
    worker.separate(decoded, partial_voice, partial_inst, worker.path / "separation")
    voice_state = fingerprint(partial_voice, frames, digest=lambda path: worker.hash_file(path, "校验分离人声", 9, 9.5))
    inst_state = fingerprint(partial_inst, frames, digest=lambda path: worker.hash_file(path, "校验分离伴奏", 9.5, 10))
    partial_voice.replace(vocals)
    partial_inst.replace(accompaniment)
    state["separated"] = {"version": SEPARATION_VERSION, "model": signature(worker.options.get("separation_model", "hp2")),
                          "decoded_sha256": state["decoded"]["sha256"],
                          "vocals": voice_state, "accompaniment": inst_state}
    db.write(state_path, state)
    return vocals
