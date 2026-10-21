"""Pinned local RoFormer models for the two SVC separation stages."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

from filelock import FileLock, Timeout
import numpy as np
import soundfile as sf

from . import storage as db
from .install import ASSETS, ROOT
from tools.workspace_resources import require_space

VOCAL_DEFAULT = "mel_deux"
KARAOKE_DEFAULT = "bs_karaoke"
VOCAL_CHOICES = [("Mel-Band RoFormer Deux · 推荐", VOCAL_DEFAULT), ("HP2 · 旧版", "hp2")]
KARAOKE_CHOICES = [("BS-RoFormer Karaoke · frazer / becruily · 推荐", KARAOKE_DEFAULT),
                   ("Mel-Band Karaoke · becruily · 旧版", "mel_karaoke")]
PROFILES = {
    "dereverb": dict(name="BS-RoFormer De-Reverb · anvuew", architecture="bs_roformer",
        repo="anvuew/dereverb_bs_roformer", revision="bd5c6b55a429b4b74ce85fe5dcb690cfe36d91ec",
        file="dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt", bytes=204496717,
        sha256="b4dfa4c4aa251c0afc303e5d512d904a1c2255e51d8d0dbfbddb18d740563d42",
        source_config="config.yaml", stems=["noreverb", "reverb_residual"], license="GPL-3.0"),
    "mel_deux": dict(name="Mel-Band RoFormer Deux", architecture="mel_band_roformer",
        repo="becruily/mel-band-roformer-deux", revision="402dd9a19baaed8c6f1b54458ce6ae2ae19f354e",
        file="becruily_deux.ckpt", bytes=435006815,
        sha256="10255c02295bf3e3865d4ee50ff752d7b19b124ed5fd93b147babc4333eda3aa",
        source_config="config_deux_becruily.yaml", stems=["vocals", "accompaniment"]),
    "bs_karaoke": dict(name="BS-RoFormer Karaoke · frazer / becruily", architecture="bs_roformer",
        repo="becruily/bs-roformer-karaoke", revision="f7849ae934209184dc288d1018cd8a76a7fc8b3c",
        file="bs_roformer_karaoke_frazer_becruily.ckpt", bytes=204436907,
        sha256="eb90ee24c1154d83fbcfd27e96182f19e061557cc6e4746953125e08c29389f9",
        source_config="config_karaoke_frazer_becruily.yaml", stems=["lead", "backing_residual"]),
}


def validate(vocal, karaoke=None):
    if vocal not in {value for _, value in VOCAL_CHOICES}:
        raise ValueError("请选择有效的人声 / 伴奏分离模型")
    if karaoke is not None and karaoke not in {value for _, value in KARAOKE_CHOICES}:
        raise ValueError("请选择有效的主唱 / 和声分离模型")


def config(model_id):
    data = db.read(Path(__file__).parent / "model_configs" / (model_id + ".json"))
    if "freqs_per_bands" in data["model"]:
        data["model"]["freqs_per_bands"] = tuple(data["model"]["freqs_per_bands"])
    return data


def signature(model_id):
    if model_id == "hp2":
        return {"id": "hp2", "version": 2}
    if model_id == "mel_karaoke":
        return {"id": model_id, "sha256": "d3aa262ac01df870b9fc033e9c7b6cad33fe04fc9c148b6c40841326a515a0e0"}
    profile = PROFILES[model_id]
    settings = config(model_id)
    return {"id": model_id, "revision": profile["revision"], "sha256": profile["sha256"], "inference_version": 1,
            "config_sha256": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()}


def install_model(model_id, check=None, progress=None):
    from .download import download
    profile = PROFILES[model_id]
    directory = ASSETS / "separation" / model_id
    directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(directory / ".install.lock"))
    while True:
        if check:
            check()
        try:
            lock.acquire(timeout=1)
            break
        except Timeout:
            continue
    try:
        path = directory / profile["file"]
        if not path.is_file() or db.digest(path, check_cancel=check) != profile["sha256"]:
            require_space(directory, profile["bytes"] * 2, "分离模型下载")
            result = download(profile["repo"], profile["file"], path,
                endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"), workers=16,
                revision=profile["revision"], check_cancel=check, progress=progress)
            if result["sha256"] != profile["sha256"]:
                raise ValueError("分离模型 SHA256 校验失败")
        db.write(directory / "manifest.json", {**profile, **signature(model_id), "inference": config(model_id)})
        return path
    finally:
        lock.release()


def load_model(model_id, path):
    import torch
    sys.path.insert(0, str(ROOT / "tools/uvr5"))
    if PROFILES[model_id]["architecture"] == "bs_roformer":
        from bs_roformer.bs_roformer import BSRoformer as Model
    else:
        from bs_roformer.mel_band_roformer import MelBandRoformer as Model
    model = Model(**config(model_id)["model"])
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights.get("state_dict", weights), strict=True)
    return model


def _predict(model, samples, device, settings, check, progress):
    import torch
    size, hop = settings["window"], settings["window"] // settings["overlap"]
    border = size - hop
    signal = np.pad(samples.T, ((0, 0), (border, border)), mode="reflect")
    result = np.zeros((2, 2, signal.shape[-1]), dtype=np.float32)
    weights = np.zeros(signal.shape[-1], dtype=np.float32)
    window = np.ones(size, dtype=np.float32)
    fade = size // 10
    window[:fade], window[-fade:] = np.linspace(0, 1, fade), np.linspace(1, 0, fade)
    for start in range(0, signal.shape[-1], hop):
        check()
        length = min(size, signal.shape[-1] - start)
        chunk = np.pad(signal[:, start:start + length], ((0, 0), (0, size - length)))
        if np.max(np.abs(chunk)) > 1e-7:
            tensor = torch.from_numpy(np.ascontiguousarray(chunk)).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
                predicted = model(tensor)[0].float().cpu().numpy()
            if settings["model"]["num_stems"] == 1:
                # Single-target models return lead or dry vocals. Preserve the
                # residual separately, without changing the shared timeline.
                if predicted.shape != (2, size):
                    raise ValueError("单目标分离模型输出形状异常")
                predicted = np.stack([predicted, chunk - predicted])
            if predicted.shape != (2, 2, size) or not np.isfinite(predicted).all():
                raise ValueError("分离模型输出形状或数值异常")
            result[:, :, start:start + length] += predicted[:, :, :length] * window[:length]
        weights[start:start + length] += window[:length]
        progress(min(max(0, start + hop - border), len(samples)))
    return result[:, :, border:border + len(samples)] / np.maximum(weights[border:border + len(samples)], 1e-6)


def separate(worker, source, first, second, model_id, stage, progress_range=None):
    import torch
    profile, settings = PROFILES[model_id], config(model_id)
    worker.update(f"准备{stage}模型：{profile['name']}")
    path = install_model(model_id, worker.check_cancel,
        lambda done, total: worker.update(f"下载{profile['name']} {done / total:.0%}"))
    worker.check()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_id, path).to(device).eval()
    try:
        worker.check()
        with sf.SoundFile(source) as src:
            if src.samplerate != 44100 or src.channels != 2 or not len(src):
                raise ValueError("分离输入需要非空 44.1 kHz 双声道音频")
            require_space(Path(first).parent, len(src) * 2 * 4 * 3 + 128 * 1024 ** 2, stage)
            with sf.SoundFile(first, "w", samplerate=44100, channels=2, subtype="FLOAT") as a, \
                 sf.SoundFile(second, "w", samplerate=44100, channels=2, subtype="FLOAT") as b:
                for start in range(0, len(src), 44100 * 30):
                    worker.check()
                    end = min(len(src), start + 44100 * 30)
                    left, right = max(0, start - settings["window"]), min(len(src), end + settings["window"])
                    src.seek(left)
                    samples = src.read(right - left, dtype="float32", always_2d=True)
                    if len(samples) != right - left or not np.isfinite(samples).all():
                        raise ValueError("分离输入读取异常")
                    def report(done):
                        position = min(end, max(start, left + done))
                        percent = None if progress_range is None else progress_range[0] + (progress_range[1] - progress_range[0]) * position / len(src)
                        worker.update(f"{stage} · {profile['name']} · {position / 44100:.1f}/{len(src) / 44100:.1f} 秒", percent)
                    result = _predict(model, samples, device, settings, worker.check, report)
                    a.write(result[0, :, start - left:end - left].T)
                    b.write(result[1, :, start - left:end - left].T)
        manifest = db.read(worker.path / "separation-models.json", {})
        manifest[model_id] = {**profile, **signature(model_id), "inference": settings}
        db.write(worker.path / "separation-models.json", manifest)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
