"""Pinned Karaoke separator for lead and backing vocals on an existing vocal stem.

Architecture: tools/uvr5/bs_roformer (existing project implementation).
Weights/config: https://huggingface.co/becruily/mel-band-roformer-karaoke
The two model outputs are Vocals (lead) and Instrumental (backing on vocal input).
"""
import gc
import os
from pathlib import Path
import sys

from filelock import FileLock, Timeout
import numpy as np
import soundfile as sf

from . import storage as db
from .install import ASSETS, ROOT
from tools.workspace_resources import require_space

REPO = "becruily/mel-band-roformer-karaoke"
REVISION = "0c149975cfaa261c7d87baf54330a9da85bcf888"
FILENAME = "mel_band_roformer_karaoke_becruily.ckpt"
SHA256 = "d3aa262ac01df870b9fc033e9c7b6cad33fe04fc9c148b6c40841326a515a0e0"
MODEL_BYTES = 1719139254
VERSION = 1
MODES = [("仅转换主唱，保留原和声", "keep"),
         ("主唱与和声分别转换", "convert"),
         ("仅保留转换主唱", "remove")]
# Inference-relevant fields from config_karaoke_becruily.yaml at REVISION.
MODEL_CONFIG = dict(dim=384, depth=6, stereo=True, num_stems=2,
                    time_transformer_depth=1, freq_transformer_depth=1, num_bands=60,
                    dim_head=64, heads=8, attn_dropout=0, ff_dropout=0, flash_attn=True,
                    dim_freqs_in=1025, sample_rate=44100, stft_n_fft=2048,
                    stft_hop_length=441, stft_win_length=2048, stft_normalized=False,
                    mask_estimator_depth=2, multi_stft_resolution_loss_weight=1.0,
                    multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
                    multi_stft_hop_size=147, multi_stft_normalized=False,
                    match_input_audio_length=True)
WINDOW = 485100
OVERLAP = 8


def install_model(check_cancel=None, progress=None):
    from .download import download
    directory = ASSETS / "vocal_split"
    directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(directory / ".install.lock"))
    while True:
        if check_cancel:
            check_cancel()
        try:
            lock.acquire(timeout=1)
            break
        except Timeout:
            continue
    try:
        path = directory / FILENAME
        if not path.is_file() or db.digest(path, check_cancel=check_cancel) != SHA256:
            require_space(directory, MODEL_BYTES * 2, "主唱和声分离模型下载")
            result = download(REPO, FILENAME, path,
                              endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                              workers=12, revision=REVISION, check_cancel=check_cancel, progress=progress)
            if result["sha256"] != SHA256:
                raise ValueError("主唱和声分离模型版本校验失败")
        db.write(directory / "manifest.json", dict(repo=REPO, revision=REVISION,
                 sha256=SHA256, file=FILENAME, config=MODEL_CONFIG,
                 model_outputs=["lead", "backing"], window=WINDOW, overlap=OVERLAP))
        return path
    finally:
        lock.release()


def _predict(model, samples, device, check, progress):
    """Bounded block inference with the pinned config's eight-fold overlap."""
    import torch
    size, hop = WINDOW, WINDOW // OVERLAP
    border = size - hop
    signal = np.pad(samples.T, ((0, 0), (border, border)), mode="reflect")
    result = np.zeros((2, 2, signal.shape[-1]), dtype=np.float32)
    weights = np.zeros(signal.shape[-1], dtype=np.float32)
    window = np.ones(size, dtype=np.float32)
    fade = size // 10
    window[:fade] = np.linspace(0, 1, fade)
    window[-fade:] = np.linspace(1, 0, fade)
    for start in range(0, signal.shape[-1], hop):
        check()
        length = min(size, signal.shape[-1] - start)
        chunk = np.pad(signal[:, start:start + length], ((0, 0), (0, size - length)))
        if np.max(np.abs(chunk)) > 1e-7:
            tensor = torch.from_numpy(np.ascontiguousarray(chunk)).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
                predicted = model(tensor)[0].float().cpu().numpy()
            if predicted.shape != (2, 2, size) or not np.all(np.isfinite(predicted)):
                raise ValueError("主唱和声分离输出形状或数值异常")
            result[:, :, start:start + length] += predicted[:, :, :length] * window[:length]
        weights[start:start + length] += window[:length]
        progress(min(max(0, start + hop - border), len(samples)))
    return result[:, :, border:border + len(samples)] / np.maximum(weights[border:border + len(samples)], 1e-6)


def separate(worker, source, lead, backing):
    from . import separation
    selected = worker.options.get("vocal_split_model", "mel_karaoke")
    separation.validate("hp2", selected)
    if selected != "mel_karaoke":
        return separation.separate(worker, source, lead, backing, selected, "分离主唱 / 和声", (11, 28))
    import torch
    worker.update("准备主唱 / 和声分离模型（首次使用需下载）", 10)
    model_path = install_model(worker.check_cancel,
                              lambda done, total: worker.update(f"下载主唱和声模型 {done / total:.0%}", 10))
    if worker.hash_file(model_path, "校验主唱和声分离模型", 10, 11) != SHA256:
        raise ValueError("主唱和声分离模型被修改，请重新安装")
    sys.path.insert(0, str(ROOT / "tools/uvr5"))
    from bs_roformer.mel_band_roformer import MelBandRoformer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MelBandRoformer(**MODEL_CONFIG)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    try:
        with sf.SoundFile(source) as src, sf.SoundFile(lead, "w", samplerate=44100, channels=2, subtype="FLOAT") as a, sf.SoundFile(backing, "w", samplerate=44100, channels=2, subtype="FLOAT") as b:
            if src.samplerate != 44100 or src.channels != 2 or not len(src):
                raise ValueError("主唱和声分离需要非空 44.1 kHz 双声道人声")
            # Context is trimmed only at the block boundary; original positions are retained.
            for start in range(0, len(src), 44100 * 30):
                worker.check()
                end = min(len(src), start + 44100 * 30)
                left, right = max(0, start - WINDOW), min(len(src), end + WINDOW)
                src.seek(left)
                samples = src.read(right - left, dtype="float32", always_2d=True)
                result = _predict(model, samples, device, worker.check,
                    lambda done: worker.update(f"分离主唱 / 和声 {min(end, max(start, left + done)) / 44100:.1f}/{len(src) / 44100:.1f} 秒",
                                               11 + 17 * min(end, max(start, left + done)) / len(src)))
                a.write(result[0, :, start - left:end - left].T)
                b.write(result[1, :, start - left:end - left].T)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def prepare(worker, source):
    """Commit both stems together and reuse only a complete, verified pair."""
    from .convert_cache import fingerprint, matches
    source = Path(source)
    key = dict(version=VERSION, model_sha256=SHA256,
               source_sha256=worker.hash_file(source, "校验主唱分离输入", 9, 10))
    options = getattr(worker, "options", {})
    if options.get("vocal_split_model", "mel_karaoke") != "mel_karaoke":
        from .separation import signature
        key["model"] = signature(options["vocal_split_model"])
        key["model_sha256"] = key["model"]["sha256"]
    state_path = worker.path / "vocal-split.json"
    try:
        state = db.read(state_path, {})
    except (OSError, ValueError):
        state = {}
    lead, backing = worker.path / "lead_vocals.wav", worker.path / "backing_vocals.wav"
    frames = sf.info(source).frames
    if (isinstance(state, dict) and state.get("key") == key
            and matches(lead, state.get("lead"), frames, digest=lambda p: worker.hash_file(p, "校验主唱缓存", 10, 11))
            and matches(backing, state.get("backing"), frames, digest=lambda p: worker.hash_file(p, "校验和声缓存", 11, 12))):
        worker.update("复用已校验的主唱和和声音轨", 28)
        return lead, backing
    state_path.unlink(missing_ok=True)
    require_space(worker.path, frames * 2 * 4 * 10 + 64 * 1024 ** 2, "主唱和声分离与转换")
    a, b = worker.path / "lead_vocals.part.wav", worker.path / "backing_vocals.part.wav"
    try:
        separate(worker, source, a, b)
        state = dict(key=key, lead=fingerprint(a, frames), backing=fingerprint(b, frames))
        a.replace(lead)
        b.replace(backing)
        db.write(state_path, state)
        return lead, backing
    finally:
        a.unlink(missing_ok=True)
        b.unlink(missing_ok=True)
