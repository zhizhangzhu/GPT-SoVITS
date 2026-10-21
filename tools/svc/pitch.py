"""Pinned neural F0 analysis, channel diagnostics and reusable aligned curves."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import soundfile as sf

from . import storage as db
from .install import ASSETS, ROOT, RUNTIME

METHODS = {"dio", "harvest", "pm", "rmvpe", "fcpe"}
CHOICES = [("RMVPE · 歌声音高", "rmvpe"), ("FCPE · 快速歌声音高", "fcpe"),
           ("DIO", "dio"), ("Harvest", "harvest"), ("PM · Praat", "pm")]
RMVPE_SHA = "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"
RMVPE_REV = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
VERSION = 2


def install_models(check=None):
    from filelock import FileLock
    from .download import download
    directory = ASSETS / "pitch"
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".install.lock")):
        path = directory / "rmvpe.pt"
        if not path.is_file() or db.digest(path, check_cancel=check) != RMVPE_SHA:
            result = download("lj1995/VoiceConversionWebUI", "rmvpe.pt", path,
                              revision=RMVPE_REV, check_cancel=check)
            if result["sha256"] != RMVPE_SHA:
                raise ValueError("RMVPE 权重校验失败")
        db.write(directory / "rmvpe-manifest.json", {"repo": "lj1995/VoiceConversionWebUI",
                 "revision": RMVPE_REV, "sha256": RMVPE_SHA, "file": "rmvpe.pt"})
    return path


def signature(method):
    if method == "rmvpe":
        return {"method": method, "sha256": RMVPE_SHA, "version": VERSION}
    if method == "fcpe":
        import torchfcpe
        from importlib.metadata import version
        path = Path(torchfcpe.__file__).parent / "assets/fcpe_c_v001.pt"
        return {"method": method, "package": version("torchfcpe"), "sha256": db.digest(path), "version": VERSION}
    return {"method": method, "version": VERSION}


class Predictor:
    """Expose raw voiced frames for reports; interpolation is only for SVC input."""
    def __init__(self, method, rate=44100, hop=512, device=None):
        import torch
        if method not in METHODS:
            raise ValueError("未知音高提取方式")
        self.method, self.rate, self.hop = method, rate, hop
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        sys.path.insert(0, str(RUNTIME))
        if method == "rmvpe":
            from modules.F0Predictor.rmvpe.inference import RMVPE
            from modules.F0Predictor.rmvpe.model import E2E0
            from modules.F0Predictor.rmvpe.spec import MelSpectrogram
            from modules.F0Predictor.rmvpe.constants import N_MELS, SAMPLE_RATE, WINDOW_LENGTH, MEL_FMIN, MEL_FMAX
            path = install_models()
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = E2E0(4, 1, (2, 2))
            # The pinned RVC checkpoint uses the original RMVPE network.
            # SVC's copy registers an unused timbre-filter branch in E2E0;
            # its forward never calls that branch. Remove it for strict loading.
            if not any(k.startswith("unet.tf.") for k in state.get("model", state)):
                del model.unet.tf
            model.load_state_dict(state.get("model", state), strict=True)
            self.model = RMVPE.__new__(RMVPE)
            self.model.device, self.model.dtype = torch.device(self.device), torch.float32
            self.model.model = model.to(self.device).eval()
            self.model.resample_kernel = {}
            self.model.mel_extractor = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, 160, None, MEL_FMIN, MEL_FMAX)
        elif method == "fcpe":
            from torchfcpe import spawn_bundled_infer_model
            self.model = spawn_bundled_infer_model(device=self.device)
        else:
            import utils
            self.model = utils.get_f0_predictor(method, hop_length=hop, sampling_rate=rate)

    def raw(self, samples):
        import torch
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.
        if peak > 1:
            samples = samples / peak
        if self.method == "rmvpe":
            with torch.inference_mode():
                f0 = self.model.infer_from_audio(torch.as_tensor(samples, dtype=torch.float32), self.rate, .05)
            f0 = f0.detach().cpu().numpy()
            times = np.arange(len(f0)) * .01
        elif self.method == "fcpe":
            with torch.inference_mode():
                f0 = self.model.infer(torch.as_tensor(samples, dtype=torch.float32, device=self.device)[None, :, None],
                    sr=self.rate, decoder_mode="local_argmax", threshold=.006, f0_min=50, f0_max=1600,
                    interp_uv=False, output_interp_target_length=len(samples) // self.hop + 1)
            f0 = f0.detach().float().cpu().numpy().reshape(-1)
            times = np.arange(len(f0)) * self.hop / self.rate
        else:
            f0, uv = self.model.compute_f0_uv(samples)
            f0 = np.where(np.asarray(uv) > .5, f0, 0)
            times = np.arange(len(f0)) * self.hop / self.rate
        f0 = np.asarray(f0, dtype=np.float32)
        f0[~np.isfinite(f0) | (f0 < 40) | (f0 > 2000)] = 0
        return times, f0

    def compute_f0_uv(self, samples):
        times, raw = self.raw(samples)
        grid = np.arange(max(1, len(samples) // self.hop)) * self.hop / self.rate
        f0, uv = sample_curve(times, raw, grid)
        return continuous(f0, uv), uv.astype(np.float32)


def continuous(f0, voiced):
    idx = np.flatnonzero(voiced)
    return np.interp(np.arange(len(f0)), idx, np.asarray(f0)[idx]).astype(np.float32) if len(idx) else np.zeros(len(f0), np.float32)


def sample_curve(times, raw, grid):
    if not len(times):
        return np.zeros(len(grid), np.float32), np.zeros(len(grid), bool)
    nearest = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    prev = np.maximum(0, nearest - 1)
    nearest = np.where(np.abs(times[prev] - grid) < np.abs(times[nearest] - grid), prev, nearest)
    voiced = raw[nearest] > 0
    # Log-frequency interpolation within voiced runs, with UV masked afterwards.
    valid = raw > 0
    result = (np.exp(np.interp(grid, times[valid], np.log(raw[valid])))
              if valid.any() else np.zeros(len(grid)))
    return np.where(voiced, result, 0).astype(np.float32), voiced


def channels(path, check=None):
    info = sf.info(path)
    if info.channels == 1:
        return {"selection": "mono", "correlation": 1., "mid_loss_db": 0., "cancelled_fraction": 0., "suspected": False}
    sums = np.zeros(5, np.float64)
    active = bad = 0
    with sf.SoundFile(path) as src:
        while True:
            if check:
                check()
            x = src.read(4410, dtype="float32", always_2d=True)
            if not len(x):
                break
            x = x.astype(np.float64)
            l, r = x[:, 0], x[:, 1]
            el, er, cross = np.sum(l*l), np.sum(r*r), np.sum(l*r)
            mid = np.sum(((l+r)*.5)**2)
            sums += [el, er, cross, mid, len(l)]
            if max(el, er) / len(l) > 1e-6:
                active += 1
                bad += int(mid < .25 * max(el, er) and cross < 0)
    el, er, cross, mid, _ = sums
    corr = float(np.clip(cross / max(np.sqrt(el * er), 1e-12), -1, 1))
    loss = float(10*np.log10(max(mid,1e-12)/max((el+er)*.5,1e-12)))
    fraction = bad / max(active, 1)
    suspected = fraction > .02
    selection = ("left" if el >= er else "right") if suspected else "mid"
    return {"selection": selection, "correlation": round(corr,4), "mid_loss_db": round(loss,2),
            "cancelled_fraction": round(fraction,4), "suspected": suspected}


def mono(samples, selection):
    if samples.ndim == 1:
        return samples
    if selection == "left":
        return samples[:, 0]
    if selection == "right":
        return samples[:, 1]
    return samples.mean(axis=1)


def analyze(worker, path, method="rmvpe", hop=512, detector=None):
    """Bounded analysis with shared global frame grid and verified atomic cache."""
    path = Path(path)
    rate = sf.info(path).samplerate
    channel = channels(path, worker.check)
    key = {"source": db.digest(path, check_cancel=worker.check), "model": signature(method),
           "rate": rate, "hop": hop, "channel": channel, "version": VERSION}
    token = hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:24]
    directory = worker.path / "pitch-cache"
    directory.mkdir(exist_ok=True)
    target = directory / (token + ".npz")
    manifest = directory / (token + ".json")
    saved = db.read(manifest, {})
    if saved.get("key") == key and target.is_file() and db.digest(target,check_cancel=worker.check) == saved.get("sha256"):
        with np.load(target, allow_pickle=False) as v:
            return {k: v[k] for k in v.files}, channel
    detector = detector or Predictor(method, rate, hop)
    with sf.SoundFile(path) as src:
        times = np.arange(src.frames // hop + 1) * hop / rate
        f0 = np.zeros(len(times), np.float32)
        energy = np.zeros(len(times), np.float32)
        for start in range(0, len(src), rate * 30):
            worker.check()
            end = min(len(src), start + rate * 30)
            left, right = max(0,start-rate), min(len(src),end+rate)
            src.seek(left)
            x = mono(src.read(right-left,dtype="float32",always_2d=True), channel["selection"])
            t, raw = detector.raw(x)
            mask = (times >= start/rate) & (times < end/rate)
            f0[mask], _ = sample_curve(t+left/rate, raw, times[mask])
            # Windowed energy masks noise/silence instead of treating interpolated
            # SVC UV values as observed notes.
            squares = np.r_[0., np.cumsum(x.astype(np.float64)**2)]
            indexes = np.clip(np.rint(times[mask]*rate-left).astype(int),0,len(x)-1)
            a, b = np.maximum(0,indexes-441),np.minimum(len(x),indexes+441)
            energy[mask] = np.sqrt((squares[b]-squares[a])/np.maximum(b-a,1))
            worker.update(f"音高分析 · {method.upper()} · {end/rate:.1f}/{len(src)/rate:.1f} 秒")
    voiced = (f0>0) & (energy>10**(-50/20))
    f0[~voiced] = 0
    curves = dict(times=times,f0=f0,voiced=voiced,energy=energy)
    temporary = target.with_suffix(".part.npz")
    try:
        np.savez_compressed(temporary,**curves)
        temporary.replace(target)
        db.write(manifest,{"key":key,"sha256":db.digest(target,check_cancel=worker.check)})
    finally:
        temporary.unlink(missing_ok=True)
    return curves, channel
