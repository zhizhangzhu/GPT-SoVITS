"""Optional dry inputs; keep originals and commit only verified, aligned pairs."""
from pathlib import Path

import soundfile as sf

from . import storage as db
from .convert_cache import fingerprint, matches
from .separation import separate, signature

TRACKS = {"lead", "backing", "vocals"}


def prepare(worker, source, track):
    if track not in TRACKS:
        raise ValueError("无效的去混响音轨")
    source = Path(source)
    label = {"lead": "主唱", "backing": "和声", "vocals": "全部人声"}[track]
    key = {"version": 1, "model": signature("dereverb"),
           "source_sha256": db.digest(source, check_cancel=worker.check)}
    state_path = worker.path / ("dereverb-" + track + ".json")
    dry = worker.path / ("dry_" + track + ".wav")
    residual = worker.path / (track + "_reverb.wav")
    frames = sf.info(source).frames
    try:
        state = db.read(state_path, {})
    except (OSError, ValueError):
        state = {}
    digest = lambda path: db.digest(path, check_cancel=worker.check)
    if (isinstance(state, dict) and state.get("key") == key
            and matches(dry, state.get("dry"), frames, digest)
            and matches(residual, state.get("residual"), frames, digest)):
        worker.update("复用已校验的" + label + "去混响音轨")
        return dry, residual
    state_path.unlink(missing_ok=True)
    a, b = dry.with_suffix(".part.wav"), residual.with_suffix(".part.wav")
    try:
        separate(worker, source, a, b, "dereverb", label + "去混响")
        state = {"key": key, "dry": fingerprint(a, frames, digest),
                 "residual": fingerprint(b, frames, digest)}
        a.replace(dry)
        b.replace(residual)
        db.write(state_path, state)
        return dry, residual
    finally:
        a.unlink(missing_ok=True)
        b.unlink(missing_ok=True)


def inputs(worker, vocals, backing, outputs):
    """The caller publishes outputs after each complete stage."""
    if worker.options.get("dereverb_lead", False):
        track = "lead" if backing is not None else "vocals"
        vocals, residual = prepare(worker, vocals, track)
        outputs.update({"dry_" + track: str(vocals), track + "_reverb": str(residual)})
        worker.update("去混响音轨已准备，可分别试听处理前后", outputs=dict(outputs))
    if backing is not None and worker.options.get("dereverb_backing", False):
        backing, residual = prepare(worker, backing, "backing")
        outputs.update(dry_backing=str(backing), backing_reverb=str(residual))
        worker.update("和声去混响音轨已准备", outputs=dict(outputs))
    return vocals, backing
