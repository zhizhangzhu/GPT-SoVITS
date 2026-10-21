"""Bounded, local history for completed TTS generations."""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import threading
import uuid

from .output_speed import OUTPUT_DIR, OUTPUT_LOCK, source_path

MAX_RESULTS = 10
MANIFEST = OUTPUT_DIR / "history.json"
_lock = threading.RLock()


def _read():
    try:
        records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return []
    if not isinstance(records, list):
        return []
    valid = []
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get("id"), str)
                or not isinstance(record.get("label"), str)
                or not isinstance(record.get("text"), str)
                or not isinstance(record.get("config"), dict)):
            continue
        name = record.get("filename")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".wav"):
            continue
        path = (OUTPUT_DIR / name).resolve()
        if path.parent == OUTPUT_DIR.resolve() and path.is_file():
            valid.append(record)
    return valid[:MAX_RESULTS]


def _write(records):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_name(f"history.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, MANIFEST)
    finally:
        temporary.unlink(missing_ok=True)


def _title(text):
    words = re.sub(r"\s+", " ", text).strip()
    safe = re.sub(r"[^\w\u4e00-\u9fff]+", "-", words).strip("-_ ")[:24].strip("-_ ")
    return safe or "语音合成"


def _choices(records):
    return [(record["label"], record["id"]) for record in records]


def choices():
    with _lock:
        return _choices(_read())


def get(record_id):
    with _lock:
        return next((record for record in _read() if record["id"] == record_id), None)


def add(source, text, config):
    """Name the WAV, commit its metadata, then remove files beyond the newest ten."""
    with _lock, OUTPUT_LOCK:
        source = source_path(source)
        created = datetime.now().astimezone()
        record_id = uuid.uuid4().hex
        title = _title(text)
        filename = f"{created:%Y%m%d-%H%M%S}_{title}_{record_id[:8]}.wav"
        destination = OUTPUT_DIR / filename
        source.rename(destination)
        record = {
            "id": record_id,
            "filename": filename,
            "label": f"{created:%m-%d %H:%M} · {title} · {record_id[:4]}",
            "created_at": created.isoformat(timespec="seconds"),
            "text": text,
            "config": config,
        }
        try:
            records = [record, *_read()][:MAX_RESULTS]
            _write(records)
        except Exception:
            destination.rename(source)
            raise
        # This directory contains only TTS results and their derived speed files.
        active = [Path(item["filename"]).stem for item in records]
        for path in OUTPUT_DIR.glob("*.wav"):
            if not any(path.stem == stem or path.stem.startswith(stem + "-speed-") for stem in active):
                path.unlink(missing_ok=True)
        return record, _choices(records), str(destination)
