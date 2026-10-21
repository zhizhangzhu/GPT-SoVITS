"""Persistent records. IDs, never display names, determine filesystem paths."""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from datetime import datetime, timezone

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GPT_SOVITS_SVC_DATA", str(ROOT / "svc_data"))).expanduser().resolve()
EXPERIMENTS = Path(os.environ.get("GPT_SOVITS_SVC_EXPERIMENTS", str(ROOT / "svc_experiments"))).expanduser().resolve()
VERSION = "svc-workspace-1"


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier(value):
    if not re.fullmatch(r"[a-f0-9]{32}", str(value or "")):
        raise ValueError("无效的 SVC 记录编号")
    return str(value)


def new_id():
    return uuid.uuid4().hex


def read(path, default=None):
    if not Path(path).exists() and default is not None:
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + new_id() + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def digest(path, check_cancel=None, progress=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        processed = 0
        if progress:
            progress(0, before.st_size)
        while True:
            if check_cancel:
                check_cancel()
            block = stream.read(1024 * 1024)
            if not block:
                break
            h.update(block)
            processed += len(block)
            if progress:
                progress(processed, before.st_size)
        after = os.fstat(stream.fileno())
        current = Path(path).stat()
        stamp = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if stamp(before) != stamp(after) or stamp(before) != stamp(current) or processed != before.st_size:
            raise ValueError("校验过程中文件发生变化，请重新处理")
    return h.hexdigest()


def lock():
    DATA.mkdir(parents=True, exist_ok=True)
    return FileLock(str(DATA / ".records.lock"), timeout=30)


def singers():
    return read(DATA / "singers.json", [])


def add_singer(name):
    name = str(name or "").strip()
    if not name or len(name) > 100:
        raise ValueError("请填写 1–100 字的歌手名称")
    with lock():
        records = singers()
        # A new identity is deliberate, even when display names are identical.
        singer = {"id": new_id(), "name": name, "created_at": now()}
        records.append(singer)
        write(DATA / "singers.json", records)
    return singer


def singer(value):
    value = identifier(value)
    for item in singers():
        if item["id"] == value:
            return item
    raise ValueError("歌手不存在，请先创建或选择歌手")


def rename_singer(singer_id, name):
    singer(singer_id)
    name = str(name or "").strip()
    if not 1 <= len(name) <= 100:
        raise ValueError("请填写 1–100 字的歌手名称")
    with lock():
        records = singers()
        for row in records:
            if row["id"] == singer_id:
                row["name"] = name
        write(DATA / "singers.json", records)


def write_manifest(singer_id, rows):
    directory = DATA / "datasets" / identifier(singer_id)
    write(directory / "manifest.json", rows)
    temporary = directory / ("manifest.jsonl." + new_id() + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, directory / "manifest.jsonl")


def job_dir(value):
    return DATA / "jobs" / identifier(value)


def source_dir(singer_id, source_id):
    return DATA / "sources" / identifier(singer_id) / identifier(source_id)


def experiment_dir(value):
    return EXPERIMENTS / identifier(value)


def jobs():
    result = []
    for path in (DATA / "jobs").glob("*/job.json"):
        try:
            item = read(path)
            if not isinstance(item, dict) or not {"id", "kind", "status", "created_at", "options"} <= item.keys():
                continue
            identifier(item["id"])
            if item["kind"] not in {"dataset", "train", "convert", "report", "import_model", "pitch", "reverb"} or not isinstance(item["options"], dict):
                continue
            result.append(item)
        except (OSError, ValueError):
            continue
    return sorted(result, key=lambda item: item["created_at"], reverse=True)


def clips(singer_id):
    singer(singer_id)
    return read(DATA / "datasets" / singer_id / "manifest.json", [])


def update_job(job_id, **changes):
    with lock():
        path = job_dir(job_id) / "job.json"
        record = read(path)
        record.update(changes, updated_at=now())
        write(path, record)
    return record


def source_audio(row):
    """Resolve the precise preparation track; old rows retain their old input."""
    if not row.get("preparation_job"):
        return source_dir(row["singer_id"], row["source_id"]) / "vocals.wav"
    directory = job_dir(row["preparation_job"]) / "source-stems" / identifier(row["source_id"])
    name = row.get("source_audio", "vocals.wav")
    if name not in {"vocals.wav", "lead_vocals.wav", "backing_vocals.wav",
                    "dry_lead.wav", "dry_backing.wav", "dry_vocals.wav"}:
        raise ValueError("素材音轨路径无效")
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("素材音轨路径超出准备目录")
    return path


def source_stream(row):
    return (row["source_id"], row.get("preparation_job"), row.get("source_audio", "vocals.wav"))
