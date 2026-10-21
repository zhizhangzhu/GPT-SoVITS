"""User-saved reference audio and prompt settings for TTS inference."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import soundfile as sf

from .service import ROOT

LIBRARY = ROOT / "ref_audios" / "snapshots"
IDENTIFIER = re.compile(r"[0-9a-f]{32}\Z")
MAX_BYTES = 512 * 1024 * 1024


def _directory(snapshot_id):
    if not IDENTIFIER.fullmatch(str(snapshot_id or "")):
        raise ValueError("请选择有效的参考音色快照")
    directory = LIBRARY / snapshot_id
    if directory.is_symlink():
        raise ValueError("参考音色快照目录无效")
    return directory


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source(path):
    path = Path(path or "")
    if not path.is_file() or path.is_symlink():
        raise ValueError("参考音频文件不存在，请重新上传")
    try:
        info = sf.info(path)
    except (OSError, RuntimeError) as error:
        raise ValueError("参考音频无法读取，请换用 WAV、FLAC 等音频文件") from error
    if info.frames <= 0:
        raise ValueError("参考音频为空")
    return path, info


def _read(directory):
    data = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("快照元数据无效")
    entries = [data.get("reference"), *(data.get("auxiliary") if isinstance(data.get("auxiliary"), list) else [])]
    if (data.get("id") != directory.name or not isinstance(data.get("auxiliary"), list)
            or not all(isinstance(data.get(key), str) for key in ("name", "created_at", "prompt", "language", "gpt", "sovits"))
            or not isinstance(data.get("ref_free"), bool)
            or not all(isinstance(item, dict) and isinstance(item.get("file"), str)
                       and isinstance(item.get("sha256"), str) and isinstance(item.get("bytes"), int)
                       for item in entries)):
        raise ValueError("快照元数据无效")
    return data


def list_snapshots():
    if not LIBRARY.exists():
        return []
    rows = []
    for directory in LIBRARY.iterdir():
        if not directory.is_dir() or directory.is_symlink() or not IDENTIFIER.fullmatch(directory.name):
            continue
        try:
            rows.append(_read(directory))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(rows, key=lambda row: row["created_at"], reverse=True)


def choices():
    return [(f"{row['name']} · {row['created_at'][:10]} · {row['id'][:6]}", row["id"]) for row in list_snapshots()]


def local_reference_choices():
    folder = LIBRARY.parent
    if not folder.exists():
        return []
    result = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
            result.append((path.name, path.name))
    return result


def local_reference(filename):
    if Path(str(filename or "")).name != filename or filename not in {value for _, value in local_reference_choices()}:
        raise ValueError("请选择本机已有的参考音频")
    path, info = _source(LIBRARY.parent / filename)
    if not 3 <= info.duration <= 10:
        raise ValueError(f"参考音频为 {info.duration:.2f} 秒，请先裁剪为 3–10 秒")
    return str(path)


def save(name, reference, prompt, language, auxiliary, ref_free, gpt, sovits):
    name = str(name or "").strip()
    if not 1 <= len(name) <= 80:
        raise ValueError("请填写 1–80 字的快照名称")
    if not gpt or not sovits:
        raise ValueError("请先选择 GPT 和 SoVITS 模型")
    source, info = _source(reference)
    if not 3 <= info.duration <= 10:
        raise ValueError(f"参考音频为 {info.duration:.2f} 秒，请先裁剪为 3–10 秒")
    if not ref_free and not str(prompt or "").strip():
        raise ValueError("请填写参考音频实际说出的文字")
    auxiliary = [_source(path)[0] for path in (auxiliary or [])]
    total = source.stat().st_size + sum(path.stat().st_size for path in auxiliary)
    if total > MAX_BYTES:
        raise ValueError("快照音频合计超过 512 MiB，请缩短辅助参考音频")
    snapshot_id = uuid.uuid4().hex
    LIBRARY.mkdir(parents=True, exist_ok=True)
    temporary = LIBRARY / (snapshot_id + ".part")
    temporary.mkdir()
    try:
        def copy_audio(path, stem):
            suffix = path.suffix.lower() if path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"} else ".audio"
            filename = stem + suffix
            destination = temporary / filename
            shutil.copy2(path, destination)
            return {"file": filename, "sha256": _digest(destination), "bytes": destination.stat().st_size}

        primary = copy_audio(source, "reference")
        extra = [copy_audio(path, f"aux-{index:02d}") for index, path in enumerate(auxiliary)]
        data = {"id": snapshot_id, "name": name, "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "reference": primary, "auxiliary": extra, "prompt": str(prompt or "").strip(),
                "language": str(language), "ref_free": bool(ref_free), "gpt": str(gpt), "sovits": str(sovits)}
        (temporary / "snapshot.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.rename(LIBRARY / snapshot_id)
        return data
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load(snapshot_id):
    directory = _directory(snapshot_id)
    data = _read(directory)
    for entry in [data["reference"], *data["auxiliary"]]:
        filename = entry["file"]
        if Path(filename).name != filename:
            raise ValueError("快照音频文件名无效")
        path = directory / filename
        if not path.is_file() or path.is_symlink() or path.stat().st_size != entry["bytes"] or _digest(path) != entry["sha256"]:
            raise ValueError("快照音频丢失或已改变，请重新保存")
    return data, str(directory / data["reference"]["file"]), [str(directory / item["file"]) for item in data["auxiliary"]]


def delete(snapshot_id):
    directory = _directory(snapshot_id)
    _read(directory)
    shutil.rmtree(directory)
