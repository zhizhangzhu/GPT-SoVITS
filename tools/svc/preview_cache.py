"""Owned, regenerable audition files; dataset revisions are never cache entries."""
import hashlib
import json
import math
from pathlib import Path
import re
import time

from filelock import FileLock

from . import storage as db
from .audio import crop
from tools.workspace_resources import require_space

LEGACY = re.compile(r"[a-f0-9]{32}(?:-context|-r\d+-[a-f0-9]{10}-context|-[a-f0-9]{10}-(?:start|end)-\d+\.\d{4})\.wav")


def read_index(directory):
    try:
        if (directory / "index.json").exists():
            result = db.read(directory / "index.json")
            if isinstance(result, dict):
                return result
    except (OSError, ValueError):
        pass
    return {path.name: {} for path in directory.glob("*.wav") if re.fullmatch(r"[a-f0-9]{64}\.wav", path.name)}


def cache_lock(job_id):
    directory = db.job_dir(job_id)
    if not (directory / "job.json").is_file():
        raise ValueError("请先选择有效任务")
    return FileLock(str(directory / ".preview-cache.lock"), timeout=30)


def excerpt(job_id, source, start, end):
    source = Path(source).resolve()
    stat = source.stat()
    key = hashlib.sha256(json.dumps([str(source), stat.st_size, stat.st_mtime_ns, start, end, 1]).encode()).hexdigest()
    with cache_lock(job_id):
        directory = db.job_dir(job_id) / "preview-cache"
        if directory.is_symlink():
            raise ValueError("试听缓存目录不能为符号链接")
        directory.mkdir(exist_ok=True)
        target = directory / (key + ".wav")
        if target.is_symlink():
            raise ValueError("试听缓存路径异常，请移除符号链接后重试")
        index_path = directory / "index.json"
        index = read_index(directory)
        if not target.exists():
            require_space(directory, purpose="生成切点试听")
            index[target.name] = {"accessed_at": time.time(), "bytes": 0}
            db.write(index_path, index)
            temporary = directory / (key + "-" + db.new_id() + ".tmp.wav")
            try:
                crop(source, temporary, start, end)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        target.touch()
        index[target.name] = {"accessed_at": time.time(), "bytes": target.stat().st_size}
        db.write(index_path, index)
    return str(target)


def entries(job_id):
    directory = db.job_dir(job_id)
    cache = directory / "preview-cache"
    if cache.is_symlink():
        raise ValueError("试听缓存目录不能为符号链接")
    index = read_index(cache)
    result = []
    for name in index:
        if not re.fullmatch(r"[a-f0-9]{64}\.wav", name):
            continue
        path = cache / name
        if path.is_file() and not path.is_symlink():
            result.append(path)
    for path in cache.glob("*.tmp.wav"):
        if re.fullmatch(r"[a-f0-9]{64}-[a-f0-9]{32}\.tmp\.wav", path.name) and path.is_file() and not path.is_symlink():
            result.append(path)
    protected_data = {Path(row["file"]).resolve() for row in db.read(directory / "segments.json", [])}
    for entry in db.read(directory / "edits.json", []):
        protected_data.update(Path(row["file"]).resolve() for row in entry["before"] + entry["after"])
    # Earlier versions put these explicitly named derived snippets beside clips.
    for path in (directory / "clips").glob("*.wav"):
        if LEGACY.fullmatch(path.name) and path.is_file() and not path.is_symlink() and path.resolve() not in protected_data:
            result.append(path)
    return [path for path in result if path.resolve() not in protected_data]


def summary(job_id):
    if not job_id:
        return "选择任务后显示试听缓存大小。"
    with cache_lock(job_id):
        files = entries(job_id)
        size = sum(path.stat().st_size for path in files)
    return f"当前任务试听缓存：**{len(files)} 个文件 · {size / 1024 ** 2:.2f} MiB**。原始音频、切片修订和训练文件不计入缓存。"


def clean(job_id, days=7, protected=None):
    days = float(days)
    if not math.isfinite(days) or not 0 <= days <= 3650:
        raise ValueError("缓存保留天数须为 0–3650；0 表示清理全部未在当前试听中使用的缓存")
    protected = {Path(path).resolve() for path in (protected or []) if path}
    cutoff = time.time() - days * 86400
    count, size = 0, 0
    with cache_lock(job_id):
        for path in entries(job_id):
            stat = path.stat()
            if path.resolve() in protected or stat.st_mtime > cutoff:
                continue
            size += stat.st_size
            path.unlink()
            count += 1
        index_path = db.job_dir(job_id) / "preview-cache/index.json"
        index = read_index(index_path.parent)
        index = {name: value for name, value in index.items()
                 if re.fullmatch(r"[a-f0-9]{64}\.wav", name) and (index_path.parent / name).is_file()}
        if index_path.exists():
            db.write(index_path, index)
    return f"已清理 {count} 个试听缓存，释放 {size / 1024 ** 2:.2f} MiB；当前试听与切片修订保留。"
