"""Resumable HTTP range downloads with upstream SHA256 verification."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import os
import time
import threading

import requests


def download(repo, filename, target, endpoint="https://huggingface.co", workers=16, revision=None,
             check_cancel=None, progress=None):
    if check_cancel:
        check_cancel()
    api_path = f"{endpoint}/api/models/{repo}" + (f"/revision/{revision}" if revision else "")
    metadata = requests.get(api_path, params={"blobs": "true"}, timeout=60)
    metadata.raise_for_status()
    info = metadata.json()
    entry = next(row for row in info["siblings"] if row["rfilename"] == filename)
    lfs = entry.get("lfs") or {}
    expected, size = lfs.get("sha256"), lfs.get("size", entry.get("size"))
    if not expected or not size:
        raise ValueError("模型没有可校验的 LFS SHA256，拒绝自动下载")
    target = Path(target)
    if target.exists():
        with target.open("rb") as stream:
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                if check_cancel:
                    check_cancel()
                h.update(block)
        if h.hexdigest() == expected:
            return {"revision": info["sha"], "sha256": expected}
    parts = target.parent / ("." + target.name + "." + expected[:12] + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    url = f"{endpoint}/{repo}/resolve/{info['sha']}/{filename}"
    width = 4 * 1024 * 1024
    count = (size + width - 1) // width
    clients = threading.local()
    sessions = []

    def part(index):
        if check_cancel:
            check_cancel()
        begin, end = index * width, min(size, (index + 1) * width) - 1
        path = parts / str(index)
        if path.exists() and path.stat().st_size == end - begin + 1:
            return path
        if not hasattr(clients, "session"):
            clients.session = requests.Session()
            sessions.append(clients.session)
        for attempt in range(4):
            temporary = path.with_suffix(".tmp")
            try:
                with clients.session.get(url, headers={"Range": f"bytes={begin}-{end}"}, stream=True, timeout=(30, 60)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{end}/{size}":
                        raise ValueError("下载服务返回了错误的分段范围")
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(256 * 1024):
                            if check_cancel:
                                check_cancel()
                            stream.write(block)
                if temporary.stat().st_size != end - begin + 1:
                    raise ValueError("下载分段不完整")
                os.replace(temporary, path)
                return path
            except Exception:
                temporary.unlink(missing_ok=True)
                if check_cancel:
                    check_cancel()
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(part, index) for index in range(count)]
            for done, future in enumerate(as_completed(futures), 1):
                future.result()
                if progress:
                    progress(min(done * width, size), size)
                if done % 8 == 0 or done == count:
                    print(f"{target.name}：{done}/{count} 块", flush=True)
    finally:
        for session in sessions:
            session.close()
    temporary = target.with_suffix(target.suffix + ".tmp")
    checksum = hashlib.sha256()
    with temporary.open("wb") as stream:
        for index in range(count):
            with (parts / str(index)).open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    if check_cancel:
                        check_cancel()
                    checksum.update(block)
                    stream.write(block)
    if checksum.hexdigest() != expected:
        temporary.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(parts)
        raise ValueError("下载模型 SHA256 不一致，请重新下载")
    os.replace(temporary, target)
    import shutil
    shutil.rmtree(parts)
    return {"revision": info["sha"], "sha256": expected}
