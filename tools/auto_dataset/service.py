"""Job storage and annotation validation; no GPU models are loaded here."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = {"zh", "en", "ja", "ko", "yue"}
TERMINAL = {"awaiting_review", "reviewed", "failed", "cancelled"}
PROCESSES = {}
PROCESS_LOCK = threading.Lock()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def job_dir(value):
    if not value:
        raise ValueError("请先创建或选择任务")
    path = Path(value).resolve()
    if not path.is_relative_to((ROOT / "output").resolve()) or not (path / "job.json").is_file():
        raise ValueError("无效的任务目录")
    return path


def create_job(files, speaker):
    speaker = speaker or ""
    if not speaker or len(speaker) > 80 or any(c in speaker for c in "|/\\\r\n\x00"):
        raise ValueError("请输入说话人名（最多80字，不含路径分隔符、竖线或换行）")
    speaker = speaker.strip()
    if not speaker:
        raise ValueError("请输入说话人名")
    if isinstance(files, (str, Path)):
        files = [files]
    if not files:
        raise ValueError("请先上传音频")
    slug = re.sub(r"[^\w-]", "_", speaker).strip("_.") or "voice"
    task_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    inputs = ROOT / "input" / slug / task_id
    outputs = ROOT / "output" / slug / task_id
    inputs.mkdir(parents=True)
    outputs.mkdir(parents=True)
    originals = []
    for i, filename in enumerate(files):
        src = Path(filename)
        if not src.is_file():
            raise ValueError("上传文件不存在")
        target = inputs / (f"{i:03d}" + src.suffix.lower())
        shutil.copy2(src, target)
        originals.append({"path": str(target), "name": src.name})
    atomic_json(outputs / "job.json", {
        "id": task_id, "speaker": speaker, "experiment": slug + "_" + task_id,
        "originals": originals, "status": "queued", "stage": "等待处理", "progress": 0,
        "message": "任务已创建", "completed": [], "created_at": time.time(),
    })
    return str(outputs)


def launch_job(value):
    path = job_dir(value)
    with PROCESS_LOCK:
        proc = PROCESSES.get(str(path))
        if proc and proc.poll() is None:
            raise ValueError("这个任务仍在处理中")
        job = read_json(path / "job.json")
        if worker_alive(job, path):
            raise ValueError("这个任务的后台进程仍在处理中，请查看当前任务")
        if job["status"] not in {"queued", "failed", "cancelled"}:
            raise ValueError("任务已完成，请审核结果或创建新任务")
        (path / "cancel").unlink(missing_ok=True)
        job.update(status="queued", stage="等待 GPU 队列", progress=0, message="正在排队", pid=None)
        atomic_json(path / "job.json", job)
        with (path / "worker.log").open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "tools.auto_dataset.worker", str(path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=(os.name != "nt"),
            )
        PROCESSES[str(path)] = proc
    return proc


def cancel_job(value):
    path = job_dir(value)
    job = read_json(path / "job.json")
    if job["status"] in TERMINAL:
        return "任务已经结束"
    (path / "cancel").touch()
    return "已请求停止；当前模型操作结束后停止，保留已完成的数据。"


def snapshot(value):
    path = job_dir(value)
    job = read_json(path / "job.json")
    proc = PROCESSES.get(str(path))
    if proc and proc.poll() is not None and job["status"] not in TERMINAL:
        job.update(status="failed", message="后台进程异常退出，请查看任务日志后重试")
        atomic_json(path / "job.json", job)
    elif not proc and job.get("pid") and job["status"] not in TERMINAL and not worker_alive(job, path):
        job.update(status="failed", message="后台进程已结束；已完成步骤保留，可重试")
        atomic_json(path / "job.json", job)
    records = read_json(path / "metadata.json") if (path / "metadata.json").exists() else []
    return job, records


def worker_alive(job, path):
    try:
        command = psutil.Process(job.get("pid") or -1).cmdline()
        return "tools.auto_dataset.worker" in command and str(path) in command
    except (psutil.Error, ValueError):
        return False


def write_annotations(path, records):
    lines = [f"{r['path']}|{r['speaker']}|{r['language']}|{r['text']}" for r in records]
    target = Path(path) / "annotations.list"
    temp = target.with_suffix(".list.tmp")
    temp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(temp, target)


def import_annotations(value):
    """Import edits/deletions/splits made in the existing annotation WebUI."""
    path = job_dir(value)
    job, old = snapshot(path)
    if job["status"] not in {"awaiting_review", "reviewed"}:
        raise ValueError("请等待自动处理完成")
    by_path = {str(Path(r["path"]).resolve()): r for r in old}
    rows = []
    seen = set()
    for line in (path / "annotations.list").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 4:
            raise ValueError("标注必须是：音频路径|说话人|语言|文字")
        filename, _, language, text = fields
        audio = Path(filename).resolve()
        if audio.parent != path / "clips" or not audio.is_file():
            raise ValueError("标注音频必须位于当前任务目录内")
        if str(audio) in seen:
            raise ValueError("标注含重复音频")
        seen.add(str(audio))
        row = dict(by_path.get(str(audio), {
            "id": hashlib.sha256(str(audio).encode()).hexdigest()[:16], "path": str(audio),
            "duration": 0, "flags": ["外部标注新增片段，请试听"], "selected": False,
        }))
        row.update(speaker=job["speaker"], text=text.strip(), language=language.lower())
        rows.append(row)
    # Deleted annotation rows stay in metadata but cannot enter train.list.
    for r in old:
        if str(Path(r["path"]).resolve()) not in seen:
            r = dict(r, selected=False)
            r["flags"] = list(dict.fromkeys(r.get("flags", []) + ["标注中已排除"]))
            rows.append(r)
    if rows != old:
        train = path / "train.list"
        if train.exists():
            train.rename(path / ("train.before-edit-" + uuid.uuid4().hex[:8] + ".list"))
        job.update(status="awaiting_review", stage="等待审核", message="标注修改已导入，请重新保存审核结果")
        atomic_json(path / "job.json", job)
    atomic_json(path / "metadata.json", rows)
    return job, rows


def approve(value, table):
    path = job_dir(value)
    job, records = snapshot(path)
    if job["status"] not in {"awaiting_review", "reviewed"}:
        raise ValueError("自动处理尚未完成")
    if hasattr(table, "values"):
        table = table.values.tolist()
    by_id = {r["id"]: dict(r) for r in records}
    seen = set()
    approved = []
    for enabled, row_id, language, text, *_ in table:
        if row_id not in by_id or row_id in seen:
            raise ValueError("审核表的片段编号无效或重复，请重新读取审核结果")
        seen.add(row_id)
        row = by_id[row_id]
        language, text = str(language).strip().lower(), str(text).strip()
        if any(c in text for c in "|\r\n\x00"):
            raise ValueError("标注文字不能含竖线或换行")
        selected = enabled is True or str(enabled).lower() == "true"
        if selected and (language not in LANGUAGES or not text):
            raise ValueError(f"片段 {row_id} 必须有文字和有效语言：zh/en/ja/ko/yue")
        audio = Path(row["path"]).resolve()
        if selected and (audio.parent != path / "clips" or not audio.is_file()):
            raise ValueError("审核音频不存在或位于任务目录外")
        row.update(text=text, language=language, selected=selected)
        if selected:
            approved.append(row)
    if seen != set(by_id):
        raise ValueError("审核表缺少片段；请用‘使用’列排除音频，不要删除整行")
    if not approved:
        raise ValueError("至少选择一条有效音频")
    lines = [f"{r['path']}|{job['speaker']}|{r['language']}|{r['text']}" for r in approved]
    train_path = path / "train.list"
    temp = train_path.with_suffix(".list.tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temp, train_path)
    atomic_json(path / "metadata.json", list(by_id.values()))
    # Keep excluded clips visible in the annotation editor.
    write_annotations(path, list(by_id.values()))
    job.update(status="reviewed", stage="审核已保存", message=f"已选入训练集 {len(approved)} 条")
    atomic_json(path / "job.json", job)
    return str(train_path), str(path / "clips"), job["experiment"], job["message"]
