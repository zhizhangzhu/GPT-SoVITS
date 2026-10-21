"""UI-facing commands, immutable snapshots, background task lifecycle."""
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import psutil
import soundfile as sf
from filelock import FileLock

from . import storage as db
from .audio import crop, parse_lrc
from .review import analyze_boundaries, annotate_source, neighbors
from . import feedback
from . import validation
from . import separation
from .history import next_revision
from .install import ASSETS, ENV, RUNTIME, COMMIT, ENCODER_FILE, ENCODER_NAME
from tools.workspace_resources import require_space, disk_summary

PROCESSES = {}
SPAWN_LOCK = threading.Lock()
TERMINAL = {"completed", "awaiting_review", "failed", "cancelled"}


def backend_status():
    missing = [str(p.relative_to(db.ROOT)) for p in [ENV / "bin/python", RUNTIME / "models.py",
               ASSETS / ENCODER_FILE, ASSETS / "G_0.pth", ASSETS / "D_0.pth"] if not p.exists()]
    return "后端待安装：" + "、".join(missing) if missing else "SVC 后端已安装 · 默认底模 CC-BY-NC-4.0（非商业）"


def resource_status():
    return backend_status() + "\n\n素材与任务：" + disk_summary(db.DATA) + " · 训练实验：" + disk_summary(db.EXPERIMENTS)


def generate_dataset_report(singer_id, predictor="dio"):
    db.singer(singer_id)
    if not db.clips(singer_id):
        raise ValueError("请先试听审核并入库片段，再生成训练前报告")
    if predictor not in {"dio", "harvest", "pm", "rmvpe", "fcpe"}:
        raise ValueError("请选择有效的训练 F0 提取方式")
    python = ENV / "bin/python"
    if not python.exists():
        raise ValueError("请先安装 SVC 环境")
    directory = db.DATA / "datasets" / singer_id / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(GPT_SOVITS_SVC_DATA=str(db.DATA), GPT_SOVITS_SVC_EXPERIMENTS=str(db.EXPERIMENTS),
                       PYTHONPATH=str(db.ROOT), OMP_NUM_THREADS="2", MPLCONFIGDIR=str(directory / ".matplotlib"))
    with (directory / "processing.log").open("ab") as log:
        try:
            result = subprocess.run([str(python), "-u", "-m", "tools.svc.dataset_report", singer_id,
                                     "--predictor", predictor], cwd=db.ROOT, env=environment,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, timeout=900)
        except subprocess.TimeoutExpired as error:
            raise ValueError("数据报告分析超过 15 分钟，已停止；已完成的片段分析缓存保留") from error
    if result.returncode:
        raise ValueError("训练前报告生成失败，请查看数据报告日志")
    from .dataset_report import load
    report, _, stale = load(singer_id)
    if report is None or stale:
        raise ValueError("分析过程中入库数据发生变化，请重新生成报告")
    return report


def create_report(singer_id, predictor="dio"):
    db.singer(singer_id)
    if not db.clips(singer_id):
        raise ValueError("请先试听审核并入库片段，再生成训练前报告")
    if predictor not in {"dio", "harvest", "pm", "rmvpe", "fcpe"}:
        raise ValueError("请选择有效的训练 F0 提取方式")
    return submit_report(singer_id, predictor)


def submit_report(singer_id, predictor, retry_id=None):
    directory = db.DATA / "datasets" / db.identifier(singer_id) / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    # Serialize lookup, reservation and spawn across threads and WebUI processes.
    with FileLock(str(directory / ".submission.lock"), timeout=30):
        for record in history():
            if record["kind"] == "report" and record["singer_id"] == singer_id and record["status"] not in TERMINAL and live(record):
                if record["options"].get("predictor") == predictor:
                    db.write(directory / "report-job.json", {"id": record["id"]})
                    return record["id"]
                raise ValueError("当前歌手已有数据报告正在分析；完成或停止后再切换 F0 工具")
        if retry_id is None:
            previous = report_task(singer_id)
            if previous and previous["status"] == "completed" and previous["options"].get("predictor") == predictor:
                from .dataset_report import load
                report, _, stale = load(singer_id)
                if report and not stale and report["predictor"] == predictor:
                    return previous["id"]
        record = db.read(db.job_dir(retry_id) / "job.json") if retry_id else new_job("report", singer_id, predictor=predictor)
        try:
            task = start(record["id"])
        except Exception as error:
            db.update_job(record["id"], status="failed", message=f"报告启动失败：{error}")
            raise
        db.write(directory / "report-job.json", {"id": task})
        return task


def report_task(singer_id):
    if not singer_id:
        return None
    pointer = db.read(db.DATA / "datasets" / db.identifier(singer_id) / "analysis/report-job.json", {})
    if pointer:
        return recover_job(db.read(db.job_dir(pointer["id"]) / "job.json"))
    return next((record for record in history() if record["kind"] == "report" and record.get("singer_id") == singer_id), None)


def live(job):
    try:
        process = psutil.Process(int(job.get("pid", 0)))
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE and job["id"] in " ".join(process.cmdline()) and abs(process.create_time() - job.get("process_time", 0)) < 2
    except (psutil.Error, ValueError):
        return False


def recover_job(record):
    """Refresh one task without scanning every historical dataset on each tick."""
    with SPAWN_LOCK:
        process = PROCESSES.get(record["id"])
        if process is not None and process.poll() is not None:
            PROCESSES.pop(record["id"], None)
    if record["status"] not in TERMINAL and record.get("pid") and not live(record):
        with db.lock():
            path = db.job_dir(record["id"]) / "job.json"
            record = db.read(path)
            if record["status"] not in TERMINAL and record.get("pid") and not live(record):
                record.update(status="failed", message="后台进程已退出，可重试或续训", updated_at=db.now())
                db.write(path, record)
    return record


def history():
    return [recover_job(record) for record in db.jobs()]


def busy():
    return any(record["kind"] != "report" and record["status"] not in TERMINAL and live(record) for record in history())


def new_job(kind, singer_id=None, **options):
    if kind not in {"dataset", "train", "convert", "report", "import_model", "pitch", "reverb"}:
        raise ValueError("未知任务类型")
    if kind in {"dataset", "train", "convert", "pitch", "reverb"}:
        device = str(options.get("gpu", "0"))
        if not device.isdigit():
            raise ValueError("GPU 编号须为一个非负整数")
        from tools.gpu_utils import subprocess_gpu_index
        subprocess_gpu_index(device)
    record = {"id": db.new_id(), "kind": kind, "singer_id": singer_id, "status": "queued", "progress": 0,
              "message": "等待处理", "created_at": db.now(), "options": options, "version": db.VERSION}
    db.write(db.job_dir(record["id"]) / "job.json", record)
    return record


def start(job_id):
    with SPAWN_LOCK, db.lock():
        directory = db.job_dir(job_id)
        record = db.read(directory / "job.json")
        if live(record):
            return job_id
        from .audio_edit import active as editing
        if editing(job_id):
            raise ValueError("正在保存此任务的音轨，请先等待或停止编辑后再重试")
        python = ENV / "bin/python"
        if not python.exists():
            raise ValueError("请先运行 python -m tools.svc.install 安装 SVC 环境")
        require_space(directory, purpose="启动 SVC 任务")
        if record["kind"] == "train":
            require_space(db.experiment_dir(record["options"]["experiment_id"]), purpose="启动 SVC 训练")
        (directory / "cancel").unlink(missing_ok=True)
        gpu_task = record["kind"] in {"dataset", "train", "convert", "pitch", "reverb"}
        record.update(status="queued", message="排队等待 GPU" if gpu_task else "准备校验模型" if record["kind"] == "import_model" else "准备分析数据报告",
                      progress=0, attempt=record.get("attempt", 0) + 1)
        record.pop("outputs", None)
        record.pop("audio_edit", None)
        # Regeneration starts a new editing baseline. Older audio files remain
        # on disk; active editors were excluded above.
        (directory / "audio-edit-base.json").unlink(missing_ok=True)
        (directory / "audio-edit.json").unlink(missing_ok=True)
        db.write(directory / "job.json", record)
        environment = os.environ.copy()
        environment["GPT_SOVITS_SVC_DATA"] = str(db.DATA)
        environment["GPT_SOVITS_SVC_EXPERIMENTS"] = str(db.EXPERIMENTS)
        if gpu_task:
            from tools.gpu_utils import subprocess_gpu_index
            device = str(record["options"].get("gpu", "0"))
            if not device.isdigit():
                raise ValueError("GPU 编号须为一个非负整数")
            environment["CUDA_VISIBLE_DEVICES"] = subprocess_gpu_index(device)
        else:
            environment["CUDA_VISIBLE_DEVICES"] = ""
            environment["MPLCONFIGDIR"] = str(directory / ".matplotlib")
            environment["GPT_SOVITS_SVC_DATA"] = str(db.DATA)
            environment["GPT_SOVITS_SVC_EXPERIMENTS"] = str(db.EXPERIMENTS)
        environment["PATH"] = str(Path(sys.base_prefix) / "bin") + os.pathsep + environment.get("PATH", "")
        environment["PYTHONPATH"] = str(db.ROOT)
        environment["OMP_NUM_THREADS"] = "4"
        with (directory / "processing.log").open("ab") as log:
            process = subprocess.Popen([str(python), "-u", "-m", "tools.svc.worker", job_id], cwd=db.ROOT,
                                       stdout=log, stderr=log, env=environment, start_new_session=True)
        record.update(pid=process.pid, process_time=psutil.Process(process.pid).create_time())
        db.write(directory / "job.json", record)
        PROCESSES[job_id] = process
    return job_id


def cancel(job_id):
    from . import audio_edit
    if audio_edit.active(job_id):
        return audio_edit.cancel(job_id)
    record = db.read(db.job_dir(job_id) / "job.json")
    if record["status"] in TERMINAL:
        return "任务已结束，无需停止"
    if not live(record):
        return "当前任务未运行"
    (db.job_dir(job_id) / "cancel").touch()
    return "已请求停止；当前处理块或训练步结束后停止，训练会保存可续训状态"


def create_dataset(singer_id, files, lyric_files=None, lyric_text="", mixed=False, offset=0, minimum=5, maximum=15, gpu="0", quality_settings=None,
                   separation_model=separation.VOCAL_DEFAULT, split_vocals=False, include_backing=False,
                   vocal_split_model=separation.KARAOKE_DEFAULT, dereverb_lead=False, dereverb_backing=False):
    db.singer(singer_id)
    separation.validate(separation_model, vocal_split_model)
    if (include_backing or dereverb_backing) and not split_vocals:
        raise ValueError("处理和声前请开启主唱 / 和声分离")
    if not files:
        raise ValueError("请上传至少一份音频")
    if not math.isfinite(float(offset)) or not 1 <= float(minimum) < float(maximum) <= 30:
        raise ValueError("切片范围须满足 1 ≤ 最短 < 最长 ≤ 30 秒")
    quality_settings = quality_settings or {"silence_db": -45, "max_silence_ratio": .5, "max_clipping_ratio": .001}
    if not -80 <= quality_settings["silence_db"] <= -20 or not 0 <= quality_settings["max_silence_ratio"] <= 1 or not 0 <= quality_settings["max_clipping_ratio"] <= 1:
        raise ValueError("质检阈值超出可用范围")
    if lyric_text.strip() and len(files) > 1:
        raise ValueError("多份音频请使用同名 LRC 文件配对；粘贴歌词仅适用于单份音频")
    lrc_map = {Path(path).stem.casefold(): Path(path).read_text(encoding="utf-8-sig") for path in (lyric_files or [])}
    unmatched = set(lrc_map) - {Path(path).stem.casefold() for path in files}
    if unmatched:
        raise ValueError("歌词文件没有同名音频：" + "、".join(sorted(unmatched)))
    if any(not Path(path).is_file() for path in files):
        raise ValueError("上传音频不存在，请重新上传")
    require_space(db.DATA, sum(Path(path).stat().st_size for path in files), "保存歌手素材")
    sources, duplicate_jobs = [], []
    with db.lock():
        index_path = db.DATA / "sources" / singer_id / "index.json"
        index = db.read(index_path, {})
        for path in files:
            path = Path(path)
            checksum = db.digest(path)
            if checksum in index:
                duplicate_jobs.append(index[checksum]["job_id"])
                continue
            source_id = db.new_id()
            directory = db.source_dir(singer_id, source_id)
            directory.mkdir(parents=True, exist_ok=True)
            original = directory / ("original" + path.suffix.lower())
            shutil.copy2(path, original)
            lyrics = lrc_map.get(path.stem.casefold(), lyric_text)
            (directory / "lyrics.lrc").write_text(lyrics, encoding="utf-8")
            source = {"id": source_id, "singer_id": singer_id, "name": path.name, "original": str(original),
                      "sha256": checksum, "lyrics_sha256": db.digest(directory / "lyrics.lrc"), "created_at": db.now()}
            db.write(directory / "source.json", source)
            sources.append(source)
            index[checksum] = {"source_id": source_id, "job_id": None}
        if not sources:
            if duplicate_jobs:
                return duplicate_jobs[0], "素材已存在，打开已有任务；若需重新处理请使用重试"
            raise ValueError("没有新素材")
        record = new_job("dataset", singer_id, sources=[s["id"] for s in sources], mixed=bool(mixed), offset=float(offset),
                         minimum=float(minimum), maximum=float(maximum), gpu=str(gpu), duplicate_count=len(duplicate_jobs),
                         quality_settings=quality_settings, boundary_profile=feedback.profile(singer_id),
                         separation_model=separation_model, vocal_split_model=vocal_split_model,
                         split_vocals=bool(split_vocals), include_backing=bool(include_backing),
                         dereverb_lead=bool(dereverb_lead), dereverb_backing=bool(dereverb_backing))
        for source in sources:
            index[source["sha256"]]["job_id"] = record["id"]
        db.write(index_path, index)
    return start(record["id"]), f"已提交 {len(sources)} 份素材，跳过 {len(duplicate_jobs)} 份重复素材"


def records(job_id):
    return db.read(db.job_dir(job_id) / "segments.json", [])


def find_clip(job_id, clip_id):
    rows = records(job_id)
    for index, row in enumerate(rows):
        if row["id"] == clip_id:
            return rows, index, row
    raise ValueError("片段不存在，请重新选择")


def recut(row, start, end, directory):
    revised = dict(row, start=float(start), end=float(end), revision=next_revision(directory, row), review="pending")
    revised.pop("feedback_id", None)
    path = directory / "clips" / f"{row['id']}-r{revised['revision']}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    source = db.source_audio(row)
    measurements = crop(source, path, start, end, row.get("quality_settings"))
    revised.update(measurements, file=str(path), sha256=db.digest(path))
    revised["alignment_flags"] = [flag for flag in revised.get("alignment_flags", [])
                                   if flag != "持续发声段达到最大长度，切点需要试听"]
    return revised


def boundary_plan(job_id, clip_id, existing_rows=None):
    rows = existing_rows if existing_rows is not None else records(job_id)
    row = next((item for item in rows if item["id"] == clip_id), None)
    if row is None:
        raise ValueError("片段不存在，请重新选择")
    related = [item for item in rows if db.source_stream(item) == db.source_stream(row)]
    previous, following = neighbors(related)[clip_id]
    source = db.source_audio(row)
    info = source.stat()
    basis = [{key: item.get(key) for key in ["id", "revision", "sha256", "start", "end", "review", "quality_settings"]}
             for item in [previous, row, following] if item is not None]
    settings = feedback.profile(row["singer_id"])
    def context(item):
        return {key: item.get(key) for key in ["id", "singer_id", "source_id", "start", "end", "quality_settings", "source_track", "preparation_job", "source_audio"]} if item else None
    return {"job_id": job_id, "clip_id": clip_id, "basis": basis,
            "source_stat": [info.st_size, info.st_mtime_ns],
            "profile": settings, "context": {"row": context(row), "previous": context(previous), "following": context(following)},
            "analysis": analyze_boundaries(source, row, previous, following, profile=settings)}


def refresh_boundaries(job, rows, changed):
    for stream in {db.source_stream(item) for item in changed}:
        related = [item for item in rows if db.source_stream(item) == stream]
        if not related:
            continue
        source = db.source_audio(related[0])
        adjacent = neighbors(related)
        affected = {item["id"] for item in changed if db.source_stream(item) == stream}
        for item in changed:
            if item["id"] in adjacent:
                affected.update(neighbor["id"] for neighbor in adjacent[item["id"]] if neighbor)
        annotate_source(related, source, only=affected, profile=feedback.profile(job["singer_id"]))
        lyric_path = db.source_dir(job["singer_id"], stream[0]) / "lyrics.lrc"
        lyrics = parse_lrc(lyric_path.read_text(encoding="utf-8")) if lyric_path.exists() else []
        offset = float(job.get("options", {}).get("offset", 0))
        if lyrics:
            for item in changed:
                if db.source_stream(item) == stream:
                    item["lyrics"] = " / ".join(line["text"] for line in lyrics
                                                if line["text"] and item["start"] <= line["time"] + offset < item["end"])


def decline_recommendation(job_id, clip_id, recommendation, reason=""):
    with db.lock():
        job = db.read(db.job_dir(job_id) / "job.json")
        if live(job):
            raise ValueError("请等待任务结束后记录反馈")
        rows, _, row = find_clip(job_id, clip_id)
        plan = boundary_plan(job_id, clip_id, rows)
        if not recommendation or any(recommendation.get(key) != plan[key] for key in ["job_id", "clip_id", "basis", "source_stat", "profile"]):
            raise ValueError("边界建议已变化，请重新选择片段")
        if not any(cut["suggested"] is not None for cut in plan["analysis"]["cuts"].values()):
            raise ValueError("当前没有建议可记录为不采用")
        feedback.invalidate(job_id, row.get("feedback_id"))
        row["feedback_id"] = feedback.record(job_id, plan, "declined", {key: row[key] for key in ["start", "end"]}, reason=reason)
        if row["review"] == "approved":
            feedback.confirm(job_id, row, "approved")
        db.write(db.job_dir(job_id) / "segments.json", rows)


def recheck_job(job_id):
    with db.lock():
        job = db.read(db.job_dir(job_id) / "job.json")
        if job["kind"] != "dataset" or live(job):
            raise ValueError("请等待数据集任务结束后重新检查边界")
        rows = records(job_id)
        refresh_boundaries(job, rows, rows)
        db.write(db.job_dir(job_id) / "segments.json", rows)
        job["options"]["boundary_profile"] = feedback.profile(job["singer_id"])
        db.write(db.job_dir(job_id) / "job.json", job)


def edit_clip(job_id, clip_id, action, start=None, end=None, point=None, recommendation=None):
    with db.lock():
        job = db.read(db.job_dir(job_id) / "job.json")
        if live(job):
            raise ValueError("任务正在处理，请结束后编辑")
        rows, index, row = find_clip(job_id, clip_id)
        edit_id = db.new_id()
        feedback_plan = boundary_plan(job_id, clip_id, rows) if action in {"bounds", "recommend"} else None
        old_rows = [dict(row)]
        replacements = []
        if action == "recommend":
            plan = feedback_plan
            if not recommendation or any(recommendation.get(key) != plan[key] for key in ["job_id", "clip_id", "basis", "source_stat", "profile"]):
                raise ValueError("片段或相邻边界已变化，请重新选择片段后查看最新推荐")
            analysis = plan["analysis"]
            a = analysis["cuts"]["start"]["suggested"]
            b = analysis["cuts"]["end"]["suggested"]
            if a is None and b is None:
                raise ValueError("当前没有可应用的推荐切点，请试听后手动修正")
            a = row["start"] if a is None else a
            b = row["end"] if b is None else b
            changes = {clip_id: (a, b)}
            if analysis["shared_start"] and a != row["start"]:
                previous = next(item for item in rows if item["id"] == analysis["shared_start"])
                changes[previous["id"]] = (previous["start"], a)
            if analysis["shared_end"] and b != row["end"]:
                following = next(item for item in rows if item["id"] == analysis["shared_end"])
                changes[following["id"]] = (b, following["end"])
            if any(end - start < .4 for start, end in changes.values()):
                raise ValueError("推荐边界会使片段过短，请手动调整")
            old_rows = [dict(item) for item in rows if item["id"] in changes]
            replacements = [recut(item, *changes[item["id"]], db.job_dir(job_id)) for item in old_rows]
            revised_by_id = {item["id"]: item for item in replacements}
            rows = [revised_by_id.get(item["id"], item) for item in rows]
        elif action == "bounds":
            replacements = [recut(row, start, end, db.job_dir(job_id))]
        elif action == "split":
            if not row["start"] + .4 < float(point) < row["end"] - .4:
                raise ValueError("拆分点须位于片段内部，并为两侧各保留至少 0.4 秒")
            replacements = [recut(dict(row, id=db.new_id()), a, b, db.job_dir(job_id))
                            for a, b in [(row["start"], float(point)), (float(point), row["end"])]]
        elif action == "merge":
            if index + 1 >= len(rows):
                raise ValueError("没有下一片段")
            following = rows[index + 1]
            if db.source_stream(following) != db.source_stream(row) or following["start"] < row["end"] - .1 or following["start"] - row["end"] > 2:
                raise ValueError("只能合并同一素材、同一音轨时间相邻的两个片段（间隔不超过 2 秒）")
            old_rows.append(dict(following))
            merged = dict(row, lyrics=" / ".join(filter(None, [row.get("lyrics"), following.get("lyrics")])) )
            replacements = [recut(merged, row["start"], following["end"], db.job_dir(job_id))]
        elif action in {"approved", "rejected", "pending"}:
            if action == "approved" and row["quality"] == "rejected":
                raise ValueError("近乎静音或过短的片段不可入库，请先修正边界重新计算质量")
            replacements = [dict(row, review=action)]
        else:
            raise ValueError("未知审核操作")
        if action != "recommend":
            rows[index:index + len(old_rows)] = replacements
        if action in {"bounds", "split", "merge", "recommend"}:
            refresh_boundaries(job, rows, replacements)
            for old in old_rows:
                feedback.invalidate(job_id, old.get("feedback_id"))
            if feedback_plan is not None:
                selected = next(item for item in replacements if item["id"] == clip_id)
                selected["feedback_id"] = feedback.record(job_id, feedback_plan, "recommend" if action == "recommend" else "manual",
                                                         {key: selected[key] for key in ["start", "end"]}, edit_id=edit_id)
        elif action in {"approved", "rejected", "pending"}:
            feedback.confirm(job_id, replacements[0], action)
        db.write(db.job_dir(job_id) / "segments.json", rows)
        # Remove edited/rejected clips immediately from future training manifests.
        manifest_path = db.DATA / "datasets" / row["singer_id"] / "manifest.json"
        invalid = {old["id"] for old in old_rows}
        manifest = [clip for clip in db.read(manifest_path, []) if clip["id"] not in invalid]
        db.write_manifest(row["singer_id"], manifest)
        history_path = db.job_dir(job_id) / "edits.json"
        history_rows = db.read(history_path, [])
        history_rows.append({"id": edit_id, "at": db.now(), "action": action, "selected_clip": clip_id, "before": old_rows, "after": replacements})
        db.write(history_path, history_rows)
    return clip_id if action == "recommend" else replacements[0]["id"]


def publish(job_id, approve_passed=False):
    with db.lock():
        job = db.read(db.job_dir(job_id) / "job.json")
        if job["kind"] != "dataset" or live(job):
            raise ValueError("请选择已经完成处理的数据集任务")
        rows = records(job_id)
        if approve_passed:
            for row in rows:
                if row["quality"] == "passed" and not row.get("alignment_flags") and row["review"] == "pending":
                    row["review"] = "approved"
        directory = db.DATA / "datasets" / job["singer_id"]
        (directory / "clips").mkdir(parents=True, exist_ok=True)
        manifest = {row["id"]: row for row in db.read(directory / "manifest.json", [])}
        count = 0
        for row in rows:
            manifest.pop(row["id"], None)
            if row["review"] != "approved":
                continue
            target = directory / "clips" / f"{row['source_id']}_{row['id']}_{row['sha256'][:12]}.wav"
            if not target.exists():
                shutil.copy2(row["file"], target)
            manifest[row["id"]] = dict(row, file=str(target), job_id=job_id)
            count += 1
        db.write_manifest(job["singer_id"], list(manifest.values()))
        db.write(db.job_dir(job_id) / "segments.json", rows)
    return f"当前任务入库 {count} 个片段；歌手共 {len(manifest)} 个已审核片段"


def review_location(singer_id, clip_id, preferred_job=None):
    """Find the original review task for a published clip, including old manifests."""
    db.singer(singer_id)
    db.identifier(clip_id)
    candidates = [row["id"] for row in db.jobs() if row["kind"] == "dataset" and row.get("singer_id") == singer_id]
    if preferred_job in candidates:
        candidates.remove(preferred_job)
        candidates.insert(0, preferred_job)
    for job_id in candidates:
        if any(row["id"] == clip_id for row in records(job_id)):
            return job_id
    raise ValueError("找不到该片段的原审核任务；可从任务历史手动查找")


def validation_plan(singer_id, selected_sources=None):
    return validation.plan(db.clips(singer_id), selected_sources)


def create_experiment(singer_id, name, steps=1000, batch=2, save_every=100, gpu="0", predictor="dio", generator=None, discriminator=None,
                      validation_source_ids=None):
    identity = db.singer(singer_id)
    if not 1 <= int(steps) <= 10000000 or not 1 <= int(batch) <= 64 or int(save_every) < 1:
        raise ValueError("训练步数、批大小和保存频率须为正整数（batch ≤ 64）")
    generator = Path(generator or ASSETS / "G_0.pth").resolve()
    discriminator = Path(discriminator or ASSETS / "D_0.pth").resolve()
    if not generator.is_file() or not discriminator.is_file():
        raise ValueError("SVC 预训练 G/D 不齐全，请完成安装或上传兼容权重")
    with db.lock():
        rows = db.clips(singer_id)
        if len(rows) < 2:
            raise ValueError("训练前至少需要两个已审核入库的片段；建议先积累多首歌曲")
        split = validation.plan(rows, validation_source_ids)
        snapshot_size = sum(Path(row["file"]).stat().st_size for row in rows)
        weights_size = generator.stat().st_size + discriminator.stat().st_size
        require_space(db.EXPERIMENTS, snapshot_size * 8 + weights_size * 4, "训练快照、特征与首组检查点")
        experiment_id = db.new_id()
        directory = db.experiment_dir(experiment_id)
        speaker_dir = directory / "features" / singer_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        snapshot = []
        for row in rows:
            if db.digest(row["file"]) != row["sha256"]:
                raise ValueError("数据集文件已在外部修改，请重新审核入库")
            target = speaker_dir / f"{row['id']}.wav"
            shutil.copy2(row["file"], target)
            snapshot.append(dict(row, snapshot_file=str(target)))
        training_ids = {row["id"] for row in split["training"]}
        validation_ids = {row["id"] for row in split["validation"]}
        train = [row["snapshot_file"] for row in snapshot if row["id"] in training_ids]
        val = [row["snapshot_file"] for row in snapshot if row["id"] in validation_ids]
        warning = split["warning"]
        (directory / "filelists").mkdir()
        for label, paths in [("train", train), ("val", val)]:
            (directory / "filelists" / f"{label}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        config = db.read(RUNTIME / "configs_template/config_template.json")
        config["spk"] = {singer_id: 0}
        config["model"].update(speech_encoder=ENCODER_NAME, n_speakers=1, ssl_dim=256, gin_channels=256)
        config["train"].update(batch_size=int(batch), max_steps=int(steps), eval_interval=int(save_every), fp16_run=False)
        config["data"].update(training_files=str(directory / "filelists/train.txt"), validation_files=str(directory / "filelists/val.txt"))
        db.write(directory / "config.json", config)
        db.write(directory / "dataset_snapshot.json", {"clips": snapshot, "warning": warning,
                 "validation_source_ids": split["selected_sources"], "train_seconds": split["train_seconds"],
                 "validation_seconds": split["validation_seconds"], "created_at": db.now(), "backend_commit": COMMIT,
                 "feature_key": {"encoder": ENCODER_NAME, "f0": predictor, "sampling_rate": 44100, "version": db.VERSION},
                 "sample_order": "shuffled-per-epoch-v2"})
        (directory / "initial").mkdir()
        for source, label in [(generator, "G_0.pth"), (discriminator, "D_0.pth")]:
            shutil.copy2(source, directory / "initial" / label)
        db.write(directory / "experiment.json", {"id": experiment_id, "name": str(name or identity["name"] + " SVC")[:100],
                 "singer_id": singer_id, "created_at": db.now(), "warning": warning, "status": "created",
                 "initial_weights": [{"path": str(generator), "sha256": db.digest(generator)}, {"path": str(discriminator), "sha256": db.digest(discriminator)}]})
        job = new_job("train", singer_id, experiment_id=experiment_id, gpu=str(gpu), predictor=predictor)
    return start(job["id"])


def experiments():
    result = []
    for path in db.EXPERIMENTS.glob("*/experiment.json"):
        try:
            item = db.read(path)
            if not isinstance(item, dict) or not {"id", "name", "created_at", "singer_id"} <= item.keys():
                raise ValueError("实验记录缺少必要字段")
            db.identifier(item["id"])
            if item["id"] != path.parent.name:
                raise ValueError("实验编号与目录不一致")
            result.append(item)
        except (OSError, ValueError) as error:
            logging.getLogger(__name__).warning("跳过无法读取的 SVC 实验 %s: %s", path, error)
    return sorted(result, key=lambda item: item["created_at"], reverse=True)


def checkpoints(experiment_id):
    if not experiment_id:
        return []
    directory = db.experiment_dir(experiment_id) / "checkpoints"
    latest = db.read(directory / "latest.json", {"step": 0})
    index = db.read(directory / "checkpoint-index.json", {})
    result = []
    for generator in directory.glob("G_*.pth"):
        try:
            step = int(generator.stem.split("_")[-1])
        except ValueError:
            continue
        if step <= 0 or step > int(latest.get("step", 0)):
            continue
        group = {key: directory / f"{prefix}_{step}.pth" for key, prefix in
                 [("generator", "G"), ("discriminator", "D"), ("state", "state")]}
        saved = index.get(str(step))
        if any(not path.is_file() or path.stat().st_size <= 0 for path in group.values()):
            continue
        if saved and (not isinstance(saved, dict) or any(
                not isinstance(saved.get(key), dict) or saved[key].get("name") != path.name or
                saved[key].get("bytes") != path.stat().st_size for key, path in group.items())):
            continue
        result.append(generator)
    return sorted(result, key=lambda path: int(path.stem.split("_")[-1]))


def delete_checkpoint(experiment_id, checkpoint):
    directory = db.experiment_dir(experiment_id) / "checkpoints"
    selected = Path(checkpoint or "").resolve()
    with db.lock():
        available = {path.resolve() for path in checkpoints(experiment_id)}
        if selected not in available:
            raise ValueError("请选择当前实验中完整的旧检查点")
        step = int(selected.stem.split("_")[-1])
        latest = db.read(directory / "latest.json", {"step": 0})
        if step >= int(latest["step"]):
            raise ValueError("最新检查点用于续训，不能清理")
        for job in db.jobs():
            if live(job) and (job.get("options", {}).get("experiment_id") == experiment_id
                              or job.get("options", {}).get("checkpoint") == str(selected)):
                raise ValueError("该实验仍有运行任务，请结束后再清理检查点")
        files = [directory / f"{prefix}_{step}.pth" for prefix in ("G", "D", "state")]
        previews = list((directory.parent / "previews").glob(f"step-{step}-*.wav"))
        legacy = directory.parent / "previews" / f"step-{step}.wav"
        if legacy.exists():
            previews.append(legacy)
        reclaimed = sum(path.stat().st_size for path in files + previews if path.exists())
        for path in files + previews:
            path.unlink(missing_ok=True)
        index_path = directory / "checkpoint-index.json"
        index = db.read(index_path, {})
        if str(step) in index:
            index.pop(str(step))
            db.write(index_path, index)
    return f"已删除第 {step} 步检查点及其试听文件，释放约 {reclaimed / 1024 ** 2:.1f} MiB"


def verify_checkpoint(experiment_id, checkpoint, check_cancel=None, progress=None):
    path = Path(checkpoint).resolve()
    if path not in {candidate.resolve() for candidate in checkpoints(experiment_id)}:
        raise ValueError("检查点不完整或不属于当前实验，请刷新列表")
    directory = db.experiment_dir(experiment_id) / "checkpoints"
    step = int(path.stem.split("_")[-1])
    saved = db.read(directory / "checkpoint-index.json", {}).get(str(step))
    if saved:
        total = sum(saved[key]["bytes"] for key in ("generator", "discriminator", "state"))
        processed = 0
        for key, prefix in (("generator", "G"), ("discriminator", "D"), ("state", "state")):
            file = directory / f"{prefix}_{step}.pth"
            callback = (lambda read, size: progress(processed + read, total, file.name)) if progress else None
            if db.digest(file, check_cancel=check_cancel, progress=callback) != saved[key].get("sha256"):
                raise ValueError(f"第 {step} 步检查点校验失败，请选择其他检查点或续训")
            processed += saved[key]["bytes"]
    return path


def submit_model_import(candidate):
    from .external_models import source_for
    source_for(candidate)
    job = new_job("import_model", candidate=str(candidate))
    return start(job["id"])


def _queue_conversion(singer_id, model_options, audio, mixed, transpose, predictor, chunk, gpu,
                      vocal_gain, instrumental_gain, accompaniment=None, cleanup="none",
                      vocal_mode="off", harmony_transpose=0, harmony_gain=1, retry_from=None,
                      separation_model=separation.VOCAL_DEFAULT, vocal_split_model=separation.KARAOKE_DEFAULT,
                      dereverb_lead=False, dereverb_backing=False, output_reverb=None):
    from .reverb import settings as reverb_settings
    output_reverb = reverb_settings(output_reverb)
    separation.validate(separation_model, vocal_split_model)
    if dereverb_backing and vocal_mode == "off":
        raise ValueError("和声去混响需要先开启主唱 / 和声分离")
    if not audio or not Path(audio).is_file():
        raise ValueError("请上传待转换的人声音频")
    if accompaniment and (not Path(accompaniment).is_file() or Path(accompaniment).resolve() == Path(audio).resolve()):
        raise ValueError("请分别上传人声和对应的伴奏文件")
    if not 2 <= float(chunk) <= 60 or not -36 <= float(transpose) <= 36:
        raise ValueError("分段长度须为 2–60 秒，音高偏移须为 -36 至 36 半音")
    if predictor not in {"dio", "harvest", "pm", "rmvpe", "fcpe"}:
        raise ValueError("请选择有效的 F0 提取方式")
    if not 0 <= float(vocal_gain) <= 2 or not 0 <= float(instrumental_gain) <= 2:
        raise ValueError("人声与伴奏增益须为 0–2")
    size = Path(audio).stat().st_size + (Path(accompaniment).stat().st_size if accompaniment else 0)
    require_space(db.DATA, size * 2, "保存转换素材")
    if cleanup not in {"none", "denoise", "separate"}:
        raise ValueError("请选择有效的人声清理方式")
    if mixed and not accompaniment and cleanup == "none":
        cleanup = "separate"
    if accompaniment and cleanup != "none":
        raise ValueError("已上传独立伴奏时不需要额外降噪或分离")
    if vocal_mode not in {"off", "keep", "convert", "remove"}:
        raise ValueError("请选择有效的主唱和声处理方式")
    if not 0 <= float(harmony_gain) <= 2 or not -24 <= float(harmony_transpose) <= 24:
        raise ValueError("和声增益须为 0–2，和声额外音高偏移须为 -24 至 24 半音")
    if vocal_mode == "convert" and not -36 <= float(transpose) + float(harmony_transpose) <= 36:
        raise ValueError("主唱偏移加和声额外偏移须在 -36 至 36 半音之间")
    job = new_job("convert", singer_id, **model_options, mixed=bool(mixed or accompaniment), cleanup=cleanup,
                  separate=bool(mixed and not accompaniment), transpose=float(transpose), predictor=predictor,
                  chunk=float(chunk), gpu=str(gpu), vocal_gain=float(vocal_gain),
                  instrumental_gain=float(instrumental_gain), vocal_mode=vocal_mode,
                  harmony_transpose=float(harmony_transpose), harmony_gain=float(harmony_gain),
                  separation_model=separation_model, vocal_split_model=vocal_split_model,
                  dereverb_lead=bool(dereverb_lead), dereverb_backing=bool(dereverb_backing),output_reverb=output_reverb)
    directory = db.job_dir(job["id"])
    source = directory / ("original" + Path(audio).suffix.lower())
    shutil.copy2(audio, source)
    from .presentation import describe
    previous=db.read(db.job_dir(retry_from)/'job.json',{}) if retry_from else {}
    names=describe(previous) if previous else {}
    changes = {"input": str(source),"source_name":names.get('source_name') or (Path(audio).name if not retry_from else ''),
               "model_name":describe(job)['model_name']}
    if retry_from:
        changes["retried_from"] = db.identifier(retry_from)
    if accompaniment:
        backing = directory / ("original_accompaniment" + Path(accompaniment).suffix.lower())
        shutil.copy2(accompaniment, backing)
        changes["accompaniment_input"] = str(backing)
    db.update_job(job["id"], **changes)
    return start(job["id"])


def create_conversion(experiment_id, checkpoint, audio, mixed=False, transpose=0, predictor="dio", chunk=15, gpu="0", vocal_gain=1, instrumental_gain=1, accompaniment=None, cleanup="none",
                      vocal_mode="off", harmony_transpose=0, harmony_gain=1,
                      separation_model=separation.VOCAL_DEFAULT, vocal_split_model=separation.KARAOKE_DEFAULT,
                      dereverb_lead=False, dereverb_backing=False, output_reverb=None):
    if not experiment_id:
        raise ValueError("请先选择当前 SVC 实验及检查点")
    directory = db.experiment_dir(experiment_id)
    experiment = db.read(directory / "experiment.json")
    checkpoint = Path(checkpoint or "").resolve()
    if checkpoint not in {candidate.resolve() for candidate in checkpoints(experiment_id)}:
        raise ValueError("请选择该 SVC 实验的完整检查点")
    return _queue_conversion(experiment["singer_id"], {"experiment_id": experiment_id, "checkpoint": str(checkpoint)},
                             audio, mixed, transpose, predictor, chunk, gpu, vocal_gain, instrumental_gain, accompaniment,
                             cleanup, vocal_mode, harmony_transpose, harmony_gain,
                             separation_model=separation_model, vocal_split_model=vocal_split_model,
                  dereverb_lead=bool(dereverb_lead), dereverb_backing=bool(dereverb_backing),output_reverb=output_reverb)


def create_imported_conversion(model_id, speaker_name, audio, mixed=False, transpose=0, predictor="dio", chunk=15,
                               gpu="0", vocal_gain=1, instrumental_gain=1, accompaniment=None,
                               shallow_diffusion=False, diffusion_steps=35, cleanup="none",
                               vocal_mode="off", harmony_transpose=0, harmony_gain=1,
                               separation_model=separation.VOCAL_DEFAULT, vocal_split_model=separation.KARAOKE_DEFAULT,
                      dereverb_lead=False, dereverb_backing=False, output_reverb=None):
    from .external_models import record, LIBRARY
    item = record(model_id)
    if speaker_name not in item["speakers"]:
        raise ValueError("请选择导入模型中存在的目标音色")
    if shallow_diffusion and not item.get("diffusion"):
        raise ValueError("当前音色模型未导入浅扩散文件")
    maximum = int(item.get("diffusion_max_steps") or 100)
    if shallow_diffusion and (not isinstance(diffusion_steps, (int, float)) or
                              not float(diffusion_steps).is_integer() or not 1 <= diffusion_steps <= maximum):
        raise ValueError(f"浅扩散步数须为 1–{maximum} 的整数")
    directory = LIBRARY / item["id"]
    options = {"model_source": "imported", "model_id": item["id"],
               "checkpoint": str(directory / "G.pth"), "checkpoint_sha256": item["sha256"],
               "config_sha256": item["config_sha256"], "speaker_id": item["speakers"][speaker_name],
               "speaker_name": speaker_name, "shallow_diffusion": bool(shallow_diffusion),
               "diffusion_steps": int(diffusion_steps) if shallow_diffusion else None,
               "diffusion_files": item.get("diffusion") if shallow_diffusion else None}
    return _queue_conversion(None, options, audio, mixed, transpose, predictor, chunk, gpu,
                             vocal_gain, instrumental_gain, accompaniment, cleanup,
                             vocal_mode, harmony_transpose, harmony_gain,
                             separation_model=separation_model, vocal_split_model=vocal_split_model,
                  dereverb_lead=bool(dereverb_lead), dereverb_backing=bool(dereverb_backing),output_reverb=output_reverb)


def retry(job_id, extra_steps=0):
    record = db.read(db.job_dir(job_id) / "job.json")
    if record["kind"] == "convert":
        options = record["options"]
        return retry_conversion(job_id, options["transpose"], options["predictor"], options["chunk"],
                                options["gpu"], options["vocal_gain"], options["instrumental_gain"])
    if live(record):
        if record["status"] not in TERMINAL:
            raise ValueError("任务仍在运行")
        try:
            # The worker writes its final status before releasing the GPU lock
            # and exiting. Let that cleanup finish before reusing this job dir.
            psutil.Process(record["pid"]).wait(timeout=30)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired as error:
            raise ValueError("任务已结束，后台仍在释放资源；请稍后重试") from error
    if record["kind"] == "report":
        return submit_report(record["singer_id"], record["options"]["predictor"], retry_id=job_id)
    if record["kind"] == "train" and int(extra_steps) > 0:
        directory = db.experiment_dir(record["options"]["experiment_id"])
        config = db.read(directory / "config.json")
        state = db.read(directory / "checkpoints/latest.json", {"step": 0})
        config["train"]["max_steps"] = int(state["step"]) + int(extra_steps)
        db.write(directory / "config.json", config)
    return start(job_id)


def retry_conversion(job_id, transpose, predictor, chunk, gpu, vocal_gain, instrumental_gain,
                     vocal_mode=None, harmony_transpose=None, harmony_gain=None,
                     separation_model=None, vocal_split_model=None, dereverb_lead=None, dereverb_backing=None, output_reverb=None):
    """Create a separate conversion attempt so the previous output stays available."""
    job_id = db.identifier(job_id)
    record = db.read(db.job_dir(job_id) / "job.json")
    if record["kind"] != "convert":
        raise ValueError("所选任务不是音色转换任务")
    if record["status"] not in TERMINAL:
        raise ValueError("请等待当前转换任务结束后再重试")
    options = record["options"]
    source = Path(record.get("input") or "").resolve()
    directory = db.job_dir(job_id).resolve()
    if source.parent != directory or not source.is_file():
        raise ValueError("原始音频已丢失，无法重新转换")
    accompaniment = None
    if record.get("accompaniment_input"):
        accompaniment = Path(record["accompaniment_input"]).resolve()
        if accompaniment.parent != directory or not accompaniment.is_file():
            raise ValueError("原始伴奏已丢失，无法重新转换")
    conversion_keys = {"mixed", "separate", "cleanup", "transpose", "predictor", "chunk", "gpu",
                       "vocal_gain", "instrumental_gain", "vocal_mode", "harmony_transpose", "harmony_gain",
                       "separation_model", "vocal_split_model", "dereverb_lead", "dereverb_backing", "output_reverb"}
    model_options = {key: value for key, value in options.items() if key not in conversion_keys}
    new_task = _queue_conversion(record.get("singer_id"), model_options, str(source), options["separate"],
                                 transpose, predictor, chunk, gpu, vocal_gain, instrumental_gain,
                                 str(accompaniment) if accompaniment else None, options.get("cleanup", "none"),
                                 options.get("vocal_mode", "off") if vocal_mode is None else vocal_mode,
                                 options.get("harmony_transpose", 0) if harmony_transpose is None else harmony_transpose,
                                 options.get("harmony_gain", 1) if harmony_gain is None else harmony_gain,
                                 retry_from=job_id,
                                 separation_model=options.get("separation_model", "hp2") if separation_model is None else separation_model,
                                 vocal_split_model=options.get("vocal_split_model", "mel_karaoke") if vocal_split_model is None else vocal_split_model,
                                 dereverb_lead=options.get("dereverb_lead", False) if dereverb_lead is None else dereverb_lead,
                                 dereverb_backing=options.get("dereverb_backing", False) if dereverb_backing is None else dereverb_backing,
                                 output_reverb=options.get("output_reverb") if output_reverb is None else output_reverb)
    return new_task


def delete_job(job_id):
    """Delete a terminal task without removing shared training or dataset data."""
    job_id = db.identifier(job_id)
    with SPAWN_LOCK, db.lock():
        directory = db.job_dir(job_id)
        record = db.read(directory / "job.json")
        from .audio_edit import active as editing
        if editing(job_id):
            raise ValueError("音轨编辑仍在保存，请等待完成或停止后再删除")
        if record["status"] not in TERMINAL or live(record):
            raise ValueError("请等待任务结束后再删除")
        if any(j["kind"] in {"pitch", "reverb"} and
               (j["options"].get("source_job") == job_id or job_id in j["options"].get("dependencies", []))
               and j["status"] not in TERMINAL for j in db.jobs()):
            raise ValueError("该任务的音轨正在用于音准或混响处理，请等待处理结束后再删除")
        if record["kind"] == "dataset":
            raise ValueError("数据整理任务含有审核片段，不能从任务中心直接删除")
        if record["kind"] == "report":
            pointer = db.DATA / "datasets" / db.identifier(record["singer_id"]) / "analysis/report-job.json"
            if db.read(pointer, {}).get("id") == job_id:
                pointer.unlink(missing_ok=True)
        shutil.rmtree(directory)


def snapshot(job_id):
    if not job_id:
        return {}, ""
    directory = db.job_dir(job_id)
    record = recover_job(db.read(directory / "job.json"))
    logfile = directory / "processing.log"
    if logfile.exists():
        with logfile.open("rb") as stream:
            stream.seek(max(0, logfile.stat().st_size - 16000))
            log = stream.read().decode("utf-8", errors="replace")
    else:
        log = ""
    return record, log


def conversion_outputs(record, source_id=None):
    """Expose committed intermediate stems, including pre-feature conversion tasks."""
    outputs = dict(record.get("outputs", {}))
    if record.get("kind") == "dataset":
        directory = db.job_dir(record["id"])
        stems = db.read(directory / "dataset-stems.json", {})
        candidates = record.get("options", {}).get("sources", [])
        selected = source_id if source_id in candidates else (candidates[0] if candidates else None)
        return {key: value for key, value in stems.get(selected, {}).items() if Path(value).is_file()}
    if record.get("kind") != "convert":
        return outputs
    directory = db.job_dir(record["id"])
    def load(name):
        try:
            value = db.read(directory / name, {})
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}
    state = load("convert-stages.json")
    separated = state.get("separated") if isinstance(state.get("separated"), dict) else {}
    processed = state.get("processed") if isinstance(state.get("processed"), dict) else {}
    if state.get("mode") in {"vocals", "provided"} and state.get("decoded"):
        outputs.setdefault("source_vocals", str(directory / "decoded.wav"))
    elif separated.get("vocals") or processed.get("vocals"):
        outputs.setdefault("source_vocals", str(directory / "vocals.wav"))
    if separated.get("accompaniment") or separated.get("backing"):
        outputs.setdefault("accompaniment", str(directory / "accompaniment.wav"))
    split = load("vocal-split.json")
    if split.get("lead") and split.get("backing"):
        outputs.setdefault("lead_vocals", str(directory / "lead_vocals.wav"))
        outputs.setdefault("backing_vocals", str(directory / "backing_vocals.wav"))
    for key in ["vocals", "mix", "accompaniment", "source_vocals", "lead_vocals", "backing_vocals",
                "converted_lead", "converted_backing", "dry_lead", "dry_backing", "dry_vocals",
                "lead_reverb", "backing_reverb", "vocals_reverb"]:
        if outputs.get(key) and not Path(outputs[key]).is_file():
            outputs.pop(key)
    return outputs


def create_reverb(job_id, version="auto", style="reference", lead_strength=1., backing_strength=1.,
                  decay=1.5, predelay_ms=25., wet=.22, clean_output=False, gpu="0"):
    from .reverb import sources, VERSIONS, STYLES
    if version not in {value for _,value in VERSIONS} or style not in {value for _,value in STYLES}:
        raise ValueError("请选择有效的音轨版本与混响方式")
    values = {"lead_strength":(lead_strength,0,2), "backing_strength":(backing_strength,0,2),
              "decay":(decay,.3,4), "predelay_ms":(predelay_ms,0,120), "wet":(wet,0,.65)}
    for name,(value,low,high) in values.items():
        if not math.isfinite(float(value)) or not low <= float(value) <= high:
            raise ValueError("混响参数超出有效范围："+name)
    with db.lock():
        original = db.read(db.job_dir(job_id)/"job.json")
        if original["kind"] == "reverb":
            if original["status"] not in TERMINAL:
                raise ValueError("请等待当前混响任务结束后再调整参数")
            if version == "auto":
                version = original["options"]["version"]
            original = db.read(db.job_dir(original["options"]["source_job"])/"job.json", {})
            if not original:
                raise ValueError("原转换／校准任务已删除，无法重新添加混响")
        resolved = sources(original,version)
        if style == "reference" and any(not row.get("wet") and float(lead_strength if key=="lead" else backing_strength)>0
                                        for key,row in resolved["tracks"].items()):
            raise ValueError("原曲参考湿声已丢失，请选择手动设置混响")
        size = sum(sf.info(row["input"]).frames*8 for row in resolved["tracks"].values())
        require_space(db.DATA, additional=size*8, purpose="生成参考混响音轨")
        task = new_job("reverb",original.get("singer_id"),source_job=original["id"],
                       dependencies=resolved["dependencies"],version=resolved["version"],style=style,
                       **{key:float(value[0]) for key,value in values.items()},
                       clean_output=bool(clean_output),gpu=str(gpu))
        # Already holding the records lock: update_job() would acquire a new
        # FileLock instance on the same file and time out before start().
        from .presentation import describe
        task.update(target_name=(original.get("target_name") or original["id"][:8])+" · 混响",updated_at=db.now(),**describe(original))
        db.write(db.job_dir(task["id"])/"job.json",task)
    try:
        return start(task["id"])
    except Exception as error:
        db.update_job(task["id"],status="failed",message="混响任务启动失败："+str(error))
        raise


def create_pitch(job_id, predictor="rmvpe", backend="report", threshold=50, strength=1., gpu="0"):
    from .pitch_job import BACKENDS
    if predictor not in {"rmvpe", "fcpe"}:
        raise ValueError("参考音准分析请选择 RMVPE 或 FCPE")
    if backend not in {value for _,value in BACKENDS}:
        raise ValueError("请选择有效的校准方式")
    if not math.isfinite(float(threshold)) or not 20 <= float(threshold) <= 200:
        raise ValueError("触发校准的偏差须为 20–200 音分")
    if not math.isfinite(float(strength)) or not 0 <= float(strength) <= 1:
        raise ValueError("校准强度须为 0–1")
    with db.lock():
        previous = db.read(db.job_dir(job_id) / "job.json")
        if previous["kind"] == "pitch" and previous["options"].get("input_mode") == "upload":
            raise ValueError("独立上传校准任务请在左侧“一键校准”页面重新处理或重试")
        if previous["kind"] == "pitch":
            previous = db.read(db.job_dir(previous["options"]["source_job"]) / "job.json")
        if previous["kind"] != "convert" or previous["status"] != "completed":
            raise ValueError("请先选择一条已完成的音色转换任务")
        outputs = conversion_outputs(previous)
        if not outputs.get("vocals"):
            raise ValueError("原任务转换音轨已丢失")
        task = new_job("pitch", previous.get("singer_id"), source_job=previous["id"], predictor=predictor,
                       backend=backend, threshold=float(threshold), strength=float(strength), gpu=str(gpu))
    from .presentation import describe
    db.update_job(task['id'],**describe(previous))
    return start(task["id"])


def create_uploaded_pitch(reference, target, reference_mixed=False, target_mixed=False,
                          predictor="rmvpe", backend="psola", threshold=50, strength=1., gpu="0",
                          vocal_mode="keep", alignment="auto", reference_offset=0, reference_transpose=0,
                          dereverb_reference=False, dereverb_target=False,
                          separation_model=separation.VOCAL_DEFAULT, vocal_split_model=separation.KARAOKE_DEFAULT):
    """Persist both uploads; audio work runs only in the normal GPU worker."""
    from .pitch_job import BACKENDS
    if predictor not in {"rmvpe", "fcpe"} or backend not in {v for _,v in BACKENDS}:
        raise ValueError("请选择有效的音高分析和校准方式")
    if vocal_mode not in {"off", "keep", "convert"} or alignment not in {"auto", "timeline"}:
        raise ValueError("请选择有效的声部处理和对齐方式")
    separation.validate(separation_model,vocal_split_model)
    for value,low,high,label in [(threshold,20,200,"校准阈值"),(strength,0,1,"校准强度"),
                                (reference_offset,-1200,1200,"参考时间偏移"),(reference_transpose,-36,36,"参考音高偏移")]:
        if not math.isfinite(float(value)) or not low<=float(value)<=high:
            raise ValueError(f"{label}须为 {low}–{high}")
    for value,label in [(reference,"参考音频"),(target,"待校准音频")]:
        if not value or not Path(value).is_file():
            raise ValueError("请上传"+label)
    require_space(db.DATA,(Path(reference).stat().st_size+Path(target).stat().st_size)*2,"保存独立校准音频")
    task=new_job("pitch",input_mode="upload",predictor=predictor,backend=backend,
                 threshold=float(threshold),strength=float(strength),gpu=str(gpu),
                 reference_mixed=bool(reference_mixed),target_mixed=bool(target_mixed),vocal_mode=vocal_mode,
                 alignment=alignment,reference_offset=float(reference_offset),reference_transpose=float(reference_transpose),
                 dereverb_reference=bool(dereverb_reference),dereverb_target=bool(dereverb_target),
                 separation_model=separation_model,vocal_split_model=vocal_split_model)
    directory=db.job_dir(task['id'])
    changes={}
    try:
        for side,source in [('reference',reference),('target',target)]:
            source=Path(source)
            suffix=source.suffix.lower() if source.suffix else '.audio'
            destination=directory/(side+'_input'+suffix)
            shutil.copy2(source,destination)
            changes[side+'_input']=str(destination)
            changes[side+'_name']=source.name
        db.update_job(task['id'],**changes)
    except Exception:
        shutil.rmtree(directory)
        raise
    return start(task['id'])
