"""Immutable audition revisions and a validated undo stack for geometry edits."""
import hashlib
import json
from pathlib import Path
import re
import shutil

from . import storage as db
from tools.workspace_resources import require_space

GEOMETRY = {"bounds", "split", "merge", "recommend"}
ACTION_NAMES = {"bounds": "修改边界", "split": "拆分", "merge": "合并", "recommend": "采用推荐", "undo": "撤销"}


def next_revision(directory, row):
    revision = int(row.get("revision", 0))
    pattern = re.compile(re.escape(row["id"]) + r"-r(\d+)\.wav")
    for path in (Path(directory) / "clips").glob(row["id"] + "-r*.wav"):
        match = pattern.fullmatch(path.name)
        if match:
            revision = max(revision, int(match[1]))
    return revision + 1


def key(entry):
    return entry.get("id") or hashlib.sha256(json.dumps(entry, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def edits(job_id):
    return db.read(db.job_dir(job_id) / "edits.json", [])


def undo_candidate(job_id):
    history = edits(job_id)
    undone = {entry.get("undo_of") for entry in history if entry["action"] == "undo"}
    return next((entry for entry in reversed(history) if entry["action"] in GEOMETRY and key(entry) not in undone), None)


def undo(job_id):
    from . import service
    from . import feedback
    with db.lock():
        directory = db.job_dir(job_id)
        job = db.read(directory / "job.json")
        if job["kind"] != "dataset" or service.live(job):
            raise ValueError("请等待数据集任务结束后撤销修改")
        entry = undo_candidate(job_id)
        if entry is None:
            raise ValueError("没有可撤销的边界修改")
        rows = service.records(job_id)
        current = {row["id"]: row for row in rows}
        after = entry["after"]
        if any(saved["id"] not in current or any(current[saved["id"]].get(field) != saved.get(field)
               for field in ["sha256", "start", "end", "source_id"]) for saved in after):
            raise ValueError("当前片段与修改记录不一致，无法覆盖；请检查最近修订")
        before_ids, after_ids = {row["id"] for row in entry["before"]}, {row["id"] for row in after}
        if (before_ids - after_ids) & set(current):
            raise ValueError("恢复的片段编号已经存在，请检查修改记录")
        originals = []
        for old in entry["before"]:
            path = Path(old["file"]).resolve()
            if path.parent != (directory / "clips").resolve() or not path.is_file() or db.digest(path) != old["sha256"]:
                raise ValueError("历史修订音频不存在或已被修改，无法撤销")
            originals.append(path)
        require_space(directory, sum(path.stat().st_size for path in originals), "恢复历史修订")
        restored = []
        for old, path in zip(entry["before"], originals):
            row = dict(old, revision=next_revision(directory, old), review="pending")
            target = directory / "clips" / f"{row['id']}-r{row['revision']}.wav"
            shutil.copy2(path, target)
            row.update(file=str(target), sha256=db.digest(target))
            row.pop("feedback_id", None)
            restored.append(row)
        first = min(index for index, row in enumerate(rows) if row["id"] in after_ids)
        position = sum(row["id"] not in after_ids for row in rows[:first])
        updated = [row for row in rows if row["id"] not in after_ids]
        updated[position:position] = restored
        service.refresh_boundaries(job, updated, restored)
        db.write(directory / "segments.json", updated)
        invalid = before_ids | after_ids
        manifest = [row for row in db.clips(job["singer_id"]) if row["id"] not in invalid]
        db.write_manifest(job["singer_id"], manifest)
        for row in after:
            feedback.invalidate(job_id, current[row["id"]].get("feedback_id"), "undone")
        feedback.invalidate_edit(job_id, key(entry))
        history = edits(job_id)
        history.append({"id": db.new_id(), "at": db.now(), "action": "undo", "undo_of": key(entry),
                        "before": [current[row["id"]] for row in after], "after": restored})
        db.write(directory / "edits.json", history)
        return entry.get("selected_clip") if entry.get("selected_clip") in before_ids else restored[0]["id"]


def progress(job_id):
    if not job_id:
        return {"total": 0, "approved": 0, "rejected": 0, "pending": 0}
    rows = db.read(db.job_dir(job_id) / "segments.json", [])
    return dict(total=len(rows), **{name: sum(row["review"] == name for row in rows)
                                   for name in ["approved", "rejected", "pending"]})


def remember(job_id, clip_id):
    with db.lock():
        db.write(db.job_dir(job_id) / "review_cursor.json", {"clip_id": clip_id, "at": db.now()})


def cursor(job_id):
    return db.read(db.job_dir(job_id) / "review_cursor.json", {}).get("clip_id") if job_id else None
