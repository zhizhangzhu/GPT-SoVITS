"""Confirmed local boundary labels and previewable, singer-specific calibration."""
from collections import Counter
import hashlib
from itertools import product
import json

from . import storage as db
from .review import analyze_boundaries, DEFAULT_PROFILE, validate_profile

CONFIRMED = {"confirmed_recommend", "confirmed_manual", "confirmed_declined"}


def profile(singer_id):
    path = db.DATA / "boundary_profiles" / (db.identifier(singer_id) + ".json")
    return validate_profile(db.read(path, {}).get("settings", DEFAULT_PROFILE))


def rows(job_id):
    return db.read(db.job_dir(job_id) / "boundary_feedback.json", [])


def invalidate(job_id, sample_id, status="superseded"):
    if not sample_id:
        return
    records = rows(job_id)
    changed = False
    for row in records:
        if row["id"] == sample_id:
            row.update(status=status, updated_at=db.now())
            changed = True
    if changed:
        db.write(db.job_dir(job_id) / "boundary_feedback.json", records)


def invalidate_edit(job_id, edit_id):
    records = rows(job_id)
    for row in records:
        if row.get("edit_id") == edit_id:
            row.update(status="undone", updated_at=db.now())
    if records:
        db.write(db.job_dir(job_id) / "boundary_feedback.json", records)


def record(job_id, plan, kind, target, edit_id=None, reason=""):
    sample = {"id": db.new_id(), "job_id": job_id, "kind": kind, "status": kind + "_pending",
              "plan": plan, "target": target, "edit_id": edit_id, "reason": str(reason)[:300], "created_at": db.now()}
    records = rows(job_id)
    records.append(sample)
    db.write(db.job_dir(job_id) / "boundary_feedback.json", records)
    return sample["id"]


def confirm(job_id, row, decision):
    sample_id = row.get("feedback_id")
    if not sample_id:
        return
    records = rows(job_id)
    for sample in records:
        if sample["id"] != sample_id:
            continue
        if any(abs(sample["target"][key] - row[key]) > .0001 for key in ["start", "end"]):
            sample["status"] = "superseded"
        elif decision == "approved":
            sample["status"] = "confirmed_" + sample["kind"]
        elif decision == "rejected":
            sample["status"] = "clip_rejected"
        else:
            sample["status"] = sample["kind"] + "_pending"
        sample["updated_at"] = db.now()
    db.write(db.job_dir(job_id) / "boundary_feedback.json", records)


def singer_samples(singer_id):
    db.singer(singer_id)
    result = []
    for job in db.jobs():
        if job.get("singer_id") == singer_id and job["kind"] == "dataset":
            result.extend(rows(job["id"]))
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def usable_samples(singer_id):
    records = singer_samples(singer_id)
    unique, skipped = {}, 0
    for sample in sorted(records, key=lambda item: item["created_at"]):
        if sample["status"] not in CONFIRMED:
            continue
        context = sample["plan"]["context"]
        source = db.source_audio(context["row"])
        try:
            stat = source.stat()
        except OSError:
            skipped += 1
            continue
        if [stat.st_size, stat.st_mtime_ns] != sample["plan"]["source_stat"]:
            skipped += 1
            continue
        identity = (db.source_stream(context["row"]), context["row"]["start"], context["row"]["end"])
        unique[identity] = sample
    return sorted(unique.values(), key=lambda item: item["created_at"])[-60:], skipped, records


def summary(singer_id):
    if not singer_id:
        return "选择歌手后显示边界反馈。"
    usable, skipped, all_rows = usable_samples(singer_id)
    counts = Counter(row["kind"] for row in all_rows)
    current = profile(singer_id)
    return (f"累计反馈（含已撤销或替换）：采用建议 {counts['recommend']} · 手动修正 {counts['manual']} · 不采用建议 {counts['declined']}。"
            f"可用于校准的已确认片段 **{len(usable)}**（最多使用最近 60 条；源文件变化跳过 {skipped} 条）。  \n"
            f"当前参数：搜索 ±{current['search_radius']:g} 秒 · 停顿至少 {current['pause_seconds'] * 1000:g} 毫秒 · 相对能量阈值 {current['energy_ratio']:g}。")


def evaluate(samples, settings):
    weighted_error, weights, matches, count = 0., 0., 0, 0
    for sample in samples:
        context = sample["plan"]["context"]
        row = context["row"]
        source = db.source_audio(row)
        analysis = analyze_boundaries(source, row, context["previous"], context["following"], profile=settings)
        for key in ["start", "end"]:
            cut = analysis["cuts"][key]
            predicted = cut["point"] if cut["suggested"] is None else cut["suggested"]
            error = float(abs(predicted - sample["target"][key]))
            weight = 2. if abs(sample["target"][key] - row[key]) > .03 else 1.
            weighted_error += error * weight
            weights += weight
            matches += int(error <= .08)
            count += 1
    return {"mean_error_seconds": weighted_error / max(weights, 1), "matches": matches, "boundaries": count}


def calibrate(singer_id):
    samples, skipped, all_records = usable_samples(singer_id)
    if len(samples) < 5:
        raise ValueError("至少需要 5 个试听确认过的反馈片段。采用或手动修改建议后确认通过；不采用建议后也需确认原片段")
    sources = list(dict.fromkeys(sample["plan"]["context"]["row"]["source_id"] for sample in samples))
    holdout = set(sources[-max(1, round(len(sources) * .2)):]) if len(sources) > 1 else set()
    training = [sample for sample in samples if sample["plan"]["context"]["row"]["source_id"] not in holdout]
    validation = [sample for sample in samples if sample not in training]
    if len(training) < 3:
        raise ValueError("按源素材划分后校准组不足 3 个确认片段，请增加其他源素材的审核反馈")
    current = profile(singer_id)
    baseline = evaluate(training, current)
    best, score = current, baseline
    candidates = [dict(search_radius=radius, pause_seconds=pause, energy_ratio=ratio)
                  for radius, pause, ratio in product([.8, 1.2, 1.8], [.08, .12, .2], [.04, .06, .08])]
    for candidate in candidates:
        result = evaluate(training, candidate)
        if result["mean_error_seconds"] < score["mean_error_seconds"] - 1e-9:
            best, score = candidate, result
    before_validation = evaluate(validation, current) if validation else None
    after_validation = evaluate(validation, best) if validation else None
    improved = baseline["mean_error_seconds"] - score["mean_error_seconds"] >= .02
    if validation and after_validation["mean_error_seconds"] > before_validation["mean_error_seconds"] + .02:
        improved = False
    report = {"id": db.new_id(), "singer_id": singer_id, "created_at": db.now(),
              "feedback_digest": digest(all_records), "current": current, "candidate": best,
              "train_before": baseline, "train_after": score, "validation_before": before_validation,
              "validation_after": after_validation, "can_apply": improved, "samples": len(samples), "sources": len(sources),
              "skipped": skipped, "sample_sources": {sample["id"]: sample["plan"]["source_stat"] for sample in samples}}
    db.write(db.DATA / "calibration_reports" / (singer_id + ".json"), report)
    return report


def report_text(report):
    if not report:
        return "生成参数建议后，可以比较当前规则与候选规则对已确认边界的匹配情况。"
    before, after = report["train_before"], report["train_after"]
    candidate = report["candidate"]
    text = (f"使用 **{report['samples']} 个确认片段 · {report['sources']} 份源素材**。  \n"
            f"校准组加权平均偏差：{before['mean_error_seconds']:.3f} → {after['mean_error_seconds']:.3f} 秒。  \n"
            f"候选参数：搜索 ±{candidate['search_radius']:g} 秒 · 停顿 {candidate['pause_seconds'] * 1000:g} 毫秒 · 相对能量阈值 {candidate['energy_ratio']:g}。  \n")
    if report["validation_after"]:
        first, second = report["validation_before"], report["validation_after"]
        text += f"独立源素材保留组平均偏差：{first['mean_error_seconds']:.3f} → {second['mean_error_seconds']:.3f} 秒。  \n"
    else:
        text += "只有一份源素材，结果只描述这份素材内的匹配情况。  \n"
    return text + ("可应用到当前歌手；应用后重新检查当前任务，不改变音频或审核决定。" if report["can_apply"] else "改善不足或保留组退步，保持当前参数。")


def apply(singer_id, report):
    with db.lock():
        saved = db.read(db.DATA / "calibration_reports" / (db.identifier(singer_id) + ".json"), {})
        if not report or saved.get("id") != report.get("id") or report.get("singer_id") != singer_id or not saved.get("can_apply"):
            raise ValueError("没有可应用的当前歌手参数建议，请重新生成")
        usable, _, all_records = usable_samples(singer_id)
        if digest(all_records) != saved["feedback_digest"] or profile(singer_id) != saved["current"]:
            raise ValueError("审核反馈或当前参数已变化，请重新生成建议")
        if {sample["id"]: sample["plan"]["source_stat"] for sample in usable} != saved["sample_sources"]:
            raise ValueError("校准素材已变化，请重新生成建议")
        db.write(db.DATA / "boundary_profiles" / (singer_id + ".json"),
                 {"settings": validate_profile(saved["candidate"]), "report_id": saved["id"], "updated_at": db.now()})


def reset(singer_id):
    db.singer(singer_id)
    with db.lock():
        db.write(db.DATA / "boundary_profiles" / (singer_id + ".json"),
                 {"settings": dict(DEFAULT_PROFILE), "updated_at": db.now()})
