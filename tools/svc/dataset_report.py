"""CPU-only pre-training coverage report, executed in the isolated SVC runtime."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from html import escape
import io
import json
from pathlib import Path
import sys
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf
from filelock import FileLock

from . import storage as db
from .install import RUNTIME
from tools.workspace_resources import require_space

VERSION = 2
HOP = 512
RATE = 44100
LENGTH_BINS = [("<3 s", 0, 3), ("3–5 s", 3, 5), ("5–10 s", 5, 10),
               ("10–15.5 s", 10, 15.500001), (">15.5 s", 15.500001, float("inf"))]


def signature(rows):
    fields = ["id", "sha256", "source_id", "file", "start", "end", "duration"]
    payload = [{key: row.get(key) for key in fields} for row in sorted(rows, key=lambda row: row["id"])]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def note_name(note):
    return ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][note % 12] + str(note // 12 - 1)


def pitch_summary(counts):
    ordered = sorted((int(note), count) for note, count in counts.items() if count > 0)
    if not ordered:
        return None
    cumulative = np.cumsum([count for _, count in ordered])
    def quantile(fraction):
        position = min(len(ordered) - 1, int(np.searchsorted(cumulative, max(1, cumulative[-1] * fraction))))
        note = ordered[position][0]
        return {"note": note, "name": note_name(note), "hz": round(440 * 2 ** ((note - 69) / 12), 1)}
    return {"p5": quantile(.05), "p50": quantile(.5), "p95": quantile(.95)}


def analyze_clip(path, predictor, check_cancel=None):
    counts, total_frames, voiced_frames = Counter(), 0, 0
    near_low, near_high, octave_jumps = 0, 0, 0
    with sf.SoundFile(path) as stream:
        if stream.samplerate != RATE or stream.channels != 1 or len(stream) == 0:
            raise ValueError("应为非空 44.1 kHz 单声道入库人声")
        frames = len(stream)
        # Bounded RAM even for manually merged, unusually long clips.
        for start in range(0, frames, RATE * 15):
            if check_cancel:
                check_cancel()
            end = min(frames, start + RATE * 15)
            left, right = max(0, start - round(RATE * .3)), min(frames, end + round(RATE * .3))
            stream.seek(left)
            wave = stream.read(right - left, dtype="float32")
            f0, uv = predictor.compute_f0_uv(wave)
            times = np.arange(len(f0)) * HOP / RATE
            core = (times >= (start - left) / RATE) & (times < (end - left) / RATE)
            valid = core & (np.asarray(uv) > 0) & np.isfinite(f0) & (np.asarray(f0) > 0)
            total_frames += int(core.sum())
            voiced_frames += int(valid.sum())
            near_low += int(np.count_nonzero(valid & (np.asarray(f0) < 65)))
            near_high += int(np.count_nonzero(valid & (np.asarray(f0) > 900)))
            frequency = np.asarray(f0)
            adjacent = valid[1:] & valid[:-1]
            octave_jumps += int(np.count_nonzero(adjacent & (np.abs(np.log2(
                np.maximum(frequency[1:], 1) / np.maximum(frequency[:-1], 1))) >= .9)))
            notes = np.rint(69 + 12 * np.log2(np.asarray(f0)[valid] / 440)).astype(int)
            counts.update(map(int, notes))
    return {"duration": frames / RATE, "samples": frames, "total_frames": total_frames,
            "voiced_frames": voiced_frames, "pitch_counts": dict(counts), "near_low_frames": near_low,
            "near_high_frames": near_high, "octave_jumps": octave_jumps}


def summarize(rows, analyses, invalid, predictor):
    pitch, lengths = Counter(), Counter({label: 0 for label, _, _ in LENGTH_BINS})
    sources = defaultdict(lambda: {"clips": 0, "seconds": 0.})
    duration, voiced, frames = 0., 0, 0
    by_id = {row["id"]: row for row in rows}
    problems = []
    for identifier, data in analyses.items():
        row = by_id[identifier]
        duration += data["duration"]
        voiced += data["voiced_frames"]
        frames += data["total_frames"]
        pitch.update({int(note): count for note, count in data["pitch_counts"].items()})
        for label, low, high in LENGTH_BINS:
            if low <= data["duration"] < high:
                lengths[label] += 1
                break
        group = sources[row["source_id"]]
        group.update(name=row.get("source_name") or "未命名源素材", source_id=row["source_id"])
        group["clips"] += 1
        group["seconds"] += data["duration"]
        reasons = []
        ratio = data["voiced_frames"] / data["total_frames"] if data["total_frames"] else 0
        if ratio < .2:
            reasons.append(f"有声 F0 帧仅 {ratio:.0%}，请试听并检查提取器")
        if data["voiced_frames"] and data.get("near_low_frames", 0) / data["voiced_frames"] >= .1:
            reasons.append("较多 F0 接近 50 Hz 下限，请检查低音或倍频错误")
        if data["voiced_frames"] and data.get("near_high_frames", 0) / data["voiced_frames"] >= .1:
            reasons.append("较多 F0 接近 1100 Hz 上限，请检查高音或倍频错误")
        if data.get("octave_jumps", 0) >= 3 and data["voiced_frames"] and data["octave_jumps"] / data["voiced_frames"] >= .01:
            reasons.append("连续有声帧出现频繁大幅音高跳变，请检查 F0")
        if data["duration"] < 3 or data["duration"] > 15.5:
            reasons.append("片段时长超出建议的 3–15.5 秒")
        if reasons:
            problems.append({"id": identifier, "job_id": row.get("job_id"), "source_name": row.get("source_name", ""),
                             "start": row.get("start"), "end": row.get("end"), "duration": data["duration"],
                             "voiced_ratio": ratio, "reasons": reasons})
    for item in invalid:
        row = by_id[item["id"]]
        problems.append({"id": item["id"], "job_id": row.get("job_id"), "source_name": row.get("source_name", ""),
                         "start": row.get("start"), "end": row.get("end"), "duration": row.get("duration", 0),
                         "voiced_ratio": None, "reasons": ["入库文件校验失败：" + item["reason"]]})
    problems.sort(key=lambda item: (item["voiced_ratio"] is not None, item["voiced_ratio"] or 0, -len(item["reasons"])))
    source_rows = sorted(sources.values(), key=lambda row: row["seconds"], reverse=True)
    for row in source_rows:
        row["share"] = row["seconds"] / duration if duration else 0.
    warnings = []
    if len(source_rows) == 1:
        warnings.append("只有一份有效源素材，训练与验证将来自同一源，不能据此评价跨歌曲表现")
    if len(source_rows) > 1 and source_rows[0]["share"] >= .8:
        warnings.append(f"最多的一份源素材占有效时长 {source_rows[0]['share']:.0%}，请检查素材是否过度集中")
    if lengths["<3 s"] or lengths[">15.5 s"]:
        warnings.append(f"{lengths['<3 s'] + lengths['>15.5 s']} 个片段超出当前建议的 3–15.5 秒范围")
    if invalid:
        warnings.append(f"{len(invalid)} 个入库文件校验失败，未纳入时长和音高统计；训练前需重新审核")
    if not voiced:
        warnings.append("未检测到有效有声 F0，请试听素材并尝试其他 F0 提取方式")
    return {"clips": len(rows), "valid_clips": len(analyses), "invalid": invalid, "duration_seconds": duration,
            "voiced_ratio": voiced / frames if frames else 0., "voiced_frames": voiced,
            "pitch_counts": dict(pitch), "pitch_summary": pitch_summary(pitch),
            "lengths": dict(lengths), "sources": source_rows, "warnings": warnings, "problem_clips": problems,
            "predictor": predictor, "sampling_rate": RATE, "hop_length": HOP, "f0_range_hz": [50, 1100]}


def charts(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    settings = {"font.size": 14, "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(settings):
        fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
        counts = sorted((int(note), count) for note, count in report["pitch_counts"].items())
        if counts:
            ax.bar([note for note, _ in counts], [count * HOP / RATE for _, count in counts], color="#2161d4")
            ticks = list(range((counts[0][0] // 12) * 12, counts[-1][0] + 1, 12))
            ax.set_xticks(ticks, [note_name(note) for note in ticks])
        else:
            ax.text(.5, .5, "No voiced F0 detected", ha="center", transform=ax.transAxes)
        ax.set(xlabel="Pitch (one semitone per bin)", ylabel="Approx. voiced seconds", title="Voiced pitch coverage")
        fig.savefig(directory / "pitch.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
        labels = [label for label, _, _ in LENGTH_BINS]
        ax.bar(labels, [report["lengths"][label] for label in labels], color="#2161d4")
        ax.set(xlabel="Clip duration", ylabel="Clips", title="Clip length distribution")
        fig.savefig(directory / "lengths.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
        shown = report["sources"][:10]
        values = [row["seconds"] / 60 for row in shown]
        labels = [f"Source {index + 1}" for index in range(len(shown))]
        if len(report["sources"]) > 10:
            values.append(sum(row["seconds"] for row in report["sources"][10:]) / 60)
            labels.append("Other sources")
        ax.barh(labels, values, color="#2161d4")
        ax.invert_yaxis()
        ax.set(xlabel="Audited minutes", title="Source recording contribution")
        fig.savefig(directory / "sources.png", dpi=140)
        plt.close(fig)


def export(report, directory):
    import base64
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["source", "clips", "seconds", "share"])
    for row in report["sources"]:
        name = row["name"]
        # Export labels as text when opened by spreadsheet applications.
        if name.lstrip().startswith(("=", "+", "-", "@")):
            name = "'" + name
        writer.writerow([name, row["clips"], round(row["seconds"], 3), round(row["share"], 5)])
    (directory / "sources.csv").write_text(output.getvalue(), encoding="utf-8-sig")
    images = "".join(f'<img alt="{title}" src="data:image/png;base64,{base64.b64encode((directory / filename).read_bytes()).decode()}">'
                     for filename, title in [("pitch.png", "音高分布"), ("lengths.png", "片段时长分布"), ("sources.png", "源素材占比")])
    source_table = "".join(f'<tr><td>{index + 1}</td><td>{escape(row["name"])}</td><td>{row["clips"]}</td><td>{row["seconds"] / 60:.2f}</td><td>{row["share"]:.1%}</td></tr>'
                           for index, row in enumerate(report["sources"]))
    problem_table = "".join(f'<tr><td>{escape(row["source_name"])}</td><td>{row["start"] if row["start"] is not None else "-"}</td>'
                            f'<td>{row["end"] if row["end"] is not None else "-"}</td><td>{escape("；".join(row["reasons"]))}</td></tr>'
                            for row in report["problem_clips"])
    notes = "".join(f"<li>{escape(message)}</li>" for message in report["warnings"])
    stamp = datetime.fromisoformat(report["created_at"]).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
    page = (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1000px;margin:32px auto;padding:0 20px;color:#172435}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #d6deea}</style>'
            f'<title>SVC 训练前数据报告</title><h1>{escape(report["singer_name"])} · 训练前数据报告</h1><p>{stamp} · F0 {report["predictor"].upper()}</p>'
            f'<p>入库 {report["clips"]} 段 · 校验有效 {report["valid_clips"]} 段 · 有效时长 {report["duration_seconds"] / 60:.2f} 分钟 · 有声帧 {report["voiced_ratio"]:.1%}</p>'
            f'<ul>{notes}</ul>{images}<table><thead><tr><th>图中序号</th><th>源素材</th><th>片段</th><th>分钟</th><th>时长占比</th></tr></thead><tbody>{source_table}</tbody></table>'
            f'<h2>建议试听的片段（{len(report["problem_clips"])}）</h2><table><thead><tr><th>源素材</th><th>开始 / 秒</th><th>结束 / 秒</th><th>检查原因</th></tr></thead><tbody>{problem_table}</tbody></table>'
            '<p>音高按半音分桶，仅统计提取器判断为有声的帧。提取范围为 50–1100 Hz；统计反映数据覆盖，不代表歌手身份、音色质量或训练效果。</p></html>')
    (directory / "report.html").write_text(page, encoding="utf-8")


def generate(singer_id, predictor_name="dio", progress=None, check_cancel=None):
    identity = db.singer(singer_id)
    rows = db.clips(singer_id)
    if not rows:
        raise ValueError("请先试听审核并入库片段，再生成训练前报告")
    sys.path.insert(0, str(RUNTIME))
    from .pitch import Predictor, METHODS
    if predictor_name not in METHODS:
        raise ValueError("请选择有效的 F0 提取方式")
    # Dataset reports can run alongside a conversion; keep their model on CPU.
    predictor = Predictor(predictor_name, RATE, HOP, device="cpu")
    root = db.DATA / "datasets" / singer_id / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".report.lock"), timeout=1):
        require_space(root, 16 * 1024 ** 2, "保存训练前报告")
        cached_path = root / "clip-cache.json"
        try:
            cache = db.read(cached_path, {})
        except (OSError, ValueError):
            cache = {}
        if not isinstance(cache, dict):
            cache = {}
        analyses, invalid, stats = {}, [], {}
        for index, row in enumerate(rows):
            if check_cancel:
                check_cancel()
            if progress:
                progress(f"分析片段 {index + 1}/{len(rows)}", 2 + 86 * index / len(rows))
            path = Path(row["file"]).resolve()
            try:
                stat = path.stat()
                stamp = [stat.st_size, stat.st_mtime_ns]
            except OSError:
                stamp = None
            stats[row["id"]] = {"path": str(path), "stat": stamp}
            try:
                if path.parent != (root.parent / "clips").resolve() or db.digest(path, check_cancel=check_cancel) != row["sha256"]:
                    raise ValueError("入库文件不匹配审核记录")
                key = hashlib.sha256(f"{VERSION}:{predictor_name}:{row['sha256']}".encode()).hexdigest()
                data = cache.get(key)
                if not isinstance(data, dict) or not {"duration", "samples", "total_frames", "voiced_frames",
                                                      "pitch_counts", "near_low_frames", "near_high_frames", "octave_jumps"} <= data.keys():
                    data = None
                if data is None:
                    data = analyze_clip(path, predictor, check_cancel)
                    cache[key] = data
                    db.write(cached_path, cache)
                current = path.stat()
                if stamp != [current.st_size, current.st_mtime_ns]:
                    raise ValueError("分析期间文件发生变化，请重新生成")
                analyses[row["id"]] = data
            except (OSError, ValueError, RuntimeError) as error:
                invalid.append({"id": row["id"], "source_name": row.get("source_name", ""), "reason": str(error)})
            print(f"已分析 {index + 1}/{len(rows)} 个入库片段", flush=True)
            if progress:
                progress(f"已分析 {index + 1}/{len(rows)} 个入库片段", 2 + 86 * (index + 1) / len(rows))
        if check_cancel:
            check_cancel()
        report = dict(summarize(rows, analyses, invalid, predictor_name), id=db.new_id(), version=VERSION,
                      singer_id=singer_id, singer_name=identity["name"], created_at=db.now(),
                      dataset_signature=signature(rows), file_stats=stats)
        if signature(db.clips(singer_id)) != report["dataset_signature"]:
            raise ValueError("入库清单已变化，请重新生成报告")
        final = root / "reports" / report["id"]
        final.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".report-", dir=root) as staging:
            directory = Path(staging)
            if progress:
                progress("生成图表与可下载报告", 92)
            charts(report, directory)
            if check_cancel:
                check_cancel()
            export(report, directory)
            db.write(directory / "report.json", report)
            directory.rename(final)
        db.write(root / "latest.json", {"id": report["id"], "dataset_signature": report["dataset_signature"]})
    return report


def load(singer_id):
    root = db.DATA / "datasets" / db.identifier(singer_id) / "analysis"
    latest = db.read(root / "latest.json", {})
    if not latest:
        return None, None, False
    directory = root / "reports" / db.identifier(latest["id"])
    report = db.read(directory / "report.json")
    stale = report["version"] != VERSION or report["dataset_signature"] != signature(db.clips(singer_id))
    for info in report["file_stats"].values():
        try:
            stat = Path(info["path"]).stat()
            current = [stat.st_size, stat.st_mtime_ns]
        except OSError:
            current = None
        stale = stale or current != info["stat"]
    return report, directory, stale


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("singer_id")
    parser.add_argument("--predictor", choices=["dio", "harvest", "pm", "rmvpe", "fcpe"], default="dio")
    args = parser.parse_args()
    generate(args.singer_id, args.predictor)
