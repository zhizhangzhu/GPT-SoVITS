"""Local energy checks, bounded pause suggestions and explicit review priorities."""
import math

import numpy as np
import soundfile as sf

from .audio import envelope

VERSION = 1
DEFAULT_PROFILE = {"search_radius": 1.2, "pause_seconds": .12, "energy_ratio": .06}
PRIORITIES = {4: "不可直接入库", 3: "优先复核", 2: "需要检查", 1: "试听确认", 0: "常规试听"}


def validate_profile(profile=None):
    result = dict(DEFAULT_PROFILE, **(profile or {}))
    limits = {"search_radius": (.3, 3.), "pause_seconds": (.08, .4), "energy_ratio": (.02, .15)}
    for key, (low, high) in limits.items():
        value = float(result[key])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"边界参数 {key} 须为 {low}–{high}")
        result[key] = value
    return {key: result[key] for key in DEFAULT_PROFILE}


def inspect_cut(stream, point, lower, upper, settings, profile):
    radius = profile["search_radius"]
    rate, total = stream.samplerate, len(stream)
    width = max(1, round(rate * .02))
    left = max(0, math.floor((point - radius) * rate / width) * width)
    right = min(total, math.ceil((point + radius) * rate / width) * width)
    stream.seek(left)
    samples = stream.read(right - left, dtype="float32", always_2d=True).mean(axis=1)
    rms, frame = envelope(samples, rate)
    threshold = max(10 ** (settings.get("silence_db", -45) / 20),
                    float(np.percentile(rms, 85)) * profile["energy_ratio"]) if len(rms) else 1.
    origin = left / rate
    quiet = rms <= threshold
    transitions = np.diff(np.r_[False, quiet, False].astype(np.int8))
    pauses = [(origin + a * frame, min(total / rate, origin + b * frame))
              for a, b in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))
              if (b - a) * frame >= profile["pause_seconds"]]
    safe = any(a + .03 <= point <= b - .03
               or (point < .04 and a == 0 and point <= b - .03)
               or (total / rate - point < .04 and b == total / rate and point >= a + .03)
               for a, b in pauses)
    centers = origin + (np.arange(len(rms)) + .5) * frame
    near = rms[np.abs(centers - point) <= .06]
    active = bool(len(near) and np.mean(near > threshold) >= .5)
    edge = point < .04 or total / rate - point < .04
    state = "quiet" if safe else "edge_active" if active and edge else "voiced" if active else "uncertain"
    result = {"point": round(point, 4), "state": state, "suggested": None,
              "threshold_db": round(20 * math.log10(max(threshold, 1e-12)), 2), "search_radius": radius}
    if safe:
        result["reason"] = "当前切点位于持续低能量停顿内"
        return result
    candidates = []
    for a, b in pauses:
        low, high = max(lower, point - radius, a + .04), min(upper, point + radius, b - .04)
        if low <= high:
            candidate = min(high, max(low, (a + b) / 2))
            if abs(candidate - point) >= .03:
                candidates.append((abs(candidate - point), candidate, b - a))
    if candidates:
        _, candidate, length = min(candidates)
        result.update(suggested=round(candidate, 4), quiet_seconds=round(length, 3),
                      reason="附近找到持续低能量停顿，建议移动切点并试听")
    else:
        result["reason"] = "附近未找到满足时长与相邻片段约束的停顿，保留原切点"
    return result


def neighbors(rows):
    from .storage import source_stream
    groups = {}
    for row in rows:
        groups.setdefault(source_stream(row), []).append(row)
    result = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda row: (row["start"], row["end"], row["id"]))
        result.update({row["id"]: (ordered[i - 1] if i else None, ordered[i + 1] if i + 1 < len(ordered) else None)
                       for i, row in enumerate(ordered)})
    return result


def analyze_boundaries(source, row, previous=None, following=None, profile=None):
    profile = validate_profile(profile)
    start, end = float(row["start"]), float(row["end"])
    with sf.SoundFile(source) as stream:
        duration = len(stream) / stream.samplerate
        if not 0 <= start < end <= duration + 1e-6:
            raise ValueError("片段边界超出源人声音轨，请重新整理或修正")
        shared_start = bool(previous and abs(previous["end"] - start) <= .0001)
        shared_end = bool(following and abs(following["start"] - end) <= .0001)
        midpoint = (start + end) / 2
        low = max(0., previous["start"] + .41 if shared_start else previous["end"] if previous else 0.)
        high = min(duration, following["end"] - .41 if shared_end else following["start"] if following else duration)
        start_high = min(end - .41, midpoint - .205)
        end_low = max(start + .41, midpoint + .205)
        settings = row.get("quality_settings") or {}
        cuts = {"start": inspect_cut(stream, start, low, start_high, settings, profile),
                "end": inspect_cut(stream, end, end_low, high, settings, profile)}
    flags = []
    for key, name in [("start", "起点"), ("end", "终点")]:
        if cuts[key]["state"] == "voiced":
            flags.append(name + "附近仍有持续声音，疑似截字，请试听")
        elif cuts[key]["state"] == "edge_active":
            flags.append(name + "位于源音轨边缘且仍有声音，请确认素材是否已截断")
        elif cuts[key]["state"] == "uncertain":
            flags.append(name + "附近停顿不足，切点需要试听")
    return {"version": VERSION, "profile": profile, "cuts": cuts, "flags": flags,
            "shared_start": previous["id"] if shared_start else None,
            "shared_end": following["id"] if shared_end else None}


def annotate(row, analysis):
    row["boundary_analysis"] = analysis
    # Contextual analysis replaces the old first/last-frame-only warning.
    row["flags"] = [flag for flag in row.get("flags", []) if flag != "边界可能仍在发声"]
    row["boundary_flags"] = analysis["flags"]
    if row["quality"] != "rejected":
        row["quality"] = "needs_review" if row["flags"] or analysis["flags"] or row.get("alignment_flags") else "passed"
    return row


def annotate_source(rows, source, only=None, check_cancel=None, profile=None):
    adjacent = neighbors(rows)
    for row in rows:
        if only is not None and row["id"] not in only:
            continue
        if check_cancel:
            check_cancel()
        annotate(row, analyze_boundaries(source, row, *adjacent[row["id"]], profile=profile))


def priority(row):
    flags = list(dict.fromkeys(row.get("flags", []) + row.get("boundary_flags", []) + row.get("alignment_flags", [])))
    if row.get("quality") == "rejected":
        return {"level": 4, "label": PRIORITIES[4], "reasons": ["近乎静音或过短，需先修正"]}
    severe = [flag for flag in flags if any(word in flag for word in ["截字", "截断", "削波", "边界可能", "持续发声"])]
    if severe:
        return {"level": 3, "label": PRIORITIES[3], "reasons": severe}
    moderate = [flag for flag in flags if any(word in flag for word in ["时长", "静音", "音量", "停顿不足"])]
    if moderate:
        return {"level": 2, "label": PRIORITIES[2], "reasons": moderate}
    level = 1 if flags else 0
    return {"level": level, "label": PRIORITIES[level], "reasons": flags}
