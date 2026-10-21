"""Signal-only phrase proposals and transparent quality measurements.

Energy activity is not a lyric recognizer or a singer/chorus detector.
"""
import math
from bisect import bisect_left, bisect_right
import re
import shutil
import subprocess
import time
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def ffmpeg():
    executable = shutil.which("ffmpeg")
    if not executable:
        import sys
        candidate = Path(sys.base_prefix) / "bin/ffmpeg"
        if candidate.exists():
            executable = str(candidate)
    if not executable:
        raise ValueError("未找到 FFmpeg，请安装后将其加入 PATH")
    return executable


def decode(source, target, rate=44100, channels=2, check_cancel=None):
    command = [ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source), "-ar", str(rate), "-ac", str(channels),
                    "-c:a", "pcm_s16le", str(target)]
    process = subprocess.Popen(command)
    deadline = time.monotonic() + 3600
    try:
        while process.poll() is None:
            if check_cancel:
                check_cancel()
            if time.monotonic() > deadline:
                raise ValueError("解码超过一小时，已停止")
            time.sleep(.1)
        if process.returncode:
            raise ValueError("音频无法解码，请检查格式或源文件是否损坏")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def parse_lrc(text):
    offset = re.search(r"\[offset:([+-]?\d+)\]", text, flags=re.I)
    offset = int(offset.group(1)) / 1000 if offset else 0
    lines = []
    for line in text.splitlines():
        stamps = list(re.finditer(r"\[(\d+):([0-5]?\d(?:\.\d+)?)\]", line))
        lyric = re.sub(r"\[[^]]*\]", "", line).strip()
        for stamp in stamps:
            lines.append({"time": max(0, int(stamp[1]) * 60 + float(stamp[2]) + offset), "text": lyric})
    return sorted(lines, key=lambda row: row["time"])


def envelope(samples, rate, frame=.02):
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    width = max(1, round(rate * frame))
    pad = (-len(mono)) % width
    frames = np.pad(mono, (0, pad)).reshape(-1, width)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return rms, width / rate


def proposals(samples, rate, lyric_text="", offset=0, minimum=5, maximum=15, padding=.15, settings=None):
    rms, frame = envelope(samples, rate)
    return proposals_from_envelope(rms, frame, len(samples) / rate, lyric_text, offset, minimum, maximum, padding, settings)


def file_proposals(source, lyric_text="", offset=0, minimum=5, maximum=15, padding=.15, settings=None, check_cancel=None):
    """Keep only the 20 ms energy curve; never load the entire recording."""
    with sf.SoundFile(source) as stream:
        width = max(1, round(stream.samplerate * .02))
        count = (len(stream) + width - 1) // width
        rms = np.empty(count, dtype=np.float32)
        position = 0
        while position < count:
            if check_cancel:
                check_cancel()
            samples = stream.read(width * 2048, dtype="float32", always_2d=True).mean(axis=1)
            if not len(samples):
                raise ValueError("源音轨提前结束，请检查文件是否被修改")
            values, _ = envelope(samples, stream.samplerate)
            rms[position:position + len(values)] = values
            position += len(values)
        duration = len(stream) / stream.samplerate
        frame = width / stream.samplerate
    return proposals_from_envelope(rms, frame, duration, lyric_text, offset, minimum, maximum, padding, settings, check_cancel)


def proposals_from_envelope(rms, frame, duration, lyric_text="", offset=0, minimum=5, maximum=15, padding=.15, settings=None, check_cancel=None):
    if not len(rms):
        return [], ["音频为空，没有可用片段"]
    settings = settings or {}
    threshold = max(10 ** (settings.get("silence_db", -45) / 20), float(np.percentile(rms, 85)) * .06)
    active = rms > threshold
    # Only quiet runs >= 200 ms qualify as reliable musical pauses.
    runs = []
    begin = None
    for i, quiet in enumerate(~active):
        if check_cancel and i % 2048 == 0:
            check_cancel()
        if quiet and begin is None:
            begin = i
        if begin is not None and (not quiet or i == len(active) - 1):
            end = i if not quiet else i + 1
            if (end - begin) * frame >= .2:
                runs.append(((begin + end) * frame / 2, begin * frame, end * frame))
            begin = None
    lrc = parse_lrc(lyric_text)
    lrc = [{**row, "time": row["time"] + float(offset)} for row in lrc if row["text"]]
    warnings = []
    if lyric_text and not lrc:
        warnings.append("歌词无有效 LRC 时间戳，已按停顿切片")
    if lrc:
        warnings.append("LRC 仅作时间定位；能量校准不验证歌词内容，请试听核对版本与偏移")
        if any(row["time"] < 0 or row["time"] >= duration for row in lrc):
            warnings.append("部分歌词时间超出音频范围")
        if any("<" in row["text"] for row in lrc):
            warnings.append("增强 LRC 逐字标签不参与校准")
    pauses = [point for point, _, _ in runs]
    lyric_times = [row["time"] for row in lrc]
    voiced = np.flatnonzero(active)
    if not len(voiced):
        return [], warnings + ["音频近乎静音，没有可用片段"]
    first = max(0, voiced[0] * frame - padding)
    last = min(duration, (voiced[-1] + 1) * frame + padding)
    boundaries = [first]
    cursor = first
    while last - cursor > maximum:
        if check_cancel:
            check_cancel()
        candidates = pauses[bisect_left(pauses, cursor + minimum):bisect_right(pauses, cursor + maximum)]
        lyric_candidates = lyric_times[bisect_left(lyric_times, cursor + minimum):bisect_right(lyric_times, cursor + maximum)]
        if candidates:
            target = lyric_candidates[0] if lyric_candidates else cursor + (minimum + maximum) / 2
            cut = min(candidates, key=lambda p: abs(p - target))
        else:
            cut = cursor + maximum
        boundaries.append(cut)
        cursor = cut
    if last - boundaries[-1] < minimum and len(boundaries) > 1 and last - boundaries[-2] <= maximum:
        boundaries.pop()
    boundaries.append(last)
    records = []
    for start, end in zip(boundaries, boundaries[1:]):
        if check_cancel:
            check_cancel()
        # Avoid adjoining clip overlap; pad only into known quiet regions.
        text = " / ".join(row["text"] for row in lrc[bisect_left(lyric_times, start):bisect_left(lyric_times, end)])
        flags = list(warnings)
        near = bisect_left(pauses, start)
        if not any(abs(start - p) < .05 for p in pauses[max(0, near - 1):near + 1]) and start > first:
            flags.append("持续发声段达到最大长度，切点需要试听")
        records.append({"start": round(start, 4), "end": round(end, 4), "lyrics": text, "alignment": "energy_hint" if lrc else "pause_only", "alignment_flags": flags})
    return records, warnings


class OverlapWriter:
    """Finalize each hop immediately, retaining only the overlap in RAM."""
    def __init__(self, stream, hop):
        self.stream = stream
        self.hop = hop
        self.tail = np.empty(0, dtype=np.float32)
        self.weights = np.empty(0, dtype=np.float32)
        self.peak = 0.
        self.frames = 0

    def append(self, start, samples, window):
        if start != self.frames or len(self.tail) > len(samples):
            raise ValueError("转换片段时间轴不连续，已阻止错位写入")
        output = np.zeros(len(samples), dtype=np.float32)
        weights = np.zeros(len(samples), dtype=np.float32)
        output[:len(self.tail)] = self.tail
        weights[:len(self.weights)] = self.weights
        output += samples * window
        weights += window
        count = min(self.hop, len(output))
        ready = output[:count] / np.maximum(weights[:count], 1e-6)
        if not np.all(np.isfinite(ready)):
            raise ValueError("转换输出出现非有限数值，已停止写入")
        self.peak = max(self.peak, float(np.max(np.abs(ready))) if count else 0.)
        self.stream.write(ready)
        self.frames += count
        self.tail = output[count:].copy()
        self.weights = weights[count:].copy()


def write_audio(source, target, check_cancel=None):
    """Export exact floating amplitudes, including peaks above digital full scale.

    Gain is a user decision. FLOAT WAV prevents integer export from clipping
    values above 1; playback hardware may still clip them. Never normalize or
    limit a stem or mix here, or let one export change any other export.
    """
    target = Path(target)
    with tempfile.TemporaryDirectory(prefix=".svc-write-", dir=target.parent) as directory:
        temporary = Path(directory) / "audio.wav"
        with sf.SoundFile(source) as src, sf.SoundFile(temporary, "w", samplerate=src.samplerate, channels=src.channels, subtype="FLOAT") as dst:
            while True:
                if check_cancel:
                    check_cancel()
                samples = src.read(65536, dtype="float32", always_2d=True)
                if not len(samples):
                    break
                if not np.isfinite(samples).all():
                    raise ValueError("音轨出现非有限数值，未写入音频")
                dst.write(samples)
        temporary.replace(target)
    return 1.


def mix_tracks(vocals, accompaniment, target, vocal_gain, instrumental_gain, source_gain=1., check_cancel=None):
    """Mix aligned tracks at the requested gains without automatic attenuation."""
    target = Path(target)
    with tempfile.TemporaryDirectory(prefix=".svc-mix-", dir=target.parent) as directory:
        floating = Path(directory) / "mix.wav"
        with sf.SoundFile(vocals) as voice, sf.SoundFile(accompaniment) as inst:
            if voice.samplerate != inst.samplerate or len(voice) != len(inst) or voice.channels not in (1, inst.channels):
                raise ValueError("伴奏长度或采样率与转换人声不同，已阻止错位混音")
            with sf.SoundFile(floating, "w", samplerate=voice.samplerate, channels=inst.channels, subtype="FLOAT") as dst:
                while True:
                    if check_cancel:
                        check_cancel()
                    samples = voice.read(65536, dtype="float32", always_2d=True)
                    backing = inst.read(len(samples), dtype="float32", always_2d=True)
                    if not len(samples):
                        break
                    samples *= source_gain
                    mixed = backing * instrumental_gain + samples * vocal_gain
                    if not np.all(np.isfinite(mixed)):
                        raise ValueError("混音出现非有限数值，已停止写入")
                    dst.write(mixed)
        return write_audio(floating, target, check_cancel)


def sum_tracks(tracks, target, check_cancel=None):
    """Sum aligned mono/stereo tracks into a floating file without clipping."""
    from contextlib import ExitStack
    if not tracks:
        raise ValueError("没有可合并的音轨")
    with ExitStack() as stack:
        sources = [(stack.enter_context(sf.SoundFile(path)), float(gain)) for path, gain in tracks]
        first = sources[0][0]
        channels = max(source.channels for source, _ in sources)
        if channels not in (1, 2) or any(source.samplerate != first.samplerate or len(source) != len(first)
                                       or source.channels not in (1, channels) for source, _ in sources):
            raise ValueError("人声音轨长度、声道或采样率不一致，已阻止错位合并")
        dst = stack.enter_context(sf.SoundFile(target, "w", samplerate=first.samplerate,
                                               channels=channels, subtype="FLOAT"))
        peak = 0.
        for start in range(0, len(first), 65536):
            if check_cancel:
                check_cancel()
            length = min(65536, len(first) - start)
            mixed = np.zeros((length, channels), dtype=np.float32)
            for source, gain in sources:
                samples = source.read(length, dtype="float32", always_2d=True)
                if len(samples) != length:
                    raise ValueError("合并音轨读取长度异常")
                mixed += samples * gain
            if not np.all(np.isfinite(mixed)):
                raise ValueError("合并音轨出现非有限数值")
            peak = max(peak, float(np.max(np.abs(mixed))))
            dst.write(mixed)
    return peak


def quality(samples, rate, settings=None):
    settings = settings or {}
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    if not len(mono):
        return {"duration": 0., "rms_db": -240., "peak": 0., "silence_ratio": 1.,
                "clipping_ratio": 0., "flags": ["近乎静音或过短"], "quality": "rejected"}
    rms, _ = envelope(mono, rate)
    level = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0
    threshold = max(10 ** (settings.get("silence_db", -45) / 20), float(np.percentile(rms, 85)) * .06)
    silent = float(np.mean(rms <= threshold))
    clipped = float(np.mean(np.abs(mono) >= .999))
    duration = len(mono) / rate
    flags = []
    rejected = level < 1e-5 or duration < .4
    if rejected:
        flags.append("近乎静音或过短")
    if duration < 3 or duration > 15.5:
        flags.append("时长超出建议范围")
    if silent > settings.get("max_silence_ratio", .5):
        flags.append("静音占比偏高")
    if clipped > settings.get("max_clipping_ratio", .001):
        flags.append("疑似削波")
    if level < .005:
        flags.append("音量偏低")
    if rms[0] > threshold or rms[-1] > threshold:
        flags.append("边界可能仍在发声")
    return {"duration": round(duration, 4), "rms_db": round(20 * math.log10(max(level, 1e-12)), 2),
            "peak": round(peak, 5), "silence_ratio": round(silent, 4), "clipping_ratio": round(clipped, 5),
            "flags": flags, "quality": "rejected" if rejected else "needs_review" if flags else "passed"}


def crop(source, target, start, end, settings=None):
    with sf.SoundFile(source) as stream:
        if not 0 <= float(start) < float(end) <= len(stream) / stream.samplerate + 1e-6:
            raise ValueError("起止时间必须位于原始人声音轨内，结束时间须大于开始时间")
        stream.seek(round(float(start) * stream.samplerate))
        samples = stream.read(round(float(end) * stream.samplerate) - round(float(start) * stream.samplerate), dtype="float32")
        rate = stream.samplerate
    if not len(samples):
        raise ValueError("切片不足一个采样点，请扩大起止时间范围")
    sf.write(target, samples.mean(axis=1) if samples.ndim == 2 else samples, rate, subtype="PCM_16")
    return quality(samples, rate, settings)
