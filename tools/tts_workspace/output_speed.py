"""Create pitch-preserving TTS exports at the selected playback rate."""

import math
import os
from pathlib import Path
import subprocess
import threading
import uuid

import soundfile as sf

from tools.portable_runtime import FFMPEG_EXE

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "tts-workspace"
OUTPUT_LOCK = threading.RLock()


def rate_value(value):
    try:
        rate = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("请填写 0.25–5 倍之间的数字") from error
    if not math.isfinite(rate) or not 0.25 <= rate <= 5:
        raise ValueError("倍速须在 0.25–5 倍之间")
    return round(rate, 3)


def source_path(value):
    if not value:
        raise ValueError("请先合成语音，再调整倍速")
    path = Path(value).resolve()
    if path.parent != OUTPUT_DIR.resolve() or path.suffix.lower() != ".wav" or not path.is_file():
        raise ValueError("原始合成音频已不存在，请重新合成")
    return path


def atempo_filter(rate):
    factors = []
    while rate < 0.5:
        factors.append(0.5)
        rate /= 0.5
    while rate > 2:
        factors.append(2.0)
        rate /= 2.0
    factors.append(rate)
    return ",".join(f"atempo={factor:.6g}" for factor in factors)


def format_duration(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}:{seconds:04.1f}"


def result(source, selected_rate):
    with OUTPUT_LOCK:
        return _result(source, selected_rate)


def _result(source, selected_rate):
    source = source_path(source)
    rate = rate_value(selected_rate)
    original_seconds = sf.info(source).duration
    if rate == 1:
        destination = source
    else:
        destination = source.with_name(f"{source.stem}-speed-{rate:.3f}.wav")
        if not destination.is_file():
            temporary = destination.with_name(f"{destination.stem}.{uuid.uuid4().hex}.part.wav")
            try:
                command = [str(FFMPEG_EXE), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                           "-i", str(source), "-vn", "-filter:a", atempo_filter(rate),
                           "-c:a", "pcm_s16le", str(temporary)]
                try:
                    process = subprocess.run(command, capture_output=True, text=True, timeout=300)
                except subprocess.TimeoutExpired as error:
                    raise ValueError("调整倍速超时，请重试或缩短音频") from error
                if process.returncode or not temporary.is_file():
                    raise ValueError("调整倍速失败：" + (process.stderr.strip()[-300:] or "音频处理工具没有生成文件"))
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    adjusted_seconds = sf.info(destination).duration
    message = (f"**当前 {rate:g}×** · 原始 {format_duration(original_seconds)} · "
               f"调整后 {format_duration(adjusted_seconds)}。播放器的下载按钮保存当前倍速音频。")
    return str(destination), message
