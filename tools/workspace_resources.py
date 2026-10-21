"""Disk preflight checks shared by the local dataset workspaces."""
import math
import os
from pathlib import Path
import shutil


GIB = 1024 ** 3


def disk_usage(path):
    directory = Path(path).resolve()
    while not directory.exists():
        directory = directory.parent
    return shutil.disk_usage(directory)


def reserve_bytes():
    try:
        value = float(os.environ.get("GPT_SOVITS_DISK_RESERVE_GIB", "2"))
    except ValueError as error:
        raise ValueError("GPT_SOVITS_DISK_RESERVE_GIB 必须是非负数") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError("GPT_SOVITS_DISK_RESERVE_GIB 必须是有限非负数")
    return math.ceil(value * GIB)


def require_space(path, additional=0, purpose="任务写入"):
    required = max(0, int(additional)) + reserve_bytes()
    free = disk_usage(path).free
    if free < required:
        raise ValueError(f"{purpose}空间不足：可用 {free / GIB:.2f} GiB，"
                         f"预计至少需要 {required / GIB:.2f} GiB（含保留空间）。请释放空间后重试")


def disk_summary(path):
    try:
        usage = disk_usage(path)
        reserve = reserve_bytes()
        label = "空间偏低" if usage.free < reserve else "磁盘可用"
        return f"{label} {usage.free / GIB:.1f} GiB / 总计 {usage.total / GIB:.1f} GiB"
    except (OSError, ValueError) as error:
        return f"无法读取磁盘余量：{error}"
