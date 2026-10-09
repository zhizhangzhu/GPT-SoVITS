"""Map WebUI logical CUDA indices before a child resets device visibility."""

import os


def subprocess_gpu_index(index, visible=None):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if visible is None else visible
    devices = [value.strip() for value in visible.split(",")] if visible else []
    # Legacy training configs use '-' between numbers, so UUIDs cannot be
    # serialized through that format. Keep their existing behavior.
    if devices and all(value.isdigit() for value in devices):
        number = int(index)
        if 0 <= number < len(devices):
            return devices[number]
        raise ValueError("GPU 逻辑编号超出 CUDA_VISIBLE_DEVICES 范围")
    return index
