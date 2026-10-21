"""Lazy model lifecycle and audio generation for the main Gradio application."""

import atexit
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

from filelock import FileLock, Timeout

ROOT = Path(__file__).resolve().parents[2]


def timeout_seconds(name, default):
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是正数秒数") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是正数秒数")
    return value


def gpu_lease():
    (ROOT / "output").mkdir(exist_ok=True)
    lease = FileLock(str(ROOT / "output/auto_dataset-gpu.lock"))
    try:
        lease.acquire(timeout=0)
    except Timeout as error:
        raise ValueError("数据整理或 SVC 任务正在使用 GPU，请等待结束") from error
    return lease


class InferenceService:
    def __init__(self):
        self.process = None
        self.metadata = None
        self.spec = None
        self.events = None
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.generating = False
        self.cancel_requested = False
        self.cancel_deadline = None
        self.config_path = None

    def close(self):
        process, self.process = self.process, None
        self.metadata = self.spec = None
        self.generating = False
        self.events = None
        self.cancel_deadline = None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
        if self.config_path is not None:
            self.config_path.unlink(missing_ok=True)
            self.config_path = None

    def _read(self, process, events):
        try:
            for line in process.stdout:
                try:
                    events.put(json.loads(line))
                except ValueError:
                    continue
        except (ValueError, OSError):
            pass
        finally:
            events.put({"kind": "error", "message": "推理模型已退出，请重新加载模型"})

    def _receive(self, request_id=None, *, process, events, deadline, response_timeout=None):
        response_deadline = time.monotonic() + response_timeout if response_timeout else deadline
        while True:
            if self.process is not process:
                raise ValueError("推理模型已卸载或加载已取消，请重新加载模型")
            now = time.monotonic()
            if request_id and self.cancel_deadline is not None and now >= self.cancel_deadline:
                raise ValueError("停止请求超时，已终止后台进程；请重新加载模型")
            if now >= deadline:
                raise ValueError("语音合成超过总时长限制，已释放模型；请缩短文本或调整超时设置" if request_id else "推理模型加载超时，已释放资源；请检查模型文件及后台日志")
            if now >= response_deadline:
                raise ValueError("后台长时间没有返回合成结果，已释放模型；请检查后台日志")
            try:
                event = events.get(timeout=min(.25, deadline - now, response_deadline - now))
            except queue.Empty:
                if process.poll() is not None:
                    raise ValueError("推理后台进程已退出，请查看后台日志并重新加载模型")
                continue
            if request_id and event.get("id") not in {None, request_id}:
                continue
            if event["kind"] == "error":
                raise ValueError(event["message"])
            return event

    def load(self, spec):
        if not self.lock.acquire(blocking=False):
            raise ValueError("模型正在加载或合成，请等待结束，或先卸载模型")
        lease = None
        try:
            timeout = timeout_seconds("GPT_SOVITS_TTS_LOAD_TIMEOUT", 600)
            lease = gpu_lease()
            self.close()
            directory = ROOT / "TEMP" / "tts-workspace"
            directory.mkdir(parents=True, exist_ok=True)
            config_path = directory / f"model-{uuid.uuid4().hex}.json"
            config_path.write_text(
                json.dumps(spec, ensure_ascii=False), encoding="utf-8"
            )
            self.config_path = config_path
            environment = os.environ.copy()
            environment.update(
                GPT_SOVITS_HEADLESS="1",
                BROWSER="true",
                language=spec["language"],
                version=spec["version"],
                gpt_path=spec["gpt_path"],
                sovits_path=spec["sovits_path"],
                bert_path=spec["bert_path"],
                cnhubert_base_path=spec["cnhubert_path"],
                _CUDA_VISIBLE_DEVICES=spec["gpu"],
                CUDA_VISIBLE_DEVICES=spec["gpu"],
            )
            (ROOT / "logs").mkdir(exist_ok=True)
            with (ROOT / "logs" / "tts_workspace.log").open("ab") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "tools.tts_workspace.worker",
                        str(config_path),
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            self.process = process
            self.events = queue.Queue()
            threading.Thread(
                target=self._read, args=(process, self.events), daemon=True
            ).start()
            event = self._receive(process=process, events=self.events, deadline=time.monotonic() + timeout)
            if event["kind"] != "loaded":
                raise ValueError("推理模型未完成加载")
            if self.process is not process or process.poll() is not None:
                raise ValueError("模型加载已取消，请重新加载")
            self.metadata, self.spec = event, dict(spec)
            return event
        except Exception:
            if lease is not None:
                self.close()
            raise
        finally:
            try:
                if lease is not None:
                    lease.release()
            finally:
                self.lock.release()

    def _send(self, message):
        with self.write_lock:
            if self.process is None or self.process.poll() is not None:
                raise ValueError("请先加载推理模型")
            self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.process.stdin.flush()

    def generate(self, inputs):
        if not self.lock.acquire(blocking=False):
            raise ValueError("已有加载或合成任务正在运行，请等待结束")
        finished = False
        lease = None
        try:
            total_timeout = timeout_seconds("GPT_SOVITS_TTS_GENERATE_TIMEOUT", 3600)
            response_timeout = timeout_seconds("GPT_SOVITS_TTS_RESPONSE_TIMEOUT", 300)
            lease = gpu_lease()
            if not self.metadata:
                raise ValueError("请先加载推理模型")
            self.generating, self.cancel_requested = True, False
            self.cancel_deadline = None
            process, events = self.process, self.events
            deadline = time.monotonic() + total_timeout
            request_id = uuid.uuid4().hex
            self._send(
                {
                    "command": "generate",
                    "id": request_id,
                    "inputs": inputs,
                    "output_dir": str(ROOT / "output" / "tts-workspace"),
                }
            )
            while True:
                event = self._receive(request_id, process=process, events=events, deadline=deadline, response_timeout=response_timeout)
                if event["kind"] == "done":
                    finished = True
                yield event
                if finished:
                    break
        finally:
            try:
                if self.generating and not finished:
                    # A failed or abandoned stream must not leave GPU work running.
                    self.close()
            finally:
                self.generating = False
                self.cancel_deadline = None
                try:
                    if lease is not None:
                        lease.release()
                finally:
                    self.lock.release()

    def cancel(self):
        if not self.generating:
            return "当前没有正在生成的音频"
        grace = timeout_seconds("GPT_SOVITS_TTS_CANCEL_TIMEOUT", 30)
        if self.cancel_deadline is None:
            self.cancel_deadline = time.monotonic() + grace
        self.cancel_requested = True
        self._send({"command": "cancel"})
        return "已请求停止，当前计算片段结束后停止"


service = InferenceService()
atexit.register(service.close)
