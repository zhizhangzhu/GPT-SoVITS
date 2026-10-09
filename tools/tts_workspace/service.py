"""Lazy model lifecycle and audio generation for the main Gradio application."""

import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[2]


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

    def close(self):
        process, self.process = self.process, None
        self.metadata = self.spec = None
        self.generating = False
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

    def _receive(self, request_id=None):
        while True:
            event = self.events.get()
            if request_id and event.get("id") not in {None, request_id}:
                continue
            if event["kind"] == "error":
                raise ValueError(event["message"])
            return event

    def load(self, spec):
        if not self.lock.acquire(blocking=False):
            raise ValueError("模型正在加载或合成，请等待结束，或先卸载模型")
        try:
            self.close()
            directory = ROOT / "TEMP" / "tts-workspace"
            directory.mkdir(parents=True, exist_ok=True)
            config_path = directory / f"model-{uuid.uuid4().hex}.json"
            config_path.write_text(
                json.dumps(spec, ensure_ascii=False), encoding="utf-8"
            )
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
            event = self._receive()
            if event["kind"] != "loaded":
                raise ValueError("推理模型未完成加载")
            if self.process is not process or process.poll() is not None:
                raise ValueError("模型加载已取消，请重新加载")
            self.metadata, self.spec = event, dict(spec)
            return event
        except Exception:
            self.close()
            raise
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
        try:
            if not self.metadata:
                raise ValueError("请先加载推理模型")
            self.generating, self.cancel_requested = True, False
            request_id = uuid.uuid4().hex
            self._send(
                {
                    "command": "generate",
                    "id": request_id,
                    "inputs": inputs,
                    "output_dir": str(ROOT / "TEMP" / "tts-workspace"),
                }
            )
            while True:
                event = self._receive(request_id)
                if event["kind"] == "done":
                    finished = True
                yield event
                if finished:
                    break
        finally:
            if self.generating and not finished:
                try:
                    self._send({"command": "cancel"})
                except (ValueError, OSError):
                    pass
            self.generating = False
            self.lock.release()

    def cancel(self):
        if not self.generating:
            return "当前没有正在生成的音频"
        self.cancel_requested = True
        self._send({"command": "cancel"})
        return "已请求停止，当前计算片段结束后停止"


service = InferenceService()
atexit.register(service.close)
