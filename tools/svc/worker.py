"""Cooperative persisted tasks. The browser never owns the compute process."""
import os
from pathlib import Path
import sys
import time
import traceback

from filelock import FileLock, Timeout
import numpy as np
import soundfile as sf

from . import storage as db
from .audio import decode, file_proposals, crop
from tools.workspace_resources import require_space


class Cancelled(Exception):
    pass


class Worker:
    def __init__(self, job_id):
        self.job_id = db.identifier(job_id)
        self.path = db.job_dir(job_id)
        self.job = db.read(self.path / "job.json")
        self.options = self.job["options"]

    def check(self):
        self.check_cancel()
        require_space(self.path, purpose="SVC 音频处理")
        if self.job["kind"] == "train":
            require_space(db.experiment_dir(self.options["experiment_id"]), purpose="SVC 训练")

    def check_cancel(self):
        if (self.path / "cancel").exists():
            raise Cancelled("任务已停止")

    def hash_progress(self, label, begin, end):
        last = None
        def progress(processed, total, filename=None):
            nonlocal last
            now = time.monotonic()
            if last is None or processed >= total or now - last >= .5:
                ratio = processed / total if total else 1
                name = f" · {filename}" if filename else ""
                self.update(f"{label}{name} · {processed / 1024 ** 2:.1f}/{total / 1024 ** 2:.1f} MiB · {ratio:.0%}",
                            begin + (end - begin) * ratio)
                last = now
        return progress

    def hash_file(self, path, label, begin, end):
        self.check()
        return db.digest(path, check_cancel=self.check_cancel, progress=self.hash_progress(label, begin, end))

    def update(self, message, progress=None, **extra):
        print(message, flush=True)
        changes = dict(message=message, **extra)
        if progress is not None:
            changes["progress"] = round(float(progress), 1)
        db.update_job(self.job_id, **changes)

    def separate(self, source, vocals, accompaniment, work):
        from . import separation
        selected = self.options.get("separation_model", "hp2")
        separation.validate(selected)
        if selected != "hp2":
            return separation.separate(self, source, vocals, accompaniment, selected, "分离人声 / 伴奏")
        import torch
        from .install import ROOT
        model = ROOT / "tools/uvr5/uvr5_weights/HP2_all_vocals.pth"
        if not model.exists():
            from huggingface_hub import hf_hub_download
            import shutil
            downloaded = hf_hub_download("lj1995/VoiceConversionWebUI", "uvr5_weights/HP2_all_vocals.pth", endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
            model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, model)
        sys.path.insert(0, str(ROOT / "tools/uvr5"))
        from vr import AudioPre
        separator = AudioPre(agg=10, model_path=str(model), device="cuda:0" if torch.cuda.is_available() else "cpu", is_half=False)
        work.mkdir(parents=True, exist_ok=True)
        for name in ["vocal", "instrument"]:
            (work / name).mkdir(exist_ok=True)
        with sf.SoundFile(source) as src, sf.SoundFile(vocals, "w", samplerate=44100, channels=2, subtype="PCM_16") as voice, sf.SoundFile(accompaniment, "w", samplerate=44100, channels=2, subtype="PCM_16") as inst:
            for start in range(0, len(src), 44100 * 60):
                self.check()
                end = min(len(src), start + 44100 * 60)
                left, right = max(0, start - 44100), min(len(src), end + 44100)
                src.seek(left)
                raw = src.read(right - left, dtype="float32", always_2d=True)
                block = work / "block.wav"
                sf.write(block, raw, 44100, subtype="PCM_16")
                separator._path_audio_(str(block), str(work / "instrument"), str(work / "vocal"), "wav")
                stems = []
                for output, path in [(voice, work / "vocal/vocal_block.wav_10.wav"), (inst, work / "instrument/instrument_block.wav_10.wav")]:
                    samples, rate = sf.read(path, dtype="float32", always_2d=True)
                    if rate != 44100 or samples.shape[1] != 2 or abs(len(samples) - len(raw)) > 44100 * .1:
                        raise ValueError("分离输出时长或采样率异常，已停止以避免时间错位")
                    samples = np.pad(samples[:len(raw)], ((0, max(0, len(raw) - len(samples))), (0, 0)))
                    stems.append((output, samples[start - left:end - left]))
                original = raw[start - left:end - left]
                residual = stems[0][1] + stems[1][1] - original
                source_rms = float(np.sqrt(np.mean(original.astype(np.float64) ** 2)))
                error_rms = float(np.sqrt(np.mean(residual.astype(np.float64) ** 2)))
                if not np.isfinite(error_rms) or error_rms > max(.08, source_rms * 1.5):
                    raise ValueError(f"UVR5 在 {start / 44100:.0f}–{end / 44100:.0f} 秒输出异常，请检查分离模型或改用独立人声与伴奏")
                for output, samples in stems:
                    output.write(samples)
                self.update(f"人声分离 {end / 44100:.1f}/{len(src) / 44100:.1f} 秒")
        del separator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def denoise(self, source, vocals):
        """Use the project's speech denoiser, then normalize its output to SVC audio."""
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except ImportError as error:
            raise ValueError("语音降噪依赖未安装；请运行 python -m tools.svc.install 更新独立 SVC 环境") from error
        from .install import ROOT
        model = ROOT / "tools/denoise-model/speech_frcrn_ans_cirm_16k"
        model = str(model) if model.exists() else "damo/speech_frcrn_ans_cirm_16k"
        raw = self.path / "denoised-model-output.wav"
        raw.unlink(missing_ok=True)
        self.check()
        enhancer = pipeline(Tasks.acoustic_noise_suppression, model=model)
        try:
            enhancer(str(source), output_path=str(raw))
            self.check()
            if not raw.is_file():
                raise ValueError("语音降噪模型没有生成结果，请检查模型安装和后台日志")
            from .audio import decode
            decode(raw, vocals, check_cancel=self.check)
        finally:
            raw.unlink(missing_ok=True)
            del enhancer

    def dataset(self):
        import shutil
        rows = db.read(self.path / "segments.json", [])
        finished = set(db.read(self.path / "prepared_sources.json", []))
        sources = self.options["sources"]
        warnings = []
        for position, source_id in enumerate(sources):
            self.check()
            if source_id in finished:
                continue
            directory = db.source_dir(self.job["singer_id"], source_id)
            source = db.read(directory / "source.json")
            # Each preparation owns its inputs, so later edits always use the
            # exact track that produced the clip, including after a retry.
            from types import SimpleNamespace
            work = self.path / "source-stems" / source_id
            work.mkdir(parents=True, exist_ok=True)
            scoped = SimpleNamespace(path=work, options=self.options, check=self.check,
                                     check_cancel=self.check_cancel, hash_file=self.hash_file, update=self.update)
            decoded, vocals = work / "decoded.wav", work / "vocals.wav"
            self.update(f"解码素材 {position + 1}/{len(sources)}：{source['name']}", 5 + position / len(sources) * 85)
            decode(source["original"], decoded, check_cancel=self.check)
            require_space(work, decoded.stat().st_size * 12 + 64 * 1024 ** 2, "分离、去混响与候选切片")
            tracks = {"source_vocals": str(vocals)}
            if self.options["mixed"]:
                self.separate(decoded, vocals, work / "accompaniment.wav", work / "separation")
                tracks["accompaniment"] = str(work / "accompaniment.wav")
            else:
                shutil.copy2(decoded, vocals)
            def publish():
                manifest = db.read(self.path / "dataset-stems.json", {})
                manifest[source_id] = dict(tracks)
                db.write(self.path / "dataset-stems.json", manifest)
            publish()
            backing = None
            if self.options.get("split_vocals", False):
                from .vocal_split import prepare as split
                vocals, backing = split(scoped, vocals)
                tracks.update(lead_vocals=str(vocals), backing_vocals=str(backing))
                publish()
            from .dereverb import prepare as dry
            if self.options.get("dereverb_lead", False):
                track = "lead" if backing is not None else "vocals"
                vocals, residual = dry(scoped, vocals, track)
                tracks.update({"dry_" + track: str(vocals), track + "_reverb": str(residual)})
                publish()
            if backing is not None and self.options.get("dereverb_backing", False):
                backing, residual = dry(scoped, backing, "backing")
                tracks.update(dry_backing=str(backing), backing_reverb=str(residual))
                publish()
            inputs = [("lead" if backing is not None else "vocals", vocals)]
            if backing is not None and self.options.get("include_backing", False):
                inputs.append(("backing", backing))
            source_rows = []
            clips = self.path / "clips"
            clips.mkdir(exist_ok=True)
            for track, audio in inputs:
                generated, source_warnings = file_proposals(audio, (directory / "lyrics.lrc").read_text(encoding="utf-8"),
                    self.options["offset"], self.options["minimum"], self.options["maximum"],
                    settings=self.options.get("quality_settings"), check_cancel=self.check)
                warnings.extend([source["name"] + "：" + text for text in source_warnings])
                track_rows = []
                for proposal in generated:
                    self.check()
                    clip_id = db.new_id()
                    target = clips / f"{clip_id}-r0.wav"
                    measurements = crop(audio, target, proposal["start"], proposal["end"], self.options.get("quality_settings"))
                    if self.options["mixed"] or backing is not None:
                        proposal["alignment_flags"].append("分离人声需试听确认残留和失真")
                    if track == "backing":
                        proposal["alignment_flags"].append("和声需确认属于目标歌手且为单声部；多声部混合请排除")
                    if self.options.get("dereverb_" + ("backing" if track == "backing" else "lead"), False):
                        proposal["alignment_flags"].append("去混响素材需试听确认齿音、尾音和失真")
                    if proposal["alignment_flags"] and measurements["quality"] == "passed":
                        measurements["quality"] = "needs_review"
                    label = {"lead": "主唱", "backing": "和声", "vocals": "人声"}[track]
                    track_rows.append({"id": clip_id, "singer_id": self.job["singer_id"], "source_id": source_id,
                        "source_name": source["name"] + " · " + label, "source_track": track,
                        "preparation_job": self.job_id, "source_audio": audio.name,
                        **proposal, **measurements, "file": str(target), "sha256": db.digest(target),
                        "revision": 0, "review": "pending", "created_at": db.now(), "quality_settings": self.options.get("quality_settings", {})})
                from .review import annotate_source
                from .feedback import profile
                self.update(f"检查{track}切片边界 {position + 1}/{len(sources)}")
                annotate_source(track_rows, audio, check_cancel=self.check,
                                profile=self.options.get("boundary_profile") or profile(self.job["singer_id"]))
                source_rows.extend(track_rows)
            rows = [row for row in rows if row["source_id"] != source_id] + source_rows
            finished.add(source_id)
            with db.lock():
                db.write(self.path / "segments.json", rows)
                db.write(self.path / "prepared_sources.json", sorted(finished))
        self.update(f"已生成 {len(rows)} 个片段，请试听审核后入库", 100, status="awaiting_review", warnings=warnings)

    def train(self):
        from .engine import train
        directory = db.experiment_dir(self.options["experiment_id"])
        try:
            train(self)
        except Exception as error:
            record = db.read(directory / "experiment.json")
            record["status"] = "cancelled" if isinstance(error, Cancelled) else "failed"
            db.write(directory / "experiment.json", record)
            raise

    def convert(self):
        from .convert_cache import prepare
        if self.options.get("model_source") == "imported":
            from .external_models import record, LIBRARY
            from .shallow_diffusion import FILES as DIFFUSION_FILES
            item = record(self.options["model_id"])
            directory = LIBRARY / item["id"]
            files = [
                ("G.pth", self.options["checkpoint_sha256"], 1, 2.5),
                ("config.json", self.options["config_sha256"], 2.5, 3),
            ]
            if self.options.get("shallow_diffusion"):
                saved = self.options.get("diffusion_files") or {}
                if set(saved) != set(DIFFUSION_FILES):
                    raise ValueError("浅扩散文件记录不完整，请重新导入")
                files.extend((relative, saved[key]["sha256"], 3 + index, 4 + index)
                             for index, (key, relative) in enumerate(DIFFUSION_FILES.items()))
            for name, expected, begin, end in files:
                if self.hash_file(directory / name, "校验导入模型", begin, end) != expected:
                    raise ValueError("导入模型文件已更改，请重新导入")
        else:
            from .service import verify_checkpoint
            verify_checkpoint(self.options["experiment_id"], self.options["checkpoint"],
                              check_cancel=self.check_cancel, progress=self.hash_progress("校验检查点", 1, 3))
        vocals = prepare(self)
        sources = {"source_vocals": str(vocals)}
        accompaniment = self.path / "accompaniment.wav"
        if accompaniment.is_file():
            sources["accompaniment"] = str(accompaniment)
        self.update("转换前人声已准备，可在任务中心试听", 10, outputs=sources)
        harmony = None
        if self.options.get("vocal_mode", "off") != "off":
            from .vocal_split import prepare as prepare_vocal_split
            vocals, harmony = prepare_vocal_split(self, vocals)
            sources.update(lead_vocals=str(vocals), backing_vocals=str(harmony))
            self.update("主唱和和声已分离，可在任务中心分别试听", 28, outputs=sources)
        from .dereverb import inputs
        vocals, harmony = inputs(self, vocals, harmony, sources)
        from .engine import convert
        convert(self, vocals, harmony=harmony)
        if self.options.get("output_reverb", {}).get("enabled"):
            from .reverb import execute
            self.update("音色转换完成，开始自动添加输出混响",90)
            execute(self,inline=True)

    def pitch(self):
        from .pitch_job import execute
        execute(self)

    def reverb(self):
        from .reverb import execute
        execute(self)

    def import_model(self):
        from .external_models import import_candidate
        import_candidate(self)

    def report(self):
        from .dataset_report import generate
        def progress(message, percent):
            self.check()
            self.update(message, percent)
        def check_cancel():
            if (self.path / "cancel").exists():
                raise Cancelled("报告分析已停止；已完成的片段缓存保留")
        result = generate(self.job["singer_id"], self.options["predictor"], progress=progress, check_cancel=check_cancel)
        self.update("训练前数据报告已生成", 100, outputs={"report_id": result["id"]})

    def run(self):
        (db.ROOT / "output").mkdir(exist_ok=True)
        gpu_task = self.job["kind"] in {"dataset", "train", "convert", "pitch", "reverb"}
        gpu_lock = FileLock(str(db.ROOT / "output/auto_dataset-gpu.lock")) if gpu_task else None
        started, computing = time.monotonic(), None
        try:
            if gpu_task:
                while True:
                    self.check()
                    try:
                        gpu_lock.acquire(timeout=1)
                        break
                    except Timeout:
                        pass
            self.check()
            computing = time.monotonic()
            if gpu_task:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            self.update("开始处理", 1, status="running")
            getattr(self, self.job["kind"])()
            if self.job["kind"] != "dataset":
                self.update("处理完成", 100, status="completed")
        except Cancelled as error:
            self.update(str(error), status="cancelled")
        except Exception as error:
            traceback.print_exc()
            self.update(f"处理失败：{type(error).__name__}: {error}", status="failed")
        finally:
            metrics = {"elapsed_seconds": round(time.monotonic() - started, 3), "compute_seconds": round(time.monotonic() - computing, 3) if computing else 0}
            if computing and gpu_task:
                import torch
                if torch.cuda.is_available():
                    metrics["cuda_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
                    metrics["cuda_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1)
                    metrics["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                db.write(self.path / "metrics.json", metrics)
                db.write(self.path / f"metrics-attempt-{self.job.get('attempt', 0)}.json", metrics)
                db.update_job(self.job_id, metrics=metrics)
            finally:
                if gpu_lock and gpu_lock.is_locked:
                    gpu_lock.release()


if __name__ == "__main__":
    Worker(sys.argv[1]).run()
