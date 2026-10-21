"""Failure recovery at SVC job, cache and UI boundaries using isolated files."""
import asyncio
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
import numpy as np
import soundfile as sf

from tools.svc import convert_cache, service, storage as db, ui
from tools.svc.worker import Cancelled


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="svc-reliability-")
        self.root = Path(self.temp.name)
        self.patches = [patch.object(db, "DATA", self.root / "data"),
                        patch.object(db, "EXPERIMENTS", self.root / "experiments"),
                        patch.dict(os.environ, {"GPT_SOVITS_DISK_RESERVE_GIB": "0", "GRADIO_TEMP_DIR": str(self.root / "gradio")})]
        for item in self.patches:
            item.start()
        self.singer = db.add_singer("测试歌手")

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_report_double_submit_reuses_one_job(self):
        db.write_manifest(self.singer["id"], [{"id": db.new_id()}])
        calls = []
        def slow_start(task):
            time.sleep(.15)
            db.update_job(task, pid=999999)
            calls.append(task)
            return task
        barrier = threading.Barrier(3)
        results = []
        def submit():
            barrier.wait()
            results.append(service.create_report(self.singer["id"], "dio"))
        with patch.object(service, "start", side_effect=slow_start), patch.object(service, "live", side_effect=lambda row: bool(row.get("pid"))):
            threads = [threading.Thread(target=submit) for _ in range(2)]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=5)
            self.assertFalse(service.busy())
        self.assertEqual(len(results), 2)
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(calls, [results[0]])

    def test_manual_validation_preview_and_frozen_snapshot(self):
        clip_dir = db.DATA / "datasets" / self.singer["id"] / "clips"
        clip_dir.mkdir(parents=True)
        rows = []
        for source_index, duration in enumerate([2, 4, 6]):
            source_id = db.new_id()
            for clip_index in range(2):
                path = clip_dir / f"{source_index}-{clip_index}.wav"
                sf.write(path, np.zeros(4410, dtype=np.float32), 44100)
                rows.append({"id": db.new_id(), "source_id": source_id, "source_name": f"素材 {source_index + 1}",
                             "file": str(path), "sha256": db.digest(path), "duration": float(duration),
                             "start": 0., "end": float(duration)})
        db.write_manifest(self.singer["id"], rows)
        selected = rows[2]["source_id"]
        plan = service.validation_plan(self.singer["id"], [selected])
        self.assertEqual({row["source_id"] for row in plan["validation"]}, {selected})
        self.assertEqual(plan["validation_seconds"], 8)
        with gr.Blocks(analytics_enabled=False) as app:
            ui.build_page()
        callback = next(event for event in app.fns.values() if event.fn and event.fn.__name__ == "preview_validation")
        result = callback.fn(self.singer["id"], [selected])
        self.assertEqual(len(result), len(callback.outputs))
        self.assertIn("素材 2", result[1])
        self.assertIn("33.3%", result[1])
        asyncio.run(app.postprocess_data(callback, list(result), SessionState(app)))
        generator, discriminator = self.root / "G.pth", self.root / "D.pth"
        generator.write_bytes(b"G")
        discriminator.write_bytes(b"D")
        with patch.object(service, "start", side_effect=lambda task: task):
            task = service.create_experiment(self.singer["id"], "手动划分", generator=generator,
                                             discriminator=discriminator, validation_source_ids=[selected])
        experiment = db.read(db.job_dir(task) / "job.json")["options"]["experiment_id"]
        snapshot = db.read(db.experiment_dir(experiment) / "dataset_snapshot.json")
        self.assertEqual(snapshot["validation_source_ids"], [selected])
        self.assertEqual(len((db.experiment_dir(experiment) / "filelists/val.txt").read_text().splitlines()), 2)

    def test_conversion_retry_reuses_verified_stages_and_repairs_damage(self):
        audio = self.root / "source.wav"
        samples = np.zeros((44100, 2), dtype=np.float32)
        sf.write(audio, samples, 44100)
        folder = self.root / "job"
        folder.mkdir()
        class FakeWorker:
            path = folder
            job = {"input": str(audio)}
            options = {"mixed": True}
            separations = 0
            def check(self):
                return None
            def update(self, *args):
                return None
            def hash_file(self, path, *args):
                return db.digest(path)
            def separate(self, decoded, vocals, accompaniment, work):
                self.separations += 1
                data, rate = sf.read(decoded, always_2d=True)
                sf.write(vocals, data, rate)
                sf.write(accompaniment, data, rate)
        worker = FakeWorker()
        first = convert_cache.prepare(worker)
        self.assertEqual(worker.separations, 1)
        self.assertEqual(convert_cache.prepare(worker), first)
        self.assertEqual(worker.separations, 1)
        first.write_bytes(b"damaged")
        convert_cache.prepare(worker)
        self.assertEqual(worker.separations, 2)
        (folder / "convert-stages.json").write_text("{damaged", encoding="utf-8")
        convert_cache.prepare(worker)
        self.assertEqual(worker.separations, 3)
        self.assertEqual(sf.info(first).frames, 44100)

    def test_large_digest_cancel_and_checkpoint_cleanup(self):
        target = self.root / "large.bin"
        target.write_bytes(b"a" * (3 * 1024 * 1024))
        progress = []
        def stop():
            if progress and progress[-1] > 1024 * 1024:
                raise Cancelled("停止校验")
        with self.assertRaises(Cancelled):
            db.digest(target, check_cancel=stop, progress=lambda done, total: progress.append(done))
        self.assertGreater(progress[-1], 1024 * 1024)
        experiment = db.new_id()
        checkpoint_dir = db.experiment_dir(experiment) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        db.write(checkpoint_dir.parent / "experiment.json", {"id": experiment, "name": "测试实验",
                 "singer_id": self.singer["id"], "created_at": db.now()})
        for step in [1, 2]:
            for prefix in ["G", "D", "state"]:
                (checkpoint_dir / f"{prefix}_{step}.pth").write_bytes(b"checkpoint")
        db.write(checkpoint_dir / "latest.json", {"step": 2})
        preview = checkpoint_dir.parent / "previews/step-1-example.wav"
        preview.parent.mkdir(parents=True)
        sf.write(preview, np.zeros(4410), 44100)
        selected = str(checkpoint_dir / "G_1.pth")
        with self.assertRaises(ValueError):
            service.delete_checkpoint(experiment, checkpoint_dir / "G_2.pth")
        with gr.Blocks(analytics_enabled=False) as app:
            ui.build_page()
        event = next(event for event in app.fns.values() if event.fn and event.fn.__name__ == "remove_checkpoint")
        rejected = event.fn(experiment, selected, False)
        self.assertEqual(len(rejected), len(event.outputs))
        self.assertTrue((checkpoint_dir / "G_1.pth").exists())
        accepted = event.fn(experiment, selected, True)
        self.assertEqual(len(accepted), len(event.outputs))
        self.assertIn("已删除", accepted[-1])
        asyncio.run(app.postprocess_data(event, list(accepted), SessionState(app)))
        self.assertFalse((checkpoint_dir / "G_1.pth").exists())
        self.assertFalse(preview.exists())
        self.assertTrue((checkpoint_dir / "G_2.pth").exists())

    def test_checkpoint_integrity_is_checked_in_conversion_worker(self):
        experiment = db.new_id()
        directory = db.experiment_dir(experiment)
        checkpoints = directory / "checkpoints"
        checkpoints.mkdir(parents=True)
        db.write(directory / "experiment.json", {"id": experiment, "name": "校验实验",
                 "singer_id": self.singer["id"], "created_at": db.now()})
        indexed = {}
        for key, prefix in (("generator", "G"), ("discriminator", "D"), ("state", "state")):
            path = checkpoints / f"{prefix}_1.pth"
            path.write_bytes(b"valid checkpoint")
            indexed[key] = {"name": path.name, "bytes": path.stat().st_size, "sha256": db.digest(path)}
        db.write(checkpoints / "latest.json", {"step": 1})
        db.write(checkpoints / "checkpoint-index.json", {"1": indexed})
        audio = self.root / "source.wav"
        sf.write(audio, np.zeros(44100), 44100)
        selected = checkpoints / "G_1.pth"
        selected.write_bytes(b"wrong checkpoint")
        self.assertEqual(selected.stat().st_size, indexed["generator"]["bytes"])
        with patch.object(service, "start", side_effect=lambda task: task):
            task = service.create_conversion(experiment, selected, audio)
        from tools.svc.worker import Worker
        with self.assertRaisesRegex(ValueError, "检查点校验失败"):
            Worker(task).convert()


if __name__ == "__main__":
    unittest.main()
