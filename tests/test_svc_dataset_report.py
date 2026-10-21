"""Dataset coverage statistics and real isolated-runtime F0/report generation."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tools.svc import storage as db, service, dataset_report as report
from tools.svc.audio import quality
from tools.svc.worker import Worker


class DatasetReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="svc-report-")
        self.root = Path(self.temp.name)
        self.patches = [patch.object(db, "DATA", self.root / "data"), patch.object(db, "EXPERIMENTS", self.root / "experiments"),
                        patch.dict(os.environ, {"GPT_SOVITS_DISK_RESERVE_GIB": "0"})]
        for item in self.patches:
            item.start()
        self.singer = db.add_singer("报告测试 <歌手>")

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def publish_fixture(self):
        directory = db.DATA / "datasets" / self.singer["id"] / "clips"
        directory.mkdir(parents=True)
        rows = []
        for index, frequency in enumerate([220, 440]):
            t = np.arange(44100 * 3) / 44100
            wave = sum(amplitude * np.sin(2 * np.pi * frequency * harmonic * t)
                       for harmonic, amplitude in [(1, .12), (2, .06), (3, .03)])
            wave[:4410] = 0
            wave[-4410:] = 0
            path = directory / f"fixture-{index}.wav"
            sf.write(path, wave, 44100, subtype="PCM_16")
            rows.append({"id": db.new_id(), "singer_id": self.singer["id"], "source_id": db.new_id(),
                         "source_name": f"录音 {index + 1}", "file": str(path), "sha256": db.digest(path),
                         "start": 0., "end": 3., "review": "approved", **quality(wave, 44100)})
        db.write_manifest(self.singer["id"], rows)
        return rows

    def test_empty_dataset_cannot_generate_report(self):
        with self.assertRaises(ValueError):
            service.generate_dataset_report(self.singer["id"])

    def test_distribution_uses_valid_clips_and_duration_weighted_source_share(self):
        rows = [{"id": "a", "source_id": "one", "source_name": "长录音"},
                {"id": "b", "source_id": "two", "source_name": "短录音"},
                {"id": "invalid", "source_id": "two"}]
        data = {"a": {"duration": 12., "voiced_frames": 8, "total_frames": 10, "pitch_counts": {57: 8}},
                "b": {"duration": 2., "voiced_frames": 2, "total_frames": 10, "pitch_counts": {69: 2}}}
        result = report.summarize(rows, data, [{"id": "invalid", "reason": "bad file"}], "dio")
        self.assertEqual(result["clips"], 3)
        self.assertEqual(result["valid_clips"], 2)
        self.assertEqual(result["duration_seconds"], 14.)
        self.assertAlmostEqual(result["sources"][0]["share"], 12 / 14)
        self.assertEqual(result["voiced_ratio"], .5)
        self.assertEqual(result["lengths"]["<3 s"], 1)
        self.assertEqual(result["lengths"]["10–15.5 s"], 1)
        self.assertEqual(result["pitch_summary"]["p50"]["name"], "A3")
        self.assertTrue(result["warnings"])

    def test_pitch_histogram_excludes_unvoiced_and_invalid_frames(self):
        path = self.root / "clip.wav"
        sf.write(path, np.ones(44100 * 2, dtype=np.float32) * .1, 44100)
        class Predictor:
            def compute_f0_uv(self, wave):
                size = len(wave) // report.HOP
                f0 = np.full(size, 220.)
                uv = np.ones(size)
                uv[:10] = 0
                f0[10] = np.nan
                f0[11] = 0
                return f0, uv
        result = report.analyze_clip(path, Predictor())
        self.assertEqual(result["voiced_frames"], result["total_frames"] - 12)
        self.assertEqual(result["pitch_counts"], {57: result["voiced_frames"]})

    def test_real_dio_report_cache_exports_and_stale_detection(self):
        rows = self.publish_fixture()
        first = service.generate_dataset_report(self.singer["id"], "dio")
        self.assertEqual(first["valid_clips"], 2)
        self.assertGreater(first["voiced_ratio"], .5)
        self.assertEqual(first["pitch_summary"]["p5"]["name"], "A3")
        self.assertEqual(first["pitch_summary"]["p95"]["name"], "A4")
        loaded, directory, stale = report.load(self.singer["id"])
        self.assertFalse(stale)
        self.assertEqual(loaded["id"], first["id"])
        for filename in ["report.html", "report.json", "sources.csv", "pitch.png", "lengths.png", "sources.png"]:
            self.assertGreater((directory / filename).stat().st_size, 0)
        html = (directory / "report.html").read_text()
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("&lt;歌手&gt;", html)
        cache_path = directory.parent.parent / "clip-cache.json"
        before = cache_path.read_bytes()
        second = service.generate_dataset_report(self.singer["id"], "dio")
        self.assertEqual(second["pitch_counts"], first["pitch_counts"])
        self.assertEqual(cache_path.read_bytes(), before)
        self.assertNotEqual(second["id"], first["id"])
        self.assertTrue((directory / "report.json").is_file())
        db.write_manifest(self.singer["id"], rows[:1])
        self.assertTrue(report.load(self.singer["id"])[2])

    def test_modified_published_file_is_marked_invalid_and_not_counted(self):
        rows = self.publish_fixture()
        original = Path(rows[0]["file"]).read_bytes()
        sf.write(rows[0]["file"], np.zeros(44100 * 3), 44100)
        result = service.generate_dataset_report(self.singer["id"], "dio")
        self.assertEqual(result["valid_clips"], 1)
        self.assertEqual(len(result["invalid"]), 1)
        self.assertEqual(result["duration_seconds"], 3.)
        self.assertFalse(report.load(self.singer["id"])[2])
        Path(rows[0]["file"]).write_bytes(original)
        self.assertTrue(report.load(self.singer["id"])[2])

    def test_report_supports_all_three_training_predictors(self):
        self.publish_fixture()
        for name in ["dio", "harvest", "pm"]:
            with self.subTest(predictor=name):
                result = service.generate_dataset_report(self.singer["id"], name)
                self.assertEqual(result["predictor"], name)
                self.assertEqual(result["valid_clips"], 2)
                self.assertGreater(result["voiced_frames"], 0)

    def test_missing_published_file_is_reported_and_new_file_invalidates_report(self):
        rows = self.publish_fixture()
        removed = Path(rows[0]["file"])
        original = removed.read_bytes()
        removed.unlink()
        result = service.generate_dataset_report(self.singer["id"])
        self.assertEqual(result["valid_clips"], 1)
        self.assertFalse(report.load(self.singer["id"])[2])
        removed.write_bytes(original)
        self.assertTrue(report.load(self.singer["id"])[2])

    def test_background_report_stop_retry_and_corrupt_cache_rebuild(self):
        self.publish_fixture()
        task = service.new_job("report", self.singer["id"], predictor="dio")["id"]
        def stop_during_analysis(*args, progress, check_cancel):
            progress("分析片段 1/2", 15)
            (db.job_dir(task) / "cancel").touch()
            check_cancel()
        with patch.object(report, "generate", side_effect=stop_during_analysis):
            Worker(task).run()
        self.assertEqual(service.snapshot(task)[0]["status"], "cancelled")
        self.assertGreaterEqual(service.snapshot(task)[0]["progress"], 15)
        retried = service.retry(task)
        self.assertEqual(retried, task)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            record, _ = service.snapshot(task)
            if record["status"] in service.TERMINAL:
                break
            time.sleep(.2)
        self.assertEqual(record["status"], "completed", record["message"])
        self.assertEqual(record["outputs"]["report_id"], report.load(self.singer["id"])[0]["id"])
        self.assertEqual(service.create_report(self.singer["id"], "dio"), task)
        cache_path = db.DATA / "datasets" / self.singer["id"] / "analysis/clip-cache.json"
        cache_path.write_text("{damaged", encoding="utf-8")
        service.retry(task)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            record, _ = service.snapshot(task)
            if record["status"] in service.TERMINAL:
                break
            time.sleep(.2)
        self.assertEqual(record["status"], "completed", record["message"])
        self.assertEqual(report.load(self.singer["id"])[0]["valid_clips"], 2)
        self.assertIsInstance(db.read(cache_path), dict)


if __name__ == "__main__":
    unittest.main()
