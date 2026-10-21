"""Plan acceptance checks for persistence, review and immutable snapshots."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tools.svc import storage as db, service
from tools.svc.audio import parse_lrc, proposals, quality, crop
from tools.svc.worker import Worker


class SVCDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(db, "DATA", self.root / "data"), patch.object(db, "EXPERIMENTS", self.root / "experiments"),
                        patch.object(service, "start", side_effect=lambda value: value)]
        for item in self.patches:
            item.start()
        self.singer = db.add_singer("测试歌手 / ../../")
        rate = 44100
        t = np.arange(rate * 22) / rate
        samples = .1 * np.sin(2 * np.pi * 220 * t)
        for a, b in [(0, .3), (6, 6.5), (12, 12.5), (18, 18.5), (21.5, 22)]:
            samples[int(a * rate):int(b * rate)] = 0
        self.audio = self.root / "song.wav"
        sf.write(self.audio, samples, rate)

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def prepare(self):
        task, _ = service.create_dataset(self.singer["id"], [self.audio])
        Worker(task).dataset()
        return task

    def test_lrc_repeated_stamps_and_offset(self):
        rows = parse_lrc("[offset:-200]\n[00:01.50][00:20.05]同一句\n[ar:歌手]")
        self.assertEqual([row["time"] for row in rows], [1.3, 19.85])
        self.assertEqual(rows[0]["text"], "同一句")

    def test_identity_rename_and_path_safety(self):
        other = db.add_singer(self.singer["name"])
        self.assertNotEqual(self.singer["id"], other["id"])
        db.rename_singer(self.singer["id"], "新名字")
        self.assertEqual(db.singer(self.singer["id"])["name"], "新名字")
        with self.assertRaises(ValueError):
            db.job_dir("../../etc")

    def test_duplicate_upload_and_resume_do_not_duplicate(self):
        task = self.prepare()
        duplicate, _ = service.create_dataset(self.singer["id"], [self.audio])
        self.assertEqual(task, duplicate)
        before = service.records(task)
        Worker(task).dataset()
        self.assertEqual(before, service.records(task))

    def test_review_publish_and_edit_invalidate_future_manifest(self):
        task = self.prepare()
        rows = service.records(task)
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(db.clips(self.singer["id"]), [])
        for row in rows:
            service.edit_clip(task, row["id"], "approved")
        service.publish(task)
        service.publish(task)
        self.assertEqual(len(db.clips(self.singer["id"])), len(rows))
        selected = rows[0]
        service.edit_clip(task, selected["id"], "bounds", selected["start"] + .05, selected["end"] - .05)
        _, _, edited = service.find_clip(task, selected["id"])
        self.assertEqual(edited["review"], "pending")
        self.assertNotEqual(selected["sha256"], edited["sha256"])
        self.assertNotIn(selected["id"], [row["id"] for row in db.clips(self.singer["id"])])
        lines = (db.DATA / "datasets" / self.singer["id"] / "manifest.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), len(rows) - 1)

    def test_same_singer_accumulates_distinct_uploads_and_splits_by_source(self):
        first = self.prepare()
        second_audio = self.root / "song-2.wav"
        samples, rate = sf.read(self.audio)
        sf.write(second_audio, samples * .9, rate)
        second, _ = service.create_dataset(self.singer["id"], [second_audio])
        Worker(second).dataset()
        for task in [first, second]:
            for row in service.records(task):
                service.edit_clip(task, row["id"], "approved")
            service.publish(task)
        manifest = db.clips(self.singer["id"])
        self.assertEqual(len({row["source_id"] for row in manifest}), 2)
        dummy_g, dummy_d = self.root / "G.pth", self.root / "D.pth"
        dummy_g.write_bytes(b"G")
        dummy_d.write_bytes(b"D")
        task = service.create_experiment(self.singer["id"], "两首录音", generator=dummy_g, discriminator=dummy_d)
        experiment = db.read(db.job_dir(task) / "job.json")["options"]["experiment_id"]
        directory = db.experiment_dir(experiment)
        snapshot = db.read(directory / "dataset_snapshot.json")
        self.assertIn("验证集占总时长", snapshot["warning"])
        sources = {row["snapshot_file"]: row["source_id"] for row in snapshot["clips"]}
        train = {sources[path] for path in (directory / "filelists/train.txt").read_text().splitlines()}
        val = {sources[path] for path in (directory / "filelists/val.txt").read_text().splitlines()}
        self.assertFalse(train & val)

    def test_quality_thresholds_are_recorded_and_reused_for_edits(self):
        settings = {"silence_db": -40, "max_silence_ratio": .2, "max_clipping_ratio": .002}
        task, _ = service.create_dataset(self.singer["id"], [self.audio], quality_settings=settings)
        Worker(task).dataset()
        row = service.records(task)[0]
        self.assertEqual(row["quality_settings"], settings)
        service.edit_clip(task, row["id"], "bounds", row["start"] + .1, row["end"] - .1)
        self.assertEqual(service.records(task)[0]["quality_settings"], settings)

    def test_split_and_merge_are_source_constrained(self):
        task = self.prepare()
        selected = service.records(task)[0]
        split_id = service.edit_clip(task, selected["id"], "split", point=(selected["start"] + selected["end"]) / 2)
        rows = service.records(task)
        self.assertEqual(rows[0]["end"], rows[1]["start"])
        service.edit_clip(task, split_id, "merge")
        merged = service.records(task)[0]
        self.assertAlmostEqual(merged["start"], selected["start"])
        self.assertAlmostEqual(merged["end"], selected["end"])
        rows = service.records(task)
        rows[1]["source_id"] = db.new_id()
        db.write(db.job_dir(task) / "segments.json", rows)
        with self.assertRaises(ValueError):
            service.edit_clip(task, rows[0]["id"], "merge")

    def test_lyric_hints_are_not_claimed_as_alignment(self):
        samples, rate = sf.read(self.audio, dtype="float32")
        rows, warnings = proposals(samples, rate, "[00:04.00]一句\n[99:00.00]错误版本")
        self.assertTrue(warnings)
        self.assertTrue(all(row["alignment"] == "energy_hint" for row in rows))
        self.assertTrue(all(row["alignment_flags"] for row in rows))

    def test_silence_and_bad_boundaries(self):
        self.assertEqual(quality(np.zeros(44100), 44100)["quality"], "rejected")
        with self.assertRaises(ValueError):
            crop(self.audio, self.root / "bad.wav", 5, 4)

    def test_snapshot_survives_edits_and_single_source_warning(self):
        task = self.prepare()
        for row in service.records(task):
            service.edit_clip(task, row["id"], "approved")
        service.publish(task)
        dummy_g, dummy_d = self.root / "G.pth", self.root / "D.pth"
        dummy_g.write_bytes(b"G")
        dummy_d.write_bytes(b"D")
        train_job = service.create_experiment(self.singer["id"], "实验", steps=1, generator=dummy_g, discriminator=dummy_d)
        experiment = db.read(db.job_dir(train_job) / "job.json")["options"]["experiment_id"]
        snapshot = db.read(db.experiment_dir(experiment) / "dataset_snapshot.json")
        self.assertTrue(snapshot["warning"])
        first = snapshot["clips"][0]
        original = Path(first["snapshot_file"]).read_bytes()
        service.edit_clip(task, first["id"], "bounds", first["start"] + .05, first["end"] - .05)
        self.assertEqual(Path(first["snapshot_file"]).read_bytes(), original)

    def test_lyrics_pairing_rejects_unmatched_file(self):
        lrc = self.root / "different.lrc"
        lrc.write_text("[00:01]歌词")
        with self.assertRaises(ValueError):
            service.create_dataset(self.singer["id"], [self.audio], [lrc])


if __name__ == "__main__":
    unittest.main()
