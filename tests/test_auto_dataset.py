import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.auto_dataset import service
from tools.auto_dataset.worker import align_separation, apply_reviews, parse_transcription, text_flags
from tools.gpu_utils import subprocess_gpu_index


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.root_patch = patch.object(service, "ROOT", self.root)
        self.root_patch.start()
        self.upload = self.root / "sample.wav"
        self.upload.write_bytes(b"audio fixture")

    def tearDown(self):
        self.root_patch.stop()
        self.tmp.cleanup()

    def ready_job(self):
        path = Path(service.create_job([self.upload], "测试说话人"))
        clip = path / "clips/clip.wav"
        clip.parent.mkdir()
        clip.write_bytes(b"audio fixture")
        record = dict(id="clip", path=str(clip), speaker="测试说话人", language="zh",
                      text="你好。", selected=True, flags=[], duration=3.0)
        service.atomic_json(path / "metadata.json", [record])
        job = service.read_json(path / "job.json")
        job["status"] = "awaiting_review"
        service.atomic_json(path / "job.json", job)
        service.write_annotations(path, [record])
        return path, record

    def test_separate_jobs_keep_uploads_and_unique_directories(self):
        one = Path(service.create_job([self.upload], "voice"))
        two = Path(service.create_job([self.upload], "voice"))
        self.assertNotEqual(one, two)
        original = service.read_json(one / "job.json")["originals"][0]["path"]
        self.assertEqual(Path(original).read_bytes(), self.upload.read_bytes())

    def test_invalid_speaker_cannot_escape_input_directory(self):
        for speaker in ("../voice", "voice|zh", "voice\n", ""):
            with self.subTest(speaker=speaker), self.assertRaises(ValueError):
                service.create_job([self.upload], speaker)

    def test_approval_validates_language_and_exports_exact_reviewed_text(self):
        path, row = self.ready_job()
        with self.assertRaises(ValueError):
            service.approve(path, [[True, row["id"], "AUTO", "你好", "", 3]])
        train, clips, _, _ = service.approve(path, [[True, row["id"], "en", "Hello.", "", 3]])
        self.assertEqual(Path(train).read_text(), f"{row['path']}|测试说话人|en|Hello.\n")
        self.assertEqual(clips, str(path / "clips"))
        self.assertEqual(service.read_json(path / "job.json")["status"], "reviewed")

    def test_repeated_or_missing_review_rows_cannot_corrupt_dataset(self):
        path, row = self.ready_job()
        data = [True, row["id"], "zh", "你好", "", 3]
        with self.assertRaises(ValueError):
            service.approve(path, [data, data])
        with self.assertRaises(ValueError):
            service.approve(path, [])
        self.assertFalse((path / "train.list").exists())

    def test_external_annotation_edits_and_deletions_import(self):
        path, row = self.ready_job()
        (path / "annotations.list").write_text(f"{row['path']}|ignored|EN|Hello\n", encoding="utf-8")
        _, rows = service.import_annotations(path)
        self.assertEqual(rows[0]["text"], "Hello")
        self.assertEqual(rows[0]["language"], "en")
        self.assertEqual(rows[0]["speaker"], "测试说话人")
        (path / "annotations.list").write_text("", encoding="utf-8")
        _, rows = service.import_annotations(path)
        self.assertFalse(rows[0]["selected"])

    def test_external_audio_path_cannot_enter_training(self):
        path, _ = self.ready_job()
        (path / "annotations.list").write_text(f"{self.upload}|voice|zh|你好\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            service.import_annotations(path)

    def test_external_edit_invalidates_old_training_export_and_keeps_backup(self):
        path, row = self.ready_job()
        service.approve(path, [[True, row["id"], "zh", "你好", "", 3]])
        (path / "annotations.list").write_text(f"{row['path']}|voice|zh|再见\n", encoding="utf-8")
        service.import_annotations(path)
        self.assertFalse((path / "train.list").exists())
        self.assertEqual(len(list(path.glob("train.before-edit-*.list"))), 1)
        self.assertEqual(service.read_json(path / "job.json")["status"], "awaiting_review")

    def test_special_tags_are_removed_without_emotion_metadata(self):
        language, text = parse_transcription("<|zh|><|HAPPY|><|Speech|><|withitn|>大家好。")
        self.assertEqual((language, text), ("zh", "大家好。"))
        self.assertTrue(text_flags("今天去 shopping", "zh"))
        self.assertTrue(text_flags("bonjour", "und"))

    def test_codex_cannot_invent_transcript_or_clear_existing_flags(self):
        rows = [dict(id="a", text="你好", language="zh", flags=["疑似削波"], selected=False)]
        apply_reviews(rows, [dict(id="a", text="你好，欢迎！", language="zh", needs_review=False, reason="")])
        self.assertEqual(rows[0]["text"], "你好")
        self.assertFalse(rows[0]["selected"])
        self.assertIn("疑似削波", rows[0]["flags"])

    def test_codex_must_cover_each_clip_once(self):
        rows = [dict(id="a", text="你好", language="zh", flags=[])]
        with self.assertRaises(ValueError):
            apply_reviews(rows, [])

    def test_separator_rounding_preserves_timestamps_but_rejects_large_drift(self):
        import numpy as np

        audio = np.ones((44100 - 66, 2), dtype=np.float32)
        aligned = align_separation(audio, 44100, 44100)
        self.assertEqual(aligned.shape, (44100, 2))
        self.assertTrue(np.all(aligned[-66:] == 0))
        with self.assertRaises(ValueError):
            align_separation(audio, 50000, 44100)

    def test_child_gpu_visibility_preserves_selected_physical_cards(self):
        self.assertEqual(subprocess_gpu_index(0, "2"), "2")
        self.assertEqual(subprocess_gpu_index(1, "2,0"), "0")
        self.assertEqual(subprocess_gpu_index(1, ""), 1)
        with self.assertRaises(ValueError):
            subprocess_gpu_index(2, "2")


if __name__ == "__main__":
    unittest.main()
