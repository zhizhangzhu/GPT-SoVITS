"""Real WAV/FFmpeg regression checks; all records and deletions live in /tmp."""
import os
import asyncio
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import gradio as gr
import numpy as np
import soundfile as sf
from gradio.state_holder import SessionState

from tools.svc import storage as db, service, history, feedback, preview_cache, ui
from tools.svc.review import annotate_source
from tools.svc.worker import Worker


class ReviewFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="svc-review-flow-")
        self.root = Path(self.temp.name)
        self.patches = [patch.object(db, "DATA", self.root / "data"),
                        patch.object(db, "EXPERIMENTS", self.root / "experiments"),
                        patch.object(service, "start", side_effect=lambda value: value),
                        patch.dict(os.environ, {"GPT_SOVITS_DISK_RESERVE_GIB": "0", "GRADIO_TEMP_DIR": str(self.root / "gradio-cache")})]
        for item in self.patches:
            item.start()
        self.singer = db.add_singer("回归测试歌手")
        rate = 44100
        t = np.arange(rate * 22) / rate
        wave = .1 * np.sin(2 * np.pi * 220 * t)
        for a, b in [(0, .3), (6, 6.5), (12, 12.5), (18, 18.5), (21.5, 22)]:
            wave[int(a * rate):int(b * rate)] = 0
        self.audio = self.root / "source.wav"
        sf.write(self.audio, wave, rate)
        self.task, _ = service.create_dataset(self.singer["id"], [self.audio])
        Worker(self.task).dataset()
        self.rows = service.records(self.task)

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def force_voiced_shared_cut(self):
        rows = service.records(self.task)
        rows[0] = service.recut(rows[0], rows[0]["start"], 11.8, db.job_dir(self.task))
        rows[1] = service.recut(rows[1], 11.8, rows[1]["end"], db.job_dir(self.task))
        source = db.source_dir(self.singer["id"], rows[0]["source_id"]) / "vocals.wav"
        annotate_source(rows, source)
        db.write(db.job_dir(self.task) / "segments.json", rows)
        return rows

    def test_repeated_undo_and_new_branch_preserve_old_audio(self):
        original = self.rows[0]
        archive = {Path(original["file"]): Path(original["file"]).read_bytes()}
        for delta in [.05, .10]:
            service.edit_clip(self.task, original["id"], "bounds", original["start"] + delta, original["end"] - delta)
            row = service.find_clip(self.task, original["id"])[2]
            archive[Path(row["file"])] = Path(row["file"]).read_bytes()
        history.undo(self.task)
        history.undo(self.task)
        restored = service.find_clip(self.task, original["id"])[2]
        self.assertEqual(restored["sha256"], original["sha256"])
        self.assertEqual(restored["review"], "pending")
        self.assertGreater(restored["revision"], 2)
        self.assertIsNone(history.undo_candidate(self.task))
        service.edit_clip(self.task, original["id"], "bounds", original["start"] + .02, original["end"] - .02)
        for path, data in archive.items():
            self.assertEqual(path.read_bytes(), data)

    def test_split_merge_then_undo_twice_restores_ids_and_waveforms(self):
        original = self.rows[0]
        split = service.edit_clip(self.task, original["id"], "split", point=(original["start"] + original["end"]) / 2)
        split_rows = service.records(self.task)[:2]
        service.edit_clip(self.task, split, "merge")
        history.undo(self.task)
        restored = service.records(self.task)[:2]
        self.assertEqual([(r["id"], r["sha256"]) for r in restored], [(r["id"], r["sha256"]) for r in split_rows])
        history.undo(self.task)
        self.assertEqual(service.records(self.task)[0]["id"], original["id"])
        self.assertEqual(service.records(self.task)[0]["sha256"], original["sha256"])
        self.assertEqual(len(service.records(self.task)), len(self.rows))

    def test_shared_cut_recommendation_preserves_sample_coverage(self):
        rows = self.force_voiced_shared_cut()
        before = sum(sf.info(row["file"]).frames for row in rows)
        plan = service.boundary_plan(self.task, rows[1]["id"])
        self.assertIsNotNone(plan["analysis"]["cuts"]["start"]["suggested"])
        service.edit_clip(self.task, rows[1]["id"], "recommend", recommendation=plan)
        changed = service.records(self.task)
        self.assertEqual(changed[0]["end"], changed[1]["start"])
        self.assertNotEqual(changed[1]["start"], 11.8)
        self.assertEqual(sum(sf.info(row["file"]).frames for row in changed), before)
        self.assertTrue(all(row["review"] == "pending" for row in changed))
        history.undo(self.task)
        restored = service.records(self.task)
        self.assertEqual(restored[0]["end"], 11.8)
        self.assertEqual(restored[1]["start"], 11.8)

    def test_stale_recommendation_and_changed_undo_state_do_not_write(self):
        rows = self.force_voiced_shared_cut()
        plan = service.boundary_plan(self.task, rows[1]["id"])
        service.edit_clip(self.task, rows[1]["id"], "bounds", 11.9, rows[1]["end"])
        before = (db.job_dir(self.task) / "segments.json").read_bytes()
        with self.assertRaises(ValueError):
            service.edit_clip(self.task, rows[1]["id"], "recommend", recommendation=plan)
        self.assertEqual((db.job_dir(self.task) / "segments.json").read_bytes(), before)
        changed = service.records(self.task)
        changed[1]["start"] += .01
        db.write(db.job_dir(self.task) / "segments.json", changed)
        before = (db.job_dir(self.task) / "segments.json").read_bytes()
        with self.assertRaises(ValueError):
            history.undo(self.task)
        self.assertEqual((db.job_dir(self.task) / "segments.json").read_bytes(), before)

    def test_undo_revokes_manifest_but_keeps_experiment_snapshot(self):
        for row in self.rows:
            service.edit_clip(self.task, row["id"], "approved")
        service.publish(self.task)
        g, d = self.root / "G.pth", self.root / "D.pth"
        g.write_bytes(b"G")
        d.write_bytes(b"D")
        train = service.create_experiment(self.singer["id"], "测试快照", generator=g, discriminator=d)
        exp = db.read(db.job_dir(train) / "job.json")["options"]["experiment_id"]
        snapshot = db.read(db.experiment_dir(exp) / "dataset_snapshot.json")
        files = {Path(row["snapshot_file"]): Path(row["snapshot_file"]).read_bytes() for row in snapshot["clips"]}
        row = self.rows[0]
        service.edit_clip(self.task, row["id"], "bounds", row["start"] + .05, row["end"] - .05)
        service.edit_clip(self.task, row["id"], "approved")
        service.publish(self.task)
        history.undo(self.task)
        self.assertNotIn(row["id"], {r["id"] for r in db.clips(self.singer["id"])})
        for path, data in files.items():
            self.assertEqual(path.read_bytes(), data)

    def test_feedback_requires_confirmation_and_is_invalidated_by_undo(self):
        row = self.rows[0]
        service.edit_clip(self.task, row["id"], "bounds", row["start"] + .05, row["end"] - .05)
        self.assertEqual(len(feedback.usable_samples(self.singer["id"])[0]), 0)
        service.edit_clip(self.task, row["id"], "approved")
        self.assertEqual(len(feedback.usable_samples(self.singer["id"])[0]), 1)
        service.edit_clip(self.task, row["id"], "pending")
        self.assertEqual(len(feedback.usable_samples(self.singer["id"])[0]), 0)
        service.edit_clip(self.task, row["id"], "approved")
        history.undo(self.task)
        self.assertEqual(len(feedback.usable_samples(self.singer["id"])[0]), 0)
        self.assertEqual(feedback.rows(self.task)[0]["status"], "undone")

    def test_declined_suggestion_only_becomes_label_after_approval(self):
        rows = self.force_voiced_shared_cut()
        plan = service.boundary_plan(self.task, rows[1]["id"])
        service.decline_recommendation(self.task, rows[1]["id"], plan, "保留当前唱字")
        self.assertEqual(len(feedback.usable_samples(self.singer["id"])[0]), 0)
        service.edit_clip(self.task, rows[1]["id"], "approved")
        samples = feedback.usable_samples(self.singer["id"])[0]
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["status"], "confirmed_declined")

    def test_calibration_applies_confirmed_labels_and_rejects_stale_profile(self):
        tasks = [self.task]
        samples, rate = sf.read(self.audio)
        for index in range(4):
            path = self.root / f"calibration-{index}.wav"
            sf.write(path, samples * (.9 ** (index + 1)), rate)
            task, _ = service.create_dataset(self.singer["id"], [path])
            Worker(task).dataset()
            tasks.append(task)
        for task in tasks:
            rows = service.records(task)
            rows[0] = service.recut(rows[0], rows[0]["start"], 11.1, db.job_dir(task))
            rows[1] = service.recut(rows[1], 11.1, rows[1]["end"], db.job_dir(task))
            db.write(db.job_dir(task) / "segments.json", rows)
            plan = service.boundary_plan(task, rows[1]["id"])
            self.assertIsNotNone(plan["analysis"]["cuts"]["start"]["suggested"])
            service.decline_recommendation(task, rows[1]["id"], plan, "确认保留当前切点")
            service.edit_clip(task, rows[1]["id"], "approved")
        result = feedback.calibrate(self.singer["id"])
        self.assertTrue(result["can_apply"])
        self.assertIsNotNone(result["validation_after"])
        selected = service.records(self.task)[1]
        old_plan = service.boundary_plan(self.task, selected["id"])
        feedback.apply(self.singer["id"], result)
        self.assertEqual(feedback.profile(self.singer["id"]), result["candidate"])
        with self.assertRaises(ValueError):
            service.edit_clip(self.task, selected["id"], "recommend", recommendation=old_plan)
        with self.assertRaises(ValueError):
            feedback.apply(self.singer["id"], result)

    def test_source_change_removes_confirmed_label_from_calibration(self):
        row = self.rows[0]
        service.edit_clip(self.task, row["id"], "bounds", row["start"] + .05, row["end"] - .05)
        service.edit_clip(self.task, row["id"], "approved")
        source = db.source_dir(self.singer["id"], row["source_id"]) / "vocals.wav"
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        usable, skipped, _ = feedback.usable_samples(self.singer["id"])
        self.assertEqual(len(usable), 0)
        self.assertEqual(skipped, 1)

    def test_cache_reuse_cleanup_protects_current_revisions_and_history(self):
        row = self.rows[0]
        source = db.source_dir(self.singer["id"], row["source_id"]) / "vocals.wav"
        current = Path(preview_cache.excerpt(self.task, source, 1, 2))
        old = Path(preview_cache.excerpt(self.task, source, 3, 4))
        self.assertEqual(str(current), preview_cache.excerpt(self.task, source, 1, 2))
        os.utime(old, (time.time() - 10 * 86400,) * 2)
        service.edit_clip(self.task, row["id"], "bounds", row["start"] + .05, row["end"] - .05)
        originals = {Path(r["file"]): Path(r["file"]).read_bytes() for r in self.rows + service.records(self.task)}
        preview_cache.clean(self.task, 7, [str(current)])
        self.assertTrue(current.is_file())
        self.assertFalse(old.exists())
        preview_cache.clean(self.task, 0, [str(current)])
        self.assertTrue(current.is_file())
        for path, data in originals.items():
            self.assertEqual(path.read_bytes(), data)

    def test_cache_corrupt_index_recovers_and_ignores_unowned_files(self):
        source = db.source_dir(self.singer["id"], self.rows[0]["source_id"]) / "vocals.wav"
        cached = Path(preview_cache.excerpt(self.task, source, 1, 2))
        (cached.parent / "index.json").write_text("broken json")
        outside = self.root / "outside.wav"
        outside.write_bytes(b"keep")
        unknown = cached.parent / "user-audio.wav"
        unknown.write_bytes(b"keep")
        self.assertIn(cached, preview_cache.entries(self.task))
        preview_cache.clean(self.task, 0)
        self.assertTrue(outside.is_file())
        self.assertTrue(unknown.is_file())
        self.assertFalse(cached.exists())

    def test_actual_ui_callbacks_advance_and_preserve_output_contracts(self):
        with gr.Blocks(analytics_enabled=False) as app:
            ui.build_page()
        def callback(label):
            control = next(component for component in app.blocks.values() if isinstance(component, gr.Button) and component.value == label)
            dependency = next(dep for dep in app.config["dependencies"]
                              if any(target[0] == control._id and target[1] == "click" for target in dep["targets"]))
            return app.fns[dependency["id"]]
        first = self.rows[0]
        inputs = [self.task, first["id"], first["start"], first["end"], 5., "全部", "风险优先", service.boundary_plan(self.task, first["id"])]
        event = callback("确认并下一条")
        result = event.fn(*inputs)
        self.assertEqual(len(result), len(event.outputs))
        current = result[1]["value"]
        processed = asyncio.run(app.postprocess_data(event, list(result), SessionState(app)))
        self.assertEqual(len(processed), len(result))
        self.assertEqual(processed[1]["value"], current)
        self.assertNotEqual(current, first["id"])
        second = service.find_clip(self.task, current)[2]
        inputs = [self.task, current, second["start"], second["end"], 15., "全部", "风险优先", service.boundary_plan(self.task, current)]
        event = callback("排除并下一条")
        result = event.fn(*inputs)
        self.assertEqual(len(result), len(event.outputs))
        self.assertIsNone(result[1]["value"])
        self.assertEqual(history.progress(self.task), {"total": 2, "approved": 1, "rejected": 1, "pending": 0})

    def test_ui_undo_serializes_restored_audio_and_revision_comparison(self):
        original = self.rows[0]
        service.edit_clip(self.task, original["id"], "bounds", original["start"] + .05, original["end"] - .05)
        with gr.Blocks(analytics_enabled=False) as app:
            ui.build_page()
        event = next(event for event in app.fns.values() if event.fn and event.fn.__name__ == "undo_edit")
        result = event.fn(self.task, "已确认", "风险优先")
        self.assertEqual(len(result), len(event.outputs))
        self.assertEqual(result[-1]["value"], "待复核")
        self.assertEqual(service.find_clip(self.task, original["id"])[2]["sha256"], original["sha256"])
        processed = asyncio.run(app.postprocess_data(event, list(result), SessionState(app)))
        self.assertEqual(len(processed), len(result))

    def test_training_report_ui_callbacks_clear_stale_results(self):
        for row in self.rows:
            service.edit_clip(self.task, row["id"], "approved")
        service.publish(self.task)
        with gr.Blocks(analytics_enabled=False) as app:
            ui.build_page()
        event = next(event for event in app.fns.values() if event.fn and event.fn.__name__ == "build_report")
        result = event.fn(self.singer["id"], "dio")
        self.assertEqual(len(result), len(event.outputs))
        self.assertIn("后台分析", result[1])
        report_job = result[0]["value"]
        self.assertEqual(db.read(db.job_dir(report_job) / "job.json")["kind"], "report")
        processed = asyncio.run(app.postprocess_data(event, list(result), SessionState(app)))
        self.assertEqual(len(processed), len(result))
        service.generate_dataset_report(self.singer["id"], "dio")
        event = next(event for event in app.fns.values() if event.fn and event.fn.__name__ == "open_report")
        opened = event.fn(self.singer["id"], "dio")
        self.assertEqual(len(opened), len(event.outputs))
        self.assertTrue(opened[0]["open"])
        asyncio.run(app.postprocess_data(event, list(opened), SessionState(app)))
        row = self.rows[0]
        service.edit_clip(self.task, row["id"], "bounds", row["start"] + .05, row["end"] - .05)
        event = next(event for event in app.fns.values() if event.fn and event.fn.__name__ == "report_view")
        result = event.fn(self.singer["id"], "dio")
        self.assertEqual(len(result), len(event.outputs))
        self.assertIn("过期", result[0])
        self.assertIsNone(result[-1])

    def test_report_problem_opens_original_review_clip(self):
        first = self.rows[0]
        service.edit_clip(self.task, first["id"], "bounds", first["start"], first["start"] + 2)
        for row in service.records(self.task):
            service.edit_clip(self.task, row["id"], "approved")
        service.publish(self.task)
        report = service.generate_dataset_report(self.singer["id"], "dio")
        self.assertIn(first["id"], {row["id"] for row in report["problem_clips"]})
        with gr.Blocks(analytics_enabled=False) as app:
            ui.build_page()
        event = next(item for item in app.fns.values() if item.fn and item.fn.__name__ == "open_report_problem")
        result = event.fn(self.singer["id"], first["id"])
        self.assertEqual(len(result), len(event.outputs))
        self.assertEqual(result[0]["selected"], "dataset")
        self.assertEqual(result[2]["value"], self.task)
        self.assertEqual(result[6]["value"], first["id"])
        processed = asyncio.run(app.postprocess_data(event, list(result), SessionState(app)))
        self.assertEqual(len(processed), len(result))

    def test_checkpoint_comparison_ui_uses_same_fixed_clip(self):
        experiment = db.new_id()
        directory = db.experiment_dir(experiment)
        (directory / "checkpoints").mkdir(parents=True)
        (directory / "previews").mkdir()
        db.write(directory / "experiment.json", {"id": experiment, "name": "对比实验", "warning": "",
                                                 "singer_id": self.singer["id"], "created_at": db.now()})
        selected = {"id": self.rows[0]["id"], "file": self.rows[0]["file"], "source_name": "素材 A",
                    "median_note": 57., "duration": 5.}
        db.write(directory / "previews/selection.json", [selected])
        for step in [100, 200]:
            (directory / f"checkpoints/G_{step}.pth").write_bytes(b"fixture")
            (directory / f"checkpoints/D_{step}.pth").write_bytes(b"fixture")
            (directory / f"checkpoints/state_{step}.pth").write_bytes(b"fixture")
            sf.write(directory / f"previews/step-{step}-{selected['id']}.wav", np.zeros(4410), 44100)
        db.write(directory / "checkpoints/latest.json", {"step": 200})
        sf.write(directory / f"previews/reference-{selected['id']}.wav", np.zeros(4410), 44100)
        with gr.Blocks(analytics_enabled=False) as app:
            ui.build_page()
        details = next(item for item in app.fns.values() if item.fn and item.fn.__name__ == "experiment_details")
        values = details.fn(experiment)
        self.assertEqual(len(values), len(details.outputs))
        self.assertEqual(values[3]["value"], selected["id"])
        event = next(item for item in app.fns.values() if item.fn and item.fn.__name__ == "load_checkpoint_preview")
        result = event.fn(experiment, values[1]["value"], values[2]["value"], selected["id"])
        self.assertEqual(len(result), len(event.outputs))
        self.assertTrue(all(Path(path).is_file() for path in result[:3]))
        processed = asyncio.run(app.postprocess_data(event, list(result), SessionState(app)))
        self.assertEqual(len(processed), len(result))


if __name__ == "__main__":
    unittest.main()
