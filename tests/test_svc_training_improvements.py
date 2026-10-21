"""Meaningful recovery, fixed-preview and clip-diagnostic checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from tools.svc import dataset_report, engine, storage as db
from tools.svc.training_order import epoch_order, resumed_order


class TrainingImprovementTests(unittest.TestCase):
    def test_epoch_shuffle_recovers_exact_remaining_batches(self):
        first = epoch_order(24, 1234, 1)
        self.assertEqual(set(first), set(range(24)))
        self.assertNotEqual(first, list(range(24)))
        self.assertNotEqual(first, epoch_order(24, 1234, 2))
        self.assertEqual(first, epoch_order(24, 1234, 1))
        saved = {"epoch": 1, "next_batch": 2, "sample_order": first}
        self.assertEqual(resumed_order(24, 1234, 1, saved)[8:], first[8:])
        self.assertEqual(resumed_order(24, 1234, 2, saved), epoch_order(24, 1234, 2))
        self.assertEqual(resumed_order(24, 1234, 1, {"epoch": 1}), list(range(24)))
        with self.assertRaises(ValueError):
            resumed_order(24, 1234, 1, {"epoch": 1, "sample_order": first[:-1] + [first[0]]})

    def test_validation_selection_spreads_sources_and_is_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "filelists").mkdir()
            paths, rows = [], []
            for index, (source, frequency) in enumerate([("one", 220), ("one", 330), ("two", 440), ("three", 660)]):
                path = root / f"{index}.wav"
                sf.write(path, np.zeros(4410), 44100)
                np.save(str(path) + ".f0.npy", np.array([np.full(10, frequency), np.ones(10)], dtype=object))
                paths.append(str(path))
                rows.append({"id": f"clip-{index}", "snapshot_file": str(path), "source_id": source,
                             "source_name": source, "duration": 4.})
            (root / "filelists/val.txt").write_text("\n".join(paths) + "\n")
            selected = engine.validation_previews(root, {"clips": rows})
            self.assertEqual(len(selected), 3)
            self.assertEqual(len({row["source_id"] for row in selected}), 3)
            self.assertEqual(selected, db.read(root / "previews/selection.json"))
            np.save(paths[0] + ".f0.npy", np.array([np.full(10, 1000), np.ones(10)], dtype=object))
            self.assertEqual(engine.validation_previews(root, {"clips": rows}), selected)

    def test_report_flags_specific_clips_without_declaring_audio_bad(self):
        rows = [{"id": "quiet", "source_id": "one", "source_name": "第一首", "start": 1., "end": 3.},
                {"id": "high", "source_id": "two", "source_name": "第二首", "start": 4., "end": 8.}]
        analyses = {"quiet": {"duration": 2., "total_frames": 100, "voiced_frames": 10,
                               "pitch_counts": {57: 10}, "near_low_frames": 0, "near_high_frames": 0, "octave_jumps": 0},
                    "high": {"duration": 4., "total_frames": 100, "voiced_frames": 80,
                             "pitch_counts": {83: 80}, "near_low_frames": 0, "near_high_frames": 20, "octave_jumps": 5}}
        result = dataset_report.summarize(rows, analyses, [], "dio")
        by_id = {row["id"]: row for row in result["problem_clips"]}
        self.assertIn("quiet", by_id)
        self.assertIn("high", by_id)
        self.assertTrue(any("有声 F0" in text for text in by_id["quiet"]["reasons"]))
        self.assertTrue(any("音高跳变" in text for text in by_id["high"]["reasons"]))
        self.assertEqual(result["valid_clips"], 2)


if __name__ == "__main__":
    unittest.main()
