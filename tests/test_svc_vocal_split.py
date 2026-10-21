"""Lead/backing separation, independent conversion and task-center regressions."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from tools.svc import convert_cache, engine, external_models, service, storage as db, ui, vocal_split
from tools.svc.worker import Cancelled


class VocalSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="svc-stems-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def wav(self, name, samples, rate=44100):
        path = self.root / name
        sf.write(path, samples, rate, subtype="FLOAT")
        return path

    def worker(self, folder=None):
        path = folder or self.root
        return SimpleNamespace(path=path, check=lambda: None, check_cancel=lambda: None,
                               hash_file=lambda path, *args: db.digest(path), update=lambda *args, **kw: None)

    def test_overlap_separation_preserves_stereo_edges_and_short_audio(self):
        class Model:
            def __call__(self, signal):
                return torch.stack([signal * .75, signal * .25], dim=1)
        rng = np.random.default_rng(4)
        with patch.object(vocal_split, "WINDOW", 256), patch.object(vocal_split, "OVERLAP", 4):
            for length in [1, 137, 1234]:
                samples = rng.normal(0, .1, (length, 2)).astype(np.float32)
                result = vocal_split._predict(Model(), samples, torch.device("cpu"), lambda: None, lambda done: None)
                self.assertEqual(result.shape, (2, 2, length))
                np.testing.assert_allclose(result[0].T, samples * .75, atol=1e-6)
                np.testing.assert_allclose(result[1].T, samples * .25, atol=1e-6)

    def test_split_cache_repairs_pair_and_never_commits_cancelled_output(self):
        source = self.wav("source.wav", np.ones((4411, 2), dtype=np.float32) * .1)
        worker = self.worker()
        calls = []
        def separate(w, src, lead, backing):
            calls.append(1)
            data, rate = sf.read(src, always_2d=True)
            sf.write(lead, data * .75, rate, subtype="FLOAT")
            sf.write(backing, data * .25, rate, subtype="FLOAT")
        with patch.object(vocal_split, "separate", side_effect=separate), patch.object(vocal_split, "require_space"):
            lead, backing = vocal_split.prepare(worker, source)
            self.assertEqual(vocal_split.prepare(worker, source), (lead, backing))
            self.assertEqual(len(calls), 1)
            backing.write_bytes(b"corrupt")
            vocal_split.prepare(worker, source)
            self.assertEqual(len(calls), 2)
            source.write_bytes(b"changed")
            self.assertNotEqual(db.digest(source), db.read(self.root / "vocal-split.json")["key"]["source_sha256"])
        cancelled = self.root / "cancelled"
        cancelled.mkdir()
        source = self.wav("valid.wav", np.ones((100, 2), dtype=np.float32) * .1)
        def cancel(w, src, lead, backing):
            lead.write_bytes(b"partial")
            raise Cancelled("stop")
        with patch.object(vocal_split, "separate", side_effect=cancel), patch.object(vocal_split, "require_space"):
            with self.assertRaises(Cancelled):
                vocal_split.prepare(self.worker(cancelled), source)
        self.assertFalse((cancelled / "vocal-split.json").exists())
        self.assertFalse((cancelled / "lead_vocals.part.wav").exists())

    def test_three_conversion_policies_keep_independent_pitch_and_mix_alignment(self):
        length = 44100 * 2 + 123
        lead = self.wav("lead.wav", np.full((length, 2), .1, dtype=np.float32))
        backing = self.wav("backing.wav", np.tile([.2, .3], (length, 1)).astype(np.float32))
        calls = []
        class Predictor:
            def compute_f0_uv(self, samples):
                value = 100. if samples.mean() < .15 else 200.
                calls.append(value)
                return np.full(10, value), np.ones(10)
        class Model:
            def eval(self):
                return self
            def infer(self, content, pitch, uv, **kwargs):
                return torch.full((1, 1, 44100 * 4), float(pitch.mean()) / 1000), None
        class Diffusion:
            k_step_max = 100
            def __call__(self, *args, **kwargs):
                return kwargs["gt_spec"]
        vocoder = SimpleNamespace(extract=lambda audio, rate: torch.zeros((1, 10, 128)),
                                  infer=lambda mel, pitch: torch.full((1, 44100 * 4), float(pitch.mean()) / 1000))
        config = dict(model=dict(speech_encoder="vec256l9-onnx"), data=dict(sampling_rate=44100,
                      unit_interpolate_mode="nearest", hop_length=512))
        library = self.root / "models"
        db.write(library / "m" / "config.json", config)
        encoder = SimpleNamespace(encoder=lambda wave: torch.zeros((1, 256, 10)))
        utils = SimpleNamespace(repeat_expand_2d=lambda values, frames, mode: torch.zeros((256, frames)))
        utils.Volume_Extractor = lambda hop: SimpleNamespace(extract=lambda audio: torch.ones(10))
        with patch.object(engine, "upstream", return_value=utils), patch.object(engine, "model_for", return_value=Model()), \
             patch.object(engine, "compatible_load"), patch.object(engine, "encoder_for", return_value=encoder), \
             patch.object(engine, "predictor_for", return_value=Predictor()), patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(external_models, "LIBRARY", library), \
             patch("tools.svc.shallow_diffusion.load", return_value=(Diffusion(), vocoder, {"infer": {"speedup": 1, "method": "ddim"}})):
            for mode in ["keep", "convert", "remove"]:
                calls.clear()
                directory = self.root / mode
                directory.mkdir()
                sf.write(directory / "accompaniment.wav", np.full((length, 2), .05), 44100, subtype="FLOAT")
                db.write(directory / "job.json", {"outputs": {"source_vocals": str(lead)}})
                worker = self.worker(directory)
                worker.options = dict(model_source="imported", model_id="m", speaker_id=0,
                                      checkpoint="unused", predictor="dio", chunk=2, transpose=0,
                                      vocal_mode=mode, harmony_transpose=12, harmony_gain=.5,
                                      mixed=True, vocal_gain=1, instrumental_gain=1,
                                      shallow_diffusion=mode == "convert", diffusion_steps=35)
                engine.convert(worker, lead, harmony=backing)
                outputs = db.read(directory / "outputs.json")
                vocals, rate = sf.read(outputs["vocals"], always_2d=True)
                self.assertEqual(rate, 44100)
                self.assertEqual(len(vocals), length)
                if mode == "keep":
                    np.testing.assert_allclose(vocals, np.tile([.2, .25], (length, 1)), atol=5e-5)
                elif mode == "convert":
                    np.testing.assert_allclose(vocals, .3, atol=5e-5)
                    self.assertIn("converted_backing", outputs)
                else:
                    np.testing.assert_allclose(vocals, .1, atol=5e-5)
                self.assertEqual(calls, [100., 100., 200., 200.] if mode == "convert" else [100., 100.])
                mix, _ = sf.read(outputs["mix"], always_2d=True)
                np.testing.assert_allclose(mix, np.broadcast_to(vocals + .05, mix.shape), atol=5e-5)
                self.assertEqual(outputs["source_vocals"], str(lead))

    def test_retry_carries_policy_and_ui_exposes_all_stems_without_poll_reload(self):
        with patch.object(db, "DATA", self.root / "data"), patch.object(service, "require_space"), \
             patch.object(service, "start", side_effect=lambda task: task):
            audio = self.wav("original.wav", np.ones((4411, 2), dtype=np.float32) * .1)
            task = service._queue_conversion(None, {}, audio, True, 0, "dio", 2, "0", 1, 1,
                                            vocal_mode="keep", harmony_transpose=2, harmony_gain=.7)
            db.update_job(task, status="completed")
            new = service.retry_conversion(task, 0, "harvest", 2, "0", 1, 1, "convert")
            options = db.read(db.job_dir(new) / "job.json")["options"]
            self.assertEqual((options["vocal_mode"], options["harmony_transpose"], options["harmony_gain"]), ("convert", 2, .7))
            self.assertEqual(options["cleanup"], "separate")
            outputs = {}
            for key, _, _ in ui.CONVERSION_TRACKS:
                path = db.job_dir(task) / (key + ".wav")
                sf.write(path, np.zeros((4411, 2)), 44100)
                outputs[key] = str(path)
            db.update_job(task, outputs=outputs)
            with gr.Blocks(analytics_enabled=False) as app:
                ui.build_page()
            callbacks = {event.fn.__name__: event for event in app.fns.values() if event.fn}
            poll = callbacks["poll"]
            result = poll.fn(task, {})
            self.assertEqual(len(result), len(poll.outputs))
            for index, (key, _, _) in enumerate(ui.CONVERSION_TRACKS, 2):
                self.assertEqual(result[index]["value"], outputs[key])
                self.assertTrue(result[index]["visible"])
            state_index = next(i for i, component in enumerate(poll.outputs) if isinstance(component, gr.State))
            again = poll.fn(task, result[state_index])
            self.assertTrue(all("value" not in update for update in again[2:2 + len(ui.CONVERSION_TRACKS)]))
            retry = callbacks["open_retry"]
            self.assertEqual(len(retry.fn(task)), len(retry.outputs))
            refresh = callbacks["refresh_workspace"]
            refreshed = refresh.fn(None, None, None, task, None, "全部", "风险优先", None, {}, "dio")
            self.assertEqual(len(refreshed), len(refresh.outputs))

    def test_inherited_stages_are_copied_then_verified_not_blindly_reused(self):
        with patch.object(db, "DATA", self.root / "data"), patch.object(convert_cache, "require_space"):
            old, new = db.new_id(), db.new_id()
            old_dir, new_dir = db.job_dir(old), db.job_dir(new)
            old_dir.mkdir(parents=True)
            new_dir.mkdir(parents=True)
            original = self.wav("source-original.wav", np.full((4411, 2), .1, dtype=np.float32))
            source = old_dir / "decoded.wav"
            import shutil
            shutil.copy2(original, source)
            info = convert_cache.fingerprint(source)
            db.write(old_dir / "convert-stages.json", dict(version=convert_cache.VERSION,
                     input_sha256=db.digest(original), mode="vocals", accompaniment_sha256=None, decoded=info))
            worker = self.worker(new_dir)
            worker.job = dict(id=new, retried_from=old, input=str(original))
            worker.options = dict(mixed=False)
            with patch.object(convert_cache, "decode", side_effect=AssertionError("cache should be reused")):
                self.assertEqual(convert_cache.prepare(worker), new_dir / "decoded.wav")
            self.assertEqual(db.digest(new_dir / "decoded.wav"), db.digest(source))
            (new_dir / "decoded.wav").write_bytes(b"corrupt")
            with patch.object(convert_cache, "decode", side_effect=lambda src, dst, **kw: shutil.copy2(src, dst)) as decode:
                convert_cache.prepare(worker)
                decode.assert_called_once()


if __name__ == "__main__":
    unittest.main()
