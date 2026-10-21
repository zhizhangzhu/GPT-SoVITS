"""Review persistence and UI construction checks without loading model weights."""

import ast
import builtins
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr

from tools import assets
from tools.auto_dataset import service, ui
from tools.tts_workspace.ui import build_page as build_tts_workspace
from tools.svc.ui import build_page as build_svc_workspace


ROOT = Path(__file__).resolve().parents[1]


def build_ui(file, model_version="v2Pro"):
    """Execute the actual Blocks declaration with model callbacks stubbed.

    This avoids webui.py's TEMP cleanup and inference model loading. Components,
    event bindings and the review service are the real implementations.
    """
    source = (ROOT / file).read_text(encoding="utf-8")
    tree = ast.parse(source)
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.With) and any(
        isinstance(i.context_expr, ast.Call) and isinstance(i.context_expr.func, ast.Attribute)
        and i.context_expr.func.attr == "Blocks" for i in n.items
    ))
    block.body = [n for n in block.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "launch"
    )]
    loads = {n.id for n in ast.walk(block) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    stores = {n.id for n in ast.walk(block) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    namespace = {name: (lambda *args, **kwargs: None) for name in loads - stores - set(dir(builtins))}
    namespace.update({name: name for name in loads if name.startswith("process_name_")})
    versions = ["v2", "v2Pro", "v2ProPlus", "v3", "v4", "v5dev", "v5turbo"]
    namespace.update(
        gr=gr, css=assets.css, js=assets.js, theme=assets.theme, top_html=assets.top_html,
        workspace_header=assets.workspace_header,
        i18n=lambda text: text, process_info=lambda name, mode: name + " · " + mode,
        build_auto_dataset_page=ui.build_page, bind_auto_dataset_training=ui.bind_training,
        build_tts_workspace=build_tts_workspace,
        build_svc_workspace=build_svc_workspace,
        auto_dataset_busy=lambda include_svc=True: False, open_auto_dataset_editor=lambda path: path,
        version=model_version, model_version=model_version, v3v4set={"v3", "v4", "v5dev", "v5turbo"},
        V5_VERSIONS={"v5dev", "v5turbo"}, gpu_info="测试设备", gpus="0", n_cpu=4,
        default_batch_size=2, default_max_batch_size=6, default_batch_size_s1=2,
        max_sovits_epoch=25, default_sovits_epoch=8, max_sovits_save_every_epoch=25,
        default_sovits_save_every_epoch=4, is_half=False,
        pretrained_gpt_name={v: "GPT_SoVITS/pretrained_models/gpt.ckpt" for v in versions},
        pretrained_sovits_name={v: "GPT_SoVITS/pretrained_models/s2G.pth" for v in versions},
        GPT_names=["example.ckpt"], SoVITS_names=["example.pth"],
        gpt_path="example.ckpt", sovits_path="example.pth",
        name2gpt_path={"example.ckpt": "example.ckpt"}, name2sovits_path={"example.pth": "example.pth"},
        custom_sort_key=lambda name: name, dict_language={"中文": "all_zh", "英文": "en"},
        graph_available=False, flash_attn_supported=False, tts_pipeline=SimpleNamespace(stop=lambda: None),
        asr_dict={"达摩 ASR (中文经典)": {"lang": ["zh", "yue"], "size": ["large"], "precision": ["float32"]}},
        webui_port_infer_tts=9872,
    )
    exec(compile(ast.Module(body=[block], type_ignores=[]), file, "exec"), namespace)
    return namespace


class ReviewWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.root_patch = patch.object(service, "ROOT", self.root)
        self.root_patch.start()
        upload = self.root / "sample.wav"
        upload.write_bytes(b"test fixture")
        self.path = Path(service.create_job([str(upload)], "测试说话人"))
        self.records = []
        for i in range(3):
            clip = self.path / "clips" / f"clip_{i}.wav"
            clip.parent.mkdir(exist_ok=True)
            clip.write_bytes(b"test fixture")
            self.records.append(dict(id=f"clip_{i}", path=str(clip), speaker="测试说话人", language="zh",
                                     text=f"片段 {i}", selected=i != 1, flags=["待试听"] if i == 1 else [], duration=3))
        service.atomic_json(self.path / "metadata.json", self.records)
        job = service.read_json(self.path / "job.json")
        job.update(status="awaiting_review", stage="等待审核", progress=100, message="请审核")
        service.atomic_json(self.path / "job.json", job)
        service.write_annotations(self.path, self.records)

    def tearDown(self):
        self.root_patch.stop()
        self.tmp.cleanup()

    def test_clip_edit_persists_by_stable_id_and_preserves_other_clips(self):
        service.save_review_clip(str(self.path), "clip_1", "Hello.", "en", True)
        rows = service.read_json(self.path / "metadata.json")
        self.assertEqual((rows[1]["text"], rows[1]["language"], rows[1]["selected"]), ("Hello.", "en", True))
        self.assertEqual(rows[0], self.records[0])
        self.assertEqual(rows[2], self.records[2])
        self.assertIn("|en|Hello.", (self.path / "annotations.list").read_text())
        self.assertFalse((self.path / "train.list").exists())

    def test_drafts_allow_excluding_every_clip_but_publication_requires_selection(self):
        for r in self.records:
            service.save_review_clip(str(self.path), r["id"], r["text"], r["language"], False)
        rows = service.read_json(self.path / "metadata.json")
        self.assertFalse(any(r["selected"] for r in rows))
        with self.assertRaises(ValueError):
            service.approve(str(self.path), [[r["selected"], r["id"], r["language"], r["text"]] for r in rows])

    def test_changed_draft_invalidates_training_export_once(self):
        table = [[r["selected"], r["id"], r["language"], r["text"]] for r in self.records]
        service.approve(str(self.path), table)
        service.save_review_clip(str(self.path), "clip_0", "新文本", "zh", True)
        self.assertFalse((self.path / "train.list").exists())
        self.assertEqual(len(list(self.path.glob("train.before-edit-*.list"))), 1)
        self.assertEqual(service.read_json(self.path / "job.json")["status"], "awaiting_review")
        service.save_review_clip(str(self.path), "clip_0", "新文本", "zh", True)
        self.assertEqual(len(list(self.path.glob("train.before-edit-*.list"))), 1)

    def test_unchanged_navigation_preserves_published_training_export(self):
        table = [[r["selected"], r["id"], r["language"], r["text"]] for r in self.records]
        service.approve(str(self.path), table)
        ui.save_current(str(self.path), "clip_0", "片段 0", "zh", True)
        self.assertTrue((self.path / "train.list").exists())
        self.assertEqual(service.read_json(self.path / "job.json")["status"], "reviewed")

    def test_invalid_selected_clip_does_not_write_partial_draft(self):
        with self.assertRaises(ValueError):
            service.save_review_clip(str(self.path), "clip_1", "invalid|text", "en", True)
        with self.assertRaises(ValueError):
            service.save_review_clip(str(self.path), "clip_1", "Hello", "und", True)
        self.assertEqual(service.read_json(self.path / "metadata.json"), self.records)

    def test_filtered_list_retains_original_clip_numbers(self):
        rows = ui.table_rows(self.records, kind="有复核提示")
        self.assertEqual([r[1] for r in rows], ["2"])
        self.assertEqual(ui.filtered_records(self.records, "片段 2")[0][1]["id"], "clip_2")
        self.assertEqual(ui.adjacent_clip(self.records, "clip_0", kind="已选用"), "clip_2")
        self.assertEqual(ui.adjacent_clip(self.records, "clip_2", direction=1), "clip_2")

    def test_user_text_is_escaped_in_html_panels(self):
        job = service.read_json(self.path / "job.json")
        job["speaker"] = '<img src=x onerror="alert(1)">'
        job["message"] = "<script>alert(1)</script>"
        self.assertNotIn("<script>", ui.status_html(job))
        self.assertNotIn("<img", ui.status_html(job))
        self.records[0]["flags"] = ["<script>bad</script>"]
        self.assertNotIn("<script>", ui.clip_details(self.records[0], 0))

    def test_render_handles_empty_filters_without_losing_source_records(self):
        result = ui.render(str(self.path), query="不匹配的查询")
        self.assertEqual(len(result), 25)
        self.assertEqual(len(result[1]), 3)
        self.assertIsNone(result[2])
        self.assertFalse(result[5]["visible"])
        self.assertEqual(result[5]["value"], ui.EMPTY_TABLE)
        self.assertIsNone(result[7])

    def test_main_sidebar_and_training_handoff_build_without_forward_references(self):
        namespace = build_ui("webui.py")
        self.assertEqual(namespace["workspace_navigation"].elem_id, "workbench-nav")
        self.assertEqual(namespace["training_workspace"].selected, "format")
        event = next(f for f in namespace["app"].fns.values() if f.fn and f.fn.__name__ == "save")
        result = event.fn(str(self.path), "clip_1", "Hello", "en", True)
        self.assertTrue(Path(result[3]).is_file())
        self.assertEqual(result[4], str(self.path / "clips"))
        self.assertIn("|en|Hello", Path(result[3]).read_text())

    def test_standard_and_fast_inference_bindings_build_for_supported_versions(self):
        for file in ("GPT_SoVITS/inference_webui.py", "GPT_SoVITS/inference_webui_fast.py"):
            for version in ("v2Pro", "v3", "v5turbo"):
                with self.subTest(file=file, version=version):
                    namespace = build_ui(file, version)
                    self.assertIsInstance(namespace["output"], gr.Audio)
                    self.assertEqual(namespace["text"].lines, 12)
                    self.assertGreater(len(namespace["app"].fns), 2)


if __name__ == "__main__":
    unittest.main()
