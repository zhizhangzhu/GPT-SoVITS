"""Gradio page, built separately so the existing training UI can be reused."""

from pathlib import Path
import time

import gradio as gr

from . import service


HEADERS = ["使用", "片段编号", "语种", "文字（可修改）", "需复核原因", "时长（秒）"]


def table_rows(records):
    return [[r.get("selected", False), str(i + 1), r["language"], r["text"],
             "；".join(r.get("flags", [])), r.get("duration", 0)] for i, r in enumerate(records)]


def history():
    files = sorted((service.ROOT / "output").glob("*/*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    choices = []
    for path in files[:100]:
        try:
            job = service.read_json(path)
            choices.append((f"{job['speaker']} / {job['id']}", str(path.parent)))
        except (OSError, ValueError, KeyError):
            continue
    return choices


def render(value):
    path = service.job_dir(value)
    job, records = service.snapshot(path)
    ready = job["status"] in {"awaiting_review", "reviewed"}
    options = [(f"{i + 1}. {r['text'][:35]} ({r.get('duration', 0)}秒)", r["id"]) for i, r in enumerate(records)]
    log = (path / "worker.log").read_text(encoding="utf-8", errors="replace")[-7000:] if (path / "worker.log").exists() else ""
    downloads = [str(p) for p in [path / "annotations.list", path / "train.list", path / "metadata.json"] if p.exists()]
    status = f"{job['stage']} · {job['progress']}%\n{job['message']}\n任务目录：{path}"
    return (status, gr.update(value=table_rows(records), interactive=ready),
            gr.update(choices=options, value=options[0][1] if options else None), log,
            downloads, gr.update(interactive=ready))


def watch(value, proc):
    while proc.poll() is None:
        yield value, *render(value)
        time.sleep(2)
    yield value, *render(value)


def build_page(busy=None):
    with gr.TabItem("0-全自动数据集获取工具"):
        gr.Markdown("上传单个说话人的音频，填写说话人名，自动完成分离、切分、识别和 Codex 校验。"
                    "支持中文、英文、日语、韩语、粤语。处理后请试听并审核标注，再开始微调。")
        job_state = gr.State("")
        with gr.Row():
            upload = gr.File(label="上传音频（可一次选择多个文件）", file_count="multiple", type="filepath")
            speaker = gr.Textbox(label="说话人名", placeholder="例如 voice_a / 小明")
        with gr.Row():
            start = gr.Button("开始全自动处理", variant="primary")
            retry = gr.Button("重试当前任务")
            cancel = gr.Button("停止当前任务")
        status = gr.Textbox(label="当前步骤与进度", lines=4, interactive=False)
        with gr.Accordion("历史任务 / 页面刷新后继续查看", open=False):
            previous = gr.Dropdown(label="已保存的任务", choices=history())
            reload_history = gr.Button("刷新任务列表")
            load = gr.Button("查看选中任务")
        with gr.Accordion("任务日志", open=False):
            log = gr.Textbox(label="处理日志", lines=10, interactive=False)
        gr.Markdown("**标注审核**：先选择片段试听，再修改表格的文字和语种（zh/en/ja/ko/yue）。"
                    "有复核原因的片段默认不进入训练集；确认后勾选‘使用’。不要修改片段编号或删除表格行。")
        with gr.Row():
            preview = gr.Dropdown(label="选择切片试听", choices=[])
            audio = gr.Audio(label="切片音频", type="filepath", interactive=False)
        table = gr.Dataframe(headers=HEADERS, datatype=["bool", "str", "str", "str", "str", "number"],
                             type="array", value=[], col_count=(6, "fixed"), row_count=(0, "dynamic"), interactive=False)
        with gr.Row():
            save = gr.Button("保存审核并同步到训练页面", variant="primary", interactive=False)
            external = gr.Button("打开原音频标注 WebUI")
            reread = gr.Button("重新读取标注结果")
        editor_info = gr.Markdown()
        downloads = gr.File(label="标注与训练文件", file_count="multiple", interactive=False)

    outputs = [job_state, status, table, preview, log, downloads, save]

    def start_task(files, name):
        if busy and busy():
            raise gr.Error("现有 GPU 训练或数据预处理正在运行，请结束后再开始自动数据集任务")
        try:
            value = service.create_job(files, name)
            proc = service.launch_job(value)
            yield from watch(value, proc)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def retry_task(value):
        if busy and busy():
            raise gr.Error("请等待现有 GPU 训练或预处理结束")
        try:
            yield from watch(value, service.launch_job(value))
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def load_task(value):
        try:
            while True:
                yield value, *render(value)
                if service.snapshot(value)[0]["status"] in service.TERMINAL:
                    break
                time.sleep(2)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def reread_task(value):
        try:
            service.import_annotations(value)
            return render(value)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def preview_audio(value, row_id):
        if not value or not row_id:
            return None
        _, records = service.snapshot(value)
        return next((r["path"] for r in records if r["id"] == row_id), None)

    def stop(value):
        try:
            return service.cancel_job(value)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    start.click(start_task, [upload, speaker], outputs, concurrency_limit=1, concurrency_id="auto_dataset_start")
    retry.click(retry_task, [job_state], outputs, concurrency_limit=1, concurrency_id="auto_dataset_start")
    cancel.click(stop, [job_state], editor_info, queue=False)
    load.click(load_task, previous, outputs)
    reload_history.click(lambda: gr.update(choices=history()), outputs=previous)
    reread.click(reread_task, job_state, outputs[1:])
    preview.change(preview_audio, [job_state, preview], audio, queue=False)
    return {"state": job_state, "status": status, "table": table, "save": save,
            "external": external, "editor_info": editor_info, "downloads": downloads}


def bind_training(page, path_list, inp_text, inp_wav_dir, exp_name, open_editor):
    def save(value, rows):
        try:
            _, records = service.snapshot(value)
            identifiers = {str(i + 1): record["id"] for i, record in enumerate(records)}
            translated = []
            for row in rows:
                number = str(row[1])
                if number not in identifiers:
                    raise ValueError("审核片段编号无效，请重新读取标注结果")
                translated.append([row[0], identifiers[number], *row[2:]])
            annotation, clips, experiment, message = service.approve(value, translated)
            path = service.job_dir(value)
            return (message + "。训练路径已填好，请进入 1A 格式化工具，再到 1B 微调训练。",
                    [str(path / "annotations.list"), annotation, str(path / "metadata.json")],
                    str(path / "annotations.list"), annotation, clips, experiment)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def external(value):
        try:
            path = service.job_dir(value)
            job, _ = service.snapshot(path)
            if job["status"] not in {"awaiting_review", "reviewed"}:
                raise ValueError("请等待自动处理完成")
            return open_editor(str(path / "annotations.list"))
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    page["save"].click(save, [page["state"], page["table"]],
                       [page["status"], page["downloads"], path_list, inp_text, inp_wav_dir, exp_name])
    page["external"].click(external, page["state"], page["editor_info"])
