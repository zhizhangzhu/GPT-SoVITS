"""Dataset task browser and a persistent audio review workspace."""

from functools import wraps
from html import escape
from pathlib import Path

import gradio as gr

from tools.assets import workspace_header
from . import service


HEADERS = ["选用", "片段", "语种", "标注文字", "质量提示", "时长 / 秒"]
EMPTY_TABLE = [["—", "—", "", "", "", 0]]
LANGUAGE_NAMES = {"zh": "中文", "en": "英文", "ja": "日语", "ko": "韩语", "yue": "粤语"}
FILTERS = ["全部片段", "有复核提示", "已选用", "已排除"]
STATUS_NAMES = {
    "queued": "等待处理", "running": "处理中", "awaiting_review": "待审核",
    "reviewed": "训练清单已生成", "failed": "处理失败", "cancelled": "已停止",
}


def history(query=""):
    files = sorted((service.ROOT / "output").glob("*/*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    choices = []
    for path in files[:100]:
        try:
            job = service.read_json(path)
            label = f"{job['speaker']} · {STATUS_NAMES.get(job['status'], job['status'])}\n{job['id']}"
            if not query or query.casefold() in label.casefold():
                choices.append((label, str(path.parent)))
        except (OSError, ValueError, KeyError):
            continue
    return choices


def filtered_records(records, query="", kind="全部片段"):
    matches = []
    for index, record in enumerate(records):
        if kind == "有复核提示" and not record.get("flags"):
            continue
        if kind == "已选用" and not record.get("selected"):
            continue
        if kind == "已排除" and record.get("selected"):
            continue
        searchable = f"{index + 1} {record['text']} {record['language']} {' '.join(record.get('flags', []))}"
        if query and query.casefold() not in searchable.casefold():
            continue
        matches.append((index, record))
    return matches


def table_rows(records, query="", kind="全部片段"):
    return [["选用" if r.get("selected") else "排除", str(i + 1), LANGUAGE_NAMES.get(r["language"], "待确认"), r["text"],
             "；".join(r.get("flags", [])), r.get("duration", 0)]
            for i, r in filtered_records(records, query, kind)]


def adjacent_clip(records, current, query="", kind="全部片段", direction=1):
    ids = [r["id"] for _, r in filtered_records(records, query, kind)]
    if not ids:
        return None
    position = ids.index(current) if current in ids else 0
    return ids[max(0, min(len(ids) - 1, position + direction))]


def summary_html(records):
    selected = [r for r in records if r.get("selected")]
    duration = sum(float(r.get("duration") or 0) for r in selected)
    metrics = [("总片段", str(len(records))), ("训练选用", str(len(selected))),
               ("质量提示", str(sum(bool(r.get("flags")) for r in records))),
               ("选用时长", f"{duration / 60:.1f} 分钟")]
    return '<div class="review-metrics">' + ''.join(
        f'<div><span>{label}</span><strong>{value}</strong></div>' for label, value in metrics
    ) + '</div>'


def status_html(job):
    progress = max(0, min(100, float(job.get("progress", 0))))
    progress_bar = (f'<progress value="{progress}" max="100" aria-label="当前任务处理进度"></progress>'
                    if job["status"] not in service.TERMINAL else "")
    return (f'<div class="task-status"><strong>{escape(job["speaker"])}</strong>'
            f'<span class="status-label">{escape(STATUS_NAMES.get(job["status"], job["status"]))}</span>'
            f'<p>{escape(job["stage"])} · {progress:g}% · {escape(job["message"])}</p>'
            f'{progress_bar}</div>')


def clip_details(record, index):
    if not record:
        return '<div class="empty-workspace"><strong>选择片段开始试听</strong>完成自动处理后，片段会出现在下方列表中。</div>'
    flags = record.get("flags", [])
    quality = "；".join(flags) if flags else "未发现自动质量提示"
    return (f'<dl class="clip-details"><dt>片段编号</dt><dd>{index + 1:04d}</dd>'
            f'<dt>时长</dt><dd>{float(record.get("duration") or 0):.2f} 秒</dd>'
            f'<dt>音频文件</dt><dd>{escape(Path(record["path"]).name)}</dd></dl>'
            f'<div class="quality-note {"has-flags" if flags else ""}">{escape(quality)}</div>')


def protect(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc
    return wrapped


def save_current(value, current, text, language, selected):
    if value and current:
        job, _ = service.snapshot(value)
        if job["status"] in {"awaiting_review", "reviewed"}:
            service.save_review_clip(value, current, text or "", language or "und", selected)


def render(value, query="", kind="全部片段", current=None):
    path = service.job_dir(value)
    job, records = service.snapshot(path)
    ready = job["status"] in {"awaiting_review", "reviewed"}
    matches = filtered_records(records, query, kind)
    ids = [r["id"] for _, r in matches]
    current = current if current in ids else (ids[0] if ids else None)
    index, record = next(((i, r) for i, r in matches if r["id"] == current), (0, None))
    options = [(f"{i + 1:04d} · {r['text'][:55] or '暂无标注'}", r["id"]) for i, r in matches]
    log_path = path / "worker.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")[-10000:] if log_path.exists() else ""
    files = [str(p) for p in [path / "annotations.list", path / "train.list", path / "metadata.json"] if p.exists()]
    editing = ready and record is not None
    position = ids.index(current) if current in ids else -1
    return (
        value, records, current, status_html(job), summary_html(records),
        gr.update(value=table_rows(records, query, kind) or EMPTY_TABLE, visible=bool(matches)),
        gr.update(choices=options, value=current, interactive=ready and bool(ids)),
        record["path"] if record else None,
        gr.update(value=record["text"] if record else "", interactive=editing),
        gr.update(value=record["language"] if record and record["language"] in service.LANGUAGES else None, interactive=editing),
        gr.update(value=record.get("selected", False) if record else False, interactive=editing),
        ('<div class="empty-workspace"><strong>没有符合筛选的片段</strong>调整筛选条件或搜索文字，再继续审核。</div>'
         if ready and records and not record else clip_details(record, index)), log, files,
        gr.update(interactive=ready and any(r.get("selected") for r in records)),
        gr.update(interactive=editing), gr.update(interactive=editing), gr.update(interactive=editing),
        gr.update(interactive=editing and position > 0),
        gr.update(interactive=editing and position < len(ids) - 1),
        gr.update(selected="review"), gr.update(active=job["status"] not in service.TERMINAL),
        gr.update(choices=history(), value=str(path)), str(path), {"query": query, "kind": kind},
    )


def empty_view():
    def disabled():
        return gr.update(interactive=False)
    return (
        "", [], None,
        '<div class="task-status"><strong>开始建立你的数据集</strong><p>新建任务，或从左侧选择已有任务继续审核。</p></div>',
        summary_html([]), gr.update(value=EMPTY_TABLE, visible=False),
        gr.update(choices=[], value=None, interactive=False), None,
        gr.update(value="", interactive=False), gr.update(value=None, interactive=False),
        gr.update(value=False, interactive=False), clip_details(None, 0), "", [],
        disabled(), disabled(), disabled(), disabled(), disabled(), disabled(),
        gr.update(selected="new"), gr.update(active=False), gr.update(choices=history(), value=None), "",
        {"query": "", "kind": FILTERS[0]},
    )


def build_page(busy=None):
    with gr.TabItem("数据集工作台"):
        job_state = gr.State("")
        records_state = gr.State([])
        current_state = gr.State(None)
        applied_view = gr.State({"query": "", "kind": FILTERS[0]})
        timer = gr.Timer(2, active=False)
        with gr.Row(elem_classes="dataset-layout"):
            with gr.Column(scale=2, min_width=220, elem_classes=["task-rail", "workbench-panel"]):
                gr.HTML('<div class="panel-heading">数据集任务</div>')
                new_task = gr.Button("+ 新建数据集", variant="primary")
                task_search = gr.Textbox(label="搜索任务", placeholder="说话人 / 任务编号，回车搜索", container=False)
                previous = gr.Radio(label="已保存的任务", choices=history(), value=None,
                                    show_label=False, container=False, elem_classes="task-list")
                reload_history = gr.Button("刷新任务列表", size="sm")
                gr.Markdown("任务和审核草稿保存在本地。刷新页面后，选择原任务即可继续。")
            with gr.Column(scale=8, min_width=480, elem_classes="dataset-main"):
                status = gr.HTML('<div class="task-status"><strong>开始建立你的数据集</strong><p>新建任务，或从左侧选择已有任务继续审核。</p></div>')
                with gr.Tabs(selected="new") as workspace:
                    with gr.TabItem("新建任务", id="new"):
                        gr.HTML(workspace_header("准备说话人音频", "每个任务使用同一说话人的素材。上传后自动完成人声分离、切分、识别和标注检查。"))
                        with gr.Row():
                            upload = gr.File(label="音频素材", file_count="multiple", type="filepath")
                            speaker = gr.Textbox(label="说话人名称", placeholder="例如：旁白小林", info="名称用于识别这份数据集，可包含中文。")
                        start = gr.Button("开始自动处理", variant="primary")
                        gr.Markdown("**完成后**：试听片段并修正标注 → 确认训练选用 → 送入训练数据格式化。支持中文、英文、日语、韩语和粤语。")
                    with gr.TabItem("音频审核", id="review"):
                        summary = gr.HTML(summary_html([]))
                        with gr.Row(elem_classes="review-controls"):
                            kind = gr.Dropdown(label="片段筛选", choices=FILTERS, value=FILTERS[0], scale=1, min_width=150)
                            preview = gr.Dropdown(label="当前试听片段", choices=[], interactive=False, scale=2, min_width=220)
                            query = gr.Textbox(label="搜索片段", placeholder="标注 / 编号 / 提示，回车搜索", scale=1, min_width=160)
                            search = gr.Button("搜索", scale=0, min_width=76)
                        with gr.Row():
                            with gr.Column(scale=5, min_width=280, elem_classes="workbench-panel"):
                                gr.HTML('<div class="panel-heading">试听与文本校对</div>')
                                audio = gr.Audio(label="片段音频", type="filepath", interactive=False,
                                                 show_download_button=True, waveform_options={"waveform_color": "#7da6df", "waveform_progress_color": "#2161d4"})
                                text = gr.Textbox(label="标注文字", lines=3, placeholder="选择片段后，根据试听内容修正文字。", interactive=False)
                                with gr.Row():
                                    prev_clip = gr.Button("上一片段", interactive=False, elem_id="review-prev")
                                    next_clip = gr.Button("下一片段", interactive=False, elem_id="review-next")
                                    apply = gr.Button("保存草稿", interactive=False)
                                gr.Markdown("翻页时保存当前草稿。快捷键：Alt + 左右方向键切换；Ctrl / ⌘ + Enter 确认并下一条。")
                            with gr.Column(scale=2, min_width=220, elem_classes=["clip-inspector", "workbench-panel"]):
                                gr.HTML('<div class="panel-heading">片段详情</div>')
                                details = gr.HTML(clip_details(None, 0))
                                language = gr.Dropdown(label="语种", choices=[("中文", "zh"), ("英文", "en"), ("日语", "ja"), ("韩语", "ko"), ("粤语", "yue")], value=None, interactive=False)
                                selected = gr.Checkbox(label="选入训练集", value=False, interactive=False)
                                accept_next = gr.Button("确认并下一条", variant="primary", interactive=False, elem_id="review-save-next")
                                exclude_next = gr.Button("排除并下一条", interactive=False)
                        table = gr.Dataframe(label="片段列表 · 点击任意片段进入试听", headers=HEADERS,
                                             datatype=["str", "str", "str", "str", "str", "number"],
                                             type="array", value=EMPTY_TABLE, visible=False, col_count=(6, "fixed"), row_count=(1, "dynamic"),
                                             interactive=False, height=290, wrap=True, elem_classes="review-table",
                                             column_widths=[60, 65, 65, "42%", "28%", 85])
                        with gr.Row(elem_classes="review-actions"):
                            save = gr.Button("确认审核并送入训练", variant="primary", interactive=False, scale=2)
                            gr.Markdown("保存草稿或切换片段会保存修改。确认审核将生成训练清单，并自动填写训练路径。")
                        editor_info = gr.Markdown()
                    with gr.TabItem("任务详情与文件", id="details"):
                        with gr.Row():
                            retry = gr.Button("重试失败或停止的任务")
                            cancel = gr.Button("停止当前任务", variant="stop")
                        task_path = gr.Textbox(label="任务目录", interactive=False)
                        downloads = gr.File(label="标注与训练文件", file_count="multiple", interactive=False)
                        with gr.Accordion("片段边界、拆分与合并", open=False):
                            gr.Markdown("在标注编辑器中调整边界、拆分或合并片段。保存后回到这里重新读取，再确认审核。")
                            external = gr.Button("打开片段标注编辑器")
                            reread = gr.Button("重新读取标注结果")
                        with gr.Accordion("处理日志", open=False):
                            log = gr.Textbox(label="后台日志", lines=14, interactive=False, elem_classes="process-log")

    outputs = [job_state, records_state, current_state, status, summary, table, preview, audio, text, language,
               selected, details, log, downloads, save, apply, accept_next, exclude_next, prev_clip, next_clip,
               workspace, timer, previous, task_path, applied_view]
    editor_inputs = [job_state, current_state, text, language, selected]
    view_inputs = [query, kind]

    @protect
    def start_task(files, name, query, kind):
        if busy and busy():
            raise ValueError("现有 GPU 训练或数据预处理正在运行，请结束后再开始自动数据集任务")
        value = service.create_job(files, name)
        service.launch_job(value)
        return render(value, query, kind)

    @protect
    def retry_task(value, query, kind, current):
        if busy and busy():
            raise ValueError("请等待现有 GPU 训练或预处理结束")
        service.launch_job(value)
        return render(value, query, kind, current)

    @protect
    def load_task(target, value, current, text, language, selected, query, kind):
        save_current(value, current, text, language, selected)
        return render(target, query, kind)

    @protect
    def refresh_task(value, query, kind, current):
        if not value:
            return tuple(gr.skip() for _ in outputs)
        values = list(render(value, query, kind, current))
        values[outputs.index(workspace)] = gr.skip()
        return tuple(values)

    @protect
    def filter_view(value, current, text, language, selected, query, kind):
        if not value:
            return tuple(gr.skip() for _ in outputs)
        save_current(value, current, text, language, selected)
        return render(value, query, kind, current)

    @protect
    def edit_clip(value, current, text, language, selected, view, direction=0, decision=None):
        query, kind = view["query"], view["kind"]
        _, records = service.snapshot(value)
        target = adjacent_clip(records, current, query, kind, direction) if direction else current
        save_current(value, current, text, language, selected if decision is None else decision)
        return render(value, query, kind, target)

    @protect
    def choose_clip(target, value, current, text, language, selected, view):
        query, kind = view["query"], view["kind"]
        save_current(value, current, text, language, selected)
        return render(value, query, kind, target)

    @protect
    def choose_row(value, current, text, language, selected, view, records, evt: gr.SelectData):
        row = evt.row_value
        if not records or not row or row[1] == "—":
            return tuple(gr.skip() for _ in outputs)
        # Use the row the user sees, including after sorting or before search submission.
        original_index = int(row[1]) - 1
        if not 0 <= original_index < len(records):
            return tuple(gr.skip() for _ in outputs)
        target = records[original_index]["id"]
        return choose_clip(target, value, current, text, language, selected, view)

    @protect
    def open_new(value, current, text, language, selected):
        save_current(value, current, text, language, selected)
        return (*empty_view(), gr.update(value=None), "", "", gr.update(value=FILTERS[0]))

    @protect
    def reread_task(value, current, query, kind):
        service.import_annotations(value)
        return render(value, query, kind, current)

    start.click(start_task, [upload, speaker, *view_inputs], outputs, concurrency_limit=1, concurrency_id="auto_dataset_start")
    retry.click(retry_task, [job_state, query, kind, current_state], outputs, concurrency_limit=1, concurrency_id="auto_dataset_start")
    previous.input(load_task, [previous, *editor_inputs, *view_inputs], outputs)
    timer.tick(refresh_task, [job_state, query, kind, current_state], outputs, show_progress="hidden")
    reload_history.click(lambda q: gr.update(choices=history(q)), task_search, previous)
    task_search.submit(lambda q: gr.update(choices=history(q), value=None), task_search, previous)
    new_task.click(open_new, editor_inputs, [*outputs, upload, speaker, query, kind])
    kind.input(filter_view, [*editor_inputs, *view_inputs], outputs)
    query.submit(filter_view, [*editor_inputs, *view_inputs], outputs)
    search.click(filter_view, [*editor_inputs, *view_inputs], outputs)
    preview.input(choose_clip, [preview, *editor_inputs, applied_view], outputs)
    table.select(choose_row, [*editor_inputs, applied_view, records_state], outputs)
    apply.click(edit_clip, [*editor_inputs, applied_view], outputs)
    prev_clip.click(lambda *args: edit_clip(*args, direction=-1), [*editor_inputs, applied_view], outputs)
    next_clip.click(lambda *args: edit_clip(*args, direction=1), [*editor_inputs, applied_view], outputs)
    accept_next.click(lambda *args: edit_clip(*args, direction=1, decision=True), [*editor_inputs, applied_view], outputs)
    exclude_next.click(lambda *args: edit_clip(*args, direction=1, decision=False), [*editor_inputs, applied_view], outputs)
    cancel.click(protect(service.cancel_job), job_state, editor_info)
    reread.click(reread_task, [job_state, current_state, query, kind], outputs)
    return {"state": job_state, "status": status, "save": save, "external": external,
            "editor_info": editor_info, "downloads": downloads, "current": current_state,
            "text": text, "language": language, "selected": selected}


def bind_training(page, path_list, inp_text, inp_wav_dir, exp_name, open_editor):
    @protect
    def save(value, current, text, language, selected):
        save_current(value, current, text, language, selected)
        _, records = service.snapshot(value)
        rows = [[r.get("selected", False), r["id"], r["language"], r["text"]] for r in records]
        annotation, clips, experiment, _ = service.approve(value, rows)
        path = service.job_dir(value)
        job, _ = service.snapshot(path)
        return (status_html(job), [str(path / "annotations.list"), annotation, str(path / "metadata.json")],
                str(path / "annotations.list"), annotation, clips, experiment)

    @protect
    def external(value, current, text, language, selected):
        save_current(value, current, text, language, selected)
        path = service.job_dir(value)
        job, _ = service.snapshot(path)
        if job["status"] not in {"awaiting_review", "reviewed"}:
            raise ValueError("请等待自动处理完成")
        return open_editor(str(path / "annotations.list"))

    event = page["save"].click(save, [page[k] for k in ("state", "current", "text", "language", "selected")],
                              [page["status"], page["downloads"], path_list, inp_text, inp_wav_dir, exp_name])
    page["external"].click(external, [page[k] for k in ("state", "current", "text", "language", "selected")], page["editor_info"])
    return event
