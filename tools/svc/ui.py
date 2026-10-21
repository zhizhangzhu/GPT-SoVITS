"""SVC workspace following the existing Gradio workbench design."""
from functools import wraps
from html import escape
import hashlib
import math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import zipfile

import gradio as gr
import soundfile as sf

from tools.assets import workspace_header
from . import service
from . import storage as db
from .review import priority, annotate
from . import history, feedback, preview_cache
from . import dataset_report
from . import task_audio as task_players
from .presentation import describe
from . import separation, pitch, pitch_job, reverb, reverb_ui, audio_edit, audio_edit_ui
from .vocal_split import MODES as VOCAL_MODES

CONVERSION_TRACKS = [
    ("vocals", "合并人声 · 加混响前", "svc-converted-audio"),
    ("mix", "完整混音 · 加混响前（人声 + 伴奏）", "svc-mixed-audio"),
    ("lead_vocals", "分离主唱 · 转换前", "svc-lead-audio"),
    ("backing_vocals", "分离和声 · 转换前", "svc-backing-audio"),
    ("accompaniment", "分离伴奏", "svc-accompaniment-audio"),
    ("converted_lead", "转换后主唱", "svc-converted-lead-audio"),
    ("converted_backing", "转换后和声", "svc-converted-backing-audio"),
    ("source_vocals", "全部人声（主唱 / 和声拆分前）", "svc-source-vocals-audio"),
    ("dry_lead", "去混响后主唱 · 转换 / 切片输入", "svc-dry-lead-audio"),
    ("dry_backing", "去混响后和声 · 转换 / 切片输入", "svc-dry-backing-audio"),
    ("dry_vocals", "去混响后全部人声 · 未拆分音轨", "svc-dry-vocals-audio"),
    ("lead_reverb", "主唱移除的混响与残留", "svc-lead-reverb-audio"),
    ("backing_reverb", "和声移除的混响与残留", "svc-backing-reverb-audio"),
    ("vocals_reverb", "全部人声移除的混响与残留", "svc-vocals-reverb-audio"),
    ("reference_lead", "参考主唱", "svc-pitch-reference-lead"),
    ("reference_backing", "参考和声", "svc-pitch-reference-backing"),
    ("reference_vocals", "参考全部人声", "svc-pitch-reference-vocals"),
    ("before_lead", "校准前主唱", "svc-pitch-before-lead"),
    ("before_backing", "校准前和声", "svc-pitch-before-backing"),
    ("before_vocals", "校准前全部人声", "svc-pitch-before-vocals"),
    ("psola_lead", "PSOLA 校准后主唱", "svc-pitch-psola-lead"),
    ("psola_backing", "PSOLA 校准后和声", "svc-pitch-psola-backing"),
    ("psola_vocals", "PSOLA 校准后全部人声", "svc-pitch-psola-vocals"),
    ("psola_combined", "PSOLA 校准后合并后人声", "svc-pitch-psola-combined"),
    ("psola_mix", "PSOLA 校准后含伴奏混音", "svc-pitch-psola-mix"),
    ("pc_nsf_lead", "PC-NSF 校准后主唱", "svc-pitch-pc-nsf-lead"),
    ("pc_nsf_backing", "PC-NSF 校准后和声", "svc-pitch-pc-nsf-backing"),
    ("pc_nsf_vocals", "PC-NSF 校准后全部人声", "svc-pitch-pc-nsf-vocals"),
    ("pc_nsf_combined", "PC-NSF 校准后合并后人声", "svc-pitch-pc-nsf-combined"),
    ("pc_nsf_mix", "PC-NSF 校准后含伴奏混音", "svc-pitch-pc-nsf-mix"),

]
CONVERSION_TRACKS += [(key,label,'svc-'+key.replace('_','-')) for key,label in reverb.TRACKS]
CONVERSION_TRACKS += [(key,label,'svc-'+key.replace('_','-')) for key,label in audio_edit.EXTRA_TRACKS]

STATUS = {"queued": "排队中", "running": "处理中", "awaiting_review": "待试听审核", "completed": "已完成", "failed": "失败", "cancelled": "已停止"}
KINDS = {"dataset": "数据整理", "train": "模型训练", "convert": "音色转换", "report": "数据报告", "import_model": "导入模型", "pitch": "音准分析 / 校准", "reverb": "参考混响"}
REVIEW = {"pending": "待审核", "approved": "已确认", "rejected": "已排除"}
QUALITY = {"passed": "工具检查通过", "needs_review": "需复核", "rejected": "不可用提示"}


def safe(function=None, *, outputs=1, quiet=False, fallback=None):
    """Show a short toast and preserve outputs instead of failing the UI event."""
    def decorate(callback):
        @wraps(callback)
        def wrapped(*args, **kwargs):
            try:
                return callback(*args, **kwargs)
            except Exception as error:
                if not quiet:
                    gr.Warning(escape(str(error)), duration=5)
                if fallback is not None:
                    return fallback(error, *args, **kwargs)
                unchanged = tuple(gr.skip() for _ in range(outputs))
                return unchanged[0] if outputs == 1 else unchanged
        return wrapped
    return decorate(function) if function is not None else decorate


def singer_choices():
    return [(f"{row['name']} · {row['id'][:6]}", row["id"]) for row in db.singers()]


def job_choices():
    return [(f"{KINDS.get(row['kind'], row['kind'])} · {STATUS.get(row['status'], row['status'])} · {row['created_at'][:16]} · {row['id'][:6]}", row["id"]) for row in service.history()]


def task_listing(kind="全部", state="全部", search="", page=1):
    records = service.history()
    indexed={row['id']:row for row in records}
    search = str(search or "").strip().casefold()
    if kind != "全部":
        records = [item for item in records if item["kind"] == kind]
    if state != "全部":
        records = [item for item in records if item["status"] == state]
    if search:
        records = [item for item in records if search in (item["id"]+' '+str(item.get("message") or '')+' '+
                   ' '.join(describe(item,indexed).values())).casefold()]
    pages = max(1, math.ceil(len(records) / 50))
    try:
        page = max(1, min(int(page or 1), pages))
    except (TypeError, ValueError, OverflowError):
        page = 1
    shown = records[(page - 1) * 50:page * 50]
    def local_time(item):
        try:
            return datetime.fromisoformat(item["created_at"]).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return str(item.get("created_at") or "")[:19]
    data=[]
    for item in shown:
        names=describe(item,indexed)
        source=names['source_name'] or ('未保存原文件名' if item['kind'] in {'convert','pitch','reverb'} else '—')
        data.append([KINDS.get(item['kind'],item['kind']),STATUS.get(item['status'],item['status']),
                     f"{float(item.get('progress') or 0):.1f}%",source,names['model_name'] or '—',
                     local_time(item),item.get('message',''),item['id']])
    return data, f"共 {len(records)} 项 · 第 {page}/{pages} 页 · 点击行查看详情", page


def experiment_choices():
    return [(f"{row['name']} · {row['id'][:6]}", row["id"]) for row in service.experiments()]


def conversion_choices():
    from .external_models import records
    trained = [(f"自己训练 · {row['name']}", "trained:" + row["id"]) for row in service.experiments() if service.checkpoints(row["id"])]
    imported = [(f"外部导入 · {row['name']} · {row['weight_name']}", "imported:" + row["id"]) for row in records()]
    return trained + imported


def conversion_description(selection):
    if not selection:return "选择已训练实验或导入模型后开始转换。"
    source,model_id=selection.split(':',1)
    if source=='trained':
        item=next(row for row in service.experiments() if row['id']==model_id)
        return f"**自己训练 · {escape(item['name'])}** · 选择要使用的检查点。"
    from .external_models import record
    item=record(model_id)
    encoder='ContentVec 768 L12' if item.get('encoder')=='vec768l12-onnx' else 'ContentVec 256 L9'
    mode='可选浅扩散' if item.get('diffusion') else '基础生成器推理'
    return f"**外部导入 · {escape(item['name'])}** · {escape(item['weight_name'])} · {encoder} · {mode}。请选择模型内歌手。"


def import_choices():
    from .external_models import candidates
    rows, problems = candidates()
    return [(f"{row['name']} / {row['weight']}", row["key"]) for row in rows], problems


def validation_distribution(plan):
    total = plan["train_seconds"] + plan["validation_seconds"]
    rows = []
    for source in sorted(plan["source_rows"], key=lambda row: -row["seconds"]):
        train = sum(float(row["duration"]) for row in plan["training"] if row["source_id"] == source["id"])
        val = sum(float(row["duration"]) for row in plan["validation"] if row["source_id"] == source["id"])
        share = 100 * source["seconds"] / total
        train_width = 100 * train / source["seconds"] if source["seconds"] else 0
        val_width = 100 * val / source["seconds"] if source["seconds"] else 0
        rows.append(f'<tr><th scope="row">{escape(source["name"])}<small>{source["id"][:6]}</small></th>'
                    f'<td>{train / 60:.2f}</td><td>{val / 60:.2f}</td><td>{share:.1f}%'
                    f'<div class="svc-source-bar" aria-hidden="true" style="width:{share:.3f}%">'
                    f'<span class="svc-train-share" style="width:{train_width:.3f}%"></span>'
                    f'<span class="svc-val-share" style="width:{val_width:.3f}%"></span></div></td></tr>')
    return ('<div class="svc-source-distribution"><table><caption>各素材时长分配 · 分钟</caption>'
            '<thead><tr><th scope="col">源素材</th><th scope="col">训练</th><th scope="col">验证</th>'
            '<th scope="col">总时长占比</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')


def status(job):
    if not job:
        return '<div class="workbench-panel" role="status">选择历史任务，或上传素材开始。</div>'
    warning = "；".join(job.get("warnings", []))
    measured = job.get("metrics", {})
    if measured:
        warning += ("；" if warning else "") + f"运行 {measured.get('compute_seconds', 0):.1f} 秒"
        if measured.get("cuda_peak_allocated_mib"):
            warning += f" · CUDA 张量峰值 {measured['cuda_peak_allocated_mib']:.0f} MiB"
    return '<div class="workbench-panel" role="status"><strong>' + escape(STATUS.get(job["status"], job["status"])) + f" · {job.get('progress', 0)}%</strong><p>" + escape(job.get("message", "")) + "</p>" + ("<p>" + escape(warning) + "</p>" if warning else "") + "</div>"


def report_stamp(record):
    """Small metadata signature: don't rebuild unchanged report DOM on polls."""
    files=[]
    directory=db.job_dir(record['id']).resolve() if record.get('id') else None
    for key in ['pitch_report','reverb_report']:
        value=record.get('outputs',{}).get(key)
        path=Path(value).resolve() if isinstance(value,str) else None
        if path and directory and path.is_relative_to(directory) and path.is_file():
            stat=path.stat();files.append([key,str(path),stat.st_mtime_ns,stat.st_size])
    return {'files':files,'edit':record.get('audio_edit',{})}


def text_stamp(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def rows(job_id, review_filter="全部", order="风险优先"):
    records = service.records(job_id) if job_id else []
    if review_filter == "待复核":
        records = [row for row in records if row["review"] == "pending"]
    elif review_filter == "已确认":
        records = [row for row in records if row["review"] == "approved"]
    elif review_filter == "已排除":
        records = [row for row in records if row["review"] == "rejected"]
    elif review_filter == "高风险待审核":
        records = [row for row in records if row["review"] == "pending" and priority(row)["level"] >= 3]
    elif review_filter == "有质量提示":
        records = [row for row in records if priority(row)["level"] >= 2]
    if order == "风险优先":
        records.sort(key=lambda row: (row["review"] != "pending", -priority(row)["level"], row["source_id"], row["start"]))
    else:
        records.sort(key=lambda row: (row["source_id"], row["start"]))
    table = [[priority(row)["label"], row["source_name"], round(row["start"], 2), round(row["end"], 2), row["duration"], REVIEW[row["review"]],
              QUALITY[row["quality"]], row.get("lyrics", ""), "；".join(dict.fromkeys(row.get("flags", []) + row.get("boundary_flags", []) + row.get("alignment_flags", [])))] for row in records]
    choices = [(f"{i + 1} · {priority(row)['label']} · {row['source_name']} · {row['start']:.2f}–{row['end']:.2f} 秒 · {REVIEW[row['review']]}", row["id"]) for i, row in enumerate(records)]
    return table, choices


def build_page(busy=None):
    def guard():
        if busy and busy():
            raise ValueError("TTS 训练、预处理或合成正在运行，请结束后提交 SVC GPU 任务")
        from tools.tts_workspace.service import service as tts_service
        process = tts_service.process
        if process is not None and process.poll() is None:
            raise ValueError("TTS 推理模型仍驻留显存，请先在“TTS 推理”中点击“卸载模型”，再提交 SVC 任务")

    gr.HTML(workspace_header("So-VITS-SVC · 音色转换", "整理素材 → 试听入库 → 训练专属模型 → 转换完整音频"))
    notice = gr.Markdown("SVC 不需要歌词转录标注。工具检查后仍需试听，确认歌手身份、伴奏残留与和声。", elem_classes=["workspace-note"])
    with gr.Row(elem_classes=["compact-actions"]):
        refresh = gr.Button("刷新页面数据", size="sm", scale=0)
    job = gr.Dropdown(choices=job_choices(), visible=False, label="当前任务")
    with gr.Tabs(elem_classes=["workspace-tabs"]) as svc_tabs:
        with gr.Tab("素材与审核", id="dataset") as dataset_tab:
            with gr.Row(elem_classes=["svc-identity"]):
                singer = gr.Dropdown(label="数据集歌手", choices=singer_choices(), scale=3, min_width=240)
                with gr.Column(scale=2, min_width=240):
                    with gr.Accordion("创建或重命名歌手", open=False):
                        new_name = gr.Textbox(label="新歌手名称", placeholder="同一歌手后续上传请选择已有身份", elem_classes=["svc-name-input"])
                        with gr.Row(elem_classes=["compact-actions"]):
                            add = gr.Button("创建歌手")
                            rename = gr.Button("重命名当前歌手")
            with gr.Row(elem_classes=["svc-layout"]):
                with gr.Column(scale=4, min_width=280, elem_classes=["workbench-panel", "svc-source"]):
                    gr.Markdown("### 追加素材")
                    audio_files = gr.File(label="音频（支持批量上传）", file_count="multiple", file_types=["audio"], type="filepath")
                    input_kind = gr.Radio(["干净人声", "带伴奏的音乐"], value="干净人声", label="素材类型")
                    dataset_separator = gr.Dropdown(separation.VOCAL_CHOICES, value=separation.VOCAL_DEFAULT,
                        label="人声 / 伴奏分离模型", info="仅带伴奏的音乐使用；默认使用升级后的 Deux，HP2 保留供对比。")
                    with gr.Accordion("主唱 / 和声与去混响（可选）", open=False):
                        dataset_split = gr.Checkbox(label="分离主唱和和声", value=False,
                            info="开启后分别保存两轨，默认仅主唱生成候选切片。")
                        dataset_karaoke = gr.Dropdown(separation.KARAOKE_CHOICES, value=separation.KARAOKE_DEFAULT,
                            label="主唱 / 和声分离模型")
                        dataset_dry_lead = gr.Checkbox(label="主唱／未分轨人声去混响", value=False,
                            info="使用 BS-RoFormer De-Reverb；默认关闭，混响较重时可开启。")
                        with gr.Group(visible=False) as dataset_backing_options:
                            dataset_dry_backing = gr.Checkbox(label="和声去混响", value=False)
                            dataset_include_backing = gr.Checkbox(label="和声也生成候选切片", value=False,
                                info="与主唱作为不同素材审核；仅目标歌手的单声部和声适合入库。")
                        gr.Markdown("保留处理前后完整音轨，可在任务中心试听。去混响可能损伤齿音、尾音，审核后再入库。", elem_classes=["workspace-hint"])
                    with gr.Accordion("歌词时间定位（可选）", open=False):
                        lyric_files = gr.File(label="LRC 歌词（与音频文件同名，可批量）", file_count="multiple", file_types=[".lrc", ".txt"], type="filepath")
                        lyric_text = gr.Textbox(label="或粘贴带时间戳歌词（单份音频）", lines=4, placeholder="[00:12.30]第一句歌词")
                    with gr.Accordion("切片设置", open=False):
                        offset = gr.Number(label="额外歌词偏移 / 秒（正值延后）", value=0)
                        minimum = gr.Number(label="目标最短 / 秒", value=5)
                        maximum = gr.Number(label="目标最长 / 秒", value=15)
                        silence_db = gr.Slider(-80, -20, value=-45, step=1, label="静音判定下限 / dB")
                        max_silence = gr.Slider(0, 100, value=50, step=1, label="静音占比提示阈值 / %")
                        max_clipping = gr.Slider(0, 5, value=.1, step=.05, label="削波占比提示阈值 / %")
                    with gr.Accordion("处理设备", open=False):
                        dataset_gpu = gr.Textbox(label="GPU 编号（分离 / 去混响时使用）", value="0")
                    prepare = gr.Button("整理为候选切片", variant="primary")
                    with gr.Accordion("用审核反馈校准边界", open=False):
                        feedback_info = gr.Markdown("选择歌手后显示反馈与当前边界参数。")
                        calibrate_button = gr.Button("生成参数建议")
                        calibration_report = gr.Markdown(feedback.report_text(None))
                        calibration_state = gr.State(None)
                        with gr.Row(elem_classes=["compact-actions"]):
                            apply_profile = gr.Button("应用候选参数", interactive=False)
                            reset_profile = gr.Button("恢复默认参数")
                        recheck_profile = gr.Button("用当前参数重新检查任务")
                        gr.Markdown("至少 5 个试听确认的反馈片段。只使用已确认边界校准；多个源素材时保留一组作比较。", elem_classes=["workspace-hint"])
                with gr.Column(scale=7, min_width=320, elem_classes=["workbench-panel", "svc-review"]):
                    gr.Markdown("### 试听与修正")
                    review_progress = gr.HTML('<div role="status">选择任务后显示审核进度。</div>')
                    with gr.Row():
                        review_filter = gr.Dropdown(["全部", "待复核", "高风险待审核", "有质量提示", "已确认", "已排除"], value="全部", label="片段筛选")
                        review_order = gr.Radio(["风险优先", "时间顺序"], value="风险优先", label="审核顺序")
                    gr.Markdown("风险优先：待审核的疑似截字、削波及不可用片段优先展示，其次是时长与静音问题。每片仍需试听确认。", elem_classes=["workspace-hint"])
                    clip = gr.Dropdown(label="当前片段", choices=[])
                    with gr.Row(elem_classes=["compact-actions"]):
                        previous = gr.Button("上一段", interactive=False, elem_id="svc-review-prev")
                        following = gr.Button("下一段", interactive=False, elem_id="svc-review-next")
                    preview = gr.Audio(label="当前切片（波形与试听）", type="filepath", interactive=False, loop=True)
                    metrics = gr.Markdown()
                    boundary_state = gr.State(None)
                    with gr.Accordion("切点检查与推荐", open=True):
                        boundary_info = gr.Markdown("选择片段后显示起点、终点检查和可用建议。")
                        with gr.Row(elem_classes=["compact-actions"]):
                            apply_recommended = gr.Button("采用推荐切点并重新裁切", interactive=False)
                            decline_recommended = gr.Button("不采用本次建议", interactive=False)
                        decline_reason = gr.Textbox(label="不采用原因（可选）", placeholder="例如：切点会漏掉唱字", max_lines=2)
                        with gr.Accordion("试听切点前后", open=False):
                            gr.Markdown("各播放切点前后约 1 秒；播放器标签标明切点在试听片段中的位置。")
                            with gr.Row():
                                with gr.Column(min_width=220):
                                    start_preview = gr.Audio(label="当前起点", type="filepath", interactive=False)
                                    suggested_start_preview = gr.Audio(label="推荐起点", type="filepath", interactive=False, visible=False)
                                with gr.Column(min_width=220):
                                    end_preview = gr.Audio(label="当前终点", type="filepath", interactive=False)
                                    suggested_end_preview = gr.Audio(label="推荐终点", type="filepath", interactive=False, visible=False)
                    with gr.Row(elem_classes=["compact-actions"]):
                        approve_next = gr.Button("确认并下一条", interactive=False, variant="primary", elem_id="svc-review-accept-next")
                        reject_next = gr.Button("排除并下一条", interactive=False, elem_id="svc-review-reject-next")
                    gr.Markdown("快捷键：Ctrl / ⌘ + Enter 确认并下一条，加 Shift 排除并下一条；Alt + ← / → 翻页；Alt + Z 撤销最近边界修改。", elem_classes=["workspace-hint"])
                    with gr.Accordion("仅修改当前审核状态", open=False):
                        with gr.Row(elem_classes=["compact-actions"]):
                            approve = gr.Button("确认通过，留在本片")
                            reject = gr.Button("排除，留在本片")
                            restore = gr.Button("恢复待审核")
                    with gr.Accordion("修订对比与撤销", open=False):
                        revision_choice = gr.Dropdown(label="边界修改记录", choices=[])
                        revision_info = gr.Markdown("修改边界、拆分、合并后可比较修订。")
                        with gr.Row():
                            with gr.Column(min_width=220):
                                before_choice = gr.Dropdown(label="修改前的片段", choices=[])
                                before_audio = gr.Audio(label="修改前", type="filepath", interactive=False)
                            with gr.Column(min_width=220):
                                after_choice = gr.Dropdown(label="修改后的片段", choices=[])
                                after_audio = gr.Audio(label="修改后", type="filepath", interactive=False)
                        undo_button = gr.Button("撤销最近一次边界修改", interactive=False, elem_id="svc-review-undo")
                        gr.Markdown("可以连续撤销。恢复后生成新修订并重新审核；已创建的训练快照保留。", elem_classes=["workspace-hint"])
                    with gr.Accordion("修正边界、拆分与合并", open=False):
                        context_audio = gr.Audio(label="前后上下文（各 1 秒）", type="filepath", interactive=False)
                        with gr.Row():
                            start = gr.Number(label="开始时间 / 秒", precision=4)
                            end = gr.Number(label="结束时间 / 秒", precision=4)
                        bounds = gr.Button("保存边界并重新裁切")
                        split_at = gr.Number(label="拆分位置 / 原音轨秒", precision=4)
                        with gr.Row(elem_classes=["compact-actions"]):
                            split = gr.Button("在此拆分")
                            merge = gr.Button("合并时间相邻的下一片段")
                    with gr.Accordion("片段列表与工具检查", open=False):
                        table = gr.Dataframe(headers=["审核优先级", "源素材", "开始 / 秒", "结束 / 秒", "时长", "审核", "工具检查", "歌词", "提示"], interactive=False, wrap=True, height=280, elem_classes=["review-table"])
                    with gr.Row(elem_classes=["compact-actions", "review-actions"]):
                        publish = gr.Button("将已确认片段入库", variant="primary")
                        publish_safe = gr.Button("确认无提示片段并入库")
                        export = gr.Button("导出歌手数据集")
                    dataset_download = gr.File(label="数据集下载", interactive=False, visible=False)
                    with gr.Accordion("试听缓存", open=False):
                        cache_info = gr.Markdown("选择任务后显示试听缓存。")
                        cache_days = gr.Number(label="保留最近访问的缓存 / 天", value=7, precision=0,
                                               info="0 表示清理全部未在当前试听中使用的缓存。")
                        clear_cache = gr.Button("清理过期试听缓存")
                        cache_notice = gr.Markdown()
                        cached_paths = gr.State({})
        with gr.Tab("专属模型训练", id="training") as training_tab:
            with gr.Row(elem_classes=["svc-identity"]):
                training_singer = gr.Dropdown(label="训练数据集歌手", choices=singer_choices(), scale=2, min_width=220)
                experiment = gr.Dropdown(label="训练实验", choices=experiment_choices(), scale=3, min_width=260)
                checkpoint = gr.Dropdown(label="当前检查点", choices=[], scale=2, min_width=220)
            with gr.Row(elem_classes=["svc-layout"]):
                with gr.Column(scale=4, min_width=280, elem_classes=["workbench-panel", "svc-source"]):
                    gr.Markdown("### 从已审核数据创建实验")
                    dataset_stats = gr.Markdown("选择歌手后显示已入库的数据量。")
                    validation_sources = gr.CheckboxGroup(label="验证集使用的源素材", choices=[], interactive=False,
                                                          info="按时长推荐；至少留一份源素材用于训练。单份源素材时自动按片段划分。")
                    validation_summary = gr.Markdown("选择歌手后预览训练与验证数据分配。", elem_classes=["workspace-hint"])
                    with gr.Accordion("各素材时长分配", open=False):
                        validation_chart = gr.HTML()
                    experiment_name = gr.Textbox(label="实验名称", placeholder="例如：歌手名 · 第一版")
                    train_steps = gr.Number(label="目标训练步数", value=1000, precision=0)
                    batch = gr.Number(label="Batch size", value=2, precision=0)
                    save_every = gr.Number(label="检查点保存间隔 / 步", value=100, precision=0)
                    train_f0 = gr.Dropdown(pitch.CHOICES, value="dio", label="训练 F0 提取")
                    with gr.Accordion("训练设备", open=False):
                        train_gpu = gr.Textbox(label="GPU 编号", value="0")
                    with gr.Accordion("替换默认 SVC 预训练权重", open=False):
                        gr.Markdown("默认底模为 CC-BY-NC-4.0。可上传与 ContentVec256 / 4.0 兼容配置一致的 G、D；不接受 TTS 权重。")
                        initial_g = gr.File(label="SVC 预训练 G", file_types=[".pth"], type="filepath")
                        initial_d = gr.File(label="SVC 预训练 D", file_types=[".pth"], type="filepath")
                    train_button = gr.Button("创建快照并开始训练", variant="primary")
                with gr.Column(scale=7, min_width=320, elem_classes=["workbench-panel", "svc-review"]):
                    gr.Markdown("### 检查点试听与对比")
                    experiment_info = gr.Markdown("选择实验后查看固定验证片段与检查点。")
                    with gr.Row():
                        preview_clip = gr.Dropdown(label="固定验证片段", choices=[], min_width=230)
                        compare_checkpoint = gr.Dropdown(label="对比检查点", choices=[], min_width=180)
                    preview_message = gr.Markdown("所有检查点使用同一组固定验证片段。", elem_classes=["workspace-hint"])
                    with gr.Row(elem_classes=["compact-actions"]):
                        ab_switch = gr.Button("A / B 同位置切换", elem_id="svc-ab-switch")
                        match_ab_volume = gr.Checkbox(label="试听时匹配 A/B 响度", value=True, elem_id="svc-ab-match")
                    ab_status = gr.HTML('<span id="svc-ab-status" role="status" data-a-rms="0" data-b-rms="0">选择两个检查点后可同位置切换试听。</span>')
                    with gr.Row(elem_classes=["svc-preview-comparison"]):
                        reference_audio = gr.Audio(label="验证原声", interactive=False, elem_id="svc-preview-reference")
                        checkpoint_preview = gr.Audio(label="A · 当前检查点", interactive=False, elem_id="svc-preview-a")
                        comparison_audio = gr.Audio(label="B · 对比检查点", interactive=False, elem_id="svc-preview-b")
                    with gr.Row(elem_classes=["compact-actions"]):
                        extra_steps = gr.Number(label="续训追加步数", value=1000, precision=0)
                        resume_button = gr.Button("从最新检查点续训")
                    gr.Markdown("同一片段可并排比较检查点；频谱指标只作参考，请以试听为准。修改切片后需创建新实验。", elem_classes=["workspace-hint"])
                    with gr.Accordion("管理旧检查点", open=False):
                        cleanup_checkpoint = gr.Dropdown(label="选择要删除的旧检查点", choices=[])
                        confirm_cleanup = gr.Checkbox(label="我已确认不再需要此检查点和对应试听文件", value=False)
                        cleanup_button = gr.Button("永久删除选中检查点及试听文件")
                        gr.Markdown("最新检查点用于续训，不能删除。运行任务使用中的检查点也不能删除。", elem_classes=["workspace-hint"])
                    report_ready_status = gr.Markdown("尚无训练前数据报告。", elem_classes=["workspace-hint"])
                    view_report = gr.Button("查看训练前报告", visible=False)
                    with gr.Accordion("训练前数据报告 · 音高与素材覆盖", open=False) as report_panel:
                        gr.Markdown("统计当前歌手已试听审核并入库的片段。音高分析使用左侧所选 F0 工具，在 CPU 上运行。", elem_classes=["workspace-hint"])
                        with gr.Row(elem_classes=["compact-actions"]):
                            make_report = gr.Button("生成训练前报告")
                            reload_report = gr.Button("读取已有报告")
                        report_status = gr.Markdown("选择歌手后可读取报告。")
                        report_metrics = gr.HTML()
                        report_notes = gr.Markdown()
                        with gr.Row():
                            with gr.Column(min_width=340):
                                pitch_chart = gr.Image(label="有声音高分布", type="filepath", interactive=False)
                            with gr.Column(min_width=340):
                                length_chart = gr.Image(label="片段长度分布", type="filepath", interactive=False)
                        with gr.Accordion("源素材占比与统计明细", open=False):
                            source_chart = gr.Image(label="源素材时长占比", type="filepath", interactive=False)
                            source_table = gr.Dataframe(headers=["图中序号", "源素材", "片段数", "时长 / 分钟", "时长占比 / %"],
                                                        interactive=False, wrap=True, height=240, elem_classes=["review-table"])
                            length_table = gr.Dataframe(headers=["片段长度", "片段数"], interactive=False, height=200)
                        with gr.Accordion("建议复核的片段", open=False):
                            gr.Markdown("以下是 F0、时长或文件校验提示，需试听判断。选择片段后可打开原审核任务。", elem_classes=["workspace-hint"])
                            problem_table = gr.Dataframe(headers=["源素材", "开始 / 秒", "结束 / 秒", "有声帧 / %", "检查原因"],
                                                        interactive=False, wrap=True, height=220, elem_classes=["review-table"])
                            problem_clip = gr.Dropdown(label="选择待复核片段", choices=[])
                            open_problem = gr.Button("打开审核试听")
                        report_files = gr.File(label="下载报告与图表", file_count="multiple", interactive=False)
        with gr.Tab("完整音频转换", id="conversion") as conversion_tab:
            with gr.Row(elem_classes=["svc-identity"]):
                model_choice = gr.Dropdown(label="目标音色模型", choices=conversion_choices(), scale=3, min_width=270)
                conversion_checkpoint = gr.Dropdown(label="训练检查点", choices=[], visible=False, scale=2, min_width=220)
                conversion_speaker = gr.Dropdown(label="模型内歌手", choices=[], visible=False, scale=2, min_width=220)
            model_details = gr.Markdown("选择已训练实验或导入模型后开始转换。", elem_classes=["workspace-hint"])
            with gr.Accordion("导入现成 So-VITS-SVC 模型", open=False):
                gr.Markdown("把模型放入 `svc_imports/模型文件夹/`，包含 `config.json` 和 G_*.pth。支持 44.1 kHz ContentVec 256 / 768 模型。若有配套浅扩散，请一并放入 `diffusion/model_30000.pt`、`diffusion/config.yaml` 和 `diffusion/vocoder/` 下的 `model`、`config.json`。导入时会复制并校验文件。")
                import_candidate = gr.Dropdown(label="待导入文件夹", choices=import_choices()[0])
                import_info = gr.Markdown("放入文件后点击“扫描文件夹”。", elem_classes=["workspace-hint"])
                with gr.Row(elem_classes=["compact-actions"]):
                    scan_models = gr.Button("扫描文件夹")
                    import_button = gr.Button("导入并校验模型", variant="primary")
            with gr.Row(elem_classes=["svc-conversion-layout"]):
                with gr.Column(scale=1, min_width=0, elem_classes=["workbench-panel", "svc-conversion-input"]):
                    gr.Markdown("### 1 · 音频输入")
                    gr.Markdown("可上传音频，或直接录制说话/唱歌。录音可试听后重录；说话会降噪，唱歌可按是否带伴奏选择处理方式。")
                    convert_source = gr.Radio(["上传音频", "录音"], value="上传音频", label="音频来源")
                    convert_audio = gr.Audio(label="上传歌声或说话音频", type="filepath", sources=["upload"], elem_id="svc-convert-upload")
                    convert_upload_name=gr.State('')
                    convert_recording = gr.Audio(label="录制说话或唱歌（可试听并重录）", type="filepath", sources=["microphone"], visible=False, elem_id="svc-convert-recording")
                    record_kind = gr.Radio(["说话 · 降噪", "唱歌 · 清唱", "唱歌 · 带伴奏"], value="说话 · 降噪",
                                           label="录音内容与处理方式", visible=False,
                                           info="说话使用语音降噪；带伴奏的歌声先分离人声；清唱保留原录音，避免不必要的分离失真。")
                    convert_kind = gr.Radio(["干净人声", "带伴奏的音乐", "人声 + 伴奏分别上传"], value="干净人声", label="上传音频类型")
                    convert_backing = gr.Audio(label="上传对应伴奏（与人声从同一时间点开始）", type="filepath", visible=False)
                    gr.Markdown("转换保持原音轨长度。说话录音会先降噪；歌声模型用于说话时，效果取决于训练数据覆盖范围。", elem_classes=["workspace-hint"])
                with gr.Column(scale=1, min_width=0, elem_classes=["workbench-panel", "svc-conversion-settings"]):
                    gr.Markdown("### 2 · 转换配置")
                    split_enabled = gr.Checkbox(label="分离主唱和和声", value=True,
                                                info="适用于含和声的歌声。上传说话音频或无和声的清唱可关闭；说话录音会自动跳过。")
                    with gr.Group() as split_options:
                        vocal_mode = gr.Radio(VOCAL_MODES, value="keep", label="主唱 / 和声处理方式")
                        gr.Markdown("主唱与和声分别转换时会分别提取音高，并使用同一目标音色。和声仍可能包含多个声部，请试听判断。", elem_classes=["workspace-hint"])
                    with gr.Accordion("分离模型", open=False):
                        conversion_separator = gr.Dropdown(separation.VOCAL_CHOICES, value=separation.VOCAL_DEFAULT,
                            label="人声 / 伴奏分离模型", info="带伴奏音乐先提取全部人声；干净人声或已配对伴奏会跳过这一步。")
                        conversion_karaoke = gr.Dropdown(separation.KARAOKE_CHOICES, value=separation.KARAOKE_DEFAULT,
                            label="主唱 / 和声分离模型", info="开启主唱 / 和声分离时使用，旧模型可用于试听对比。")
                    with gr.Accordion("输入去混响（可选）", open=False):
                        conversion_dry_lead = gr.Checkbox(label="主唱／未分轨人声去混响", value=False)
                        conversion_dry_backing = gr.Checkbox(label="和声去混响（需开启主唱和声分离）", value=False)
                        gr.Markdown("BS-RoFormer De-Reverb 在分离后、音色转换前处理。默认关闭，适合混响较重的素材；原音轨、去混响音轨和移除的残留均可试听。保留原和声时也会使用已勾选的去混响结果。", elem_classes=["workspace-hint"])
                    conversion_reverb_inputs = reverb_ui.conversion_controls()
                    with gr.Row():
                        transpose = gr.Slider(-24, 24, value=0, step=1, label="音高偏移 / 半音", min_width=180)
                        convert_f0 = gr.Dropdown(pitch.CHOICES, value="rmvpe", label="推理 F0 提取", min_width=220,
                                                 info="F0 是源人声的音高曲线；RMVPE / FCPE 适合歌声，可与 DIO、Harvest、PM 对比。")
                    with gr.Group(visible=False) as diffusion_controls:
                        shallow_diffusion = gr.Checkbox(label="使用浅扩散细化音色", value=False,
                                                        info="可能减少部分电子音，但转换更慢，效果依素材而异。可关闭后试听对比。")
                        diffusion_steps = gr.Slider(1, 100, value=35, step=1, label="浅扩散步数",
                                                    info="模型包建议 35 步；步数越大通常越慢。")
                    with gr.Accordion("分段与混音", open=False):
                        chunk = gr.Number(label="内部转换分段 / 秒", value=15)
                        vocal_gain = gr.Slider(0, 2, value=1, step=.05, label="混音人声增益")
                        instrumental_gain = gr.Slider(0, 2, value=1, step=.05, label="伴奏增益")
                        gr.Markdown("按设定增益混音，不自动压低任何音轨；浮点 WAV 保留超满刻度幅度，超过 0 dBFS 时由你手动调整。", elem_classes=["workspace-hint"])
                        harmony_gain = gr.Slider(0, 2, value=1, step=.05, label="和声增益（仅保留 / 转换和声时生效）")
                        harmony_pitch = gr.Slider(-24, 24, value=0, step=1, label="和声额外音高偏移 / 半音",
                                                  info="仅转换和声时生效，在主唱音高偏移基础上额外调整。")
                        convert_gpu = gr.Textbox(label="GPU 编号", value="0")
                    convert_button = gr.Button("转换完整音频", variant="primary")
                    gr.Markdown("任务在后台运行。提交后前往「任务中心」查看进度、试听和下载；完成后还能调整增益与混响。", elem_classes=["workspace-hint"])
        with gr.Tab("任务中心", id="tasks") as tasks_tab:
            gr.Markdown("### 任务中心")
            gr.Markdown("查看所有 SVC 后台任务。点击表格中的任意一行，再对所选任务操作。", elem_classes=["workspace-hint"])
            with gr.Row(elem_classes=["svc-task-toolbar"]):
                task_kind = gr.Dropdown(label="任务类型", choices=[("全部", "全部"), *[(name, key) for key, name in KINDS.items()]], value="全部", scale=1)
                task_state = gr.Dropdown(label="状态", choices=[("全部", "全部"), *[(name, key) for key, name in STATUS.items()]], value="全部", scale=1)
                task_search = gr.Textbox(label="搜索音频、音色模型、任务编号或消息", placeholder="输入歌曲名、模型名或任务编号", scale=2)
                task_refresh = gr.Button("刷新任务", scale=0)
            task_rows, task_summary, task_page_value = task_listing()
            task_table = gr.Dataframe(value=task_rows, headers=["类型", "状态", "进度", "参考音频", "音色模型", "创建时间", "当前消息", "任务编号"],
                                      datatype=["str"] * 8, type="array", interactive=False, height=440,
                                      wrap=True, elem_id="svc-task-table", elem_classes=["svc-task-table"])
            selected_task_marker = gr.HTML('<span data-svc-selected-task="" hidden></span>', elem_id="svc-task-selection-marker")
            with gr.Row(elem_classes=["svc-task-pages"]):
                task_previous = gr.Button("上一页", scale=0)
                task_page = gr.Number(label="页码", value=task_page_value, precision=0, scale=0, min_width=100)
                task_next = gr.Button("下一页", scale=0)
                task_count = gr.Markdown(task_summary)
            with gr.Group(elem_id="svc-task-detail", elem_classes=["workbench-panel", "svc-task-detail"]) as task_detail:
                selected_job = gr.Markdown("尚未选择任务。")
                progress = gr.HTML(status({}))
                with gr.Row(elem_classes=["compact-actions"]):
                    stop = gr.Button("停止任务")
                    retry = gr.Button("重试 / 调整配置")
                    delete = gr.Button("删除任务")
                with gr.Group(visible=False, elem_classes=["svc-retry-panel"]) as retry_panel:
                    gr.Markdown("### 重试设置")
                    retry_info = gr.Markdown("转换任务可修改参数后作为新任务提交，原任务与结果保留。")
                    with gr.Group(visible=False) as retry_controls:
                        with gr.Row():
                            retry_pitch = gr.Slider(-24, 24, value=0, step=1, label="音高偏移 / 半音")
                            retry_f0 = gr.Dropdown(pitch.CHOICES, value="dio", label="推理 F0 提取",
                                                   info="从源人声提取音高曲线；如果出现跑调、断音，可换一种方法试听。")
                        with gr.Row():
                            retry_chunk = gr.Number(label="内部转换分段 / 秒", value=15, minimum=2, maximum=60)
                            retry_gpu = gr.Textbox(label="GPU 编号", value="0")
                        with gr.Row():
                            retry_vocal_gain = gr.Slider(0, 2, value=1, step=.05, label="混音人声增益")
                            retry_instrumental_gain = gr.Slider(0, 2, value=1, step=.05, label="伴奏增益")
                        retry_vocal_mode = gr.Dropdown([("不分离主唱和和声", "off"), *VOCAL_MODES], value="off",
                                                       label="主唱 / 和声处理方式")
                        with gr.Row():
                            retry_harmony_pitch = gr.Slider(-24, 24, value=0, step=1, label="和声额外音高偏移 / 半音")
                            retry_harmony_gain = gr.Slider(0, 2, value=1, step=.05, label="和声增益")
                        with gr.Row():
                            retry_separator = gr.Dropdown(separation.VOCAL_CHOICES, value=separation.VOCAL_DEFAULT,
                                label="人声 / 伴奏分离模型")
                            retry_karaoke = gr.Dropdown(separation.KARAOKE_CHOICES, value=separation.KARAOKE_DEFAULT,
                                label="主唱 / 和声分离模型")
                        with gr.Row():
                            retry_dry_lead = gr.Checkbox(label="主唱／未分轨人声去混响", value=False)
                            retry_dry_backing = gr.Checkbox(label="和声去混响（需分离主唱和声）", value=False)
                        gr.Markdown("改变分离模型会重新生成对应分离音轨；模型和输入匹配时复用缓存，原任务结果保留。", elem_classes=["workspace-hint"])
                    with gr.Row(elem_classes=["compact-actions"]):
                        retry_cancel = gr.Button("取消")
                        retry_confirm = gr.Button("确认重试", variant="primary")
                with gr.Accordion("参考原轨的音准分析与一键校准", open=False):
                    gr.Markdown("选择已完成的转换任务后使用。主唱参考原主唱，和声参考原和声，并计入主动移调；原结果保留。只处理两种检测器均认可的持续偏差，无法保证修复多声部混合和声。", elem_classes=["workspace-hint"])
                    pitch_method = gr.Dropdown([("RMVPE", "rmvpe"), ("FCPE", "fcpe")], value="rmvpe", label="音准分析方法（另一种用于交叉检查）")
                    pitch_backend = gr.Radio(pitch_job.BACKENDS, value="report", label="分析 / 校准方式")
                    pitch_threshold = gr.Slider(20, 200, value=50, step=5, label="持续偏差触发阈值 / 音分", info="100 音分为一个半音。")
                    pitch_strength = gr.Slider(0, 1, value=1, step=.05, label="校准强度")
                    pitch_submit = gr.Button("分析 / 一键校准所选任务", variant="primary")
                    gr.Markdown("PSOLA 与 PC-NSF 可同时生成两版供试听。处理会增加时间；神经重合成也可能影响音色。", elem_classes=["workspace-hint"])
                with gr.Group(visible=False) as delete_confirm:
                    gr.Markdown("**确认删除所选任务？** 删除后不可恢复。训练模型和已导入音色仍保留；数据整理任务不能在此删除。")
                    with gr.Row(elem_classes=["compact-actions"]):
                        delete_cancel = gr.Button("取消")
                        delete_accept = gr.Button("确认删除", variant="stop")
                with gr.Accordion("处理日志与资源状态", open=False):
                    backend = gr.Markdown(service.resource_status(), elem_classes=["workspace-hint"])
                    logs = gr.Textbox(label="后台日志", lines=12, interactive=False, elem_classes=["process-log"])
                with gr.Group(visible=False,elem_classes=['svc-task-results']) as task_result:
                    gr.Markdown("### 分离音轨、转换结果与下载")
                    gr.Markdown("各音轨保留相同时间轴，支持波形定位、试听和下载。波形使用缓存预览；点击播放或定位时才加载音频，切换任务会停止旧音频。", elem_classes=["workspace-hint"])
                    gr.HTML(task_players.AMPLITUDE_LEGEND)
                    with gr.Accordion('音准、混响与编辑报告',open=False):
                        pitch_result = gr.HTML()
                    audio_editor = audio_edit_ui.controls()
                    dataset_preview_source = gr.Dropdown(label="试听素材", choices=[], visible=False)
                    task_audio = task_players.grouped_players(CONVERSION_TRACKS)
                    downloads = gr.HTML()
    with gr.Group(visible=False, elem_id="svc-submit-dialog") as submit_dialog:
        gr.Markdown("### 转换任务已启动\n任务在后台处理，可以前往任务中心查看进度与结果。")
        with gr.Row(elem_classes=["compact-actions"]):
            dismiss_submission = gr.Button("取消")
            open_submission = gr.Button("前往任务中心", variant="primary")
    edits_download = gr.File(label="审核修改记录", interactive=False, visible=False)
    timer = gr.Timer(3, active=False)
    task_timer = gr.Timer(5, active=False)
    polled_state = gr.State({})
    audio_edit_ui.bind(audio_editor,job,timer,safe)

    def listing(kind, state, search, page):
        return task_listing(kind, state, search, page)

    def first_listing(kind, state, search):
        return task_listing(kind, state, search, 1)

    task_filters = [task_kind, task_state, task_search]
    task_outputs = [task_table, task_count, task_page]
    task_kind.change(first_listing, task_filters, task_outputs, queue=False)
    task_state.change(first_listing, task_filters, task_outputs, queue=False)
    task_search.change(first_listing, task_filters, task_outputs, queue=False)
    task_refresh.click(listing, [*task_filters, task_page], task_outputs, queue=False)
    def poll_listing(kind,state,search,page,current_rows,current_summary):
        data,summary,number=task_listing(kind,state,search,page)
        current=current_rows.tolist() if hasattr(current_rows,'tolist') else current_rows
        return (gr.skip() if current==data else data,
                gr.skip() if current_summary==summary else summary,
                gr.skip() if number==page else number)

    task_timer.tick(poll_listing, [*task_filters, task_page, task_table, task_count], task_outputs,
                    queue=False,trigger_mode='always_last',show_progress='hidden')
    tasks_tab.select(lambda kind, state, search, page: (*task_listing(kind, state, search, page), gr.update(active=True)),
                     [*task_filters, task_page], [*task_outputs, task_timer], queue=False,show_progress='hidden')
    for other_tab in (dataset_tab, training_tab, conversion_tab):
        other_tab.select(lambda: gr.update(active=False), [], [task_timer], queue=False)

    def shifted_listing(kind, state, search, page, delta):
        try:
            number = int(page or 1)
        except (TypeError, ValueError, OverflowError):
            number = 1
        return task_listing(kind, state, search, number + delta)

    task_previous.click(lambda kind, state, search, page: shifted_listing(kind, state, search, page, -1),
                        [*task_filters, task_page], task_outputs, queue=False)
    task_next.click(lambda kind, state, search, page: shifted_listing(kind, state, search, page, 1),
                    [*task_filters, task_page], task_outputs, queue=False)
    task_page.submit(listing, [*task_filters, task_page], task_outputs, queue=False)

    @safe(outputs=2)
    def select_task(data, evt: gr.SelectData):
        index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        if not isinstance(index, int) or index < 0 or index >= len(data):
            raise ValueError("请选择任务表格中的一行")
        identity = db.identifier(data[index][-1])
        return gr.update(choices=job_choices(), value=identity), gr.update(elem_classes=["workbench-panel", "svc-task-detail"])

    task_table.select(select_task, [task_table], [job, task_detail], queue=False,
                      trigger_mode="always_last", show_progress="hidden")

    def name_error(error, *args):
        name = str(args[-1] or "").strip()
        if "歌手名称" in str(error) and (not name or len(name) > 100):
            return gr.skip(), gr.update(elem_classes=["svc-name-input", "svc-input-invalid"], info=str(error)), gr.skip()
        return gr.skip(), gr.skip(), gr.skip()

    new_name.input(lambda: gr.update(elem_classes=["svc-name-input"], info=None), [], [new_name], queue=False)

    @safe(outputs=3, fallback=name_error)
    def add_identity(name):
        record = db.add_singer(name)
        return gr.update(choices=singer_choices(), value=record["id"]), gr.update(value="", elem_classes=["svc-name-input"], info=None), "歌手已创建，后续追加素材请选择此身份。"

    add.click(add_identity, [new_name], [singer, new_name, notice])

    @safe(outputs=3, fallback=name_error)
    def rename_identity(singer_id, name):
        if not singer_id:
            raise ValueError("请先选择需要重命名的歌手")
        db.rename_singer(singer_id, name)
        return gr.update(choices=singer_choices(), value=singer_id), gr.update(value="", elem_classes=["svc-name-input"], info=None), "歌手名称已更新，身份及素材归档保持一致。"

    rename.click(rename_identity, [singer, new_name], [singer, new_name, notice])
    singer.change(lambda identity: gr.update(value=identity), [singer], [training_singer], queue=False)

    def refresh_lists(current_singer, current_job, current_experiment):
        return (gr.update(choices=singer_choices(), value=current_singer), gr.update(choices=job_choices(), value=current_job),
                gr.update(choices=experiment_choices(), value=current_experiment), service.resource_status())

    @safe
    def statistics(singer_id):
        if not singer_id:
            return "请先选择歌手。"
        records = db.clips(singer_id)
        seconds = sum(row["duration"] for row in records)
        sources = len({row["source_id"] for row in records})
        return f"**{len(records)} 个已审核片段 · {seconds / 60:.1f} 分钟 · {sources} 份源素材**"

    training_singer.change(statistics, [training_singer], [dataset_stats])

    def validation_text(plan):
        train = plan["training"]
        val = plan["validation"]
        message = (f"**训练：{len(train)} 段 · {plan['train_seconds'] / 60:.2f} 分钟 · {len({row['source_id'] for row in train})} 份源素材**  \n"
                   f"验证：{len(val)} 段 · {plan['validation_seconds'] / 60:.2f} 分钟 · {len({row['source_id'] for row in val})} 份源素材")
        if plan["warning"]:
            message += "  \n" + escape(plan["warning"])
        return message

    @safe(outputs=3)
    def validation_choices(identity):
        if not identity or len(db.clips(identity)) < 2:
            return gr.update(choices=[], value=[], interactive=False), "至少入库两个片段后可预览划分。", ""
        plan = service.validation_plan(identity)
        multiple = len(plan["source_rows"]) > 1
        choices = [(f"{row['name']} · {row['clips']} 段 · {row['seconds'] / 60:.1f} 分钟 · {row['id'][:6]}", row["id"])
                   for row in plan["source_rows"]] if multiple else []
        return (gr.update(choices=choices, value=plan["selected_sources"] if multiple else [], interactive=multiple),
                validation_text(plan), validation_distribution(plan))

    def preview_validation(identity, selected):
        if not identity or len(db.clips(identity)) < 2:
            return "至少入库两个片段后可预览划分。", ""
        try:
            plan = service.validation_plan(identity, selected)
            return validation_text(plan), validation_distribution(plan)
        except ValueError as error:
            return "**请调整验证源：** " + escape(str(error)), ""

    training_singer.change(validation_choices, [training_singer], [validation_sources, validation_summary, validation_chart])
    training_tab.select(validation_choices, [training_singer], [validation_sources, validation_summary, validation_chart])
    validation_sources.change(preview_validation, [training_singer, validation_sources], [validation_summary, validation_chart])

    report_outputs = [report_status, report_metrics, report_notes, pitch_chart, length_chart,
                      source_chart, source_table, length_table, problem_table, problem_clip, report_files]

    @safe(outputs=len(report_outputs))
    def report_view(identity, predictor):
        if not identity:
            return "请先选择歌手。", "", "", None, None, None, [], [], [], gr.update(choices=[], value=None), None
        report, directory, stale = dataset_report.load(identity)
        if report is None:
            return "尚无报告；入库后点击“生成训练前报告”。", "", "", None, None, None, [], [], [], gr.update(choices=[], value=None), None
        if stale:
            return "入库清单或文件已变化，报告已过期，请重新生成。", "", "", None, None, None, [], [], [], gr.update(choices=[], value=None), None
        pitch = report["pitch_summary"]
        pitch_text = f"{pitch['p5']['name']}–{pitch['p95']['name']}" if pitch else "未检测到有声 F0"
        metrics = [("校验有效片段", f"{report['valid_clips']}/{report['clips']}"),
                   ("有效时长", f"{report['duration_seconds'] / 60:.2f} 分钟"),
                   ("有效源素材", str(len(report["sources"]))), ("有声帧占比", f"{report['voiced_ratio']:.1%}")]
        html = '<div class="review-metrics">' + "".join(f'<div><span>{label}</span><strong>{escape(value)}</strong></div>' for label, value in metrics) + '</div>'
        message = f"报告已生成 · F0 {report['predictor'].upper()} · 音高中央 90%：{pitch_text}（按半音分桶）"
        if report["predictor"] != predictor:
            message += "。报告提取器与当前训练选择不同，可重新生成对比。"
        problems = report.get("problem_clips", [])
        message += f" · 建议复核 {len(problems)} 段"
        notes = "  \n".join("• " + escape(warning) for warning in report["warnings"])
        notes += "  \n音高只统计提取器判断为有声的帧，提取范围 50–1100 Hz；数据覆盖统计不代表音色质量或训练效果。"
        table = [[index + 1, row["name"], row["clips"], round(row["seconds"] / 60, 3), round(row["share"] * 100, 2)]
                 for index, row in enumerate(report["sources"])]
        lengths = [[label, report["lengths"][label]] for label, _, _ in dataset_report.LENGTH_BINS]
        shown = problems[:100]
        problem_rows = [[row["source_name"], round(row["start"], 2) if row["start"] is not None else "—",
                         round(row["end"], 2) if row["end"] is not None else "—",
                         round(row["voiced_ratio"] * 100, 1) if row["voiced_ratio"] is not None else "—",
                         "；".join(row["reasons"])] for row in shown]
        problem_choices = [(f"{row['source_name']} · {row['start'] or 0:.1f}s · {row['reasons'][0]}", row["id"]) for row in shown]
        files = [str(directory / name) for name in ["report.html", "report.json", "sources.csv", "pitch.png", "lengths.png", "sources.png"]]
        return (message, html, notes, str(directory / "pitch.png"), str(directory / "lengths.png"),
                str(directory / "sources.png"), table, lengths, problem_rows,
                gr.update(choices=problem_choices, value=problem_choices[0][1] if problem_choices else None), files)

    report_control_outputs = [make_report, report_ready_status, view_report]

    @safe(outputs=3)
    def report_controls(identity):
        if not identity:
            return gr.update(value="生成训练前报告", interactive=False), "选择歌手后可生成训练前报告。", gr.update(visible=False)
        task = service.report_task(identity)
        active = task and task["status"] not in service.TERMINAL
        report, _, stale = dataset_report.load(identity)
        ready = bool(report and not stale)
        if active:
            message = "**报告正在后台分析** · " + escape(task.get("message", ""))
        elif ready:
            message = f"**报告已完成，可查看结果** · {report['clips']} 段 · F0 {report['predictor'].upper()}"
        elif stale:
            message = "报告已过期，请按当前入库数据重新生成。"
        else:
            message = "尚无训练前数据报告。"
        return (gr.update(value="报告分析中…" if active else "生成训练前报告", interactive=not bool(active)),
                message, gr.update(visible=ready, value="查看已有报告" if active else "查看训练前报告"))

    def report_submit_error(error, *args):
        return gr.skip(), "报告提交失败：" + escape(str(error)), gr.update(value="生成训练前报告", interactive=True)

    @safe(outputs=3, fallback=report_submit_error)
    def build_report(identity, predictor):
        task = service.create_report(identity, predictor)
        completed = db.read(db.job_dir(task) / "job.json")["status"] == "completed"
        return (gr.update(choices=job_choices(), value=task),
                "已有当前数据和 F0 的报告，可点击上方“查看训练前报告”。" if completed else "报告已提交后台分析；可在上方查看进度或停止任务。",
                gr.update(value="生成训练前报告" if completed else "报告分析中…", interactive=completed))

    make_report.click(lambda: gr.update(value="正在提交…", interactive=False), [], [make_report], queue=False).then(
        build_report, [training_singer, train_f0], [job, report_status, make_report])
    reload_report.click(report_view, [training_singer, train_f0], report_outputs)
    training_singer.change(report_view, [training_singer, train_f0], report_outputs)
    train_f0.change(report_view, [training_singer, train_f0], report_outputs)
    training_tab.select(report_view, [training_singer, train_f0], report_outputs)
    training_tab.select(statistics, [training_singer], [dataset_stats])
    training_singer.change(report_controls, [training_singer], report_control_outputs)
    train_f0.change(report_controls, [training_singer], report_control_outputs)
    training_tab.select(report_controls, [training_singer], report_control_outputs)

    @safe(outputs=1 + len(report_outputs))
    def open_report(identity, predictor):
        return gr.update(open=True), *report_view.__wrapped__(identity, predictor)

    view_report.click(open_report, [training_singer, train_f0], [report_panel, *report_outputs])

    @safe(outputs=2)
    def submit_dataset(identity, files, lrcs, text, kind, off, low, high, device, silence, silent_percent, clipping_percent, separator, split, include_backing, karaoke, dry_lead, dry_backing):
        guard()
        task, message = service.create_dataset(identity, files, lrcs, text or "", kind == "带伴奏的音乐", off, low, high, device,
            quality_settings={"silence_db": float(silence), "max_silence_ratio": float(silent_percent) / 100, "max_clipping_ratio": float(clipping_percent) / 100},
            separation_model=separator, split_vocals=split, include_backing=include_backing if split else False,
            vocal_split_model=karaoke, dereverb_lead=dry_lead, dereverb_backing=dry_backing if split else False)
        return gr.update(choices=job_choices(), value=task), message

    dataset_split.change(lambda enabled: gr.update(visible=enabled), [dataset_split], [dataset_backing_options], queue=False)

    prepare.click(submit_dataset, [singer, audio_files, lyric_files, lyric_text, input_kind, offset, minimum, maximum, dataset_gpu, silence_db, max_silence, max_clipping, dataset_separator, dataset_split, dataset_include_backing, dataset_karaoke, dataset_dry_lead, dataset_dry_backing], [job, notice])

    @safe(outputs=3)
    def load_rows(job_id, kind, order="风险优先", current=None):
        table_rows, choices = rows(job_id, kind, order)
        edit_file = db.job_dir(job_id) / "edits.json" if job_id else None
        if current not in {value for _, value in choices}:
            current = history.cursor(job_id)
        selected = current if current in {value for _, value in choices} else choices[0][1] if choices else None
        return table_rows, gr.update(choices=choices, value=selected), str(edit_file) if edit_file and edit_file.exists() else None

    for control in [job, review_filter, review_order]:
        control.change(load_rows, [job, review_filter, review_order, clip], [table, clip, edits_download])

    history_outputs = [revision_choice, before_choice, after_choice, before_audio, after_audio, revision_info, undo_button]
    clip_outputs = [preview, context_audio, start, end, split_at, metrics, boundary_state, boundary_info,
                    apply_recommended, start_preview, end_preview, suggested_start_preview, suggested_end_preview,
                    *history_outputs, review_progress, cache_info, cached_paths, feedback_info, decline_recommended,
                    approve_next, reject_next, previous, following]

    def progress_html(job_id):
        counts = history.progress(job_id)
        done = counts["approved"] + counts["rejected"]
        total = counts["total"]
        percent = round(100 * done / total) if total else 0
        return (f'<div class="task-status" role="status" aria-atomic="true"><strong>审核 {done}/{total} · {percent}%</strong>'
                f'<p>已确认 {counts["approved"]} · 已排除 {counts["rejected"]} · 待审核 {counts["pending"]}</p>'
                f'<progress value="{done}" max="{max(1, total)}" aria-label="审核完成进度"></progress></div>')

    def revision_file(job_id, row):
        path = Path(row["file"]).resolve()
        if path.parent != (db.job_dir(job_id) / "clips").resolve() or not path.is_file() or db.digest(path) != row["sha256"]:
            return None
        return str(path)

    def history_view(job_id, current=None, chosen=None):
        records = [entry for entry in history.edits(job_id) if entry["action"] in history.GEOMETRY | {"undo"}] if job_id else []
        options = []
        for index, entry in enumerate(records):
            stamp = datetime.fromisoformat(entry["at"]).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%m-%d %H:%M")
            options.append((f"{index + 1} · {history.ACTION_NAMES[entry['action']]} · {stamp}", history.key(entry)))
        matches = [entry for entry in records if any(row["id"] == current for row in entry["before"] + entry["after"])]
        entry = next((item for item in records if history.key(item) == chosen), None)
        entry = entry or (matches[-1] if matches else records[-1] if records else None)
        def side(name):
            saved = entry[name] if entry else []
            choices = [(f"{index + 1} · {row['start']:.4f}–{row['end']:.4f} 秒 · 修订 {row.get('revision', 0)}", str(index))
                       for index, row in enumerate(saved)]
            return gr.update(choices=choices, value="0" if saved else None), revision_file(job_id, saved[0]) if saved else None
        old_choices, old_audio = side("before")
        new_choices, new_audio = side("after")
        candidate = history.undo_candidate(job_id) if job_id else None
        task = db.read(db.job_dir(job_id) / "job.json") if job_id else {}
        message = (f"**{history.ACTION_NAMES[entry['action']]}**：修改前 {len(entry['before'])} 段 → 修改后 {len(entry['after'])} 段。" if entry else "还没有边界修订记录。")
        if candidate:
            message += "  \n最近可撤销：" + history.ACTION_NAMES[candidate["action"]] + "；恢复后重新等待审核。"
        return (gr.update(choices=options, value=history.key(entry) if entry else None), old_choices, new_choices,
                old_audio, new_audio, message, gr.update(interactive=bool(candidate) and not service.live(task)))

    def clip_tail(job_id, current=None, paths=None, available=False, kind="全部", order="风险优先"):
        task = db.read(db.job_dir(job_id) / "job.json") if job_id else {}
        record = next((row for row in service.records(job_id) if row["id"] == current), None) if job_id else None
        _, choices = rows(job_id, kind, order)
        ids = [value for _, value in choices]
        position = ids.index(current) if current in ids else -1
        idle = not task or not service.live(task)
        return (*history_view(job_id, current), progress_html(job_id), preview_cache.summary(job_id),
                {"task": job_id, "paths": paths or []}, feedback.summary(task["singer_id"]) if task.get("singer_id") else gr.skip(),
                gr.update(interactive=available and idle),
                gr.update(interactive=bool(record) and record["quality"] != "rejected" and idle),
                gr.update(interactive=bool(record) and idle), gr.update(interactive=position > 0),
                gr.update(interactive=0 <= position < len(ids) - 1))

    def clip_failure(error, *args):
        values = [None, None, gr.skip(), gr.skip(), gr.skip(), "无法读取片段：" + escape(str(error)),
                  None, "边界建议不可用，请重新选择片段。", gr.update(interactive=False),
                  gr.update(value=None), gr.update(value=None), gr.update(value=None, visible=False), gr.update(value=None, visible=False)]
        values.extend(gr.skip() for _ in clip_outputs[13:])
        values[clip_outputs.index(undo_button)] = gr.update(interactive=False)
        values[clip_outputs.index(decline_recommended)] = gr.update(interactive=False)
        for control in [approve_next, reject_next, previous, following]:
            values[clip_outputs.index(control)] = gr.update(interactive=False)
        return tuple(values)

    @safe(outputs=len(clip_outputs), fallback=clip_failure)
    def load_clip(job_id, clip_id, kind="全部", order="风险优先"):
        if not job_id or not clip_id:
            return (None, None, 0, 0, 0, "", None, "选择片段后显示切点检查。", gr.update(interactive=False),
                    gr.update(value=None), gr.update(value=None), gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                    *clip_tail(job_id, kind=kind, order=order))
        all_records, _, record = service.find_clip(job_id, clip_id)
        plan = service.boundary_plan(job_id, clip_id, all_records)
        record = annotate(dict(record), plan["analysis"])
        source = db.source_audio(record)
        info = sf.info(source)
        context_path = preview_cache.excerpt(job_id, source, max(0, record["start"] - 1), min(info.duration, record["end"] + 1))
        paths = [context_path]
        detail = (f"**{priority(record)['label']} · {QUALITY[record['quality']]} · {REVIEW[record['review']]}**  \n"
                  f"时长 {record['duration']:.2f} 秒 · RMS {record['rms_db']:.1f} dB · 静音占比 {record['silence_ratio']:.0%} · 削波 {record['clipping_ratio']:.2%}  \n"
                  + escape("；".join(dict.fromkeys(record.get("flags", []) + record.get("boundary_flags", []) + record.get("alignment_flags", [])))))
        summary, current_audio, suggested_audio = [], [], []
        for key, name in [("start", "起点"), ("end", "终点")]:
            cut = plan["analysis"]["cuts"][key]
            point, suggested = cut["point"], cut["suggested"]
            summary.append(f"**{name} {point:.4f} 秒**：{cut['reason']}" + (f" → **推荐 {suggested:.4f} 秒**（移动 {suggested - point:+.3f} 秒）" if suggested is not None else ""))
            def cut_audio(position, label):
                left, right = max(0, position - 1), min(info.duration, position + 1)
                path = preview_cache.excerpt(job_id, source, left, right)
                paths.append(path)
                return gr.update(value=path, visible=True,
                                 label=f"{label} · 切点在试听 {position - left:.2f} 秒处")
            current_audio.append(cut_audio(point, "当前" + name))
            suggested_audio.append(cut_audio(suggested, "推荐" + name) if suggested is not None else gr.update(value=None, visible=False))
        profile = plan["profile"]
        summary.append(f"搜索切点附近 ±{profile['search_radius']:g} 秒、至少 {profile['pause_seconds'] * 1000:g} 毫秒的低能量停顿；须试听确认。")
        if any(plan["analysis"][key] for key in ["shared_start", "shared_end"]):
            summary.append("共用切点移动时会同步修正相邻片段；受影响片段重新等待试听确认。")
        available = any(cut["suggested"] is not None for cut in plan["analysis"]["cuts"].values())
        task = db.read(db.job_dir(job_id) / "job.json")
        history.remember(job_id, clip_id)
        return (record["file"], context_path, record["start"], record["end"], (record["start"] + record["end"]) / 2, detail,
                plan, "  \n\n".join(summary), gr.update(interactive=available and not service.live(task)), *current_audio, *suggested_audio,
                *clip_tail(job_id, clip_id, paths, available, kind, order))

    clip.change(load_clip, [job, clip, review_filter, review_order], clip_outputs)

    @safe(outputs=8 + len(clip_outputs))
    def open_report_problem(identity, clip_id):
        report, _, stale = dataset_report.load(identity)
        if report is None or stale:
            raise ValueError("报告已过期，请重新生成后打开片段")
        problem = next((row for row in report.get("problem_clips", []) if row["id"] == clip_id), None)
        if problem is None:
            raise ValueError("请先选择报告中的待复核片段")
        published = next((row for row in db.clips(identity) if row["id"] == clip_id), None)
        task = service.review_location(identity, clip_id, problem.get("job_id"))
        _, _, current = service.find_clip(task, clip_id)
        if published is None or current["sha256"] != published["sha256"]:
            raise ValueError("原审核片段已被修改，请刷新报告后再定位")
        table_rows, choices = rows(task, "全部", "风险优先")
        return (gr.update(selected="dataset"), gr.update(value=identity),
                gr.update(choices=job_choices(), value=task), gr.update(value="全部"), gr.update(value="风险优先"),
                table_rows, gr.update(choices=choices, value=clip_id), "已打开报告提示的片段，请试听判断。",
                *load_clip.__wrapped__(task, clip_id, "全部", "风险优先"))

    open_problem.click(open_report_problem, [training_singer, problem_clip],
                       [svc_tabs, singer, job, review_filter, review_order, table, clip, notice, *clip_outputs])

    def navigate(job_id, current, kind, order, direction):
        _, choices = rows(job_id, kind, order)
        ids = [choice[1] for choice in choices]
        if not ids:
            return None
        position = ids.index(current) if current in ids else 0
        return ids[max(0, min(len(ids) - 1, position + direction))]

    previous.click(safe(lambda a, b, c, d: navigate(a, b, c, d, -1)), [job, clip, review_filter, review_order], [clip])
    following.click(safe(lambda a, b, c, d: navigate(a, b, c, d, 1)), [job, clip, review_filter, review_order], [clip])

    @safe(outputs=4 + len(clip_outputs))
    def edit(job_id, clip_id, a, b, point, kind, order, plan, action, advance=False):
        _, old_choices = rows(job_id, kind, order)
        selected = service.edit_clip(job_id, clip_id, action, a, b, point, recommendation=plan)
        if advance and kind in {"已确认", "已排除"}:
            kind = "待复核"
        table_rows, choices = rows(job_id, kind, order)
        values = {value for _, value in choices}
        if action in history.GEOMETRY and selected not in values:
            kind = "待复核"
            table_rows, choices = rows(job_id, kind, order)
            values = {value for _, value in choices}
        current = selected if selected in values else choices[0][1] if choices else None
        if advance:
            pending = {row["id"] for row in service.records(job_id) if row["review"] == "pending"} & values
            old_ids = [value for _, value in old_choices]
            position = old_ids.index(clip_id) if clip_id in old_ids else -1
            sequence = old_ids[position + 1:] + old_ids[:position + 1]
            current = next((value for value in sequence if value in pending), next((value for _, value in choices if value in pending), None))
        message = ("推荐切点已应用；受影响片段已恢复待审核，请重新试听确认后入库。" if action == "recommend"
                   else "审核修改已保存；边界修改后请重新试听确认。")
        if advance:
            message = "审核已保存，继续下一条待审核片段。" if current else "当前筛选内没有待审核片段；可切换筛选查看其他片段。"
        return (table_rows, gr.update(choices=choices, value=current), message,
                *load_clip(job_id, current, kind, order), gr.update(value=kind))

    for button, action in [(bounds, "bounds"), (split, "split"), (merge, "merge"), (approve, "approved"), (reject, "rejected"), (restore, "pending"), (apply_recommended, "recommend")]:
        button.click(lambda *args, _action=action: edit(*args, _action), [job, clip, start, end, split_at, review_filter, review_order, boundary_state],
                     [table, clip, notice, *clip_outputs, review_filter])

    edit_inputs = [job, clip, start, end, split_at, review_filter, review_order, boundary_state]
    edit_outputs = [table, clip, notice, *clip_outputs, review_filter]
    approve_next.click(lambda *args: edit(*args, "approved", advance=True), edit_inputs, edit_outputs)
    reject_next.click(lambda *args: edit(*args, "rejected", advance=True), edit_inputs, edit_outputs)

    @safe(outputs=len(edit_outputs))
    def undo_edit(task, kind, order):
        selected = history.undo(task)
        table_rows, choices = rows(task, kind, order)
        if selected not in {value for _, value in choices}:
            kind = "待复核"
            table_rows, choices = rows(task, kind, order)
        current = selected if selected in {value for _, value in choices} else choices[0][1] if choices else None
        return (table_rows, gr.update(choices=choices, value=current), "最近一次边界修改已撤销；恢复片段已生成新修订，请重新试听。",
                *load_clip(task, current, kind, order), gr.update(value=kind))

    undo_button.click(undo_edit, [job, review_filter, review_order], edit_outputs)

    revision_choice.input(safe(lambda task, current, chosen: history_view(task, current, chosen), outputs=len(history_outputs)),
                          [job, clip, revision_choice], history_outputs)

    @safe(fallback=lambda *args: None)
    def choose_revision_audio(task, chosen, index, side):
        if chosen is None or index is None:
            return None
        entry = next((entry for entry in history.edits(task) if history.key(entry) == chosen), None)
        if entry is None or not str(index).isdigit() or not 0 <= int(index) < len(entry[side]):
            raise ValueError("修订片段不可用，请重新选择记录")
        return revision_file(task, entry[side][int(index)])

    before_choice.change(lambda *args: choose_revision_audio(*args, "before"), [job, revision_choice, before_choice], before_audio)
    after_choice.change(lambda *args: choose_revision_audio(*args, "after"), [job, revision_choice, after_choice], after_audio)

    @safe(outputs=2)
    def decline(task, current, plan, reason):
        service.decline_recommendation(task, current, plan, reason)
        record = db.read(db.job_dir(task) / "job.json")
        return "已记录不采用建议；已确认通过的原片段可用于校准。", feedback.summary(record["singer_id"])

    decline_recommended.click(decline, [job, clip, boundary_state, decline_reason], [notice, feedback_info])

    @safe(outputs=2)
    def clean_cache(task, days, protected):
        record = db.read(db.job_dir(task) / "job.json")
        if service.live(record):
            raise ValueError("请等待任务结束后清理试听缓存")
        paths = protected.get("paths", []) if protected and protected.get("task") == task else []
        message = preview_cache.clean(task, days, paths)
        return preview_cache.summary(task), message

    clear_cache.click(clean_cache, [job, cache_days, cached_paths], [cache_info, cache_notice])

    @safe(outputs=4)
    def calibration_view(identity):
        return feedback.summary(identity), feedback.report_text(None), None, gr.update(interactive=False)

    singer.change(calibration_view, [singer], [feedback_info, calibration_report, calibration_state, apply_profile])

    @safe(outputs=4)
    def generate_calibration(identity):
        report = feedback.calibrate(identity)
        return feedback.summary(identity), feedback.report_text(report), report, gr.update(interactive=report["can_apply"])

    calibrate_button.click(generate_calibration, [singer], [feedback_info, calibration_report, calibration_state, apply_profile])

    profile_outputs = [table, clip, notice, *clip_outputs, calibration_report, calibration_state, apply_profile]

    @safe(outputs=len(profile_outputs))
    def change_profile(identity, task, current, kind, order, report, reset=False, recheck_only=False):
        if task:
            record = db.read(db.job_dir(task) / "job.json")
            if record.get("singer_id") != identity:
                raise ValueError("当前任务与歌手不一致，请先选择该歌手的数据集任务")
            if service.live(record):
                raise ValueError("请等待当前任务结束后修改边界参数")
        if recheck_only and (not task or record["kind"] != "dataset"):
            raise ValueError("请先选择当前歌手的数据集任务")
        if reset:
            feedback.reset(identity)
        elif not recheck_only:
            feedback.apply(identity, report)
        message = "边界参数已更新；音频与审核决定保留。"
        if task and record["kind"] == "dataset":
            try:
                service.recheck_job(task)
            except Exception as error:
                message = "参数已保存，但当前任务重新检查未完成：" + escape(str(error))
                gr.Warning(message, duration=5)
        table_rows, choices = rows(task, kind, order)
        selected = current if current in {value for _, value in choices} else choices[0][1] if choices else None
        loaded = list(load_clip(task, selected, kind, order))
        loaded[clip_outputs.index(feedback_info)] = feedback.summary(identity)
        return (table_rows, gr.update(choices=choices, value=selected), message,
                *loaded, feedback.report_text(None), None, gr.update(interactive=False))

    apply_profile.click(change_profile, [singer, job, clip, review_filter, review_order, calibration_state], profile_outputs)
    reset_profile.click(lambda *args: change_profile(*args, reset=True),
                        [singer, job, clip, review_filter, review_order, calibration_state], profile_outputs)
    recheck_profile.click(lambda *args: change_profile(*args, recheck_only=True),
                          [singer, job, clip, review_filter, review_order, calibration_state], profile_outputs)

    @safe(outputs=4)
    def publish_dataset(job_id, singer_id, kind, order, current, confirm=False):
        message = service.publish(job_id, confirm)
        table_rows, choices = rows(job_id, kind, order)
        selected = current if current in {value for _, value in choices} else choices[0][1] if choices else None
        return message, table_rows, gr.update(choices=choices, value=selected), statistics.__wrapped__(singer_id)

    publish.click(lambda *args: publish_dataset(*args), [job, singer, review_filter, review_order, clip], [notice, table, clip, dataset_stats])
    publish_safe.click(lambda *args: publish_dataset(*args, confirm=True), [job, singer, review_filter, review_order, clip], [notice, table, clip, dataset_stats])

    @safe
    def export_dataset(singer_id):
        records = db.clips(singer_id)
        if not records:
            raise ValueError("当前歌手没有已审核入库片段")
        target = db.DATA / "datasets" / singer_id / "dataset.zip"
        with db.lock(), zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(target.parent / "manifest.json", "manifest.json")
            for record in records:
                archive.write(record["file"], f"{singer_id}/{Path(record['file']).name}")
        return gr.update(value=str(target), visible=True)

    export.click(export_dataset, [singer], [dataset_download])

    @safe(outputs=3)
    def submit_training(identity, name, steps, batch_size, interval, device, f0, g, d, validation_ids):
        guard()
        task = service.create_experiment(identity, name, steps, batch_size, interval, device, f0, g, d,
                                         validation_source_ids=validation_ids)
        record = db.read(db.job_dir(task) / "job.json")
        return gr.update(choices=job_choices(), value=task), gr.update(choices=experiment_choices(), value=record["options"]["experiment_id"]), "训练已提交，使用已审核切片的固定快照。"

    train_button.click(submit_training, [training_singer, experiment_name, train_steps, batch, save_every, train_gpu, train_f0, initial_g, initial_d,
                                         validation_sources], [job, experiment, notice])

    @safe(outputs=5)
    def experiment_details(experiment_id):
        if not experiment_id:
            empty = gr.update(choices=[], value=None)
            return "选择实验后显示检查点。", empty, empty, empty, empty
        directory = db.experiment_dir(experiment_id)
        record = db.read(directory / "experiment.json")
        choices = [(f"第 {path.stem.split('_')[-1]} 步", str(path)) for path in service.checkpoints(experiment_id)]
        selection = db.read(directory / "previews/selection.json", [])
        if selection:
            clips = []
            for index, row in enumerate(selection):
                note = dataset_report.note_name(round(row["median_note"])) if row.get("median_note") is not None else "F0 未检出"
                label = f"{index + 1} · {row['source_name']} · {note} · {row['duration']:.1f} 秒"
                clips.append((label, row["id"]))
        else:
            clips = [("旧实验固定验证片段", "legacy")] if (directory / "previews/reference.wav").exists() else []
        return (escape(record["name"] + " · " + record.get("warning", "")),
                gr.update(choices=choices, value=choices[-1][1] if choices else None),
                gr.update(choices=choices, value=choices[-2][1] if len(choices) > 1 else None),
                gr.update(choices=clips, value=clips[0][1] if clips else None),
                gr.update(choices=choices[:-1], value=None))

    experiment_outputs = [experiment_info, checkpoint, compare_checkpoint, preview_clip, cleanup_checkpoint]
    experiment.change(experiment_details, [experiment], experiment_outputs)
    experiment.change(lambda: gr.update(value=False), [], [confirm_cleanup], queue=False)
    cleanup_checkpoint.change(lambda: gr.update(value=False), [], [confirm_cleanup], queue=False)

    @safe(outputs=7)
    def remove_checkpoint(experiment_id, selected, confirmed):
        if not confirmed:
            raise ValueError("请先确认要永久删除所选旧检查点")
        message = service.delete_checkpoint(experiment_id, selected)
        return (*experiment_details.__wrapped__(experiment_id), gr.update(value=False), message)

    cleanup_button.click(remove_checkpoint, [experiment, cleanup_checkpoint, confirm_cleanup], [*experiment_outputs, confirm_cleanup, notice])

    def preview_rms(path):
        if not path:
            return 0.
        sum_squares, count = 0., 0
        with sf.SoundFile(path) as stream:
            while True:
                block = stream.read(65536, dtype="float32")
                if not len(block):
                    break
                sum_squares += float((block.astype("float64") ** 2).sum())
                count += block.size
        return math.sqrt(sum_squares / count) if count else 0.

    @safe(outputs=5)
    def load_checkpoint_preview(experiment_id, checkpoint_path, comparison_path, clip_id):
        if not experiment_id or not checkpoint_path or not clip_id:
            return None, None, None, "选择实验、验证片段和检查点后试听。", '<span id="svc-ab-status" role="status" data-a-rms="0" data-b-rms="0">请选择两个检查点。</span>'
        directory = db.experiment_dir(experiment_id)
        valid = {str(path) for path in service.checkpoints(experiment_id)}
        if checkpoint_path not in valid or comparison_path and comparison_path not in valid:
            raise ValueError("检查点不属于当前实验，请刷新列表")
        selection = db.read(directory / "previews/selection.json", [])
        if clip_id != "legacy" and clip_id not in {row["id"] for row in selection}:
            raise ValueError("验证片段不属于当前实验，请刷新列表")
        previews = directory / "previews"
        def path_for(checkpoint):
            if not checkpoint:
                return None
            step = Path(checkpoint).stem.split("_")[-1]
            path = previews / (f"step-{step}.wav" if clip_id == "legacy" else f"step-{step}-{clip_id}.wav")
            return str(path) if path.exists() else None
        reference = previews / ("reference.wav" if clip_id == "legacy" else f"reference-{clip_id}.wav")
        first, second = path_for(checkpoint_path), path_for(comparison_path)
        message = "两侧使用同一验证片段；请重点听音色、齿音、音高和伴奏残留。"
        if not first or comparison_path and not second:
            message += " 该检查点没有这段固定试听，可选择其他检查点。"
        levels = (f'<span id="svc-ab-status" role="status" data-a-rms="{preview_rms(first):.8f}" '
                  f'data-b-rms="{preview_rms(second):.8f}">点击“A / B 同位置切换”可从当前时间继续试听。</span>')
        return str(reference) if reference.exists() else None, first, second, message, levels

    preview_inputs = [experiment, checkpoint, compare_checkpoint, preview_clip]
    preview_outputs = [reference_audio, checkpoint_preview, comparison_audio, preview_message, ab_status]
    checkpoint.change(load_checkpoint_preview, preview_inputs, preview_outputs)
    compare_checkpoint.change(load_checkpoint_preview, preview_inputs, preview_outputs)
    preview_clip.change(load_checkpoint_preview, preview_inputs, preview_outputs)

    @safe(outputs=2)
    def resume(experiment_id, steps):
        guard()
        if int(steps) <= 0:
            raise ValueError("追加步数须大于 0")
        matches = [row for row in service.history() if row["kind"] == "train" and row["options"]["experiment_id"] == experiment_id]
        if not matches:
            raise ValueError("请选择已有训练实验")
        task = service.retry(matches[0]["id"], int(steps))
        return gr.update(choices=job_choices(), value=task), "续训已提交"

    resume_button.click(resume, [experiment, extra_steps], [job, notice])

    @safe(outputs=6)
    def select_conversion_model(selection):
        if not selection:
            return (gr.update(choices=[], value=None, visible=False), gr.update(choices=[], value=None, visible=False),
                    "选择已训练实验或导入模型后开始转换。", gr.update(visible=False),
                    gr.update(value=False), gr.update(value=35, maximum=100))
        source, model_id = selection.split(":", 1)
        if source == "trained":
            record = next((row for row in service.experiments() if row["id"] == model_id), None)
            if not record:
                raise ValueError("训练实验已不存在，请刷新模型列表")
            choices = [(f"第 {path.stem.split('_')[-1]} 步", str(path)) for path in service.checkpoints(model_id)]
            return (gr.update(choices=choices, value=choices[-1][1] if choices else None, visible=True),
                    gr.update(choices=[], value=None, visible=False),
                    conversion_description(selection),
                    gr.update(visible=False), gr.update(value=False), gr.update(value=35, maximum=100))
        if source == "imported":
            from .external_models import record
            item = record(model_id)
            choices = [(f"{name} · #{index}", name) for name, index in item["speakers"].items()]
            has_diffusion = bool(item.get("diffusion"))
            return (gr.update(choices=[], value=None, visible=False),
                    gr.update(choices=choices, value=choices[0][1], visible=True),
                    conversion_description(selection),
                    gr.update(visible=has_diffusion), gr.update(value=False),
                    gr.update(value=min(35, int(item.get("diffusion_max_steps") or 100)),
                              maximum=int(item.get("diffusion_max_steps") or 100)))
        raise ValueError("目标模型类型无效，请刷新列表")

    model_choice.change(select_conversion_model, [model_choice], [conversion_checkpoint, conversion_speaker,
                                                                 model_details, diffusion_controls, shallow_diffusion,
                                                                 diffusion_steps])
    convert_source.change(lambda source, kind: (gr.update(visible=source == "上传音频"),
                                                gr.update(visible=source == "录音"),
                                                gr.update(visible=source == "录音"),
                                                gr.update(visible=source == "上传音频"),
                                                gr.update(visible=source == "上传音频" and kind == "人声 + 伴奏分别上传", value=None)),
                          [convert_source, convert_kind],
                          [convert_audio, convert_recording, record_kind, convert_kind, convert_backing], queue=False)
    convert_kind.change(lambda kind, source: gr.update(visible=source == "上传音频" and kind == "人声 + 伴奏分别上传", value=None),
                        [convert_kind, convert_source], [convert_backing], queue=False)
    split_enabled.change(lambda enabled: gr.update(visible=enabled), [split_enabled], [split_options], queue=False)

    @safe(outputs=2)
    def scan_imports():
        choices, problems = import_choices()
        hint = "找到可导入模型，请选择权重。" if choices else "尚未找到可导入模型。请检查文件夹、config.json 和 G 权重。"
        if problems:
            hint += "  \n" + "  \n".join(escape(item) for item in problems[:8])
        return gr.update(choices=choices, value=choices[0][1] if len(choices) == 1 else None), hint

    scan_models.click(scan_imports, [], [import_candidate, import_info])

    @safe(outputs=2)
    def submit_import(candidate):
        if not candidate:
            raise ValueError("请选择要导入的模型文件夹和权重")
        task = service.submit_model_import(candidate)
        return gr.update(choices=job_choices(), value=task), "模型校验已提交。完成后点击“刷新页面数据”，即可在目标音色模型中选择。"

    import_button.click(submit_import, [import_candidate], [job, notice])

    @safe(outputs=3)
    def submit_conversion(selection, ckpt, speaker, source_mode, uploaded_audio, recording, record_type, kind, backing,
                          pitch, f0, seconds, device, voice_gain, inst_gain, shallow, steps,
                          split, voice_mode, harmony_offset, backing_gain, separator, karaoke, dry_lead, dry_backing,
                          room_enabled,room_style,room_lead,room_backing,room_decay,room_predelay,room_wet,room_clean,upload_name):
        guard()
        if not selection:
            raise ValueError("请先选择目标音色模型")
        audio = recording if source_mode == "录音" else uploaded_audio
        if not audio:
            raise ValueError("请先录制或选择一份音频")
        cleanup = "none"
        if source_mode == "录音":
            if record_type == "说话 · 降噪":
                cleanup = "denoise"
            elif record_type == "唱歌 · 带伴奏":
                cleanup = "separate"
            kind, backing = "录音", None
        if kind == "人声 + 伴奏分别上传" and not backing:
            raise ValueError("请上传与人声对应的伴奏音频")
        if kind != "人声 + 伴奏分别上传":
            backing = None
        if kind == "带伴奏的音乐":
            cleanup = "separate"
        mix_output = source_mode == "上传音频" and kind == "带伴奏的音乐"
        voice_mode = voice_mode if split and not (source_mode == "录音" and record_type == "说话 · 降噪") else "off"
        output_reverb = reverb.settings(dict(enabled=room_enabled,style=room_style,lead_strength=room_lead,
            backing_strength=room_backing,decay=room_decay,predelay_ms=room_predelay,wet=room_wet,clean_output=room_clean))
        source, model_id = selection.split(":", 1)
        if source == "trained":
            task = service.create_conversion(model_id, ckpt, audio, mix_output, pitch, f0, seconds,
                                             device, voice_gain, inst_gain, accompaniment=backing, cleanup=cleanup,
                                             vocal_mode=voice_mode, harmony_transpose=harmony_offset, harmony_gain=backing_gain,
                                             separation_model=separator, vocal_split_model=karaoke,
                                             dereverb_lead=dry_lead, dereverb_backing=dry_backing if voice_mode != "off" else False,
                                             output_reverb=output_reverb)
        elif source == "imported":
            task = service.create_imported_conversion(model_id, speaker, audio, mix_output, pitch, f0,
                                                      seconds, device, voice_gain, inst_gain, accompaniment=backing,
                                                      shallow_diffusion=shallow, diffusion_steps=steps, cleanup=cleanup,
                                                      vocal_mode=voice_mode, harmony_transpose=harmony_offset, harmony_gain=backing_gain,
                                                      separation_model=separator, vocal_split_model=karaoke,
                                             dereverb_lead=dry_lead, dereverb_backing=dry_backing if voice_mode != "off" else False,
                                             output_reverb=output_reverb)
        else:
            raise ValueError("目标模型类型无效，请刷新列表")
        if source_mode=='录音':
            name='录音_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'.wav'
        else:name=upload_name
        if name:db.update_job(task,source_name=Path(name).name)
        return gr.update(choices=job_choices(), value=task), "转换任务已启动。", gr.update(visible=True)

    def remember_upload(payload):
        if not payload:return ''
        return payload.get('orig_name') or Path(payload.get('path','')).name
    convert_audio.upload(remember_upload,[convert_audio],[convert_upload_name],preprocess=False,queue=False)
    convert_audio.clear(lambda:'',[],[convert_upload_name],queue=False)
    convert_button.click(submit_conversion,
                         [model_choice, conversion_checkpoint, conversion_speaker, convert_source, convert_audio,
                          convert_recording, record_kind, convert_kind, convert_backing, transpose, convert_f0, chunk,
                          convert_gpu, vocal_gain, instrumental_gain, shallow_diffusion, diffusion_steps,
                          split_enabled, vocal_mode, harmony_pitch, harmony_gain, conversion_separator, conversion_karaoke, conversion_dry_lead, conversion_dry_backing,*conversion_reverb_inputs,convert_upload_name],
                         [job, notice, submit_dialog])
    dismiss_submission.click(lambda: gr.update(visible=False), [], [submit_dialog], queue=False)
    def open_task_center():
        listing_data = task_listing()
        return (gr.update(visible=False), gr.update(selected="tasks"),
                gr.update(elem_classes=["workbench-panel", "svc-task-detail", "svc-task-attention"]),
                gr.update(value="全部"), gr.update(value="全部"), gr.update(value=""), *listing_data)

    open_submission.click(open_task_center, [],
                          [submit_dialog, svc_tabs, task_detail, *task_filters, *task_outputs], queue=False).then(
        lambda task: None, [job], [], queue=False,
        js="""(task) => {
            setTimeout(() => {
                const rows = document.querySelectorAll('#svc-task-table tbody tr');
                const row = [...rows].find(item => item.textContent.includes(task));
                const target = row || document.getElementById('svc-task-detail');
                if (row) { row.classList.remove('svc-task-row-attention'); void row.offsetWidth; row.classList.add('svc-task-row-attention'); }
                target?.scrollIntoView({behavior: 'smooth', block: 'center'});
            }, 180);
            return task;
        }""")
    @safe(outputs=3)
    def submit_pitch(task, method, backend, threshold, strength):
        guard()
        if not task:
            raise ValueError("请先选择已完成的转换任务")
        result = service.create_pitch(task, method, backend, threshold, strength)
        return gr.update(choices=job_choices(), value=result), "音准任务已提交，原转换结果保留。", gr.update(active=True)

    pitch_submit.click(submit_pitch, [job, pitch_method, pitch_backend, pitch_threshold, pitch_strength], [job, notice, timer])

    stop.click(safe(service.cancel), [job], [notice])

    @safe(outputs=16)
    def open_retry(task):
        if not task:
            raise ValueError("请先在任务表格中选择任务")
        record = service.snapshot(task)[0]
        if record["status"] not in service.TERMINAL:
            raise ValueError("请等待任务结束后再重试；运行中的任务可先停止")
        options = record["options"]
        conversion = record["kind"] == "convert"
        detail = ("修改下列参数后会创建一条新转换任务；原任务及其结果保留。音频输入、目标音色模型和浅扩散设置沿用原任务，分离模型可修改。"
                  if conversion else "这类任务会按原配置重试。")
        if conversion and "separation_model" not in options:
            detail += " 历史任务使用旧分离模型，下方默认推荐升级后的分离模型。"
        if conversion and options.get("output_reverb",{}).get("enabled"):
            detail += " 输出混响沿用原转换配置；如需修改，可在“完整音频转换 → 输出混响”中提交。"
        return (gr.update(visible=True), gr.update(visible=conversion), detail,
                options.get("transpose", 0), options.get("predictor", "dio"), options.get("chunk", 15),
                options.get("gpu", "0"), options.get("vocal_gain", 1), options.get("instrumental_gain", 1),
                options.get("vocal_mode", "off"), options.get("harmony_transpose", 0), options.get("harmony_gain", 1),
                options.get("separation_model", separation.VOCAL_DEFAULT),
                options.get("vocal_split_model", separation.KARAOKE_DEFAULT),
                options.get("dereverb_lead", False), options.get("dereverb_backing", False))

    retry.click(open_retry, [job], [retry_panel, retry_controls, retry_info, retry_pitch, retry_f0,
                                      retry_chunk, retry_gpu, retry_vocal_gain, retry_instrumental_gain,
                                      retry_vocal_mode, retry_harmony_pitch, retry_harmony_gain, retry_separator, retry_karaoke, retry_dry_lead, retry_dry_backing],
                queue=False, trigger_mode="always_last", show_progress="hidden")
    retry_cancel.click(lambda: gr.update(visible=False), [], [retry_panel], queue=False)

    @safe(outputs=10)
    def retry_task(task, pitch, f0, seconds, device, voice_gain, inst_gain, voice_mode, harmony_offset, backing_gain, separator, karaoke, dry_lead, dry_backing):
        guard()
        if not task:
            raise ValueError("请先在任务表格中选择任务")
        record = service.snapshot(task)[0]
        if record["kind"] == "convert":
            result = service.retry_conversion(task, pitch, f0, seconds, device, voice_gain, inst_gain,
                                               voice_mode, harmony_offset, backing_gain, separator, karaoke,
                                               dry_lead, dry_backing if voice_mode != "off" else False)
            message = "已按所选配置创建新转换任务；原任务和结果已保留。"
        else:
            result = service.retry(task)
            message = "任务已按原配置重新提交。"
        return (gr.update(choices=job_choices(), value=result), message, gr.update(visible=False),
                gr.update(active=True), gr.update(value="全部"), gr.update(value="全部"),
                gr.update(value=""), *task_listing())

    retry_confirm.click(retry_task, [job, retry_pitch, retry_f0, retry_chunk, retry_gpu,
                                     retry_vocal_gain, retry_instrumental_gain, retry_vocal_mode,
                                     retry_harmony_pitch, retry_harmony_gain, retry_separator, retry_karaoke, retry_dry_lead, retry_dry_backing],
                        [job, notice, retry_panel, timer, *task_filters, *task_outputs])

    delete.click(lambda task: gr.update(visible=bool(task)), [job], [delete_confirm], queue=False)
    delete_cancel.click(lambda: gr.update(visible=False), [], [delete_confirm], queue=False)

    @safe(outputs=7)
    def remove_task(task, kind, state, search, page):
        if not task:
            raise ValueError("请先在任务表格中选择任务")
        service.delete_job(task)
        listing_data = task_listing(kind, state, search, page)
        return (gr.update(choices=job_choices(), value=None), "任务已删除。", *listing_data,
                gr.update(visible=False), "尚未选择任务。")

    delete_accept.click(remove_task, [job, *task_filters, task_page],
                        [job, notice, *task_outputs, delete_confirm, selected_job])

    def poll_error(error, task, previous_state, *args):
        message = f"自动刷新已暂停：{error}。请点击“刷新列表”重试。"
        panel = status({"status": "failed", "message": message, "progress": 0})
        return (panel, gr.skip(), *(gr.skip() for _ in range(len(task_audio) + 1)),
                {"task": task, "poll_error": str(error)}, gr.skip(), gr.skip(), gr.update(active=False),
                *(gr.skip() for _ in [*report_outputs, *report_control_outputs]), gr.skip(), gr.skip())

    @safe(outputs=9 + len(task_audio) + len(report_outputs) + len(report_control_outputs), quiet=True, fallback=poll_error)
    def poll(task, previous_state, kind="全部", order="风险优先", current=None, identity=None, predictor="dio", preview_source=None):
        previous_state = previous_state or {}
        record, log = service.snapshot(task)
        source_choices = []
        if record.get("kind") == "dataset":
            for source_id in record.get("options", {}).get("sources", []):
                item = db.read(db.source_dir(record["singer_id"], source_id) / "source.json", {})
                source_choices.append((item.get("name", source_id), source_id))
        source_ids = [value for _, value in source_choices]
        preview_source = preview_source if preview_source in source_ids else (source_ids[0] if source_ids else None)
        source_update = (gr.update(choices=source_choices, value=preview_source, visible=len(source_choices) > 1)
                         if previous_state.get("source_choices") != source_choices or previous_state.get("preview_source") != preview_source
                         else gr.skip())
        outputs = service.conversion_outputs(record, preview_source)
        editing_state = audio_edit.snapshot(task)
        panel=status(record)
        log_stamp=text_stamp(log)
        detail_stamp={'report':report_stamp(record),'editing':editing_state}
        report_job = service.report_task(identity)
        report_state = {"singer": identity, "id": report_job.get("id"), "status": report_job.get("status"),
                        "outputs": report_job.get("outputs", {})} if report_job else {"singer": identity}
        state = {"task": task, "status": record.get("status"), "outputs": outputs, "report": report_state,
                 "source_choices": source_choices, "preview_source": preview_source,"audio_edit":editing_state,
                 "progress_html":panel,"log_stamp":log_stamp,"detail_stamp":detail_stamp}
        # Do not reload audio every tick: that would interrupt listening.
        if previous_state.get("task") == task and previous_state.get("outputs") == outputs:
            audio_updates = [gr.skip() for _ in range(len(task_audio) + 1)]
        else:
            published, previews = task_players.publish(record, {key: outputs[key] for key, _, _ in CONVERSION_TRACKS
                                                                if outputs.get(key)}, downloads)
            previous_outputs = previous_state.get("outputs", {})
            audio_updates = [
                gr.skip() if previous_state.get("task") == task and previous_outputs.get(key) == outputs.get(key)
                else gr.update(value=task_players.player(label, published.get(key), task, previews.get(key),key=key),
                               visible=bool(published.get(key)))
                for key, label, _ in CONVERSION_TRACKS]
            audio_updates.append(task_players.download_links(CONVERSION_TRACKS, published,task))
        if record.get("kind") == "dataset" and record.get("status") == "awaiting_review" and state != previous_state:
            table_rows, choices = rows(task, kind, order)
            selected = current if current in {value for _, value in choices} else choices[0][1] if choices else None
            table_update, clip_update = table_rows, gr.update(choices=choices, value=selected)
        else:
            table_update, clip_update = gr.update(), gr.update()
        report_active = report_job and report_job.get("status") not in service.TERMINAL
        polling = bool(task) and record.get("status") not in service.TERMINAL or bool(report_active) or editing_state.get("status") in audio_edit.ACTIVE
        # Leave one final tick for the editor panel to observe completion even
        # if its parallel status read happened just before the worker finished.
        polling = polling or bool(task and record.get("status")=="completed" and previous_state.get("status")!="completed")
        state['polling']=bool(polling)
        # Save/restore actions can activate the timer without changing this
        # cached state. Always explicitly stop it once there is no work left.
        timer_update = gr.update(active=bool(polling))
        report_updates = [gr.skip() for _ in report_outputs]
        control_updates = [gr.skip() for _ in report_control_outputs]
        if report_state != previous_state.get("report"):
            control_updates = list(report_controls.__wrapped__(identity))
            if report_job and report_job.get("status") == "completed":
                report_updates = list(report_view.__wrapped__(identity, predictor))
            elif report_job and report_job.get("status") in {"failed", "cancelled"}:
                report_updates[0] = report_job.get("message", "报告分析未完成")
        same_task=previous_state.get('task')==task
        progress_update=gr.skip() if same_task and previous_state.get('progress_html')==panel else panel
        log_update=gr.skip() if same_task and previous_state.get('log_stamp')==log_stamp else log
        detail_update=(gr.skip() if same_task and previous_state.get('detail_stamp')==detail_stamp else
                       pitch_job.report_html(record,downloads)+reverb.report_html(record,downloads)+audio_edit.report_html(record))
        state_update=gr.skip() if previous_state==state else state
        return progress_update, log_update, *audio_updates, state_update, table_update, clip_update, timer_update, *report_updates, *control_updates, source_update, detail_update

    poll_outputs = [progress, logs, *task_audio, downloads, polled_state, table, clip, timer,
                    *report_outputs, *report_control_outputs, dataset_preview_source, pitch_result]
    poll_inputs = [job, polled_state, review_filter, review_order, clip, training_singer, train_f0, dataset_preview_source]
    # A single event handles polling and selection. Keep only the latest pending
    # selection instead of overlapping timer/queued audio responses.
    gr.on(triggers=[timer.tick, job.change, dataset_preview_source.change], fn=poll, inputs=poll_inputs, outputs=poll_outputs,
          queue=False, trigger_mode="always_last", show_progress="hidden")
    @safe(outputs=4)
    def selected_task_info(task):
        if not task:
            return ("尚未选择任务。", gr.update(visible=False), gr.update(visible=False),
                    '<span data-svc-selected-task="" hidden></span>')
        record = service.snapshot(task)[0]
        policy = dict((value, label) for label, value in VOCAL_MODES).get(record.get("options", {}).get("vocal_mode"))
        detail = f"  \n**处理方式** · {policy}" if policy else ""
        options = record.get("options", {})
        if record["kind"] == "convert" and options.get("output_reverb",{}).get("enabled"):
            room = options["output_reverb"]
            style = "参考原曲估计" if room.get("style") == "reference" else "手动混响"
            detail += f"  \n**转换后自动添加混响** · {style} · 主唱强度 {room.get('lead_strength',1):g} · 和声强度 {room.get('backing_strength',1):g}"
        if record["kind"] == "reverb":
            style = "参考原曲估计" if options.get("style") == "reference" else "手动混响"
            detail += f"  \n**输出混响** · {style} · {escape(options.get('version', ''))} · 主唱强度 {options.get('lead_strength', 1):g} · 和声强度 {options.get('backing_strength', 1):g}"
        if options.get("cleanup") == "separate" or options.get("separate") or (record["kind"] == "dataset" and options.get("mixed")):
            name = dict((value, label) for label, value in separation.VOCAL_CHOICES).get(options.get("separation_model", "hp2"), "未知")
            detail += f"  \n**人声 / 伴奏模型** · {escape(name)}"
        if options.get("vocal_mode", "off") != "off" or options.get("split_vocals"):
            name = dict((value, label) for label, value in separation.KARAOKE_CHOICES).get(options.get("vocal_split_model", "mel_karaoke"), "未知")
            detail += f"  \n**主唱 / 和声模型** · {escape(name)}"
        if options.get("dereverb_lead") or options.get("dereverb_backing"):
            enabled = "、".join(name for name, key in [("主唱／未分轨人声", "dereverb_lead"), ("和声", "dereverb_backing")] if options.get(key))
            detail += f"  \n**输入去混响** · BS-RoFormer De-Reverb · {enabled}"
        return (f"**当前任务** · {KINDS.get(record['kind'], record['kind'])} · `{task}`" + detail,
                gr.update(visible=record["kind"] in {"convert", "dataset", "pitch", "reverb"}), gr.update(visible=False),
                f'<span data-svc-selected-task="{escape(task)}" hidden></span>')

    job.change(selected_task_info, [job], [selected_job, task_result, retry_panel, selected_task_marker],
               queue=False, trigger_mode="always_last", show_progress="hidden")

    @safe(outputs=2)
    def refresh_model_labels(selection):
        choices=conversion_choices()
        current=selection if selection in {value for _,value in choices} else None
        update=gr.update(choices=choices)
        if current!=selection:update['value']=current
        return update,conversion_description(current)

    gr.on([conversion_tab.select,refresh.click],refresh_model_labels,[model_choice],[model_choice,model_details],
          queue=False,trigger_mode='always_last',show_progress='hidden')

    @safe(outputs=7 + len(experiment_outputs) + len(poll_outputs))
    def refresh_workspace(current_singer, current_training_singer, current_model, current_job, current_experiment, kind, order, current_clip, previous_state, predictor):
        # One click produces one notification, even if multiple panels share bad metadata.
        lists = refresh_lists(current_singer, current_job, current_experiment)
        details = experiment_details.__wrapped__(current_experiment)
        clips = load_rows.__wrapped__(current_job, kind, order, current_clip)
        task_updates = list(poll.__wrapped__(current_job, previous_state, kind, order, current_clip, current_training_singer, predictor))
        for component, value in zip([table, clip], clips[:2]):
            task_updates[poll_outputs.index(component)] = value
        return (*lists, gr.update(choices=singer_choices(), value=current_training_singer),
                gr.skip(), *details, clips[2], *task_updates)

    refresh.click(refresh_workspace,
                  [singer, training_singer, model_choice, job, experiment, review_filter, review_order, clip, polled_state, train_f0],
                  [singer, job, experiment, backend, training_singer, model_choice, *experiment_outputs, edits_download, *poll_outputs])
    refresh.click(listing, [*task_filters, task_page], task_outputs, queue=False)
    return {"singer": singer, "job": job, "experiment": experiment, "checkpoint": checkpoint,
            "model_choice":model_choice,"model_details":model_details,"refresh_model_labels":refresh_model_labels}
