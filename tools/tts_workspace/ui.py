"""The complete synthesis flow embedded in the main workbench's inference tab."""

from html import escape
import os
import threading

import gradio as gr
import soundfile as sf

from tools.assets import workspace_header
from .service import service
from . import reference_snapshots as snapshots
from . import result_history as history
from .output_speed import result as speed_result
from .output_speed import OUTPUT_DIR

LANGUAGES = [
    "中文",
    "英文",
    "日文",
    "粤语",
    "韩文",
    "中英混合",
    "日英混合",
    "粤英混合",
    "韩英混合",
    "多语种混合",
    "多语种混合(粤语)",
]
CUTS = [
    "不切",
    "凑四句一切",
    "凑50字一切",
    "按中文句号。切",
    "按英文句号.切",
    "按标点符号切",
]


def status_panel(title, message, error=False):
    return f'<div class="inference-status{" feedback-error" if error else ""}" role="{"alert" if error else "status"}"><strong>{escape(title)}</strong><span>{escape(message)}</span></div>'


def loaded_models_panel(selected_gpt=None, selected_sovits=None):
    spec = service.spec if service.metadata and service.process and service.process.poll() is None else None
    if not spec:
        return '<div class="inference-loaded-models" role="status"><strong>当前已加载模型</strong><p>尚未加载。选择左侧模型并点击“加载模型”。</p></div>'
    version = escape(str(service.metadata.get("model_version", "")))
    mode = "并行" if spec.get("parallel") else "标准"
    pending = (selected_gpt is not None and selected_gpt != spec["gpt_path"]
               or selected_sovits is not None and selected_sovits != spec["sovits_path"])
    message = '<p class="inference-model-pending">左侧选择已更改，点击“加载模型”后才会生效。</p>' if pending else ""
    return ('<div class="inference-loaded-models" role="status"><strong>当前已加载模型'
            f' · {version} · {mode}推理</strong>'
            f'<div class="inference-model-line"><span>GPT</span><code>{escape(str(spec["gpt_path"]))}</code></div>'
            f'<div class="inference-model-line"><span>SoVITS</span><code>{escape(str(spec["sovits_path"]))}</code></div>'
            f'{message}</div>')


def result_details_panel(record):
    """Show only the model, text, and controls visible in the main synthesis form."""
    config = record.get("config") or {}
    model = config.get("模型") or {}
    synthesis = config.get("合成") or {}

    def value(item):
        return escape(str(item)) if item is not None else "—"

    version = value(model.get("版本"))
    mode = value(model.get("推理方式"))
    fields = [
        ("目标文本语种", synthesis.get("目标语种")),
        ("停顿边界", synthesis.get("停顿边界")),
        ("语速", synthesis.get("生成语速")),
        ("分段停顿 / 秒", synthesis.get("分段停顿（秒）")),
        ("标点停顿 / 秒", synthesis.get("标点停顿（秒）")),
    ]
    settings = "".join(
        f"<div><dt>{escape(label)}</dt><dd>{value(item)}</dd></div>"
        for label, item in fields if item is not None
    )
    return (
        '<section class="tts-history-card" aria-label="历史结果详情">'
        f'<div class="tts-history-heading"><strong>历史结果详情</strong><span>{version} · {mode}推理</span></div>'
        f'<div class="tts-history-text"><span>生成文本</span><p>{escape(record["text"])}</p></div>'
        '<dl class="tts-history-settings">'
        f'<div class="tts-history-model"><dt>GPT 模型</dt><dd>{value(model.get("GPT"))}</dd></div>'
        f'<div class="tts-history-model"><dt>SoVITS 模型</dt><dd>{value(model.get("SoVITS"))}</dd></div>'
        f'{settings}</dl></section>'
    )


def build_page(
    gpt_choices,
    sovits_choices,
    gpu,
    version_component,
    bert_component,
    hubert_component,
    gpu_mapper,
    busy=None,
    ensure_idle=None,
):
    gr.HTML(
        workspace_header(
            "TTS 推理 · 语音合成",
            "选择模型并加载，上传参考音频确定音色，再填写目标文本生成语音。所有操作都在当前工作区完成。",
        )
    )
    with gr.Row(elem_classes="integrated-synthesis"):
        with gr.Column(
            scale=3,
            min_width=280,
            elem_classes=["inference-sidebar", "workbench-panel"],
        ):
            gr.HTML('<div class="panel-heading">模型与参考音色</div>')
            gpt = gr.Dropdown(
                label="GPT 模型", choices=gpt_choices, value=gpt_choices[-1]
            )
            sovits = gr.Dropdown(
                label="SoVITS 模型", choices=sovits_choices, value=sovits_choices[0]
            )
            with gr.Row(elem_classes="compact-actions"):
                load = gr.Button("加载模型", variant="primary", size="sm", scale=0)
                unload = gr.Button("卸载模型", size="sm", scale=0)
                refresh = gr.Button("刷新", size="sm", scale=0)
            with gr.Accordion("设备与推理方式", open=False):
                gpu_number = gr.Textbox(label="GPU 逻辑编号", value=gpu, lines=1)
                parallel = gr.Checkbox(
                    label="启用并行推理",
                    value=False,
                    info="切换模型或推理方式后，点击加载模型使设置生效。",
                )
            reference = gr.Audio(
                label="参考音频 · 3 至 10 秒",
                type="filepath",
                sources=["upload", "microphone"],
            )
            prompt = gr.Textbox(
                label="参考音频中的文字",
                lines=3,
                max_lines=6,
                placeholder="填写参考音频实际说出的内容",
            )
            prompt_language = gr.Dropdown(
                label="参考音频语种", choices=LANGUAGES, value="中文"
            )
            with gr.Accordion("更多参考设置", open=False):
                aux_refs = gr.File(
                    label="辅助参考音频", file_count="multiple", type="filepath"
                )
                ref_free = gr.Checkbox(label="无需参考文本", value=False)
            with gr.Accordion("使用已有参考音频或快照", open=False):
                local_reference = gr.Dropdown(label="本机已有参考音频", choices=snapshots.local_reference_choices())
                use_local_reference = gr.Button("使用本机音频", size="sm")
                snapshot_choice = gr.Dropdown(label="已保存的参考设置快照", choices=snapshots.choices())
                gr.Markdown("快照会恢复参考音频、参考文字与语种，以及保存时选择的 GPT 和 SoVITS 模型。", elem_classes="workspace-hint")
                with gr.Row(elem_classes="compact-actions"):
                    apply_snapshot = gr.Button("使用快照", size="sm")
                    remove_snapshot = gr.Button("删除快照", size="sm")
                confirm_delete = gr.Checkbox(label="确认删除选中的快照", value=False)
                existing_notice = gr.Markdown(
                    "从本机选择音频只会填入音频；使用快照会恢复整套参考设置。",
                    elem_classes="workspace-hint",
                )
            with gr.Accordion("保存新快照", open=False):
                gr.Markdown("保存当前表单中的参考音频、参考文字与语种、辅助参考音频和模型选择。目标生成文本不会保存。", elem_classes="workspace-hint")
                snapshot_name = gr.Textbox(label="快照名称", placeholder="例如：诗歌剧 · 中文参考", lines=1)
                save_snapshot = gr.Button("保存为新快照", size="sm")
                save_notice = gr.Markdown(
                    "仅点击保存后才会留存音频和设置。",
                    elem_classes="workspace-hint",
                )
        with gr.Column(scale=7, min_width=320, elem_classes="inference-main"):
            status = gr.HTML(
                status_panel("尚未加载模型", "在左侧选择模型，点击加载模型后即可合成。")
            )
            loaded_models = gr.HTML(loaded_models_panel(gpt_choices[-1], sovits_choices[0]))
            with gr.Group(elem_classes="workbench-panel"):
                gr.HTML('<div class="panel-heading">目标文本</div>')
                text = gr.Textbox(
                    label="需要合成的文本",
                    lines=8,
                    max_lines=20,
                    placeholder="输入需要生成语音的内容",
                )
                with gr.Row():
                    text_language = gr.Dropdown(
                        label="目标文本语种", choices=LANGUAGES, value="中文"
                    )
                    cut = gr.Dropdown(
                        label="停顿边界", choices=CUTS, value="按标点符号切",
                        info="换行始终分段；逗号、句号等停顿需选择“按标点符号切”。",
                    )
                with gr.Row():
                    speed = gr.Slider(0.6, 1.65, value=1, step=0.05, label="语速")
                    pause = gr.Slider(
                        0, 2, value=0.3, step=0.05, label="分段停顿 / 秒",
                        info="换行或非标点切分形成的片段之间，额外插入的静音。",
                    )
                punctuation_pause = gr.Slider(
                    0, 1, value=0.15, step=0.05, label="标点停顿 / 秒",
                    info="仅在按标点切分的片段之间生效；会去除可识别的段首、段尾静音，再插入所选时长。",
                )
            with gr.Accordion("高级生成设置", open=False):
                with gr.Row():
                    top_k = gr.Slider(1, 100, value=15, step=1, label="Top K")
                    top_p = gr.Slider(0, 1, value=1, step=0.05, label="Top P")
                    temperature = gr.Slider(
                        0, 1, value=1, step=0.05, label="Temperature"
                    )
                with gr.Row():
                    steps = gr.Radio(
                        [4, 8, 16, 32], value=8, label="采样步数", visible=False
                    )
                    cfg = gr.Slider(
                        0, 2, value=0, step=0.05, label="CFG", visible=False
                    )
                    super_sampling = gr.Checkbox(label="V3 输出超分辨率", visible=False)
                freeze = gr.Checkbox(
                    label="复用上次生成结果调整语速与音色", value=False,
                    info="仅单段且目标文本、参考设置未变时复用；修改文本会重新生成。",
                )
                with gr.Row():
                    graph = gr.Checkbox(
                        label="CUDA Graph 加速", value=False, interactive=False
                    )
                    flash = gr.Checkbox(
                        label="FlashAttention 加速", value=False, interactive=False
                    )
                with gr.Group(visible=False) as fast_options:
                    with gr.Row():
                        batch = gr.Slider(1, 20, value=1, step=1, label="推理批大小")
                        repetition = gr.Slider(
                            1, 2, value=1.35, step=0.05, label="重复惩罚"
                        )
                        seed = gr.Number(value=-1, precision=0, label="随机种子")
                    random_seed = gr.Checkbox(label="每次使用随机种子", value=True)
                    split_bucket = gr.Checkbox(label="按文本长度分桶", value=True)
            with gr.Row(elem_classes="compact-actions synthesis-actions"):
                generate = gr.Button(
                    "合成语音", variant="primary", size="sm", scale=0, interactive=False
                )
                cancel = gr.Button("停止生成", size="sm", scale=0, interactive=False)
            with gr.Group(elem_classes="workbench-panel"):
                gr.HTML('<div class="panel-heading">生成结果</div>')
                with gr.Row(elem_classes="tts-history-picker"):
                    result_choice = gr.Dropdown(
                        label="最近 10 次生成结果", choices=history.choices(),
                        value=None, interactive=True, scale=5,
                    )
                    clear_result_choice = gr.Button("清空选择", size="sm", scale=0)
                result_details = gr.HTML(visible=False)
                with gr.Row(elem_classes="tts-result-speed-controls"):
                    playback_speed = gr.Dropdown(
                        label="试听与下载倍速",
                        choices=[(f"{rate:g}×", rate) for rate in (0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3)],
                        value=1, interactive=False, scale=3, min_width=200,
                    )
                    custom_speed_button = gr.Button("自定义倍率", size="sm", scale=1, min_width=130)
                with gr.Row(visible=False, elem_classes="tts-custom-speed") as custom_speed_controls:
                    custom_speed = gr.Number(
                        label="自定义倍速（0.25–5×）", value=1, minimum=0.25,
                        maximum=5, step=0.01, precision=3, scale=2,
                    )
                    apply_custom_speed = gr.Button("应用自定义倍率", variant="primary", size="sm", scale=1)
                playback_duration = gr.Markdown("合成后可调整试听和下载倍速。", elem_classes=["workspace-hint"])
                output = gr.Audio(
                    label="当前倍速音频 · 试听与下载", type="filepath", interactive=False,
                    show_download_button=True, elem_id="tts-result-audio", visible=False,
                )
                original_output = gr.State(None)
            gr.Markdown(
                "更换模型后请重新加载。卸载模型可释放显存；最近 10 次结果保留在本机，生成第 11 次时自动删除最早的音频。",
                elem_classes="workspace-hint",
            )

    manual_release = threading.Event()

    def ready_updates(meta):
        model_version = meta["model_version"]
        sampling = model_version in {"v3", "v4", "v5dev", "v5turbo"}
        versions5 = model_version in {"v5dev", "v5turbo"}
        languages = meta["languages"]
        return [
            gr.update(choices=languages, value="中文"),
            gr.update(choices=languages, value="中文"),
            gr.update(
                visible=sampling,
                value=4
                if model_version == "v5turbo"
                else 32
                if model_version in {"v3", "v5dev"}
                else 8,
            ),
            gr.update(visible=versions5, value=1.3 if model_version == "v5dev" else 0),
            gr.update(visible=model_version == "v3", value=False),
            gr.update(visible=not sampling, value=None),
            gr.update(interactive=not sampling, value=False),
            gr.update(visible=not meta["parallel"], value=False),
            gr.update(visible=meta["parallel"]),
            gr.update(
                interactive=meta["graph_available"], value=meta["graph_available"]
            ),
            gr.update(
                interactive=meta["flash_available"], value=meta["flash_available"]
            ),
        ]

    model_outputs = [
        status,
        generate,
        prompt_language,
        text_language,
        steps,
        cfg,
        super_sampling,
        aux_refs,
        ref_free,
        freeze,
        fast_options,
        graph,
        flash,
    ]

    def load_models(
        gpt_path,
        sovits_path,
        gpu_id,
        parallel_mode,
        model_version,
        bert_path,
        hubert_path,
    ):
        try:
            if busy and busy():
                raise ValueError("请等待当前训练或预处理完成后加载推理模型")
            if ensure_idle:
                ensure_idle()
        except (gr.Error, ValueError, OSError) as error:
            message = getattr(error, "message", str(error))
            gr.Warning(escape(message), duration=5)
            yield [status_panel("暂时无法加载", message, error=True),
                   gr.update(interactive=bool(service.metadata) and not service.lock.locked()),
                   *[gr.skip() for _ in model_outputs[2:]]]
            return
        manual_release.clear()
        yield [
            status_panel("正在加载模型", "首次加载可能需要一些时间，完成后即可合成。"),
            gr.update(interactive=False),
            *[gr.skip() for _ in model_outputs[2:]],
        ]
        try:
            spec = dict(
                gpt_path=gpt_path,
                sovits_path=sovits_path,
                gpu=str(gpu_mapper(gpu_id)),
                parallel=bool(parallel_mode),
                version=model_version,
                bert_path=bert_path,
                cnhubert_path=hubert_path,
                language=os.environ.get("language", "zh_CN"),
            )
            if manual_release.is_set():
                raise ValueError("加载已取消")
            meta = service.load(spec)
            if manual_release.is_set():
                service.close()
                raise ValueError("加载已取消")
            mode = "并行" if meta["parallel"] else "标准"
            yield [
                status_panel("模型已就绪", f"{meta['model_version']} · {mode}推理"),
                gr.update(interactive=True),
                *ready_updates(meta),
            ]
        except (ValueError, OSError) as error:
            if manual_release.is_set():
                yield [
                    status_panel("模型已卸载", "选择模型并加载后可继续合成。"),
                    gr.update(interactive=False),
                    *[gr.skip() for _ in model_outputs[2:]],
                ]
                return
            yield [
                status_panel("模型加载失败", str(error), error=True),
                gr.update(interactive=bool(service.metadata) and not service.lock.locked()),
                *[gr.skip() for _ in model_outputs[2:]],
            ]
            gr.Warning(escape(str(error)), duration=5)
            return

    generation_inputs = [
        reference,
        prompt,
        prompt_language,
        text,
        text_language,
        cut,
        speed,
        pause,
        punctuation_pause,
        aux_refs,
        ref_free,
        top_k,
        top_p,
        temperature,
        steps,
        cfg,
        super_sampling,
        freeze,
        graph,
        flash,
        batch,
        repetition,
        seed,
        random_seed,
        split_bucket,
        gpt,
        sovits,
        gpu_number,
        parallel,
    ]

    def synthesize(
        reference,
        prompt_text,
        prompt_lang,
        target_text,
        target_lang,
        split,
        speed_value,
        pause_value,
        punctuation_pause_value,
        refs,
        reference_free,
        k,
        p,
        temp,
        sample_steps,
        cfg_value,
        sr,
        cached,
        cuda_graph,
        flash_attn,
        batch_size,
        repetition_penalty,
        seed_value,
        random,
        bucket,
        gpt_path,
        sovits_path,
        gpu_id,
        parallel_mode,
    ):
        try:
            if not service.metadata:
                raise ValueError("请先加载模型")
            if busy and busy():
                raise ValueError("请等待当前训练或预处理完成后合成")
            if ensure_idle:
                ensure_idle()
            if not reference:
                raise ValueError("请上传参考音频")
            if not (target_text or "").strip():
                raise ValueError("请填写需要合成的目标文本")
            if not reference_free and not (prompt_text or "").strip():
                raise ValueError("请填写参考音频中的文字")
            actual = service.spec
            if actual is None:
                raise ValueError("模型已卸载，请重新加载")
            if (gpt_path, sovits_path, str(gpu_mapper(gpu_id)), bool(parallel_mode)) != (
                actual["gpt_path"], actual["sovits_path"], actual["gpu"], actual["parallel"],
            ):
                raise ValueError("模型或推理设置已更改，请先重新加载模型")
            if actual["parallel"] or not reference_free:
                # Check before sending GPU work; invalid inputs keep the model loaded.
                duration = sf.info(reference).duration
                if not 3 <= duration <= 10:
                    raise ValueError(f"参考音频为 {duration:.2f} 秒，请裁剪为 3–10 秒后再合成")
        except (gr.Error, ValueError, OSError, RuntimeError) as error:
            message = getattr(error, "message", str(error))
            gr.Warning(escape(message), duration=5)
            yield (status_panel("请检查合成设置", message, error=True), gr.update(value=None, visible=False),
                   gr.update(interactive=bool(service.metadata) and not service.lock.locked()),
                   gr.skip(), gr.update(interactive=service.generating),
                   None, gr.update(value=1, interactive=False), "合成后可调整试听和下载倍速。",
                   gr.update(value=None), gr.update(visible=False))
            return
        data = dict(
            reference=reference,
            prompt_text=prompt_text,
            prompt_language=prompt_lang,
            text=target_text,
            text_language=target_lang,
            cut=split,
            speed=speed_value,
            pause=pause_value,
            punctuation_pause=punctuation_pause_value,
            aux_refs=refs or [],
            ref_free=reference_free,
            top_k=k,
            top_p=p,
            temperature=temp,
            sample_steps=sample_steps,
            cfg_rate=cfg_value,
            super_sampling=sr,
            freeze=cached,
            cuda_graph=cuda_graph,
            flash_attention=flash_attn,
            batch_size=batch_size,
            repetition_penalty=repetition_penalty,
            seed=int(seed_value if seed_value is not None else -1),
            random_seed=random,
            split_bucket=bucket,
        )
        record_config = {
            "模型": {
                "GPT": actual["gpt_path"], "SoVITS": actual["sovits_path"],
                "版本": service.metadata.get("model_version"),
                "推理方式": "并行" if actual["parallel"] else "标准",
            },
            "合成": {
                "目标语种": target_lang, "停顿边界": split,
                "生成语速": speed_value, "分段停顿（秒）": pause_value,
                "标点停顿（秒）": punctuation_pause_value,
            },
        }
        manual_release.clear()
        yield (
            status_panel("正在合成", "生成完成后可在下方试听或下载。"),
            gr.update(value=None, visible=False),
            gr.update(interactive=False),
            gr.skip(),
            gr.update(interactive=True),
            None,
            gr.update(value=1, interactive=False),
            "正在生成原始音频…",
            gr.update(value=None),
            gr.update(visible=False),
        )
        try:
            for event in service.generate(data):
                if event["kind"] == "progress":
                    yield (
                        status_panel("正在合成", f"片段 {event['current']}/{event['total']} 已完成，正在拼接音频。"),
                        gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                        gr.skip(), gr.skip(),
                    )
                elif event["kind"] == "audio":
                    _, duration_label = speed_result(event["path"], 1)
                    _, choices, audio_path = history.add(event["path"], target_text, record_config)
                    yield (
                        status_panel("音频已生成", "可试听或下载。"),
                        gr.update(value=audio_path, visible=True),
                        gr.update(interactive=False),
                        event["seed"],
                        gr.update(interactive=True),
                        audio_path,
                        gr.update(value=1, interactive=True),
                        duration_label,
                        gr.update(choices=choices, value=None),
                        gr.update(visible=False),
                    )
                elif event["kind"] == "done":
                    title = "生成已停止" if event["cancelled"] else "合成完成"
                    yield (
                        status_panel(title, "可以继续修改文本并再次生成。"),
                        gr.skip(),
                        gr.update(interactive=True),
                        gr.skip(),
                        gr.update(interactive=False),
                        gr.skip(), gr.skip(), gr.skip(),
                        gr.skip(), gr.skip(),
                    )
        except (ValueError, OSError, RuntimeError) as error:
            if manual_release.is_set():
                yield (
                    status_panel("模型已卸载", "选择模型并加载后可继续合成。"),
                    gr.skip(),
                    gr.update(interactive=False),
                    gr.skip(), gr.skip(), gr.skip(),
                    gr.skip(),
                    gr.update(interactive=False),
                    gr.skip(), gr.skip(),
                )
                return
            yield (
                status_panel("合成失败", str(error), error=True),
                gr.update(value=None, visible=False),
                gr.update(interactive=bool(service.metadata)),
                gr.skip(),
                gr.update(interactive=False),
                None, gr.update(value=1, interactive=False), "合成后可调整试听和下载倍速。",
                gr.update(value=None), gr.update(visible=False),
            )
            gr.Warning(escape(str(error)), duration=5)
            return

    def release():
        manual_release.set()
        service.close()
        return (
            status_panel("模型已卸载", "选择模型并加载后可继续合成。"),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    def stop():
        try:
            return status_panel("生成状态", service.cancel())
        except (ValueError, OSError) as error:
            gr.Warning(escape(str(error)), duration=5)
            return status_panel("停止请求失败", str(error), error=True)

    def save_reference_snapshot(name, audio, words, language, extra, text_free, gpt_model, sovits_model):
        try:
            saved = snapshots.save(name, audio, words, language, extra, text_free, gpt_model, sovits_model)
            return (gr.update(choices=snapshots.choices(), value=saved["id"]),
                    gr.update(value=""), f"已保存参考设置快照：**{escape(saved['name'])}**。")
        except (ValueError, OSError, RuntimeError) as error:
            gr.Warning(escape(str(error)), duration=5)
            return gr.skip(), gr.skip(), f"保存失败：{escape(str(error))}"

    save_snapshot.click(save_reference_snapshot,
                        [snapshot_name, reference, prompt, prompt_language, aux_refs, ref_free, gpt, sovits],
                        [snapshot_choice, snapshot_name, save_notice])

    def apply_local_reference(filename):
        try:
            audio = snapshots.local_reference(filename)
            return audio, "已选择本机参考音频；请核对实际说出的文字和语种，再按需保存快照。"
        except (ValueError, OSError, RuntimeError) as error:
            gr.Warning(escape(str(error)), duration=5)
            return gr.skip(), f"无法选择本机音频：{escape(str(error))}"

    use_local_reference.click(apply_local_reference, [local_reference], [reference, existing_notice])

    def restore_reference_snapshot(snapshot_id):
        try:
            saved, audio, extra = snapshots.load(snapshot_id)
            from config import get_weights_names
            sovits_names, gpt_names = get_weights_names()
            missing = [label for label, path, choices in
                       [("GPT", saved["gpt"], gpt_names), ("SoVITS", saved["sovits"], sovits_names)] if path not in choices]
            model_message = "；保存时的 " + "、".join(missing) + " 模型已不在目录中，请重新选择" if missing else "；点击“加载模型”使模型选择生效"
            return (gr.update(choices=gpt_names, value=saved["gpt"] if saved["gpt"] in gpt_names else None),
                    gr.update(choices=sovits_names, value=saved["sovits"] if saved["sovits"] in sovits_names else None),
                    audio, saved["prompt"], saved["language"], extra or None, saved["ref_free"],
                    f"已恢复 **{escape(saved['name'])}**{model_message}。")
        except (ValueError, OSError, KeyError, TypeError) as error:
            gr.Warning(escape(str(error)), duration=5)
            return (gr.skip(),) * 7 + (f"读取快照失败：{escape(str(error))}",)

    apply_snapshot.click(restore_reference_snapshot, [snapshot_choice],
                         [gpt, sovits, reference, prompt, prompt_language, aux_refs, ref_free, existing_notice])

    def delete_reference_snapshot(snapshot_id, confirmed):
        try:
            if not confirmed:
                raise ValueError("请先勾选确认删除选中的快照")
            snapshots.delete(snapshot_id)
            return gr.update(choices=snapshots.choices(), value=None), gr.update(value=False), "快照已删除；当前表单内容仍可继续使用。"
        except (ValueError, OSError, KeyError, TypeError) as error:
            gr.Warning(escape(str(error)), duration=5)
            return gr.skip(), gr.skip(), f"删除失败：{escape(str(error))}"

    remove_snapshot.click(delete_reference_snapshot, [snapshot_choice, confirm_delete],
                          [snapshot_choice, confirm_delete, existing_notice])
    snapshot_choice.change(lambda: gr.update(value=False), [], [confirm_delete], queue=False)
    refresh.click(lambda: gr.update(choices=snapshots.choices()), [], [snapshot_choice], queue=False)
    refresh.click(lambda: gr.update(choices=snapshots.local_reference_choices()), [], [local_reference], queue=False)

    load_event = load.click(
        load_models,
        [
            gpt,
            sovits,
            gpu_number,
            parallel,
            version_component,
            bert_component,
            hubert_component,
        ],
        model_outputs,
    )
    load_event.then(loaded_models_panel, [gpt, sovits], [loaded_models], queue=False)
    unload_event = unload.click(release, outputs=[status, generate, cancel], queue=False)
    unload_event.then(loaded_models_panel, [gpt, sovits], [loaded_models], queue=False)
    gpt.change(loaded_models_panel, [gpt, sovits], [loaded_models], queue=False)
    sovits.change(loaded_models_panel, [gpt, sovits], [loaded_models], queue=False)
    cancel.click(stop, outputs=status, queue=False)
    custom_speed_button.click(lambda: gr.update(visible=True), outputs=custom_speed_controls,
                              queue=False)

    def choose_playback_speed(source, rate):
        if not source or rate is None:
            return gr.skip(), gr.skip(), gr.skip()
        try:
            audio, duration = speed_result(source, rate)
            return gr.update(value=audio, visible=True), duration, gr.update(value=rate, info=None)
        except (ValueError, OSError, RuntimeError) as error:
            gr.Warning(escape(str(error)), duration=5)
            return gr.skip(), gr.skip(), gr.skip()

    playback_speed.input(choose_playback_speed, [original_output, playback_speed],
                         [output, playback_duration, custom_speed], trigger_mode="always_last")

    def choose_custom_speed(source, rate):
        try:
            audio, duration = speed_result(source, rate)
            return gr.update(value=audio, visible=True), duration, gr.update(value=None), gr.update(info=None)
        except (ValueError, OSError, RuntimeError) as error:
            gr.Warning(escape(str(error)), duration=5)
            return gr.skip(), gr.skip(), gr.skip(), gr.update(info=str(error))

    apply_custom_speed.click(choose_custom_speed, [original_output, custom_speed],
                             [output, playback_duration, playback_speed, custom_speed])

    def choose_result(record_id):
        if not record_id:
            return gr.update(value="", visible=False), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        record = history.get(record_id)
        if record is None:
            gr.Warning("这条生成记录的音频已不存在，请选择其他结果", duration=5)
            return (gr.update(value="", visible=False), gr.update(value=None, visible=False), None,
                    gr.update(value=1, interactive=False), "请选择其他生成结果。")
        try:
            audio, duration = speed_result(OUTPUT_DIR / record["filename"], 1)
            return (gr.update(value=result_details_panel(record), visible=True),
                    gr.update(value=audio, visible=True), audio,
                    gr.update(value=1, interactive=True), duration)
        except (ValueError, OSError, RuntimeError) as error:
            gr.Warning(escape(str(error)), duration=5)
            return (gr.update(value="", visible=False), gr.update(value=None, visible=False), None,
                    gr.update(value=1, interactive=False), "请选择其他生成结果。")

    result_choice.input(choose_result, [result_choice],
                        [result_details, output, original_output, playback_speed, playback_duration],
                        trigger_mode="always_last")
    clear_result_choice.click(
        lambda: (gr.update(value=None), gr.update(value="", visible=False)),
        outputs=[result_choice, result_details], queue=False,
    )
    generate_event = generate.click(
        synthesize, generation_inputs,
        [status, output, generate, seed, cancel, original_output, playback_speed,
         playback_duration, result_choice, result_details]
    )
    generate_event.then(loaded_models_panel, [gpt, sovits], [loaded_models], queue=False)
    return {"gpt": gpt, "sovits": sovits, "parallel": parallel, "refresh": refresh}
