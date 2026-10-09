"""The complete synthesis flow embedded in the main workbench's inference tab."""

from html import escape
import os
import threading

import gradio as gr

from tools.assets import workspace_header
from .service import service

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


def status_panel(title, message):
    return f'<div class="inference-status" role="status"><strong>{escape(title)}</strong><span>{escape(message)}</span></div>'


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
        with gr.Column(scale=7, min_width=320, elem_classes="inference-main"):
            status = gr.HTML(
                status_panel("尚未加载模型", "在左侧选择模型，点击加载模型后即可合成。")
            )
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
                    cut = gr.Dropdown(label="长文本切分", choices=CUTS, value=CUTS[1])
                with gr.Row():
                    speed = gr.Slider(0.6, 1.65, value=1, step=0.05, label="语速")
                    pause = gr.Slider(
                        0.1, 0.5, value=0.3, step=0.01, label="分段停顿 / 秒"
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
                    label="复用上次生成结果调整语速与音色", value=False
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
                output = gr.Audio(
                    label="合成音频 · 试听与下载", type="filepath", interactive=False
                )
            gr.Markdown(
                "更换模型后请重新加载。卸载模型可释放显存；语音和下载结果仍保留在当前页面。",
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
        if busy and busy():
            raise gr.Error("请等待当前训练或预处理完成后加载推理模型")
        if ensure_idle:
            ensure_idle()
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
                status_panel("模型加载失败", str(error)),
                gr.update(interactive=False),
                *[gr.skip() for _ in model_outputs[2:]],
            ]
            raise gr.Error(str(error)) from error

    generation_inputs = [
        reference,
        prompt,
        prompt_language,
        text,
        text_language,
        cut,
        speed,
        pause,
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
        if not service.metadata:
            raise gr.Error("请先加载模型")
        if busy and busy():
            raise gr.Error("请等待当前训练或预处理完成后合成")
        if ensure_idle:
            ensure_idle()
        if not reference or not target_text.strip():
            raise gr.Error("请上传参考音频并填写目标文本")
        if not reference_free and not prompt_text.strip():
            raise gr.Error("请填写参考音频中的文字")
        actual = service.spec
        if actual is None:
            raise gr.Error("模型已卸载，请重新加载")
        if (gpt_path, sovits_path, str(gpu_mapper(gpu_id)), bool(parallel_mode)) != (
            actual["gpt_path"],
            actual["sovits_path"],
            actual["gpu"],
            actual["parallel"],
        ):
            raise gr.Error("模型或推理设置已更改，请先重新加载模型")
        data = dict(
            reference=reference,
            prompt_text=prompt_text,
            prompt_language=prompt_lang,
            text=target_text,
            text_language=target_lang,
            cut=split,
            speed=speed_value,
            pause=pause_value,
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
        manual_release.clear()
        yield (
            status_panel("正在合成", "生成完成后可在下方试听或下载。"),
            gr.skip(),
            gr.update(interactive=False),
            gr.skip(),
            gr.update(interactive=True),
        )
        try:
            for event in service.generate(data):
                if event["kind"] == "audio":
                    yield (
                        status_panel("音频已生成", "可试听或下载。"),
                        event["path"],
                        gr.update(interactive=False),
                        event["seed"],
                        gr.update(interactive=True),
                    )
                elif event["kind"] == "done":
                    title = "生成已停止" if event["cancelled"] else "合成完成"
                    yield (
                        status_panel(title, "可以继续修改文本并再次生成。"),
                        gr.skip(),
                        gr.update(interactive=True),
                        gr.skip(),
                        gr.update(interactive=False),
                    )
        except (ValueError, OSError) as error:
            if manual_release.is_set():
                yield (
                    status_panel("模型已卸载", "选择模型并加载后可继续合成。"),
                    gr.skip(),
                    gr.update(interactive=False),
                    gr.skip(),
                    gr.update(interactive=False),
                )
                return
            yield (
                status_panel("合成失败", str(error)),
                gr.skip(),
                gr.update(interactive=bool(service.metadata)),
                gr.skip(),
                gr.update(interactive=False),
            )
            raise gr.Error(str(error)) from error

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
            raise gr.Error(str(error)) from error

    load.click(
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
    unload.click(release, outputs=[status, generate, cancel], queue=False)
    cancel.click(stop, outputs=status, queue=False)
    generate.click(
        synthesize, generation_inputs, [status, output, generate, seed, cancel]
    )
    return {"gpt": gpt, "sovits": sovits, "parallel": parallel, "refresh": refresh}
