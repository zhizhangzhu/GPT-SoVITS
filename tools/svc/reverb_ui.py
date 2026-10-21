"""Shared, optional post-processing controls for existing completed tasks."""
import gradio as gr
from .reverb import VERSIONS, STYLES


def conversion_controls():
    with gr.Accordion('输出混响（可选）',open=False):
        enabled = gr.Checkbox(value=False,label='转换后自动添加混响',info='转换完成后直接生成带混响版本，干声和原转换结果同时保留。')
        with gr.Group(visible=False) as options:
            style = gr.Radio(STYLES,value='reference',label='输出混响方式',info='分别参考源主唱、源和声估计空间感；保留原和声时不重复处理它。')
            with gr.Row():
                lead = gr.Slider(0,2,value=1,step=.05,label='主唱／人声混响强度')
                backing = gr.Slider(0,2,value=1,step=.05,label='转换和声的混响强度')
            with gr.Accordion('手动参数与估计回退',open=False):
                decay = gr.Slider(.3,4,value=1.5,step=.05,label='衰减时间 RT60 / 秒')
                predelay = gr.Slider(0,120,value=25,step=1,label='预延迟 / 毫秒',info='参考估计和手动模式均使用此值。')
                wet = gr.Slider(0,.65,value=.22,step=.01,label='混响／干声幅度比',info='手动模式使用；参考估计不可靠时作为回退。')
                clean = gr.Checkbox(value=False,label='添加前先对生成的人声去混响',info='输出已有明显混响时才需考虑；额外处理可能影响气声和尾音。')
            gr.Markdown('混响风格为近似估计；未找到清晰尾音时会使用手动回退参数。处理在原任务内完成，结果在任务中心试听、下载。',elem_classes=['workspace-hint'])
        enabled.change(lambda value:gr.update(visible=bool(value)),[enabled],[options],queue=False)
    return [enabled,style,lead,backing,decay,predelay,wet,clean]


def controls():
    with gr.Accordion('输出混响 · 参考原曲（可选）',open=False):
        gr.Markdown('选择已完成的转换或校准任务，单独生成带混响的新音轨，原结果保留。主唱、和声分别参考原轨；保留原和声时不再给它叠加混响。选择已有混响任务再次提交时，会从原转换／校准结果重新处理。',elem_classes=['workspace-hint'])
        version = gr.Dropdown(VERSIONS,value='auto',label='待处理版本',info='校准任务包含两版结果时，自动选择优先使用 PSOLA；可指定另一版。')
        style = gr.Radio(STYLES,value='reference',label='混响方式',info='参考方式估计空间感，无法精确复刻原曲效果；需要原曲分离湿声。')
        with gr.Row():
            lead = gr.Slider(0,2,value=1,step=.05,label='主唱／人声混响强度',info='1 为估计或手动混响量；0 为不添加。')
            backing = gr.Slider(0,2,value=1,step=.05,label='生成和声的混响强度')
        with gr.Accordion('手动参数与估计回退',open=False):
            decay = gr.Slider(.3,4,value=1.5,step=.05,label='衰减时间 RT60 / 秒',info='手动模式使用；参考尾音不足时作为回退。')
            predelay = gr.Slider(0,120,value=25,step=1,label='预延迟 / 毫秒',info='两种模式均使用此值，保留干声起点。')
            wet = gr.Slider(0,.65,value=.22,step=.01,label='混响／干声幅度比',info='手动混响量；参考估计不可靠时作为回退。')
            clean = gr.Checkbox(value=False,label='先对待处理人声去混响，再添加新混响',info='输出已有明显混响时可选；通常 SVC 生成的较干人声无需开启。')
            gpu = gr.Textbox(value='0',label='混响分析 GPU 编号')
        gr.Markdown('仅分析原轨参数，混响由新的人声生成。输出保持原时长，结尾之外的尾音截断；结果支持波形、试听和下载。',elem_classes=['workspace-hint'])
        submit = gr.Button('为所选任务生成带混响音轨',variant='primary')
    return submit,[version,style,lead,backing,decay,predelay,wet,clean,gpu]
