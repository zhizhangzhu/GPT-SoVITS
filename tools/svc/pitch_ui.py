"""Independent upload-and-correct workspace sharing SVC background jobs."""
from html import escape
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr

from tools.assets import workspace_header
from . import service,separation,pitch_job,reverb,reverb_ui,audio_edit
from .pitch_upload import MODES
from . import task_audio
from .ui import safe,status,STATUS,report_stamp,text_stamp

TRACKS = [
    ('psola_mix','PSOLA · 校准后完整音频'),('pc_nsf_mix','PC-NSF · 校准后完整音频'),
    ('psola_combined','PSOLA · 校准后人声'),('pc_nsf_combined','PC-NSF · 校准后人声'),
    ('reference_full','上传的参考音频'),('target_full','上传的待校准音频'),
    ('reference_lead','提取的参考主唱'),('before_lead','校准前主唱'),
    ('reference_backing','提取的参考和声'),('before_backing','校准前和声'),
    ('reference_vocals','参考人声'),('before_vocals','校准前人声'),
    ('psola_lead','PSOLA · 校准后主唱'),('pc_nsf_lead','PC-NSF · 校准后主唱'),
    ('psola_backing','PSOLA · 校准后和声'),('pc_nsf_backing','PC-NSF · 校准后和声'),
    ('psola_vocals','PSOLA · 校准后单声部人声'),('pc_nsf_vocals','PC-NSF · 校准后单声部人声'),
    ('accompaniment','待校准音频的原伴奏'),
]
TRACKS += reverb.TRACKS
TRACKS += audio_edit.EXTRA_TRACKS


def jobs():
    rows=[]
    for record in service.history():
        if record['kind'] not in {'pitch','reverb'}:continue
        source='混响' if record['kind']=='reverb' else '上传' if record['options'].get('input_mode')=='upload' else 'SVC'
        name=record.get('target_name',record['id'][:8])
        created=datetime.fromisoformat(record['created_at']).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M')
        rows.append((f'{source} · {created} · {name} · {STATUS.get(record["status"],record["status"])}',record['id']))
    return rows


def build_page(busy=None):
    def guard():
        if busy and busy():
            raise ValueError('TTS 训练、预处理或合成正在运行，请结束后提交校准任务')
        from tools.tts_workspace.service import service as tts_service
        process=tts_service.process
        if process is not None and process.poll() is None:
            raise ValueError('TTS 推理模型仍驻留显存，请先在“TTS 推理”中卸载模型')

    gr.HTML(workspace_header('一键校准 · 参考音准修正','上传参考与待校准音频 → 提取人声与对齐 → 修正音准，试听并下载'))
    gr.Markdown('以参考演唱的音高曲线校准另一份音频，保留待校准音频的节奏与长度。参考和目标需为同一首歌、对应段落；带伴奏音乐会先提取人声。',elem_classes=['workspace-note'])
    with gr.Row(elem_classes=['svc-layout']):
        with gr.Column(scale=4,min_width=290,elem_classes=['workbench-panel','svc-source']):
            gr.Markdown('### 1 · 上传音频')
            reference=gr.File(label='参考音频',file_types=['audio'],type='filepath')
            reference_kind=gr.Radio(['纯人声','带伴奏音乐'],value='纯人声',label='参考音频类型')
            target=gr.File(label='待校准音频',file_types=['audio'],type='filepath')
            target_kind=gr.Radio(['纯人声','带伴奏音乐'],value='纯人声',label='待校准音频类型',info='带伴奏的目标会在校准后混回原伴奏。每份音频最长 20 分钟。')
            gr.Markdown('### 2 · 校准方式')
            mode=gr.Radio(MODES,value='keep',label='声部处理',info='清唱或已经分开的单声部音轨可选“直接校准”，减少分离处理；叠唱和声仍可能无法可靠修音。')
            backend=gr.Dropdown(pitch_job.BACKENDS,value='psola',label='修音后端',info='可同时生成两版试听，原始上传与处理前音轨保留。')
            alignment=gr.Radio([('同曲自动对齐','auto'),('按原时间轴','timeline')],value='auto',label='参考对齐方式',info='独立演唱可自动匹配不同起点和速度；SVC 转换前后音轨可选原时间轴。')
            with gr.Accordion('音高与对齐设置',open=False):
                method=gr.Dropdown([('RMVPE','rmvpe'),('FCPE','fcpe')],value='rmvpe',label='音高分析方法',info='另一种提取器会用于交叉检查。')
                threshold=gr.Slider(20,200,value=50,step=5,label='校准触发阈值 / 音分',info='100 音分为一个半音；只处理持续、可信的偏差。')
                strength=gr.Slider(0,1,value=1,step=.05,label='校准强度')
                transpose=gr.Slider(-24,24,value=0,step=1,label='参考音高偏移 / 半音',info='目标与参考采用不同调时，先把参考移到希望保留的调。')
                offset=gr.Number(value=0,label='参考时间偏移 / 秒',info='仅原时间轴模式生效。正值表示从参考音频更后的位置开始取音高。')
            with gr.Accordion('人声分离与去混响',open=False):
                separator=gr.Dropdown(separation.VOCAL_CHOICES,value=separation.VOCAL_DEFAULT,label='人声 / 伴奏分离模型')
                karaoke=gr.Dropdown(separation.KARAOKE_CHOICES,value=separation.KARAOKE_DEFAULT,label='主唱 / 和声分离模型')
                dry_reference=gr.Checkbox(value=False,label='参考人声去混响')
                dry_target=gr.Checkbox(value=False,label='待校准人声去混响')
                gr.Markdown('去混响默认关闭。仅校准主唱时保留原和声；分别校准时，两条人声分别处理。',elem_classes=['workspace-hint'])
                gpu=gr.Textbox(value='0',label='GPU 编号')
            submit=gr.Button('一键校准',variant='primary')
            notice=gr.Markdown('上传两份音频后开始；任务在后台运行，离开页面后仍继续处理。')
        with gr.Column(scale=5,min_width=290,elem_classes=['workbench-panel','svc-review']):
            gr.Markdown('### 3 · 进度与结果')
            with gr.Row():
                job=gr.Dropdown(label='校准任务',choices=jobs(),scale=4,min_width=200)
                refresh=gr.Button('刷新列表',scale=0)
            with gr.Row(elem_classes=['compact-actions']):
                stop=gr.Button('停止任务')
                retry=gr.Button('按原配置重试')
            reverb_submit,reverb_inputs=reverb_ui.controls()
            with gr.Group(elem_id='pitch-upload-task-detail',elem_classes=['svc-audio-detail']):
                progress=gr.HTML(status({}))
                with gr.Accordion('音准、混响与编辑报告',open=False):
                    report=gr.HTML()
                gr.Markdown('结果支持波形定位、试听与下载。自动对齐仅映射参考音高，输出继续使用待校准音频的时间轴。',elem_classes=['workspace-hint'])
                gr.HTML(task_audio.AMPLITUDE_LEGEND)
                players=task_audio.grouped_players(TRACKS,prefix='pitch-upload-')
                downloads=gr.HTML()
            with gr.Accordion('处理日志',open=False):
                logs=gr.Textbox(label='后台日志',lines=10,interactive=False,elem_classes=['process-log'])
    timer=gr.Timer(3,active=False)
    state=gr.State({})

    @safe(outputs=3)
    def submit_upload(ref,tar,ref_kind,tar_kind,voice_mode,engine,align,detector,limit,amount,shift,time_offset,sep,kara,dry_ref,dry_tar,device):
        guard()
        task=service.create_uploaded_pitch(ref,tar,ref_kind=='带伴奏音乐',tar_kind=='带伴奏音乐',
            predictor=detector,backend=engine,threshold=limit,strength=amount,gpu=device,vocal_mode=voice_mode,
            alignment=align,reference_offset=time_offset,reference_transpose=shift,dereverb_reference=dry_ref,
            dereverb_target=dry_tar,separation_model=sep,vocal_split_model=kara)
        return gr.update(choices=jobs(),value=task),'已提交校准任务，可在右侧查看进度。',gr.update(active=True)

    submit.click(submit_upload,[reference,target,reference_kind,target_kind,mode,backend,alignment,method,threshold,strength,transpose,offset,separator,karaoke,dry_reference,dry_target,gpu],[job,notice,timer])

    @safe(outputs=1)
    def refresh_jobs(current):
        choices=jobs()
        selected=current if current in {value for _,value in choices} else choices[0][1] if choices else None
        return gr.update(choices=choices,value=selected)
    refresh.click(refresh_jobs,[job],[job],queue=False)

    @safe(outputs=1)
    def stop_job(task):
        if not task:raise ValueError('请先选择校准任务')
        service.cancel(task)
        return '已请求停止任务。'
    stop.click(stop_job,[job],[notice],queue=False)

    @safe(outputs=3)
    def retry_job(task):
        guard()
        if not task:raise ValueError('请先选择校准任务')
        service.retry(task)
        return gr.update(choices=jobs(),value=task),'已按原配置重新提交。',gr.update(active=True)
    retry.click(retry_job,[job],[job,notice,timer])

    @safe(outputs=3)
    def submit_reverb(task,*settings):
        guard()
        if not task:raise ValueError('请先选择已完成的校准任务')
        result=service.create_reverb(task,*settings)
        return gr.update(choices=jobs(),value=result),'混响任务已提交，校准原结果保留。',gr.update(active=True)
    reverb_submit.click(submit_reverb,[job,*reverb_inputs],[job,notice,timer])

    def poll_error(error,task,old):
        return status({'status':'failed','message':str(error),'progress':0}),gr.skip(),gr.skip(),*(gr.skip() for _ in players),gr.skip(),{},gr.update(active=False)

    @safe(outputs=6+len(players),quiet=True,fallback=poll_error)
    def poll_upload(task,old):
        old=old or {}
        record,log=service.snapshot(task)
        if record and record['kind'] not in {'pitch','reverb'}:raise ValueError('请选择音准校准或混响任务')
        outputs=record.get('outputs',{})
        if old.get('task')==task and old.get('outputs')==outputs:
            audio=[gr.skip() for _ in players];links=gr.skip()
        else:
            published,previews=task_audio.publish(record,{key:outputs[key] for key,_ in TRACKS if outputs.get(key)},downloads)
            audio=[gr.skip() if old.get('task')==task and old.get('outputs',{}).get(key)==outputs.get(key)
                   else gr.update(value=task_audio.player(label,published.get(key),task or '',previews.get(key),key=key),visible=bool(published.get(key))) for key,label in TRACKS]
            links=task_audio.download_links([(key,label,'') for key,label in TRACKS],published,task or '')
        editing=audio_edit.snapshot(task)
        stamp={'report':report_stamp(record),'editing':editing}
        same_task=old.get('task')==task
        report_update=(gr.skip() if same_task and old.get('report_stamp')==stamp else pitch_job.report_html(record,downloads)+reverb.report_html(record,downloads)+audio_edit.report_html(record))
        active=bool(task) and (record.get('status') not in service.TERMINAL or editing.get('status') in audio_edit.ACTIVE)
        panel=status(record);log_stamp=text_stamp(log)
        new={'task':task,'outputs':outputs,'report_stamp':stamp,'progress_html':panel,'log_stamp':log_stamp,'polling':bool(active)}
        progress_update=gr.skip() if same_task and old.get('progress_html')==panel else panel
        log_update=gr.skip() if same_task and old.get('log_stamp')==log_stamp else log
        timer_update=gr.update(active=bool(active))
        return progress_update,log_update,report_update,*audio,links,(gr.skip() if old==new else new),timer_update

    gr.on([timer.tick,job.change],poll_upload,[job,state],[progress,logs,report,*players,downloads,state,timer],queue=False,trigger_mode='always_last',show_progress='hidden')
    return {'job':job}
