"""Task-center controls for post-generation mix edits."""
import gradio as gr
from . import audio_edit, service


def controls():
    with gr.Accordion('生成后编辑 · 增益与混响',open=False,visible=False) as panel:
        gr.Markdown('保存到当前任务，自动刷新分轨、合并人声与完整混音。每次都相对最初音轨计算；原曲参考、分离输入不修改。若有 PSOLA / PC-NSF 多版结果，会同步调整各版。',elem_classes=['workspace-hint'])
        with gr.Row():
            dry_lead=gr.Slider(-60,12,value=0,step=.5,label='加混响前主唱／人声 · dB')
            dry_backing=gr.Slider(-60,12,value=0,step=.5,label='加混响前和声 · dB')
        with gr.Row() as wet_group:
            wet_lead=gr.Slider(-60,12,value=0,step=.5,label='加混响后主唱／人声 · dB')
            wet_backing=gr.Slider(-60,12,value=0,step=.5,label='加混响后和声 · dB')
        accompaniment=gr.Slider(-60,12,value=0,step=.5,label='伴奏 · dB',info='同时用于干声版与带混响版的完整混音。0 dB 表示不增加额外增益；和声包括任务选择保留的原和声。')
        room=gr.Checkbox(value=False,label='重建输出混响',info='开启后可在原任务内添加或调整混响；关闭则沿用最初生成的混响轨。保留的原和声不重复添加混响。')
        with gr.Group(visible=False) as room_group:
            with gr.Row():
                lead_strength=gr.Slider(0,2,value=1,step=.05,label='主唱混响强度')
                backing_strength=gr.Slider(0,2,value=1,step=.05,label='生成和声混响强度')
            decay=gr.Slider(.3,4,value=1.5,step=.05,label='混响衰减 / 秒')
            predelay=gr.Slider(0,120,value=25,step=1,label='混响预延迟 / 毫秒')
            wet=gr.Slider(0,.65,value=.22,step=.01,label='混响／干声幅度比')
            gr.Markdown('从备份的生成干声重建，沿用已有混响的频率阻尼与空间宽度；没有既有混响时使用默认空间参数。此处不重新分离人声、不重新执行 SVC 或音准校准。',elem_classes=['workspace-hint'])
        gr.Markdown('按设定增益保存，不自动衰减或限幅；调整人声不会压低伴奏或其他分轨。浮点 WAV 保留超过 0 dBFS 的幅度，过载只提示，播放可能失真，可自行降低增益。保存后支持恢复最初结果。',elem_classes=['workspace-hint'])
        with gr.Row(elem_classes=['compact-actions']):
            save=gr.Button('确认保存增益与混响，并刷新音频',variant='primary')
            restore=gr.Button('恢复最初音频')
            stop=gr.Button('停止本次编辑')
        notice=gr.Markdown()
    context=gr.State({})
    fields=[dry_lead,dry_backing,wet_lead,wet_backing,accompaniment,room,lead_strength,backing_strength,decay,predelay,wet]
    return dict(panel=panel,fields=fields,wet_group=wet_group,room_group=room_group,
                save=save,restore=restore,stop=stop,notice=notice,context=context)


def bind(components,job,timer,safe):
    fields=components['fields']
    def load(task,previous=None):
        record=service.snapshot(task)[0] if task else {}
        editing=audio_edit.snapshot(task) if task else {}
        edit_stamp=str((editing.get('revision'),editing.get('status'),editing.get('message'),editing.get('progress')))
        notice=editing.get('message','')
        if editing.get('status')=='completed' and record.get('audio_edit',{}).get('gain_policy')=='preserve_float':
            notice+='。没有自动衰减或限幅。'
            overloaded=record['audio_edit'].get('overloaded',[])
            if overloaded:notice+=' 超过 0 dBFS：'+'、'.join(audio_edit.track_label(k) for k in overloaded)+'；可自行降低增益。'
        if previous and previous.get('job')==task and previous.get('loaded_status')==record.get('status'):
            updates=[gr.skip() for _ in load_outputs]
            if previous.get('editor_stamp')!=edit_stamp:
                updates[-2]={**previous,'editor_stamp':edit_stamp};updates[-1]=notice
            return tuple(updates)
        rows=audio_edit.layouts(record) if record and record.get('status')=='completed' else []
        enabled=bool(rows)
        values=audio_edit.defaults(record) if rows else {**{key:0 for key in audio_edit.GAIN_KEYS},
            'room':False,'lead_strength':1,'backing_strength':1,'decay':1.5,'predelay_ms':25,'wet':.22}
        has_back=any(r.get('dry_backing') for r in rows)
        has_wet=any(r.get('wet_lead') for r in rows) or bool(values['room'])
        has_inst=any(r.get('accompaniment') for r in rows)
        flags={'dry_backing':has_back,'wet_backing':has_back,'accompaniment':has_inst}
        keys=audio_edit.GAIN_KEYS+['room','lead_strength','backing_strength','decay','predelay_ms','wet']
        updates=[gr.update(value=values[key],visible=flags.get(key,True)) for key in keys]
        context={'job':task,'has_back':has_back,'has_wet':any(r.get('wet_lead') for r in rows),'loaded_status':record.get('status'),'editor_stamp':edit_stamp}
        return gr.update(visible=enabled),*updates,gr.update(visible=has_wet),gr.update(visible=bool(values['room'])),context,notice
    load_outputs=[components['panel'],*fields,components['wet_group'],components['room_group'],components['context'],components['notice']]
    loader=safe(outputs=len(load_outputs))(load)
    gr.on([job.change,timer.tick],loader,[job,components['context']],load_outputs,queue=False,trigger_mode='always_last',show_progress='hidden')
    def toggle(value,context):
        return gr.update(visible=bool(value)),gr.update(visible=bool(value) or context.get('has_wet',False))
    fields[5].change(toggle,[fields[5],components['context']],[components['room_group'],components['wet_group']],queue=False)
    @safe(outputs=2)
    def save(task,context,*values):
        if not task or context.get('job')!=task:raise ValueError('当前任务已切换，请重新选择任务后保存')
        keys=audio_edit.GAIN_KEYS+['room','lead_strength','backing_strength','decay','predelay_ms','wet']
        audio_edit.submit(task,dict(zip(keys,values)))
        return '正在当前任务中保存。完成后音轨和波形会自动刷新；离开页面后仍继续处理。',gr.update(active=True)
    components['save'].click(save,[job,components['context'],*fields],[components['notice'],timer],queue=False)
    @safe(outputs=1)
    def stop(task):return audio_edit.cancel(task)
    components['stop'].click(stop,[job],[components['notice']],queue=False)
    @safe(outputs=len(load_outputs)+1)
    def restore(task):
        audio_edit.restore(task)
        return *load(task),gr.update(active=True)
    components['restore'].click(restore,[job],[*load_outputs,timer],queue=False)
