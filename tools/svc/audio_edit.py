"""Non-destructive gain/reverb edits inside an existing completed task.

An editor subprocess has its own status, never creates a task row, and commits
all new output paths with one job.json replacement. Originals stay immutable.
"""
from copy import deepcopy
from html import escape
from pathlib import Path
import math
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone

from filelock import FileLock, Timeout
import numpy as np
import psutil
import soundfile as sf

from . import storage as db
from .audio import sum_tracks, write_audio
from .install import ENV
from tools.workspace_resources import require_space

GAIN_KEYS = ['dry_lead','dry_backing','wet_lead','wet_backing','accompaniment']
GAIN_LABELS = ['加混响前主唱／人声','加混响前和声','加混响后主唱／人声','加混响后和声','伴奏']
EXTRA_TRACKS = [('edit_kept_backing','调整后的保留和声')]+[
    (method+'_reverb_'+track,label+' · '+name)
    for method,label in [('psola','PSOLA'),('pc_nsf','PC-NSF')]
    for track,name in [('lead','加混响后主唱／人声'),('backing','加混响后和声'),('kept_backing','混响版保留和声'),
                       ('vocals','加混响后合并人声'),('mix','加混响后完整混音')]]
ACTIVE = {'queued','running'}


def track_label(key):
    labels={'converted_lead':'加混响前主唱','converted_backing':'加混响前和声',
            'vocals':'加混响前合并人声','mix':'加混响前完整混音','accompaniment':'伴奏',
            'reverb_lead':'加混响后主唱','reverb_backing':'加混响后和声',
            'reverb_vocals':'加混响后合并人声','reverb_mix':'加混响后完整混音',
            'reverb_before_vocals':'加混响前合并人声','reverb_before_mix':'加混响前完整混音'}
    for method,name in [('psola','PSOLA'),('pc_nsf','PC-NSF')]:
        if key.startswith(method+'_'):
            part=key[len(method)+1:]
            return name+' · '+labels.get(part,{'lead':'主唱','backing':'和声','combined':'合并人声'}.get(part,part))
    return labels.get(key,key)


def _path(value,directory):
    if not value: return None
    path = Path(value).resolve()
    return path if path.is_relative_to(directory.resolve()) and path.is_file() else None


def _state_path(job): return db.job_dir(job)/'audio-edit.json'
def _lock(job): return FileLock(str(db.job_dir(job)/'.audio-edit.lock'),timeout=0)


def snapshot(job):
    if not job:return {}
    state = db.read(_state_path(job),{})
    if state.get('status') in ACTIVE:
        from .service import live
        try:
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(state.get('created_at',db.now()))).total_seconds()
        except ValueError:
            age=0
        exited=not live(state) if state.get('pid') else age>45
        if exited:
            try:
                with _lock(job):
                    state = db.read(_state_path(job),{})
                    if state.get('status') in ACTIVE:
                        state.update(status='failed',message='音轨编辑进程已退出；原结果保留，可以重新保存。')
                        db.write(_state_path(job),state)
            except Timeout:
                pass
    return state


def active(job):
    if snapshot(job).get('status') in ACTIVE:return True
    # Metadata-only restore also holds the editor lock. Exclude concurrent
    # deletion/regeneration while it atomically changes output paths.
    try:
        with _lock(job):return False
    except Timeout:return True


def _config(record,depth=0):
    own = db.read(db.job_dir(record['id'])/'mix-config.json',{})
    if own:return own
    options = record['options']
    if record['kind']=='convert':return {**options,'source_normalization_gain':record.get('outputs',{}).get('vocal_normalization_gain',1)}
    if record['kind']=='pitch' and options.get('input_mode')=='upload':return options
    source = options.get('source_job')
    original = db.read(db.job_dir(source)/'job.json',{}) if source and depth<3 else {}
    return _config(original,depth+1) if original else options


def layouts(record):
    """Owned source stems and their existing mix ratios, for every version."""
    from .service import conversion_outputs
    if record.get('kind') not in {'convert','pitch','reverb'}:return []
    directory = db.job_dir(record['id'])
    outputs = conversion_outputs(record)
    config = _config(record)
    harmony = float(config.get('harmony_gain',1))
    normalization = float(outputs.get('vocal_normalization_gain',config.get('source_normalization_gain',1)))
    report = db.read(_path(outputs.get('reverb_report'),directory),{}) if _path(outputs.get('reverb_report'),directory) else {}
    wet_normalization = float(report.get('vocal_normalization_gain',1))
    result=[]
    def piece(key,weight=1.,fallback=None):
        source = _path(outputs.get(key),directory) or _path(fallback,directory)
        return dict(source=str(source),key=key,weight=weight) if source else None
    def add(name,lead,back,combined,mix,wet_prefix='reverb_',wet_lead=None,wet_back=None):
        if not lead:return
        result.append(dict(name=name,dry_lead=lead,dry_backing=back,dry_combined=combined,dry_mix=mix,
            wet_lead=wet_lead,wet_backing=wet_back,wet_prefix=wet_prefix,
            vocal_gain=float(config.get('vocal_gain',1)),instrumental_gain=float(config.get('instrumental_gain',1)),
            params=deepcopy(report.get('tracks',{}))))
    if record['kind']=='convert':
        split=bool(outputs.get('converted_lead'))
        lead=piece('converted_lead' if split else 'vocals')
        back=piece('converted_backing',harmony)
        if not back and split and config.get('vocal_mode')=='keep':
            back=piece('edit_kept_backing',harmony*normalization,outputs.get('dry_backing') or outputs.get('backing_vocals'))
            if back:back['generated']=False
        wet_back=piece('reverb_backing',harmony)
        if not wet_back and outputs.get('reverb_lead') and back:
            wet_back=piece('reverb_kept_backing',back['weight']*wet_normalization,back['source'])
        add('svc',lead,back,'vocals','mix',wet_lead=piece('reverb_lead'),wet_back=wet_back)
    elif record['kind']=='pitch':
        for method in ['psola','pc_nsf']:
            lead=piece(method+('_lead' if outputs.get(method+'_lead') else '_vocals'))
            back=piece(method+'_backing',harmony)
            if not back and config.get('vocal_mode')=='keep':
                back=piece('edit_kept_backing',harmony*normalization,directory/'kept_backing.wav')
                if back:back['generated']=False
            add(method,lead,back,method+'_combined',method+'_mix',wet_prefix=method+'_reverb_')
    else:
        lead=piece('reverb_before_lead')
        back=piece('reverb_before_backing',harmony)
        if not back:
            back=piece('reverb_kept_backing',harmony,directory/'kept_backing.wav')
            if back:back['generated']=False
        wet_back=piece('reverb_backing',harmony)
        if not wet_back and back:
            wet_back=piece('reverb_kept_backing',back['weight']*wet_normalization,back['source'])
        add('reverb',lead,back,'reverb_before_vocals','reverb_before_mix',
            wet_lead=piece('reverb_lead'),wet_back=wet_back)
    accompaniment=piece('accompaniment',1.,directory/'accompaniment.wav')
    for row in result:
        row['accompaniment']=deepcopy(accompaniment)
        for name,part in [('lead',row['dry_lead']),('backing',row['dry_backing'])]:
            if part and report.get('tracks',{}).get(name,{}).get('clean_output'):
                cleaned=_path(directory/'clean-output'/name/('dry_'+name+'.wav'),directory)
                if cleaned:part['room_source']=str(cleaned)
    return result


def defaults(record):
    result={key:0. for key in GAIN_KEYS}
    result.update(room=False,lead_strength=1.,backing_strength=1.,decay=1.5,predelay_ms=25.,wet=.22)
    report_path=_path(record.get('outputs',{}).get('reverb_report'),db.job_dir(record['id']))
    tracks=db.read(report_path,{}).get('tracks',{}) if report_path else {}
    for name,row in tracks.items():
        if name in {'lead','backing'}:result[name+'_strength']=float(row.get('strength',1))
    for key in ['decay','predelay_ms','wet']:
        if key in tracks.get('lead',{}):result[key]=float(tracks['lead'][key])
    result.update(record.get('audio_edit',{}).get('settings',{}))
    return result


def validate(values):
    result={'room':bool(values.get('room',False))}
    ranges=[(key,0.,-60,12) for key in GAIN_KEYS]+[
        ('lead_strength',1.,0,2),('backing_strength',1.,0,2),
        ('decay',1.5,.3,4),('predelay_ms',25.,0,120),('wet',.22,0,.65)]
    for key,default,low,high in ranges:
        value=float(values.get(key,default))
        if not math.isfinite(value) or not low<=value<=high:raise ValueError('音轨编辑参数超出范围：'+key)
        result[key]=value
    return result


def submit(job,values):
    from .service import live
    directory=db.job_dir(job)
    snapshot(job)  # Recover exited editors before taking the per-task lock.
    try:
        with _lock(job):
            record=db.read(directory/'job.json')
            if record['status']!='completed' or live(record):raise ValueError('请等待原任务完成并释放资源后再编辑音轨')
            if db.read(_state_path(job),{}).get('status') in ACTIVE:raise ValueError('正在保存音轨，请等待完成或停止本次编辑')
            rows=layouts(record)
            if not rows:raise ValueError('所选任务没有可编辑的生成人声音轨')
            values=validate(values)
            size=sum(sf.info(r['dry_lead']['source']).frames for r in rows)*8*16
            require_space(directory,additional=size,purpose='保存音轨编辑与原始备份')
            token=db.new_id()
            state=dict(id=job,revision=token,status='queued',progress=0,message='等待保存音轨',settings=values,created_at=db.now())
            db.write(_state_path(job),state)
            environment=os.environ.copy()
            environment.update(PYTHONPATH=str(db.ROOT),GPT_SOVITS_SVC_DATA=str(db.DATA),
                GPT_SOVITS_SVC_EXPERIMENTS=str(db.EXPERIMENTS),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            try:
                with (directory/'audio-edit.log').open('ab') as log:
                    process=subprocess.Popen([str(ENV/'bin/python'),'-u','-m','tools.svc.audio_edit',job,token],
                        cwd=db.ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                state.update(pid=process.pid,process_time=psutil.Process(process.pid).create_time())
                db.write(_state_path(job),state)
            except Exception as error:
                state.update(status='failed',message='音轨编辑启动失败：'+str(error));db.write(_state_path(job),state)
                raise
            return token
    except Timeout as error:
        raise ValueError('该任务正在保存音轨，请稍后再试') from error


def cancel(job):
    state=snapshot(job)
    if state.get('status') in ACTIVE:
        (db.job_dir(job)/('audio-edit-cancel-'+state['revision'])).touch()
        return '已请求停止本次音轨编辑；当前展示的已保存音频保留。'
    return '当前没有正在保存的音轨'


def restore(job):
    snapshot(job)
    try:
        with _lock(job):
            if db.read(_state_path(job),{}).get('status') in ACTIVE:raise ValueError('请先等待或停止当前音轨编辑')
            base=db.read(db.job_dir(job)/'audio-edit-base.json',{})
            if not base:raise ValueError('尚未保存音轨编辑，无需恢复')
            with db.lock():
                path=db.job_dir(job)/'job.json';record=db.read(path)
                if record['status']!='completed':raise ValueError('请等待原任务完成')
                record.update(outputs=base['original_outputs'],updated_at=db.now())
                record.pop('audio_edit',None);db.write(path,record)
            if record['kind']=='convert':db.write(db.job_dir(job)/'outputs.json',record['outputs'])
            state=dict(id=job,status='completed',progress=100,message='已恢复最初音频',updated_at=db.now())
            db.write(_state_path(job),state)
    except Timeout as error:raise ValueError('该任务正在保存音轨，请稍后再试') from error


class Editor:
    def __init__(self,job,token):
        self.path=db.job_dir(job);self.job=job;self.token=token
        self.destination=self.path/'.audio-edits'/token
        self.committed=False
    def check(self):
        if (self.path/('audio-edit-cancel-'+self.token)).exists():raise InterruptedError('本次音轨编辑已停止；上次保存结果保留')
        require_space(self.path,purpose='音轨编辑')
    def update(self,message,progress=None):
        self.check()
        state=db.read(_state_path(self.job),{})
        if state.get('revision')!=self.token:raise InterruptedError('编辑版本已更新')
        state.update(status='running',message=message,updated_at=db.now())
        if progress is not None:state['progress']=round(float(progress),1)
        db.write(_state_path(self.job),state)
        print(message,flush=True)
    def copy(self,source,target):
        with Path(source).open('rb') as src,Path(target).open('wb') as dst:
            while True:
                self.check();block=src.read(1024*1024)
                if not block:break
                dst.write(block)
    def baseline(self,record):
        state_path=self.path/'audio-edit-base.json'
        base=db.read(state_path,{})
        if base:return base
        rows=layouts(record)
        backup=self.path/'.audio-edits'/('base-'+self.token);backup.mkdir(parents=True,exist_ok=True)
        copies={}
        for row in rows:
            for role in [*GAIN_KEYS]:
                part=row.get(role)
                if not part:continue
                for key in ['source','room_source']:
                    source=part.get(key)
                    if not source:continue
                    if source not in copies:
                        target=backup/(str(len(copies))+'.wav');self.copy(source,target);copies[source]=str(target)
                    part[key]=copies[source]
        base=dict(version=1,original_outputs=deepcopy(record.get('outputs',{})),layouts=rows,created_at=db.now())
        db.write(state_path,base)
        return base
    def gain(self,source,target,db_gain):
        gain=10**(float(db_gain)/20)
        peak=0.
        with sf.SoundFile(source) as src,sf.SoundFile(target,'w',samplerate=src.samplerate,channels=src.channels,subtype='FLOAT') as dst:
            while True:
                self.check();x=src.read(65536,dtype='float32',always_2d=True)
                if not len(x):break
                x*=gain
                if not np.isfinite(x).all():raise ValueError('音轨中出现非有限值，未保存修改')
                peak=max(peak,float(np.max(np.abs(x))));dst.write(x)
        return peak
    def execute(self):
        from .reverb import render
        record=db.read(self.path/'job.json')
        base=self.baseline(record)
        values=db.read(_state_path(self.job))['settings']
        self.destination.mkdir(parents=True,exist_ok=True)
        outputs=deepcopy(base['original_outputs'])
        exports={};peaks={};cached={};room_stats={}
        def adjust(part,role):
            if not part:return None
            cache_key=(part['source'],role)
            if cache_key not in cached:
                target=self.destination/(str(len(cached))+'-'+role+'.float.wav')
                cached[cache_key]=(target,self.gain(part['source'],target,values[role]))
            target,peak=cached[cache_key]
            exports[part['key']]=target;peaks[part['key']]=peak
            return (target,part.get('weight',1.))
        def combine(parts,key):
            target=self.destination/(key+'.float.wav')
            peaks[key]=sum_tracks([p for p in parts if p],target,self.check);exports[key]=target
            return target
        for i,row in enumerate(base['layouts']):
            self.update('调整 '+row['name']+' 的分轨与混音',10+65*i/len(base['layouts']))
            dry=[adjust(row.get(role),role) for role in ['dry_lead','dry_backing']]
            accompaniment=adjust(row.get('accompaniment'),'accompaniment')
            combined=combine(dry,row['dry_combined'])
            if accompaniment:
                combine([(combined,row['vocal_gain']),(accompaniment[0],row['instrumental_gain'])],row['dry_mix'])
            wet=[]
            for name in ['lead','backing']:
                original=row.get('dry_'+name)
                part=row.get('wet_'+name)
                if values['room'] and original and original.get('generated',True):
                    # Generate from immutable dry stems; a wet track is never
                    # passed through reverb again when the user saves twice.
                    params={**row.get('params',{}).get(name,{}),
                            'decay':values['decay'],'predelay_ms':values['predelay_ms'],'wet':values['wet']}
                    target=self.destination/(row['name']+'-room-'+name+'.float.wav')
                    self.update('重建 '+row['name']+' '+name+' 的混响')
                    render(self,original.get('room_source',original['source']),target,params,values[name+'_strength'])
                    part=dict(source=str(target),key=row['wet_prefix']+name,weight=original.get('weight',1.))
                    room_stats[row['name']+'_'+name]=dict(decay=values['decay'],predelay_ms=values['predelay_ms'],
                        wet=values['wet'],strength=values[name+'_strength'])
                elif values['room'] and original and name=='backing' and not original.get('generated',True):
                    part={**original,'key':row['wet_prefix']+'kept_backing'}
                wet.append(adjust(part,'wet_'+name))
            if wet[0]:
                combined=combine(wet,row['wet_prefix']+'vocals')
                if accompaniment:
                    combine([(combined,row['vocal_gain']),(accompaniment[0],row['instrumental_gain'])],row['wet_prefix']+'mix')
        self.update('按设定增益写入浮点音轨，不自动衰减',85)
        paths={}
        for key,source in exports.items():
            self.check();target=self.destination/(key+'.wav')
            write_audio(source,target,self.check);paths[key]=str(target)
        outputs.update(paths)
        # Compatibility fields describe the new exports, rather than carrying
        # an old normalization factor into a later pitch/reverb operation.
        if 'mix' in paths:outputs['mix_normalization_gain']=1.
        metadata=dict(revision=self.token,settings=values,headroom_db=0.,gain_policy='preserve_float',
                      peak_dbfs={key:round(20*math.log10(peak),3) if peak>0 else None for key,peak in peaks.items()},
                      overloaded=[key for key,peak in peaks.items() if peak>1.],
                      reverb=room_stats,updated_at=db.now())
        self.check()
        with db.lock():
            latest=db.read(self.path/'job.json')
            if latest['status']!='completed':raise ValueError('原任务状态已变化，未保存修改')
            latest.update(outputs=outputs,audio_edit=metadata,updated_at=db.now())
            db.write(self.path/'job.json',latest)
            self.committed=True
        if record['kind']=='convert':db.write(self.path/'outputs.json',outputs)
        for path in self.destination.glob('*.float.wav'):path.unlink(missing_ok=True)
        return metadata


def report_html(record):
    state=snapshot(record.get('id'))
    if not state:return ''
    status={'queued':'准备保存','running':'正在保存','completed':'已保存','failed':'保存失败','cancelled':'已停止'}.get(state.get('status'),'')
    result=['<section class="svc-pitch-report"><h3>生成后音轨编辑 · '+status+'</h3><p>'+escape(state.get('message',''))+'</p>']
    current=record.get('audio_edit',{})
    if current:
        result.append('<p>各增益相对最初音轨计算；主唱／和声混音比例沿用原转换设置。</p><ul>')
        result.extend('<li>'+label+f'：{current["settings"].get(key,0):+g} dB</li>' for key,label in zip(GAIN_KEYS,GAIN_LABELS))
        if current.get('gain_policy')=='preserve_float':
            result.append('</ul><p>按设定增益保存为浮点 WAV；没有自动衰减或限幅，其他分轨不受连带影响。</p>')
            if current.get('overloaded'):
                result.append('<p>以下输出超过 0 dBFS，播放可能失真，请按需手动降低增益：</p><ul>')
                for key in current['overloaded']:
                    result.append('<li>'+escape(track_label(key))+f'：{current["peak_dbfs"][key]:+.2f} dBFS</li>')
                result.append('</ul>')
        else:
            result.append('</ul><p>此版本由旧规则保存，统一衰减：'+str(current.get('headroom_db',0))+' dB。重新保存将按当前增益原样导出，不再自动衰减。</p>')
        for key,row in current.get('reverb',{}).items():
            result.append('<p>'+escape(key)+f'：衰减 {row["decay"]:g} 秒 · 预延迟 {row["predelay_ms"]:g} ms · 混响幅度比 {row["wet"]:g} · 强度 {row["strength"]:g}</p>')
    result.append('</section>')
    return ''.join(result)


def run(job,token):
    editor=Editor(job,token)
    # The submitter holds this lock until its process metadata is persisted.
    lock=FileLock(str(editor.path/'.audio-edit.lock'),timeout=40)
    with lock:
        state=db.read(_state_path(job))
        if state.get('revision')!=token:return
        try:
            editor.update('备份最初音轨并准备编辑',1)
            metadata=editor.execute()
            state.update(status='completed',progress=100,message='增益已保存，分轨、人声合并与完整混音已刷新',metadata=metadata)
        except Exception as error:
            traceback.print_exc()
            if editor.committed:
                state.update(status='completed',progress=100,message='音轨已保存；辅助文件整理未完成：'+str(error))
            else:
                state.update(status='cancelled' if isinstance(error,InterruptedError) else 'failed',message=str(error))
                shutil.rmtree(editor.destination,ignore_errors=True)
        state['updated_at']=db.now();db.write(_state_path(job),state)
        (editor.path/('audio-edit-cancel-'+token)).unlink(missing_ok=True)


if __name__=='__main__':run(db.identifier(sys.argv[1]),db.identifier(sys.argv[2]))
