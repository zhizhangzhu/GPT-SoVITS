"""Prepare independent reference/target uploads for the common pitch worker."""
from pathlib import Path
from types import SimpleNamespace
import shutil

import numpy as np
import soundfile as sf

from . import storage as db
from .audio import decode
from .pitch import channels,mono

MODES = [("提取主唱校准，保留原和声", "keep"),
         ("主唱与和声分别校准", "convert"),
         ("单声部人声直接校准", "off")]


def prepare(worker):
    prepared = {}
    diagnostics = {}
    for side, mixed in [('reference',worker.options['reference_mixed']),('target',worker.options['target_mixed'])]:
        worker.check()
        work = worker.path/'prepared'/side
        work.mkdir(parents=True,exist_ok=True)
        decoded,vocals = work/'decoded.wav',work/'vocals.wav'
        source = Path(worker.job[side+'_input']).resolve()
        if source.parent != worker.path.resolve() or not source.is_file():
            raise ValueError('上传源文件不存在，请重新上传')
        worker.update('准备'+('参考音频' if side=='reference' else '待校准音频'),3)
        decode(source,decoded,check_cancel=worker.check)
        info = sf.info(decoded)
        if not 0 < info.duration <= 1200:
            raise ValueError('每份校准音频须非空且不超过 20 分钟')
        scoped = SimpleNamespace(path=work,options=worker.options,check=worker.check,
            check_cancel=worker.check_cancel,
            hash_file=lambda path,*args:db.digest(path,check_cancel=worker.check),
            update=lambda message,*args,**extra:worker.update(message,**extra))
        if mixed:
            worker.separate(decoded,vocals,work/'accompaniment.wav',work/'separation')
        else:
            shutil.copy2(decoded,vocals)
        back = None
        if worker.options['vocal_mode']!='off':
            from .vocal_split import prepare as split
            vocals,back = split(scoped,vocals)
        from .dereverb import prepare as dry
        if worker.options.get('dereverb_'+side):
            vocals,_ = dry(scoped,vocals,'lead' if back is not None else 'vocals')
            if back is not None and worker.options['vocal_mode']=='convert':
                back,_ = dry(scoped,back,'backing')
        prepared[side] = dict(vocals=vocals,backing=back)
        if mixed:prepared[side]['accompaniment']=work/'accompaniment.wav'
        for name,audio in [('vocals',vocals),('backing',back)]:
            if audio is not None:diagnostics[side+'_'+name]=channels(audio,worker.check)
    # The correction engines receive mono target stems, with a consistent
    # channel selected for the whole stem. Raw stereo inputs remain untouched.
    for name in ['vocals','backing']:
        audio = prepared['target'].get(name)
        if audio is None or (name=='backing' and worker.options['vocal_mode']=='keep'):continue
        selection=diagnostics['target_'+name]['selection']
        out = worker.path/('target_'+name+'_mono.wav')
        with sf.SoundFile(audio) as src,sf.SoundFile(out,'w',samplerate=44100,channels=1,subtype='FLOAT') as dst:
            while True:
                worker.check()
                x = src.read(65536,dtype='float32',always_2d=True)
                if not len(x):break
                dst.write(mono(x,selection))
        prepared['target'][name]=out
    mode=worker.options['vocal_mode']
    split=mode!='off'
    track='lead' if split else 'vocals'
    tracks=[(track,prepared['reference']['vocals'],prepared['target']['vocals'],float(worker.options['reference_transpose']))]
    if mode=='convert':
        tracks.append(('backing',prepared['reference']['backing'],prepared['target']['backing'],float(worker.options['reference_transpose'])))
    outputs={'reference_full':str(worker.path/'prepared/reference/decoded.wav'),
             'target_full':str(worker.path/'prepared/target/decoded.wav')}
    if prepared['target'].get('backing') is not None:outputs['backing_vocals']=str(prepared['target']['backing'])
    if prepared['target'].get('accompaniment'):outputs['accompaniment']=str(prepared['target']['accompaniment'])
    config={'vocal_mode':mode,'harmony_gain':1.,'mixed':worker.options['target_mixed'],
            'vocal_gain':1.,'instrumental_gain':1.}
    db.write(worker.path/'upload-preparation.json',{'channel_analysis':diagnostics,
        'reference_mixed':worker.options['reference_mixed'],'target_mixed':worker.options['target_mixed']})
    return config,outputs,tracks,split
