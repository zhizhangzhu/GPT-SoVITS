"""Reference-style reverb: measure ambience, synthesize a fresh stereo response.

Reference waveforms and dereverb residuals are NEVER mixed into the output.
The fitted parameters are an approximation, not an identified studio impulse
response. Processing is bounded in memory and keeps the target sample grid.
"""
from html import escape
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

from . import storage as db
from .audio import sum_tracks, write_audio, mix_tracks

VERSIONS = [('自动选择可用版本', 'auto'), ('SVC 转换结果', 'svc'),
            ('PSOLA 校准结果', 'psola'), ('PC-NSF 校准结果', 'pc_nsf')]
STYLES = [('参考原曲估计混响', 'reference'), ('手动设置混响', 'manual')]
TRACKS = [
    ('reverb_lead', '加混响后主唱'), ('reverb_backing', '加混响后和声'),
    ('reverb_vocals', '加混响后合并人声'), ('reverb_mix', '加混响后完整混音'),
    ('reverb_reference_mix', '混响参考完整混音'),
    ('reverb_before_lead', '加混响前主唱 · 输入快照'), ('reverb_before_backing', '加混响前和声 · 输入快照'),
    ('reverb_before_vocals', '加混响前人声 · 输入快照'), ('reverb_before_mix', '加混响前完整混音 · 输入快照'),
    ('reverb_reference_lead', '混响参考主唱／人声'),
    ('reverb_reference_backing', '混响参考和声'),
    ('reverb_kept_backing', '保留的原和声（不重复加混响）'),
]


def settings(value=None):
    """Validate the persisted optional conversion post-processing config."""
    if value is not None and not isinstance(value,dict):
        raise ValueError('输出混响配置须为对象')
    raw = value or {}
    result = {'enabled':bool(raw.get('enabled',False)), 'style':raw.get('style','reference'),
              'clean_output':bool(raw.get('clean_output',False))}
    if result['style'] not in {key for _,key in STYLES}:
        raise ValueError('请选择有效的输出混响方式')
    for key,default,low,high in [('lead_strength',1.,0,2),('backing_strength',1.,0,2),
                               ('decay',1.5,.3,4),('predelay_ms',25.,0,120),('wet',.22,0,.65)]:
        number = float(raw.get(key,default))
        if not np.isfinite(number) or not low<=number<=high:
            raise ValueError('输出混响参数超出有效范围：'+key)
        result[key] = number
    return result


def _inside(value, directory):
    if not value:
        return None
    path = Path(value).resolve()
    return path if path.is_relative_to(directory.resolve()) and path.is_file() else None


def sources(record, version='auto', inline=False):
    """Resolve only files owned by the selected task or its source conversion."""
    from .service import conversion_outputs
    directory = db.job_dir(record['id'])
    outputs = conversion_outputs(record)
    kind = record['kind']
    if kind not in {'convert', 'pitch'}:
        raise ValueError('请选择已完成的音色转换或音准校准任务；调整混响请重新选择其原任务')
    if record['status'] != 'completed' and not (inline and kind=='convert' and record['status']=='running'):
        raise ValueError('请等待转换或校准任务完成后再添加混响')
    config = record['options']
    reference_outputs = outputs
    reference_dir = directory
    reference_record = record
    dependencies = [record['id']]
    warnings = []
    uploaded = kind == 'pitch' and config.get('input_mode') == 'upload'
    if kind == 'convert':
        if version not in {'auto', 'svc'}:
            raise ValueError('所选转换任务没有校准版本，请选择 SVC 或自动选择')
        selected = 'svc'
        main = 'converted_lead' if outputs.get('converted_lead') else 'vocals'
        back = 'converted_backing'
        combined, mix = 'vocals', 'mix'
    else:
        selected = next((v for v in ['psola', 'pc_nsf'] if outputs.get(v+'_combined')), None) if version == 'auto' else version
        if selected not in {'psola', 'pc_nsf'} or not outputs.get(selected+'_combined'):
            raise ValueError('所选任务没有这个校准版本；仅分析报告的任务须先生成校准音轨')
        main = selected + ('_lead' if outputs.get(selected+'_lead') else '_vocals')
        back, combined, mix = selected+'_backing', selected+'_combined', selected+'_mix'
        if not uploaded:
            source = config.get('source_job')
            original = db.read(db.job_dir(source)/'job.json', {}) if source else {}
            if original:
                dependencies.append(source)
                config = original['options']
                reference_outputs = conversion_outputs(original)
                reference_dir = db.job_dir(source)
                reference_record = original
            else:
                reference_outputs = {}
                warnings.append('原转换任务已删除，无法取得原曲湿声；请使用手动混响。')
    split = main.endswith('lead')
    lead = _inside(outputs.get(main), directory)
    if not lead:
        raise ValueError('待添加混响的人声音轨已丢失')
    rows = {'lead': {'input': lead, 'gain': 1.}}
    converted_back = _inside(outputs.get(back), directory)
    harmony_gain = float(config.get('harmony_gain', 1))
    if converted_back:
        rows['backing'] = {'input': converted_back, 'gain': harmony_gain}
    kept = None
    if config.get('vocal_mode') == 'keep' and split:
        if kind == 'convert':
            kept = _inside(outputs.get('dry_backing') or outputs.get('backing_vocals'), directory)
        else:
            kept = _inside(directory/'kept_backing.wav', directory)
        if kept is None:
            raise ValueError('任务缺少保留的原和声，已停止以避免结果丢失和声')
    for name, row in rows.items():
        track = 'backing' if name == 'backing' else 'lead' if split else 'vocals'
        if uploaded:
            work = directory/'prepared/reference'
            wet = work/('backing_vocals.wav' if name == 'backing' else 'lead_vocals.wav' if split else 'vocals.wav')
            row['wet'] = _inside(wet, directory)
            row['dry'] = _inside(work/('dry_'+track+'.wav'), directory)
        else:
            key = 'backing_vocals' if name == 'backing' else 'lead_vocals' if split else 'source_vocals'
            row['wet'] = _inside(reference_outputs.get(key), reference_dir)
            row['dry'] = _inside(reference_outputs.get('dry_'+track), reference_dir)
    # The reference is the source song / uploaded reference, rather than the
    # converted or corrected "before_mix". Paired uploads need recombination.
    if uploaded:
        reference_accompaniment = _inside(directory/'prepared/reference/accompaniment.wav',directory)
        reference_full = _inside(outputs.get('reference_full'),directory) if reference_accompaniment else None
        reference_vocals = _inside(directory/'prepared/reference/vocals.wav',directory)
    else:
        reference_accompaniment = _inside(reference_outputs.get('accompaniment'),reference_dir)
        reference_full = (_inside(reference_dir/'decoded.wav',reference_dir)
                          if reference_accompaniment and not reference_record.get('accompaniment_input') else None)
        reference_vocals = _inside(reference_outputs.get('source_vocals'),reference_dir)
    return {'tracks': rows, 'kept': kept,
            'kept_gain': harmony_gain*float(reference_outputs.get('vocal_normalization_gain', 1)) if not uploaded else harmony_gain,
            'accompaniment': (_inside(outputs.get('accompaniment'), directory)
                              or (_inside(reference_outputs.get('accompaniment'),reference_dir) if not uploaded else None)),
            'before_vocals': _inside(outputs.get(combined), directory),
            'before_mix': _inside(outputs.get(mix), directory), 'config': config,
            'reference_mix':reference_full,
            'reference_mix_parts':(reference_vocals,reference_accompaniment) if reference_vocals and reference_accompaniment else None,
            'dependencies': dependencies, 'version': selected, 'warnings': warnings}


def add_reference_mix(worker, resolved, outputs, inline=False):
    """Publish the original full mix, owning copies in standalone jobs."""
    source = resolved.get('reference_mix')
    parts = resolved.get('reference_mix_parts')
    target = worker.path/'reverb_reference_mix.wav'
    if source:
        if inline:
            target = source
        else:
            from .pitch_job import copy_audio
            owner = next((db.job_dir(j) for j in resolved['dependencies']
                          if Path(source).resolve().is_relative_to(db.job_dir(j).resolve())),None)
            if owner is None:
                raise ValueError('参考完整混音来源无效')
            target = copy_audio(worker,source,target.name,owner)
    elif parts:
        # Reconstruct the source pair at unity gain; output vocal gains and
        # pitch corrections must never be applied to this reference track.
        mix_tracks(parts[0],parts[1],target,1.,1.,check_cancel=worker.check)
    else:
        return False
    outputs['reverb_reference_mix'] = str(target)
    return not bool(source)


def _scope(worker, directory):
    directory.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(path=directory, options=worker.options, check=worker.check,
        check_cancel=worker.check_cancel, hash_file=lambda path,*args: db.digest(path, check_cancel=worker.check),
        update=lambda message,*args,**extra: worker.update(message))


def _stereo(worker, source, target):
    """The dereverb network requires stereo, including for generated mono."""
    if sf.info(source).channels == 2:
        return source
    with sf.SoundFile(source) as src,sf.SoundFile(target,'w',samplerate=src.samplerate,channels=2,subtype='FLOAT') as dst:
        if src.channels != 1:
            raise ValueError('去混响输入须为单声道或立体声')
        while True:
            worker.check()
            x = src.read(65536,dtype='float32',always_2d=True)
            if not len(x): break
            dst.write(np.repeat(x,2,axis=1))
    return target


def estimate(worker, wet, dry, fallback):
    """Fit decay from quiet tails, level from LS residual, and broad damping.

    Dereverb dry is an estimate, so residual energy includes separator error.
    Require multiple usable tails to describe decay as reasonably supported.
    Predelay cannot be identified reliably here and stays user-controlled.
    """
    info = sf.info(wet)
    other = sf.info(dry)
    if (info.samplerate, info.frames, info.channels) != (other.samplerate, other.frames, other.channels):
        raise ValueError('参考湿声与去混响音轨没有对齐')
    rate = info.samplerate
    width = max(1, round(rate*.02))
    totals = np.zeros(3, dtype=np.float64)
    with sf.SoundFile(wet) as w, sf.SoundFile(dry) as d:
        while True:
            worker.check()
            a = w.read(65536, dtype='float32', always_2d=True)
            b = d.read(len(a), dtype='float32', always_2d=True)
            if not len(a): break
            totals += [np.sum(a.astype('float64')*b), np.sum(b.astype('float64')**2), np.sum(a.astype('float64')**2)]
    if totals[1] < 1e-8 or totals[2] < 1e-8:
        return {**fallback, 'wet':0., 'confidence':'不可估计', 'tail_count':0,
                'warnings':['参考人声几乎无声，该声部跳过混响估计与添加。']}
    alpha = float(np.clip(totals[0]/totals[1], .1, 4))
    coherence = float(np.clip(totals[0]/np.sqrt(totals[1]*totals[2]), -1, 1))
    dry_energy, residual_energy = [], []
    bands = [signal.butter(2, [200, 1500], btype='bandpass', fs=rate, output='sos'),
             signal.butter(2, [3000, 9000], btype='bandpass', fs=rate, output='sos')]
    band_energy = np.zeros((2, 2), dtype=np.float64)
    state = [[np.zeros((len(sos),2,info.channels)) for _ in range(2)] for sos in bands]
    residual_channels = np.zeros(3)
    with sf.SoundFile(wet) as w, sf.SoundFile(dry) as d:
        while True:
            worker.check()
            a = w.read(width*512, dtype='float32', always_2d=True)
            b = d.read(len(a), dtype='float32', always_2d=True)*alpha
            if not len(a): break
            residual = a-b
            count = len(a)//width
            dry_energy.extend(np.mean(b[:count*width].reshape(count,width,info.channels).astype('float64')**2, axis=(1,2)))
            residual_energy.extend(np.mean(residual[:count*width].reshape(count,width,info.channels).astype('float64')**2, axis=(1,2)))
            for j,sos in enumerate(bands):
                for k,x in enumerate([b,residual]):
                    filtered,state[j][k] = signal.sosfilt(sos,x,axis=0,zi=state[j][k])
                    band_energy[j,k] += np.sum(filtered**2)
            if info.channels == 2:
                residual_channels += [np.sum(residual[:,0].astype('float64')**2),np.sum(residual[:,1].astype('float64')**2),np.sum(residual[:,0].astype('float64')*residual[:,1])]
    direct = np.asarray(dry_energy)
    tail = np.asarray(residual_energy)
    if len(direct)<20:
        return {**fallback, 'confidence':'低', 'tail_count':0,
                'warnings':['参考人声过短，采用手动参数作为回退。']}
    amount = float(np.sqrt(np.sum(tail)/max(np.sum(direct),1e-12)))
    active = direct > max(1e-8, float(np.percentile(direct,85))*.0064)
    decays = []
    ends = np.flatnonzero(active[:-1] & ~active[1:])+1
    for end in ends:
        worker.check_cancel()
        next_voice = np.flatnonzero(active[end:end+150])
        length = int(next_voice[0]) if len(next_voice) else min(150,len(active)-end)
        if length < 18: continue
        values = signal.medfilt(tail[end:end+length],5)
        db_values = 10*np.log10(np.maximum(values,1e-14))
        floor = max(-85., float(np.percentile(10*np.log10(np.maximum(tail,1e-14)),10))+6)
        usable = (db_values > floor) & (db_values > db_values[0]-30)
        indices = np.flatnonzero(usable)
        if len(indices) < 10 or indices[-1]-indices[0] < 9: continue
        x = indices*.02
        y = db_values[indices]
        slope, _ = np.polyfit(x,y,1)
        corr = float(np.corrcoef(x,y)[0,1])
        if -200 < slope < -8 and corr < -.7 and np.ptp(y) >= 8:
            decays.append(float(np.clip(-60/slope,.3,4)))
    damping = (band_energy[1,1]/max(band_energy[1,0],1e-12)) / max(band_energy[0,1]/max(band_energy[0,0],1e-12),1e-12)
    stereo_corr = float(residual_channels[2]/np.sqrt(max(residual_channels[0]*residual_channels[1],1e-12))) if info.channels == 2 else .5
    params = dict(fallback)
    params.update(decay=round(float(np.median(decays)),3) if decays else fallback['decay'],
                  wet=round(float(np.clip(amount,0,.65)),4),
                  cutoff=round(float(np.clip(5000*np.sqrt(max(damping,.01)),2000,12000))),
                  width=round(float(np.clip(1-stereo_corr,.2,1)),3),
                  tail_count=len(decays), coherence=round(coherence,3),
                  confidence='中' if len(decays)>=3 and coherence>.8 else '低',
                  warnings=[])
    if coherence < .65:
        params.update(wet=fallback['wet'],decay=fallback['decay'],cutoff=6000,width=.8)
        params['warnings'].append('参考干湿声差异较大，估计不可靠；采用手动参数作为回退。')
    elif not decays:
        params['warnings'].append('未找到足够清晰的衰减尾音，衰减时间采用手动参数作为回退。')
    elif len(decays)<3:
        params['warnings'].append('可用尾音较少，估计的衰减时间需要试听确认。')
    params['warnings'].append('预延迟沿用手动设置；分离残留可能影响混响量与空间宽度估计。')
    return params


def impulse(params, rate):
    """Fresh, deterministic diffuse response, with sparse early reflections."""
    decay = float(params['decay'])
    delay = round(float(params['predelay_ms'])*rate/1000)
    n = round(rate*decay*1.15)
    rng = np.random.default_rng(20261020)
    t = np.arange(n)/rate
    common = rng.standard_normal(n)
    width = float(params.get('width',.8))
    noise = np.sqrt(1-width)*common[:,None]+np.sqrt(width)*rng.standard_normal((n,2))
    noise = signal.sosfilt(signal.butter(2,float(params.get('cutoff',6000)),fs=rate,output='sos'),noise,axis=0)
    noise = signal.sosfilt(signal.butter(2,120,btype='highpass',fs=rate,output='sos'),noise,axis=0)
    # Rising density keeps the response from behaving like filtered white noise
    # at its onset. A -60 dB amplitude envelope defines the reported RT60.
    response = noise*np.exp(-np.log(1000)*t/decay)[:,None]*(1-np.exp(-t/.018))[:,None]
    for seconds,gain in [(.007,.15),(.019,.11),(.037,.08),(.061,.05)]:
        i = min(n-1,round(seconds*rate))
        response[i] += gain*np.sqrt(rate*.008)*np.array([1.,.85 if i%2 else 1.1])
    response /= np.maximum(np.sqrt(np.sum(response**2,axis=0)),1e-8)
    return np.pad(response,((delay,0),(0,0))).astype('float32')


def render(worker, source, target, params, strength):
    """Convolve in blocks, RMS-calibrate the send, retain stereo and dry onset."""
    target = Path(target)
    info = sf.info(source)
    if info.samplerate != 44100 or info.channels not in (1,2) or not info.frames:
        raise ValueError('混响输入须为非空的 44.1 kHz 单／双声道人声')
    ir = impulse(params,info.samplerate)
    with tempfile.TemporaryDirectory(prefix='.reverb-',dir=target.parent) as temp:
        if strength<=0 or params['wet']<=0:
            temporary = Path(temp)/'unchanged.wav'
            peak = 0.
            with sf.SoundFile(source) as src,sf.SoundFile(temporary,'w',samplerate=info.samplerate,channels=2,subtype='FLOAT') as dst:
                while True:
                    worker.check()
                    x = src.read(65536,dtype='float32',always_2d=True)
                    if not len(x): break
                    if not np.all(np.isfinite(x)): raise ValueError('人声输入出现非有限值')
                    peak = max(peak,float(np.max(np.abs(x))))
                    dst.write(np.broadcast_to(x,(len(x),2)))
            temporary.replace(target)
            return peak
        send = Path(temp)/'send.wav'
        carry = np.zeros((len(ir)-1,2),dtype='float32')
        dry_energy = wet_energy = 0.
        with sf.SoundFile(source) as src,sf.SoundFile(send,'w',samplerate=info.samplerate,channels=2,subtype='FLOAT') as dst:
            while True:
                worker.check()
                x = src.read(65536,dtype='float32',always_2d=True)
                if not len(x): break
                stereo = np.broadcast_to(x,(len(x),2))
                y = np.column_stack([signal.fftconvolve(stereo[:,c],ir[:,c]) for c in range(2)]).astype('float32')
                y[:len(carry)] += carry
                ready = y[:len(x)]
                carry = y[len(x):].copy()
                dry_energy += float(np.sum(stereo.astype('float64')**2))
                wet_energy += float(np.sum(ready.astype('float64')**2))
                dst.write(ready)
        send_gain = float(strength)*float(params['wet'])*np.sqrt(dry_energy/max(wet_energy,1e-12))
        temporary = Path(temp)/'output.wav'
        peak = 0.
        with sf.SoundFile(source) as src,sf.SoundFile(send) as wet,sf.SoundFile(temporary,'w',samplerate=info.samplerate,channels=2,subtype='FLOAT') as dst:
            while True:
                worker.check()
                x = src.read(65536,dtype='float32',always_2d=True)
                if not len(x): break
                y = x+wet.read(len(x),dtype='float32',always_2d=True)*send_gain
                if not np.all(np.isfinite(y)): raise ValueError('混响输出出现非有限值，已停止写入')
                peak = max(peak,float(np.max(np.abs(y))))
                dst.write(y)
        temporary.replace(target)
    return peak


def execute(worker, inline=False):
    from .pitch_job import copy_audio
    from .dereverb import prepare as dereverb
    options = settings(worker.options.get('output_reverb')) if inline else worker.options
    original = db.read(worker.path/'job.json') if inline else db.read(db.job_dir(options['source_job'])/'job.json')
    resolved = sources(original,'svc' if inline else options['version'],inline=inline)
    if not inline:
        mix_config={key:resolved['config'][key] for key in ['vocal_mode','harmony_gain','vocal_gain','instrumental_gain'] if key in resolved['config']}
        harmony=float(mix_config.get('harmony_gain',1))
        mix_config['source_normalization_gain']=float(resolved['kept_gain'])/harmony if harmony else 1.
        db.write(worker.path/'mix-config.json',mix_config)
    outputs = dict(original.get('outputs',{})) if inline else {}
    def update(message,progress=None,**extra):
        worker.update(message,90+.1*progress if inline and progress is not None else progress,**extra)
    def own_copy(path,filename,owner):
        return Path(path) if inline else copy_audio(worker,path,filename,owner)
    def copy(path,filename):
        owner = next((db.job_dir(j) for j in resolved['dependencies'] if Path(path).resolve().is_relative_to(db.job_dir(j).resolve())), None)
        if owner is None: raise ValueError('混响音轨来源无效')
        return own_copy(path,filename,owner)
    report = {'version':1,'source_job':original['id'],'source_version':resolved['version'],
              'style':options['style'],'tracks':{},'warnings':resolved['warnings'],
              'timeline':'保持原样本数与干声起点，超出原曲结尾的混响尾音不导出',
              'method':'参考干湿差异估计参数 → 全新合成立体声脉冲响应 → 分块卷积；不混入参考人声或残留'}
    fallback = {'decay':options['decay'],'predelay_ms':options['predelay_ms'],
                'wet':options['wet'],'cutoff':6000,'width':.8,'confidence':'手动', 'warnings':[]}
    parts = []
    floating = []
    common_peak = 0.
    report_path = worker.path/'reverb-report.json'
    try:
        for index,(name,row) in enumerate(resolved['tracks'].items()):
            update('准备'+('主唱' if name=='lead' else '和声')+'混响',5+index*35)
            # Standalone jobs own copies; inline processing reuses its own stems.
            source = copy(row['input'],'reverb_before_'+name+'.wav')
            outputs['reverb_before_'+name] = str(source)
            params = dict(fallback)
            strength = float(options[name+'_strength'])
            if options['style']=='reference' and strength>0:
                if row.get('wet') is None: raise ValueError('原曲参考湿声已丢失，请改用手动混响')
                wet = copy(row['wet'],'reverb_reference_'+name+'.wav')
                outputs['reverb_reference_'+name] = str(wet)
                if row.get('dry'):
                    dry = copy(row['dry'],'reference_dry_'+name+'.wav')
                else:
                    dry,_ = dereverb(_scope(worker,worker.path/'analysis'/name),wet,'lead' if name=='lead' else 'backing')
                worker.update('估计原曲混响参数（近似）')
                params = estimate(worker,wet,dry,fallback)
            if options['clean_output'] and strength>0:
                source = _stereo(worker,source,worker.path/('clean_input_'+name+'.wav'))
                source,_ = dereverb(_scope(worker,worker.path/'clean-output'/name),source,'lead' if name=='lead' else 'backing')
            report['tracks'][name] = {**params,'strength':strength,'clean_output':options['clean_output'],
                                     'has_separate_lead':len(resolved['tracks'])>1 or original['options'].get('vocal_mode','off')!='off'}
            destination = worker.path/('wet_'+name+'.part.wav')
            update('为生成的人声合成新的立体声混响',25+index*35)
            peak = render(worker,source,destination,params,strength)
            common_peak = max(common_peak,peak)
            parts.append((destination,row['gain']))
            floating.append((name,destination))
            worker.update('该声部混响已合成，准备合并',outputs=dict(outputs))
        if resolved['kept']:
            kept = own_copy(resolved['kept'],'kept_backing.wav',db.job_dir(original['id']))
            parts.append((kept,resolved['kept_gain']))
            outputs['reverb_kept_backing'] = str(kept)
        update('合并带混响人声与伴奏',85)
        combined = worker.path/'wet_combined.part.wav'
        peak = sum_tracks(parts,combined,worker.check)
        common_peak = max(common_peak,peak)
        for name,path in floating:
            target = worker.path/('reverb_'+name+'.wav')
            write_audio(path,target,worker.check)
            outputs['reverb_'+name] = str(target)
        final = worker.path/'reverb_vocals.wav'
        gain = write_audio(combined,final,worker.check)
        outputs['reverb_vocals'] = str(final)
        report['vocal_normalization_gain'] = gain
        for source,key in [(resolved['before_vocals'],'reverb_before_vocals'),(resolved['before_mix'],'reverb_before_mix'),(resolved['accompaniment'],'accompaniment')]:
            if source:
                outputs[key] = str(copy(source,key+'.wav'))
        report['reference_mix_reconstructed'] = add_reference_mix(worker,resolved,outputs,inline=inline)
        if outputs.get('accompaniment'):
            mix = worker.path/'reverb_mix.wav'
            mix_tracks(final,outputs['accompaniment'],mix,float(resolved['config'].get('vocal_gain',1)),
                       float(resolved['config'].get('instrumental_gain',1)),check_cancel=worker.check)
            outputs['reverb_mix'] = str(mix)
        db.write(report_path,report)
        outputs['reverb_report'] = str(report_path)
        if inline:
            db.write(worker.path/'outputs.json',outputs)
        update('混响处理完成，原结果与新音轨均保留',100,outputs=outputs)
    finally:
        for _,path in floating: path.unlink(missing_ok=True)
        (worker.path/'wet_combined.part.wav').unlink(missing_ok=True)


def report_html(record, component=None):
    if record.get('audio_edit',{}).get('reverb'):
        return '<section class="svc-pitch-report"><h3>参考混响处理</h3><p>当前混响已在生成后编辑中重建，实际参数见下方音轨编辑报告。</p></section>'
    path = _inside(record.get('outputs',{}).get('reverb_report'),db.job_dir(record['id'])) if record.get('id') else None
    if not path: return ''
    data = db.read(path,{})
    result = ['<section class="svc-pitch-report"><h3>参考混响处理</h3><p>估计空间感并合成新混响，不能精确复刻原曲的全部效果。参考音轨与残留不会作为混响效果混回；保留原和声模式仍沿用原和声。</p>']
    for name,row in data.get('tracks',{}).items():
        label = '主唱／人声' if name=='lead' else '和声'
        result.append(f'<h4>{label}</h4><p>衰减 {row["decay"]:.2f} 秒 · 预延迟 {row["predelay_ms"]:g} ms · 混响／干声能量幅度比 {row["wet"]*row["strength"]:.2f} · 估计可信度：{escape(row["confidence"])}</p>')
        if 'tail_count' in row: result.append(f'<p>可用衰减尾音 {row["tail_count"]} 处。混响量、衰减与宽度仅为近似，需试听确认。</p>')
        result.extend('<p>'+escape(w)+'</p>' for w in row.get('warnings',[]))
    result.extend('<p>'+escape(w)+'</p>' for w in data.get('warnings',[]))
    if data.get('reference_mix_reconstructed'):
        result.append('<p>参考完整混音由原人声与配对伴奏按原时间轴合并，未使用转换或校准后的人声。</p>')
    result.append('<p>'+escape(data.get('timeline',''))+'</p>')
    if component is not None:
        from urllib.parse import quote
        component.temp_files.add(path); component.keep_in_cache.add(path)
        result.append('<p><a download href="./file='+escape(quote(str(path),safe='/'),quote=True)+'">下载混响参数 JSON</a></p>')
    result.append('</section>')
    return ''.join(result)
