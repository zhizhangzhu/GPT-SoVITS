"""Background analysis/correction of existing SVC stems; no repeated conversion."""
from html import escape
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf

from . import storage as db
from .audio import sum_tracks, write_audio, mix_tracks
from .pitch import analyze
from . import pitch_correction as correction

BACKENDS = [("仅生成音准偏差报告", "report"), ("PSOLA · 保守校准", "psola"),
            ("PC-NSF-HiFiGAN · 神经修音", "pc_nsf"), ("同时生成两版供对比", "compare")]


def copy_audio(worker, source, name, previous):
    source=Path(source).resolve()
    if not source.is_relative_to(previous.resolve()) or not source.is_file():
        raise ValueError('原任务音轨不存在或路径无效，请重新选择')
    target=worker.path/name
    if target.is_file() and db.digest(target,check_cancel=worker.check)==db.digest(source,check_cancel=worker.check):
        return target
    temporary=target.with_suffix('.part.wav')
    try:
        with source.open('rb') as src,temporary.open('wb') as dst:
            while True:
                worker.check();block=src.read(1024*1024)
                if not block:break
                dst.write(block)
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def execute(worker):
    from .service import conversion_outputs
    options=worker.options
    uploaded = options.get('input_mode') == 'upload'
    if uploaded:
        from .pitch_upload import prepare
        config,outputs,tracks,split = prepare(worker)
        source_id = None
        previous = worker.path
    else:
        source_id=db.identifier(options['source_job'])
        previous=db.job_dir(source_id)
        original=db.read(previous/'job.json')
        outputs=conversion_outputs(original)
        config=original['options']
        tracks=[]
        split=bool(outputs.get('converted_lead'))
        lead_ref=outputs.get('dry_lead') or outputs.get('lead_vocals') if split else outputs.get('dry_vocals') or outputs.get('source_vocals')
        lead_audio=outputs.get('converted_lead') if split else outputs.get('vocals')
        if not lead_ref or not lead_audio:
            raise ValueError('原任务缺少对应的转换前 / 后音轨，无法校准')
        tracks.append(('lead' if split else 'vocals',lead_ref,lead_audio,float(config.get('transpose',0))))
        if config.get('vocal_mode')=='convert' and outputs.get('converted_backing'):
            tracks.append(('backing',outputs.get('dry_backing') or outputs.get('backing_vocals'),
                           outputs['converted_backing'],float(config.get('transpose',0))+float(config.get('harmony_transpose',0))))
    db.write(worker.path/'mix-config.json',{**{key:config[key] for key in ['vocal_mode','harmony_gain','vocal_gain','instrumental_gain'] if key in config},
        'source_normalization_gain':float(outputs.get('vocal_normalization_gain',1))})
    manifest={'source_job':source_id,'input_mode':'upload' if uploaded else 'conversion','method':options['predictor'],
              'backend':options['backend'],'threshold_cents':options['threshold'],
              'strength':options['strength'],'tracks':{},'warnings':[]}
    published={key:value for key,value in outputs.items() if key in {'reference_full','target_full','accompaniment'}} if uploaded else {}
    if uploaded and config.get('vocal_mode')=='keep' and outputs.get('backing_vocals'):
        published['before_backing']=outputs['backing_vocals']
    report_path=worker.path/'pitch-report.json'
    curves={}; originals={}
    upload_channels = db.read(worker.path/'upload-preparation.json',{}).get('channel_analysis',{}) if uploaded else {}
    methods=['psola','pc_nsf'] if options['backend']=='compare' else [] if options['backend']=='report' else [options['backend']]
    for track_index,(track,reference,converted,shift) in enumerate(tracks):
        base = 5 + 85 * track_index / len(tracks)
        width = 85 / len(tracks)
        worker.check()
        ref=copy_audio(worker,reference,'reference_'+track+'.wav',previous)
        before=copy_audio(worker,converted,'before_'+track+'.wav',previous)
        if not uploaded and sf.info(ref).frames!=sf.info(before).frames:
            raise ValueError('前后音轨长度不同，请先核对对应任务')
        originals[track]=before
        published['reference_'+track]=str(ref)
        published['before_'+track]=str(before)
        worker.update('准备音准分析音轨',outputs=dict(published))
        worker.update('分析参考音准',base)
        r,rc=analyze(worker,ref,options['predictor'])
        worker.update('分析转换后音准',base+width*.1)
        c,cc=analyze(worker,before,options['predictor'])
        # Cross-check the reference AND the converted waveform to avoid turning
        # an estimator's octave error into an audible octave shift.
        other='fcpe' if options['predictor']=='rmvpe' else 'rmvpe'
        worker.update('交叉检查参考音准',base+width*.2)
        secondary,_=analyze(worker,ref,other)
        worker.update('交叉检查转换后音准',base+width*.3)
        second_output,_=analyze(worker,before,other)
        alignment = None
        if uploaded:
            from . import pitch_alignment
            if options.get('alignment','auto') == 'auto':
                mapping = pitch_alignment.build(worker,ref,before)
                alignment = mapping['info']
                r = pitch_alignment.warp(r,c['times'],mapping=mapping)
                secondary = pitch_alignment.warp(secondary,c['times'],mapping=mapping)
            else:
                offset = float(options.get('reference_offset',0))
                r = pitch_alignment.warp(r,c['times'],offset=offset)
                secondary = pitch_alignment.warp(secondary,c['times'],offset=offset)
                alignment = {'method':'timeline','reference_offset_seconds':offset}
        p=correction.plan(r,c,shift,options['threshold'],options['strength'],secondary)
        from .pitch import sample_curve
        secondary_out,secondary_uv=sample_curve(second_output['times'],second_output['f0'],p['times'])
        agreement=secondary_uv & (np.abs(1200*np.log2(np.maximum(secondary_out,1)/np.maximum(p['observed'],1)))<80)
        p['valid'] &= agreement
        p['mask'] &= agreement
        # Apply confidence masking to the desired curve as well.
        p['correction'][~agreement]=0
        p['target']=p['observed']*2**(p['correction']/1200)
        curves[track]=p
        target_channel=upload_channels.get('target_backing' if track=='backing' else 'target_vocals')
        row={'alignment':alignment,'target_source_channels':target_channel,'shift_semitones':shift,'reference_channels':rc,'converted_channels':cc,
             'before':correction.summary(p),'after':{},'curve':preview_curve(p)}
        if rc['suspected']:
            manifest['warnings'].append(track+'：发现立体声抵消风险，分析使用 '+rc['selection']+' 声道')
        if not np.any(r['voiced']) or not np.any(c['voiced']):
            manifest['warnings'].append(track+'：参考或转换轨缺少有效有声音高；几乎无声、气声或残留不能可靠校准，保持原样')
        elif row['before']['comparable_fraction']<.2:
            manifest['warnings'].append(track+'：可可靠比较的有声帧较少，请试听确认分离质量和多声部情况')
        row['secondary_disagreement_fraction']=round(float(np.mean(~agreement & c['voiced'])),4)
        manifest['tracks'][track]=row
        db.write(report_path,manifest)
        published['pitch_report']=str(report_path)
        worker.update('音准偏差报告已生成',outputs=dict(published))
        for method_index,method in enumerate(methods):
            worker.update(method+' 校准及复查',base+width*(.4+.5*method_index/max(1,len(methods))))
            target=worker.path/(method+'_'+track+'.wav')
            candidate = {key:value.copy() if isinstance(value,np.ndarray) else value for key,value in p.items()}
            correction.render(worker,before,target,candidate,method)
            after,_=analyze(worker,target,options['predictor'])
            rejected = correction.reject_worse_regions(candidate,after)
            if rejected:
                for a,b in rejected:
                    candidate['mask'][a:b] = False
                correction.render(worker,before,target,candidate,method)
                after,_=analyze(worker,target,options['predictor'])
                manifest['warnings'].append(track+'：'+method+' 跳过了 '+str(len(rejected))+' 处检测指标未改善或有声判定不可靠的区间')
            row.setdefault('rejected_regions',{})[method] = len(rejected)
            post=correction.plan(r,after,shift,options['threshold'],options['strength'])
            post['valid'] &= p['valid']
            row['after'][method]=correction.summary(post)
            indexes=np.linspace(0,len(after['times'])-1,min(1600,len(after['times']))).astype(int)
            row['curve']['after_'+method]=[[round(float(after['times'][i]),3),round(float(after['f0'][i]),2)] for i in indexes]
            published[method+'_'+track]=str(target)
            if row['after'][method]['median_abs_cents'] is not None and row['before']['median_abs_cents'] is not None and row['after'][method]['median_abs_cents']>row['before']['median_abs_cents']+10:
                manifest['warnings'].append(track+'：'+method+' 的音准指标变差，请对比试听，原音轨已保留')
            db.write(report_path,manifest)
            worker.update(method+' 校准音轨已生成',outputs=dict(published))
    worker.update('准备合并校准后音轨',92)
    for method in methods:
        main='lead' if split else 'vocals'
        parts=[(worker.path/(method+'_'+main+'.wav'),1.)]
        if config.get('vocal_mode')=='convert' and 'backing' in originals:
            parts.append((worker.path/(method+'_backing.wav'),float(config.get('harmony_gain',1))))
        elif config.get('vocal_mode')=='keep' and outputs.get('backing_vocals'):
            raw=outputs.get('dry_backing') or outputs['backing_vocals']
            kept=copy_audio(worker,raw,'kept_backing.wav',previous)
            parts.append((kept,float(config.get('harmony_gain',1))*float(outputs.get('vocal_normalization_gain',1))))
        floating=worker.path/(method+'_combined.part.wav')
        try:
            peak=sum_tracks(parts,floating,worker.check)
            target=worker.path/(method+'_vocals_final.wav')
            write_audio(floating,target,worker.check)
            published[method+'_combined']=str(target)
            if config.get('mixed') and outputs.get('accompaniment'):
                backing=copy_audio(worker,outputs['accompaniment'],'accompaniment.wav',previous)
                published['accompaniment']=str(backing)
                mix=worker.path/(method+'_mix.wav')
                mix_tracks(target,backing,mix,float(config.get('vocal_gain',1)),float(config.get('instrumental_gain',1)),check_cancel=worker.check)
                published[method+'_mix']=str(mix)
        finally:
            floating.unlink(missing_ok=True)
    db.write(report_path,manifest)
    worker.update('音准分析完成；校准版本与原版本均保留',100,outputs=published)


def preview_curve(p):
    indexes=np.linspace(0,len(p['times'])-1,min(1600,len(p['times']))).astype(int)
    return {key:[[round(float(p['times'][i]),3),round(float(p[key][i]),2)] for i in indexes]
            for key in ['reference','observed','target']}


def report_html(record, component=None):
    path=record.get('outputs',{}).get('pitch_report')
    if not path:
        return ''
    directory=db.job_dir(record['id']).resolve()
    path=Path(path).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        return ''
    data=db.read(path,{})
    download = ""
    if component is not None:
        from urllib.parse import quote
        component.temp_files.add(path)
        component.keep_in_cache.add(path)
        download = '<p><a href="./file=' + escape(quote(str(path), safe="/"), quote=True) + '" download>下载音准报告 JSON</a></p>' 
    result=['<section class="svc-pitch-report"><h3>音准偏差报告</h3><p>原轨为参考，并计入主动移调。音分 100 = 半音；检测结果需结合试听，多声部和声可能不可靠。</p>']
    for track,row in data.get('tracks',{}).items():
        name={'lead':'主唱','backing':'和声','vocals':'全部人声'}.get(track,track)
        before=row['before'];c=row['reference_channels']
        result.append(f'<h4>{escape(name)}</h4><p>可比较帧 {before["comparable_fraction"]:.1%} · 音高偏差中位数 {before["median_abs_cents"]} 音分 · P95 {before["p95_abs_cents"]} 音分</p>')
        result.append(f'<p>参考声道：{escape(c["selection"])} · 左右相关性 {c["correlation"]} · 转单声道能量变化 {c["mid_loss_db"]} dB · 疑似抵消区间占比 {c["cancelled_fraction"]:.1%}</p>')
        alignment = row.get('alignment')
        if alignment:
            if alignment['method'] == 'contentvec-dtw':
                result.append(f'<p>同曲自动对齐 · 可信内容匹配 {alignment["trusted_fraction"]:.1%} · 保持待校准音频时间轴</p>')
            else:
                result.append(f'<p>原时间轴对齐 · 参考偏移 {alignment["reference_offset_seconds"]:+g} 秒</p>')
        target_channel=row.get('target_source_channels')
        if target_channel:
            result.append(f'<p>待校准原声道：左右相关性 {target_channel["correlation"]} · 单声道能量变化 {target_channel["mid_loss_db"]} dB · 分析选择 {escape(target_channel["selection"])}</p>')
        result.append(curve_svg(row['curve'],'待校准' if data.get('input_mode')=='upload' else '转换后'))
        for method,summary in row.get('after',{}).items():
            result.append(f'<p>{escape(method)} 校准后：偏差中位数 {summary["median_abs_cents"]} 音分 · P95 {summary["p95_abs_cents"]} 音分</p>')
        ranges=before.get('intervals',[])[:20]
        if ranges:
            result.append('<details><summary>疑似跑调时间段（前 20 处）</summary><ul>')
            result.extend(f'<li>{v["start"]:.2f}–{v["end"]:.2f} 秒：{v["median_cents"]:+.0f} 音分</li>' for v in ranges)
            result.append('</ul></details>')
    result.extend('<p>'+escape(w)+'</p>' for w in data.get('warnings',[]))
    result.append(download + '</section>')
    return ''.join(result)


def curve_svg(curve, observed_label="转换后"):
    colors={'reference':('#74d69b','参考'),'observed':('#f0a86b',observed_label),'after_psola':('#8fb4ff','PSOLA'),'after_pc_nsf':('#da9bff','PC-NSF')}
    values=[(t,69+12*np.log2(hz/440)) for k in colors for t,hz in curve.get(k,[]) if hz>0]
    if not values:return '<p>没有足够的有声音高绘图。</p>'
    end=max(t for t,_ in values) or 1
    low=min(n for _,n in values)-1;high=max(n for _,n in values)+1
    svg=['<div style="overflow-x:auto"><svg viewBox="0 0 1000 220" style="min-width:700px;width:100%" role="img" aria-label="参考与转换音高曲线，横轴为秒，纵轴为音高">']
    for key,(color,label) in colors.items():
        if key not in curve:continue
        commands=[];pen=False
        for t,hz in curve[key]:
            if hz<=0:pen=False;continue
            x=40+940*t/end;y=190-160*((69+12*np.log2(hz/440))-low)/(high-low)
            commands.append(('L' if pen else 'M')+f'{x:.1f},{y:.1f}');pen=True
        svg.append(f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="1.3"/>')
    for i in range(6):
        svg.append(f'<text x="{40+940*i/5}" y="212" fill="currentColor" font-size="12">{end*i/5:.0f}s</text>')
    svg.append('</svg></div><p>'+ ' · '.join(f'<span style="color:{color}">{label}</span>' for key,(color,label) in colors.items() if key in curve)+'</p>')
    return ''.join(svg)
