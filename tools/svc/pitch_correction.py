"""Conservative reference-curve correction with interchangeable offline renderers."""
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d, median_filter

from .install import ASSETS
from .pitch import sample_curve, continuous


PC_ARCHIVE = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.oudep"
PC_URL = "https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/" + PC_ARCHIVE
PC_SHA = "ba7d43142d41f6900c8264b5662ca7125a50feb8760bb8b9615c61a8f5e6902e"


def install_model(check=None):
    import requests
    import zipfile
    from filelock import FileLock
    from . import storage as db
    from tools.workspace_resources import require_space
    directory = ASSETS / "pitch/pc_nsf"
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".install.lock")):
        archive = directory.parent / PC_ARCHIVE
        if not archive.is_file() or db.digest(archive, check_cancel=check) != PC_SHA:
            require_space(directory, 160 * 1024**2, "下载修音声码器")
            temporary = archive.with_suffix(".part")
            try:
                with requests.get(PC_URL, stream=True, timeout=(20, 45)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            if check: check()
                            stream.write(block)
                if db.digest(temporary, check_cancel=check) != PC_SHA:
                    raise ValueError("修音声码器校验失败")
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
        files = {}
        with zipfile.ZipFile(archive) as package:
            for item in package.infolist():
                if item.filename.endswith((".onnx", ".yaml", ".txt")):
                    if check: check()
                    target = directory / Path(item.filename).name
                    content = package.read(item)
                    import hashlib
                    expected = hashlib.sha256(content).hexdigest()
                    if not target.is_file() or db.digest(target, check_cancel=check) != expected:
                        temporary = target.with_suffix(target.suffix + ".part")
                        temporary.write_bytes(content)
                        temporary.replace(target)
                    files[target.name] = expected
        db.write(directory / "manifest.json", {"url": PC_URL, "archive_sha256": PC_SHA,
                 "files": files, "license": "CC-BY-NC-SA-4.0"})
        return directory


def plan(reference, converted, shift=0, threshold=50, strength=1., agreement=None):
    times = converted["times"]
    # Bound delay estimation to +/- 4 analysis frames (about 46 ms). This
    # aligns measurements only; the rendered waveform is never time-stretched.
    delay = 0.
    if len(times) > 20 and "energy" in reference and "energy" in converted:
        r = np.interp(times, reference["times"], reference["energy"])
        r, c = np.log1p(r * 100), np.log1p(converted["energy"] * 100)
        scores = {}
        for lag in range(-4, 5):
            a = r[:len(r)-lag] if lag > 0 else r[-lag:] if lag < 0 else r
            b = c[lag:] if lag > 0 else c[:len(c)+lag] if lag < 0 else c
            a, b = a - a.mean(), b - b.mean()
            scores[lag] = float(np.dot(a,b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))
        best = max(scores,key=scores.get)
        if scores[best] > scores[0] + .03:
            delay = best * float(np.median(np.diff(times)))
    target, ref_voiced = sample_curve(reference["times"], reference["f0"], times - delay)
    target *= 2 ** (shift/12)
    observed = converted["f0"]
    valid = ref_voiced & converted["voiced"]
    # An independent estimator guards common octave/voicing detection mistakes.
    if agreement is not None:
        secondary, secondary_uv = sample_curve(agreement["times"], agreement["f0"], times - delay)
        valid &= secondary_uv & (np.abs(1200*np.log2(np.maximum(secondary,1)/np.maximum(target / 2**(shift/12),1))) < 80)
    cents = np.zeros(len(times), np.float32)
    cents[valid] = 1200*np.log2(observed[valid]/target[valid])
    # A sustained discrepancy is required; avoid correcting brief transitions.
    sustained = median_filter((valid & (np.abs(cents)>threshold)).astype(np.float32),size=7) > .5
    # Confidence is not calibrated probability. Suppress rapid pitch jumps where
    # the reference itself may be following residual polyphonic material.
    reference_jump = np.r_[False,np.abs(np.diff(1200*np.log2(np.maximum(target,1))))>450]
    safe = valid & sustained & ~reference_jump & (np.abs(cents)<=1250) & (strength>0)
    correction = np.where(safe, -cents * strength, 0)
    correction = np.clip(gaussian_filter1d(correction,sigma=1.5),-1200,1200)
    correction[~valid] = 0
    result = observed * 2 ** (correction/1200)
    result[~converted["voiced"]] = 0
    return {"times":times,"reference":target,"observed":observed,"target":result,
            "valid":valid,"cents":cents,"mask":safe,"correction":correction,"analysis_delay_seconds":delay}


def intervals(mask,times,cents,min_seconds=.08):
    transitions=np.diff(np.r_[False,mask,False].astype(np.int8))
    step=float(np.median(np.diff(times))) if len(times)>1 else .01
    rows=[]
    for a,b in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)):
        if (b-a)*step>=min_seconds:
            rows.append({"start":round(float(times[a]),3),"end":round(float(times[b-1]+step),3),
                         "median_cents":round(float(np.median(cents[a:b])),1)})
    return rows


def summary(p):
    valid=p["valid"]
    errors=np.abs(p["cents"][valid])
    return {"analysis_delay_seconds":round(float(p.get("analysis_delay_seconds",0)),4),"comparable_frames":int(valid.sum()),"total_frames":len(valid),
            "comparable_fraction":round(float(valid.mean()),4) if len(valid) else 0,
            "median_abs_cents":round(float(np.median(errors)),1) if len(errors) else None,
            "p95_abs_cents":round(float(np.percentile(errors,95)),1) if len(errors) else None,
            "over_50_cents_fraction":round(float(np.mean(errors>50)),4) if len(errors) else None,
            "corrected_fraction":round(float(p["mask"].mean()),4) if len(valid) else 0,
            "suspected_octave_frames":int(np.sum(valid & (np.abs(p["cents"])>1100))),
            "intervals":intervals(valid & (np.abs(p["cents"])>50),p["times"],p["cents"])[:300]}


class PCNSF:
    def __init__(self):
        import onnxruntime as ort
        directory=install_model()
        candidates=list(directory.glob('*.onnx'))
        if len(candidates)!=1:
            raise ValueError('缺少 PC-NSF-HiFiGAN，请先安装可选修音声码器')
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        self.model=ort.InferenceSession(str(candidates[0]),sess_options=options,providers=['CPUExecutionProvider'])

    def render(self,x,rate,times,f0):
        import librosa
        # Official OpenUTAU package: log-mel, 128 bins, Slaney, 44.1k,
        # FFT/window 2048, hop512, reflect padding768, natural logarithm.
        padded=np.pad(x,(768,768),mode='reflect')
        spec=librosa.stft(padded,n_fft=2048,hop_length=512,win_length=2048,center=False)
        magnitude=np.abs(spec)
        bank=librosa.filters.mel(sr=rate,n_fft=2048,n_mels=128,fmin=40,fmax=16000,htk=False,norm='slaney')
        mel=np.log(np.maximum(bank@magnitude,1e-5)).astype(np.float32)
        grid=np.arange(mel.shape[1])*512/rate
        freq,uv=sample_curve(times,f0,grid)
        values={}
        for item in self.model.get_inputs():
            if 'f0' in item.name.lower() or 'pitch' in item.name.lower():
                values[item.name]=freq[None,:].astype(np.float32)
            elif 'mel' in item.name.lower() or item.name=='c':
                values[item.name]=mel.T[None,:,:]
            elif 'uv' in item.name.lower():
                values[item.name]=uv[None,:].astype(np.float32)
            else:
                raise ValueError('声码器输入不支持：'+item.name)
        result=self.model.run(None,values)[0].reshape(-1)
        return result


def psola(x,rate,times,target):
    import parselmouth
    from parselmouth.praat import call
    sound=parselmouth.Sound(x,sampling_frequency=rate)
    manipulation=call(sound,'To Manipulation',.01,50,1600)
    tier=call(manipulation,'Extract pitch tier')
    call(tier,'Remove points between',0,len(x)/rate)
    voiced=target>0
    if not voiced.any():
        return x.copy()
    for time,hz in zip(times[voiced],target[voiced]):
        if 0<time<len(x)/rate:
            call(tier,'Add point',float(time),float(hz))
    call([tier,manipulation],'Replace pitch tier')
    return call(manipulation,'Get resynthesis (overlap-add)').values[0].astype(np.float32)


def render(worker,source,target,curve,backend):
    """Render contextual blocks; blend only eligible regions, preserve others."""
    info=sf.info(source)
    if info.channels!=1 or info.samplerate!=44100:
        raise ValueError('校准输入需要 44.1 kHz 单声道转换人声')
    model=PCNSF() if backend=='pc_nsf' else None
    temporary=Path(target).with_suffix('.part.wav')
    peak=0.
    try:
        with sf.SoundFile(source) as src,sf.SoundFile(temporary,'w',samplerate=44100,channels=1,subtype='FLOAT') as dst:
            for start in range(0,len(src),44100*15):
                worker.check()
                end=min(len(src),start+44100*15)
                left,right=max(0,start-44100),min(len(src),end+44100)
                src.seek(left);x=src.read(right-left,dtype='float32')
                grid=np.arange(len(x))/44100+left/44100
                amount=np.interp(grid,curve['times'],curve['mask'].astype(float),left=0,right=0)
                amount=gaussian_filter1d(amount,sigma=441)
                # Do not smear processing across unvoiced boundaries.
                voiced=np.interp(grid,curve['times'],(curve['observed']>0).astype(float))>.99
                trusted=np.interp(grid,curve['times'],curve['valid'].astype(float))>.99
                amount[~voiced | ~trusted]=0
                if amount.max()>.01:
                    local=(curve['times']>=left/44100)&(curve['times']<=right/44100)
                    times=curve['times'][local]-left/44100
                    f0=curve['target'][local]
                    y=model.render(x,44100,times,f0) if model else psola(x,44100,times,f0)
                    if not np.isfinite(y).all() or abs(len(y)-len(x))>1024:
                        raise ValueError('修音输出数值或长度异常')
                    y=np.pad(y[:len(x)],(0,max(0,len(x)-len(y))))
                    # Smooth crossfade into the changed region; all other
                    # samples remain the original converted waveform.
                    x=x*(1-amount)+y*amount
                core=x[start-left:end-left]
                peak=max(peak,float(np.max(np.abs(core))) if len(core) else 0.)
                dst.write(core)
                worker.update(f'{backend} 校准 {end/44100:.1f}/{len(src)/44100:.1f} 秒')
        if sf.info(temporary).frames!=info.frames:
            raise ValueError('校准后时间轴长度异常')
        temporary.replace(target)
        return peak
    finally:
        temporary.unlink(missing_ok=True)


def reject_worse_regions(curve, after):
    """Check the same originally eligible frames; return regions to bypass."""
    output, uv = sample_curve(after["times"], after["f0"], curve["times"])
    mask = curve["mask"]
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    rejected = []
    for a,b in zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1)):
        comparable = uv[a:b] & (curve["reference"][a:b]>0)
        if comparable.mean()<.8:
            rejected.append((int(a),int(b)))
            continue
        error = np.abs(1200*np.log2(np.maximum(output[a:b][comparable],1)/curve["reference"][a:b][comparable]))
        old = np.abs(curve["cents"][a:b][comparable])
        if np.median(error)>np.median(old)+5 or np.percentile(error,90)>np.percentile(old,90)+25:
            rejected.append((int(a),int(b)))
    return rejected
