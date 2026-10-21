"""Map reference pitch onto a separate recording without stretching target audio."""
from pathlib import Path
import hashlib
import json

import numpy as np
import soundfile as sf

from . import storage as db
from .pitch import channels, mono, sample_curve

VERSION = 1


def features(worker, path, step):
    from .engine import encoder_for
    from .install import RUNTIME
    import sys
    sys.path.insert(0,str(RUNTIME))
    from .install import ASSETS, ENCODER_FILE
    import torch
    import librosa
    info = sf.info(path)
    key = {"version": VERSION, "source": db.digest(path, check_cancel=worker.check),
           "encoder": db.digest(ASSETS / ENCODER_FILE, check_cancel=worker.check), "step": step}
    directory = worker.path / "alignment-cache"
    directory.mkdir(exist_ok=True)
    token = hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:24]
    target = directory / (token + ".npz")
    state = db.read(directory / (token + ".json"), {})
    if target.is_file() and state.get("key") == key and db.digest(target,check_cancel=worker.check) == state.get("sha256"):
        with np.load(target,allow_pickle=False) as saved:
            return saved['times'], saved['features']
    selection = channels(path,worker.check)['selection']
    grid = np.arange(0,info.duration,step)
    result = np.zeros((len(grid),256),np.float32)
    model = encoder_for(torch.device('cpu'))
    try:
        with sf.SoundFile(path) as stream:
            for start in range(0,len(stream),info.samplerate*20):
                worker.check()
                end = min(len(stream),start+info.samplerate*20)
                left,right = max(0,start-info.samplerate),min(len(stream),end+info.samplerate)
                stream.seek(left)
                x = mono(stream.read(right-left,dtype='float32',always_2d=True),selection)
                core = (grid>=start/info.samplerate)&(grid<end/info.samplerate)
                if np.max(np.abs(x)) > 1e-5:
                    wave = librosa.resample(x,orig_sr=info.samplerate,target_sr=16000)
                    with torch.inference_mode():
                        encoded = model.encoder(torch.from_numpy(wave)).squeeze(0).cpu().numpy().T
                    times = left/info.samplerate + (np.arange(len(encoded))+.5)*len(x)/info.samplerate/len(encoded)
                    result[core] = np.stack([np.interp(grid[core],times,encoded[:,i]) for i in range(256)],axis=1)
                worker.update(f'对齐内容分析 · {end/info.samplerate:.1f}/{info.duration:.1f} 秒')
    finally:
        del model
    temporary = target.with_suffix('.part.npz')
    try:
        np.savez_compressed(temporary,times=grid,features=result)
        temporary.replace(target)
        db.write(directory / (token+'.json'),{"key":key,"sha256":db.digest(target,check_cancel=worker.check)})
    finally:
        temporary.unlink(missing_ok=True)
    return grid,result


def build(worker, reference, target):
    """Bound the DTW matrix to 3000 x 3000; use content, not wrong target F0."""
    import librosa
    from scipy.spatial.distance import cdist
    from scipy.ndimage import gaussian_filter1d
    ref_info,target_info = sf.info(reference),sf.info(target)
    step = max(.1,max(ref_info.duration,target_info.duration)/2990)
    rt,rf = features(worker,reference,step)
    tt,tf = features(worker,target,step)
    if len(tt)<3 or len(rt)<3:
        raise ValueError('自动对齐需要至少约 0.3 秒的有效音频')
    worker.check()
    worker.update('匹配同曲旋律对应的演唱内容')
    cost = cdist(tf,rf,metric='cosine').astype(np.float32)
    cost[~np.isfinite(cost)] = 1.
    # A shorter target may be an excerpt of the reference. Comparable full
    # recordings use an endpoint-anchored path with a bounded diagonal band.
    subsequence = target_info.duration < ref_info.duration*.85
    _,path = librosa.sequence.dtw(C=cost,subseq=subsequence,backtrack=True,
        global_constraints=not subsequence,band_rad=.25)
    path = path[::-1]
    mapped = np.full(len(tt),np.nan,np.float64)
    scores = np.ones(len(tt),np.float32)
    for index in np.unique(path[:,0]):
        candidates = path[path[:,0]==index,1]
        mapped[index] = np.median(rt[candidates])
        scores[index] = np.median(cost[index,candidates])
    found = np.flatnonzero(np.isfinite(mapped))
    if len(found)<3:
        raise ValueError('两份音频没有足够的可对齐内容，请确认是同一首歌、同一段落')
    mapped = np.interp(np.arange(len(tt)),found,mapped[found])
    mapped = np.maximum.accumulate(mapped)
    # Check local tempo over about one second; reference pauses/repeats or
    # unmatched material should be bypassed rather than force-corrected.
    slope = np.gradient(gaussian_filter1d(mapped,sigma=max(1.,.5/step)),tt)
    trusted = (scores<.55)&(slope>.25)&(slope<4.)
    quality = float(trusted.mean())
    if quality < .1:
        raise ValueError('参考与待校准音频匹配度不足；请使用同曲对应段落，或选择原时间轴并调整参考偏移')
    info = {"method":"contentvec-dtw","version":VERSION,"step_seconds":step,
            "trusted_fraction":round(quality,4),"median_content_distance":round(float(np.median(scores)),4),
            "reference_duration":ref_info.duration,"target_duration":target_info.duration}
    return {"target_times":tt,"reference_times":mapped,"trusted":trusted,"info":info}


def warp(curves, target_times, mapping=None, offset=0):
    query = (np.interp(target_times,mapping['target_times'],mapping['reference_times'])
             if mapping is not None else target_times+offset)
    f0,uv = sample_curve(curves['times'],curves['f0'],query)
    valid = uv & (query>=0)&(query<=curves['times'][-1])
    if mapping is not None:
        valid &= np.interp(target_times,mapping['target_times'],mapping['trusted'].astype(float))>.99
    energy = np.interp(query,curves['times'],curves['energy'],left=0,right=0)
    return {"times":target_times,"f0":np.where(valid,f0,0).astype(np.float32),
            "voiced":valid,"energy":energy}
