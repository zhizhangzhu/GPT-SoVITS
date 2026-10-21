"""Generate small peak envelopes outside UI requests, with bounded memory."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
from pathlib import Path
import struct
import threading
import uuid

import numpy as np
import soundfile as sf

POINTS = 768
DETAIL_POINTS = 8192
VERSION = 4
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="svc-waveform")
_pending = set()
_lock = threading.Lock()


def _stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def request(source):
    """Return the cache path immediately; at most one CPU reader runs at a time."""
    source = Path(source).resolve()
    stamp = _stamp(source)
    key = hashlib.sha256(repr((str(source), stamp, VERSION, POINTS, DETAIL_POINTS)).encode()).hexdigest()[:24]
    target = source.parent / ".waveforms" / f"{source.stem}-{key}.json"
    if not target.resolve().is_relative_to(source.parent):
        return None
    if target.is_file() and detail_path(target).is_file():
        return target
    with _lock:
        if target not in _pending:
            # Rapidly browsing a very large history must not grow an unbounded
            # work queue. Re-selecting a deferred task requests it again.
            if len(_pending) >= 128:
                return None
            _pending.add(target)
            try:
                _executor.submit(_generate, source, target, stamp)
            except RuntimeError:
                _pending.discard(target)
                return None
    return target


def detail_path(preview):
    return Path(preview).with_suffix('.detail.json')


def _pcm_layout(source,frames,channels):
    """Byte layout for bounded HTTP Range reads, without decoding full songs."""
    try:
        with source.open('rb') as stream:
            header=stream.read(12)
            if header[:4]!=b'RIFF' or header[8:]!=b'WAVE':return None
            fmt=None
            for _ in range(256):
                chunk=stream.read(8)
                if len(chunk)!=8:return None
                kind,size=chunk[:4],struct.unpack('<I',chunk[4:])[0]
                offset=stream.tell()
                if kind==b'fmt ':
                    value=stream.read(min(size,128))
                    if len(value)<16:return None
                    code,n,rate,_,align,bits=struct.unpack('<HHIIHH',value[:16])
                    if code==0xfffe and len(value)>=40:code=struct.unpack('<H',value[24:26])[0]
                    fmt=(code,n,rate,align,bits)
                if kind==b'data' and fmt:
                    code,n,rate,align,bits=fmt
                    if n!=channels or align!=n*bits//8 or frames*align>size:return None
                    if not ((code==1 and bits in (8,16,24,32)) or (code==3 and bits in (32,64))):return None
                    return dict(offset=offset,frame_bytes=align,bits=bits,format=code)
                stream.seek(offset+size+(size%2))
    except (OSError,ValueError,struct.error):
        return None
    return None


def _generate(source, target, stamp):
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    detail=detail_path(target)
    detail_temporary=detail.with_name(detail.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        # No parents=True: a task deleted during generation stays deleted.
        target.parent.mkdir(exist_ok=True)
        with sf.SoundFile(source) as audio:
            frames, rate = len(audio), audio.samplerate
            if frames < 1 or rate < 1 or audio.channels > 32:
                return
            sample_hash=hashlib.sha256(f'{frames}:{rate}:{audio.channels}:float64-le'.encode())
            count = min(POINTS, frames)
            peaks = np.zeros(count, dtype=np.float32)
            detail_count=min(DETAIL_POINTS,frames)
            low=np.full((detail_count,audio.channels),np.inf,dtype=np.float32)
            high=np.full_like(low,-np.inf)
            energy=np.zeros((detail_count,audio.channels),dtype=np.float64)
            counts=np.zeros(detail_count,dtype=np.int64)
            offset = 0
            for decoded in audio.blocks(blocksize=65536, dtype="float64", always_2d=True):
                if not np.isfinite(decoded).all():
                    return
                # Compare decoded samples, not filenames, WAV headers or
                # encodings. Treat signed zero as the same silent sample.
                decoded[decoded==0]=0
                sample_hash.update(decoded.astype('<f8',copy=False).tobytes())
                samples=decoded.astype(np.float32)
                if not np.isfinite(samples).all():return
                values = np.max(np.abs(samples), axis=1)
                bins = (np.arange(offset, offset + len(samples), dtype=np.int64) * count) // frames
                starts = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
                indexes = bins[starts]
                peaks[indexes] = np.maximum(peaks[indexes], np.maximum.reduceat(values, starts))
                detail_bins=(np.arange(offset,offset+len(samples),dtype=np.int64)*detail_count)//frames
                segments=np.r_[0,np.flatnonzero(np.diff(detail_bins))+1]
                slots=detail_bins[segments]
                low[slots]=np.minimum(low[slots],np.minimum.reduceat(samples,segments,axis=0))
                high[slots]=np.maximum(high[slots],np.maximum.reduceat(samples,segments,axis=0))
                energy[slots]+=np.add.reduceat(samples.astype(np.float64)**2,segments,axis=0)
                counts[slots]+=np.diff(np.r_[segments,len(samples)])
                offset += len(samples)
            if offset != frames or _stamp(source) != stamp:
                return
        maximum = float(peaks.max())
        rms=float(np.sqrt(energy.sum()/max(frames*low.shape[1],1)))
        data = dict(version=VERSION, duration=frames / rate, frames=frames, sample_rate=rate,
                    channels=low.shape[1],scale='absolute',
                    sample_hash=sample_hash.hexdigest(),
                    peak_dbfs=round(20*np.log10(maximum),3) if maximum>0 else None,
                    rms_dbfs=round(20*np.log10(rms),3) if rms>0 else None,
                    peaks=[round(float(value), 7) for value in peaks])
        details={key:value for key,value in data.items() if key!='peaks'}
        details.update(count=detail_count,pcm=_pcm_layout(source,frames,low.shape[1]),
            low=np.round(low.T.astype(np.float64),7).tolist(),high=np.round(high.T.astype(np.float64),7).tolist(),
            rms=np.round(np.sqrt(energy/np.maximum(counts[:,None],1)).T,7).tolist())
        with detail_temporary.open('x',encoding='utf-8') as stream:
            json.dump(details,stream,separators=(',',':'),allow_nan=False)
        os.replace(detail_temporary,detail)
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, target)
    except (OSError, RuntimeError, ValueError):
        logging.getLogger(__name__).debug("Waveform unavailable for %s", source, exc_info=True)
    finally:
        temporary.unlink(missing_ok=True)
        detail_temporary.unlink(missing_ok=True)
        with _lock:
            _pending.discard(target)
