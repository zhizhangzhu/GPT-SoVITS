"""On-demand named WAV/MP3 exports of registered task audio only."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
import subprocess
import threading
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
import soundfile as sf

from . import storage as db
from .audio import ffmpeg
from .presentation import download_name, describe
from tools.workspace_resources import require_space

router=APIRouter(prefix='/svc-download')
_pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='svc-export')
_lock=threading.Lock()
_futures={}


def asset_id(path):
    return hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:24]


def authorize(request:Request):
    app=request.app
    if app.auth is None and app.auth_dependency is None:return
    if app.auth_dependency is not None:
        if app.auth_dependency(request) is not None:return
    else:
        token=request.cookies.get(f'access-token-{app.cookie_id}') or request.cookies.get(f'access-token-unsecure-{app.cookie_id}')
        if app.tokens.get(token) is not None:return
    raise HTTPException(401,'请先登录工作台')


def resolve(job,key,asset):
    from .service import conversion_outputs
    try:
        directory=db.job_dir(job).resolve()
        record=db.read(directory/'job.json')
        if record.get('kind') not in {'convert','pitch','reverb','dataset'} or not re.fullmatch(r'[a-z0-9_]{1,80}',key):
            raise ValueError('无效的音轨')
        candidates=[conversion_outputs(record).get(key)]
        if record['kind']=='dataset':
            stems=db.read(directory/'dataset-stems.json',{})
            candidates.extend(row.get(key) for row in stems.values() if isinstance(row,dict))
        for value in candidates:
            if not isinstance(value,str):continue
            path=Path(value).resolve()
            if path.is_relative_to(directory) and path.is_file() and path.suffix.lower() in {'.wav','.flac','.mp3','.ogg','.m4a'} and asset_id(path)==asset:
                stat=path.stat()
                stamp=(str(path),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
                token=hashlib.sha256(repr(stamp).encode()).hexdigest()[:32]
                return record,path,stamp,token
    except (OSError,ValueError,KeyError):pass
    raise HTTPException(404,'音轨已更新或不存在，请重新选择任务后下载')


def target_for(source,token,format):
    if format=='wav' and source.suffix.lower()=='.wav':return source
    return source.parent/'.downloads'/(token+'.'+format)


def option(source,token,format,info):
    target=target_for(source,token,format)
    if target.is_file():return dict(status='ready',bytes=target.stat().st_size,estimated=False)
    with _lock:future=_futures.get((token,format))
    result=dict(status='idle',bytes=round(info.duration*40000)+2048 if format=='mp3' else info.frames*info.channels*4+80,estimated=True)
    if future is not None:
        if not future.done():result['status']='processing'
        elif future.exception() is not None:result.update(status='error',message=str(future.exception())[:300])
    return result


def metadata(record,source,token):
    try:info=sf.info(source)
    except Exception as error:raise HTTPException(400,'无法读取这条音轨的音频信息') from error
    names=describe(record)
    return dict(version=token,model_name=names['model_name'],source_name=names['source_name'],
                formats={format:{**option(source,token,format,info),'filename':download_name(record,source,format)} for format in ['wav','mp3']})


def encode(source,target,stamp,format):
    # No parents=True: deleting a task while queued cannot resurrect it.
    target.parent.mkdir(exist_ok=True)
    info=sf.info(source)
    estimated=info.duration*40000+8192 if format=='mp3' else info.frames*info.channels*4+80
    require_space(target.parent,int(estimated)+8*1024**2,'生成下载音频')
    temporary=target.with_name(target.stem+'.'+uuid.uuid4().hex+'.part.'+format)
    codec=['-c:a','libmp3lame','-b:a','320k'] if format=='mp3' else ['-c:a','pcm_f32le']
    command=[ffmpeg(),'-nostdin','-hide_banner','-loglevel','error','-y','-i',str(source),'-map','0:a:0','-vn',*codec,'-threads','2',str(temporary)]
    try:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        try:
            _,errors=process.communicate(timeout=900)
        except BaseException:
            process.kill();process.communicate();raise
        if process.returncode:raise ValueError('导出失败：'+errors.decode('utf-8',errors='replace')[-250:])
        stat=source.stat()
        current=(str(source),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if current!=stamp:raise ValueError('音轨在导出期间发生变化，请重新打开下载弹窗')
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)


@router.get('/{job}/{key}',dependencies=[Depends(authorize)])
def information(job:str,key:str,asset:str):
    record,source,_,token=resolve(job,key,asset)
    return metadata(record,source,token)


@router.post('/{job}/{key}',dependencies=[Depends(authorize)])
def prepare(job:str,key:str,asset:str,format:str,version:str):
    if format not in {'wav','mp3'}:raise HTTPException(400,'请选择 WAV 或 MP3')
    record,source,stamp,token=resolve(job,key,asset)
    if version!=token:raise HTTPException(409,'音轨已更新，请重新打开下载弹窗')
    target=target_for(source,token,format)
    if not target.is_file():
        with _lock:
            identity=(token,format);old=_futures.get(identity)
            if old is None or old.done():
                if sum(not future.done() for future in _futures.values())>=16:raise HTTPException(429,'下载音频准备较多，请稍后重试')
                for entry,future in list(_futures.items()):
                    if len(_futures)<128:break
                    if future.done():_futures.pop(entry)
                _futures[identity]=_pool.submit(encode,source,target,stamp,format)
    return metadata(record,source,token)


@router.get('/{job}/{key}/file',dependencies=[Depends(authorize)])
def download(job:str,key:str,asset:str,format:str='wav',version:str=''):
    if format not in {'wav','mp3'}:raise HTTPException(400,'请选择 WAV 或 MP3')
    record,source,_,token=resolve(job,key,asset)
    if version and version!=token:raise HTTPException(409,'音轨已更新，请重新打开下载弹窗')
    target=target_for(source,token,format)
    if not target.is_file():raise HTTPException(409,'请先在下载弹窗中生成所选格式')
    return FileResponse(target,filename=download_name(record,source,format),media_type='audio/wav' if format=='wav' else 'audio/mpeg',headers={'Cache-Control':'private, no-store'})
