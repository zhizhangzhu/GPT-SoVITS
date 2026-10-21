"""Stable user-facing names, also available for historical task records."""
from pathlib import Path
import re

from . import storage as db


def describe(record, records=None, seen=None):
    seen=set(seen or ())
    if not record or record.get('id') in seen or len(seen)>32:return {'source_name':'','model_name':''}
    seen.add(record.get('id'))
    options=record.get('options',{})
    source=record.get('source_name') or record.get('reference_name') or ''
    model=record.get('model_name') or ''
    parent=options.get('source_job') or record.get('retried_from')
    if parent and (not source or not model):
        try:
            previous=(records or {}).get(parent) or db.read(db.job_dir(parent)/'job.json',{})
            inherited=describe(previous,records,seen)
            source=source or inherited['source_name'];model=model or inherited['model_name']
        except (OSError,ValueError):pass
    if not model:
        try:
            if options.get('model_id'):
                item=db.read(db.DATA/'imported_models'/db.identifier(options['model_id'])/'model.json',{})
                model=item.get('name') or options.get('speaker_name') or ''
                if len(item.get('speakers',{}))>1 and options.get('speaker_name'):
                    model+=' · '+options['speaker_name']
            elif options.get('experiment_id'):
                item=db.read(db.experiment_dir(options['experiment_id'])/'experiment.json',{})
                model=item.get('name') or ''
                if not model and item.get('singer_id'):model=db.singer(item['singer_id'])['name']
        except (OSError,ValueError,KeyError):pass
    if not source and record.get('kind')=='dataset':
        names=[]
        for identity in options.get('sources',[]):
            try:
                item=db.read(db.source_dir(record['singer_id'],identity)/'source.json',{})
                if item.get('name'):names.append(item['name'])
            except (OSError,ValueError,KeyError):pass
        source='、'.join(names)
    return {'source_name':str(source),'model_name':str(model)}


def _part(value,limit):
    text=re.sub(r'[\x00-\x1f\x7f<>:"/\\|?*]+','_',str(value))
    text=re.sub(r'\s+',' ',text).strip(' ._')
    return text.encode('utf-8')[:limit].decode('utf-8',errors='ignore').rstrip(' ._') or '音频'


def download_name(record,source,extension):
    names=describe(record)
    model=names['model_name'] or ('独立校准' if record.get('kind')=='pitch' else '未指定音色')
    original=Path(names['source_name']).stem if names['source_name'] else '原始音频_'+record['id'][:8]
    return '_'.join([_part(model,70),_part(original,100),_part(Path(source).stem,65)])+'.'+extension
