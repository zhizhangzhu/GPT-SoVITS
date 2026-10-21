"""Small, on-demand players for full-length task outputs.

Do not feed these files through gr.Audio/gr.File: Gradio 4 hashes each file in
its async response path, and WaveSurfer downloads/decodes every visible stem.
Only committed audio outputs inside the selected task are published here.
"""
from html import escape
import hashlib
from pathlib import Path
from urllib.parse import quote,unquote

from . import storage as db
from . import waveform


AMPLITUDE_LEGEND = ('<div class="svc-amplitude-legend"><strong>统一幅度参考 · dBFS</strong>　'
                    '波形外缘 0 · 半幅 −6 · 中心 −∞（静音）。所有音轨使用相同尺度。<br>'
                    '右侧电平表：顶部 0，底部 −60 dBFS；彩色柱为当前位置附近的峰值，白线为 RMS。'
                    '播放或定位后显示窗口电平；波形下方另列整轨峰值与 RMS。</div>')

GROUPS = [
    ('results','生成结果',True,[
        ('mixes','完整混音 · 带伴奏',True),
        ('dry','人声 · 加混响前，不含伴奏',False),
        ('wet','人声 · 加混响后，不含伴奏',False)]),
    ('reference','原曲参考',False,[
        ('full','完整原曲／上传参考',True),
        ('vocals','参考人声 · 不含伴奏',True)]),
    ('accompaniment','伴奏',False,[]),
    ('process','处理过程与历史对照',False,[
        ('before','转换／校准前音频',False),
        ('dry','去混响输入',False),
        ('residual','移除的混响与残留',False),
        ('snapshots','输出混响前的输入／编辑前快照',False)]),
]


def track_group(key):
    if key=='accompaniment':return 'accompaniment',''
    if key.startswith('reverb_before_'):return 'process','snapshots'
    if key=='reference_full' or key=='reverb_reference_mix':return 'reference','full'
    if key.startswith(('reference_','reverb_reference_')) or key in {'lead_vocals','backing_vocals','source_vocals'}:
        return 'reference','vocals'
    if key.startswith('before_') or key=='target_full':return 'process','before'
    if key in {'dry_lead','dry_backing','dry_vocals'}:return 'process','dry'
    if key in {'lead_reverb','backing_reverb','vocals_reverb'}:return 'process','residual'
    if key=='mix' or key.endswith('_mix'):return 'results','mixes'
    if key.startswith('reverb_') or '_reverb_' in key:return 'results','wet'
    return 'results','dry'


def grouped_players(tracks,prefix=''):
    """Keep individual Gradio outputs, so polling doesn't reload other players."""
    import gradio as gr
    players={}
    gr.HTML('<p class="svc-audio-dedup-summary" role="status" hidden></p>')
    def add(rows):
        for row in rows:
            key=row[0]
            element=row[2] if len(row)>2 else prefix+key
            players[key]=gr.HTML(visible=False,elem_id=element,elem_classes=['svc-result-audio'])
    for group,label,opened,children in GROUPS:
        rows=[row for row in tracks if track_group(row[0])[0]==group]
        if not rows:continue
        with gr.Accordion(label,open=opened,elem_classes=['svc-track-group','svc-track-group-main']):
            if children:
                for child,title,expanded in children:
                    selected=[row for row in rows if track_group(row[0])[1]==child]
                    if selected:
                        with gr.Accordion(title,open=expanded,elem_classes=['svc-track-group','svc-track-group-sub']):
                            add(selected)
            else:add(rows)
    return [players[row[0]] for row in tracks]


def publish(record, outputs, component):
    if record.get("kind") not in {"convert", "dataset", "pitch", "reverb"}:
        return {}, {}
    directory = db.job_dir(record["id"]).resolve()
    published = {}
    previews = {}
    for key, value in outputs.items():
        if not isinstance(value, (str, Path)):
            continue
        path = Path(value).resolve()
        if (path.suffix.lower() not in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
                or not path.is_relative_to(directory) or not path.is_file()):
            continue
        # Gradio's /file route authorizes these exact paths. Protect persistent
        # task files from its optional temporary-file cleanup; do not whitelist
        # a directory or copy/hash hundreds of MB on every task selection.
        component.temp_files.add(path)
        component.keep_in_cache.add(path)
        published[key] = "./file=" + quote(str(path), safe="/")
        try:
            preview = waveform.request(path)
        except OSError:
            preview = None
        if preview is not None:
            component.temp_files.add(preview)
            component.keep_in_cache.add(preview)
            detail=waveform.detail_path(preview)
            component.temp_files.add(detail)
            component.keep_in_cache.add(detail)
            previews[key] = "./file=" + quote(str(preview), safe="/")
    return published, previews


def download_url(task,key,source):
    from .downloads import asset_id
    path=unquote(source.removeprefix('./file='))
    return './svc-download/'+quote(task,safe='')+'/'+quote(key,safe='')+'?asset='+asset_id(path)


def player(label, url, task, preview=None, key=None):
    if not url:
        return ""
    title, source = escape(label), escape(url, quote=True)
    clip = 'svc-wave-' + hashlib.sha256((task + label + url).encode()).hexdigest()[:20]
    preview_source = escape(preview or "", quote=True)
    download=download_url(task,key or label,url)
    download_info=escape(download,quote=True)
    download_file=escape(download.replace('?','/file?')+'&format=wav',quote=True)
    detail_source=escape(preview[:-5]+'.detail.json' if preview and preview.endswith('.json') else '',quote=True)
    return (f'<section class="svc-task-audio" data-svc-audio-task="{escape(task, quote=True)}" data-svc-track-title="{escape(label,quote=True)}" data-svc-track-key="{escape(key or label,quote=True)}">'
            f'<div class="svc-track-heading"><button type="button" class="svc-track-open" '
            f'aria-label="打开单轨视图：{title}">{title}<span aria-hidden="true"> · 单轨视图 ↗</span></button>'
            f'<a href="{download_file}" data-svc-download="{download_info}" aria-haspopup="dialog">下载音轨</a></div>'
            '<p class="svc-track-aliases" hidden></p>'
            '<p class="svc-audio-upgrade">若单轨标题无法打开，请刷新页面加载新版界面。</p>'
            '<div class="svc-wave-shell">'
            f'<div class="svc-waveform" data-svc-waveform="{preview_source}" data-svc-wave-detail="{detail_source}" aria-busy="true">'
            '<svg viewBox="0 0 768 100" preserveAspectRatio="none" aria-hidden="true">'
            f'<defs><clipPath id="{clip}"><rect class="svc-wave-clip" width="0" height="100"/></clipPath></defs>'
            '<path class="svc-wave-grid" d="M0,4H768M0,27H768M0,50H768M0,73H768M0,96H768"/>'
            '<path class="svc-wave-base"/><path class="svc-wave-played" '
            f'clip-path="url(#{clip})"/></svg>'
            f'<input class="svc-wave-seek" type="range" min="0" max="10000" step="1" value="0" '
            f'aria-label="{title} · 播放位置" disabled>'
            '<span class="svc-wave-status" role="status">正在生成波形…</span></div>'
            '<aside class="svc-track-level" aria-label="当前位置的窗口电平，单位 dBFS">'
            '<div class="svc-track-meter" aria-hidden="true"><i></i><b></b></div>'
            '<div class="svc-track-level-text"><small>dBFS</small>'
            '<span>峰值 <output class="svc-track-peak">—</output></span>'
            '<span>RMS <output class="svc-track-rms">—</output></span>'
            '<small class="svc-track-window">点击播放或定位</small></div></aside></div>'
            '<div class="svc-wave-metrics"><span class="svc-wave-levels">整轨峰值 / RMS 待加载</span></div>'
            '<div class="svc-wave-times"><span class="svc-wave-elapsed">0:00</span>'
            '<span class="svc-wave-duration">--:--</span></div>'
            f'<audio controls controlslist="nodownload" preload="none" src="{source}" aria-label="{title}" '
            'data-svc-task-audio></audio>'
            '<span class="svc-audio-error" role="status" hidden>'
            '音频加载失败，请点击“重新加载”后再播放。</span>'
            '<button type="button" class="svc-audio-reload" hidden>重新加载</button>'
            '</section>')


def download_links(tracks, published, task):
    groups=[]
    for group,title,_,children in GROUPS:
        links=[]
        for row in tracks:
            key,label=row[:2];url=published.get(key)
            if not url or track_group(key)[0]!=group:continue
            info=download_url(task,key,url)
            fallback=info.replace('?','/file?')+'&format=wav'
            links.append(f'<li data-svc-download-key="{escape(key,quote=True)}"><a href="{escape(fallback,quote=True)}" data-svc-download="{escape(info,quote=True)}" aria-haspopup="dialog">{escape(label)}</a></li>')
        if links:groups.append('<div class="svc-download-group"><h4>'+escape(title)+'</h4><ul class="svc-track-downloads">'+''.join(links)+'</ul></div>')
    if not groups:return ''
    return ('<details class="svc-download-panel"><summary>集中下载音轨 <span class="svc-download-count"></span></summary>'
            '<nav aria-label="下载输出">'+''.join(groups)+'</nav></details>')
