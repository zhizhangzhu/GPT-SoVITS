"""Shared format picker for list, grouped downloads and the single-track view."""

JS = r"""
    {
        const dialog=document.createElement('dialog');
        dialog.className='svc-download-dialog';
        dialog.setAttribute('aria-labelledby','svc-download-title');
        dialog.innerHTML=`<header><h2 id="svc-download-title">下载音轨</h2><button type="button" data-export-close aria-label="关闭下载弹窗">关闭</button></header>
            <p class="svc-export-track"></p><p class="svc-export-source"></p>
            <fieldset><legend>选择下载格式</legend>
                <label><input type="radio" name="svc-export-format" value="wav" checked><span><strong>WAV</strong><small>保留原始幅度 · 无损</small></span><span data-export-size="wav">正在读取…</span></label>
                <label><input type="radio" name="svc-export-format" value="mp3"><span><strong>MP3</strong><small>320 kbps · 有损压缩</small></span><span data-export-size="mp3">正在读取…</span></label>
            </fieldset>
            <div class="svc-export-filename"><span>保存文件名</span><output></output></div>
            <p class="svc-export-status" role="status" aria-live="polite">正在读取文件信息…</p>
            <div class="svc-export-actions"><button type="button" data-export-close>取消</button><button type="button" class="svc-export-confirm" disabled>下载 WAV</button></div>`;
        (document.querySelector('.gradio-container')||document.body).appendChild(dialog);
        const q=selector=>dialog.querySelector(selector);
        let state=null;
        const text=(el,value)=>{if(el.textContent!==value)el.textContent=value;};
        const size=value=>value>=1024**3?(value/1024**3).toFixed(2)+' GB':value>=1024**2?(value/1024**2).toFixed(1)+' MB':Math.max(1,Math.round(value/1024))+' KB';
        const selected=()=>q('input[name="svc-export-format"]:checked').value;
        const render=()=>{
            if(!state?.data)return;
            for(const format of ['wav','mp3']){
                const row=state.data.formats[format];
                text(q(`[data-export-size="${format}"]`),(row.estimated?'约 ':'')+size(row.bytes));
            }
            const format=selected();
            text(q('.svc-export-filename output'),state.data.formats[format].filename);
            text(q('.svc-export-confirm'),state.busy?'正在准备…':'下载 '+format.toUpperCase());
            q('.svc-export-confirm').disabled=state.busy;
            dialog.querySelectorAll('input[type="radio"]').forEach(el=>{el.disabled=state.busy;});
        };
        const read=async(current,url,options={})=>{
            const response=await fetch(url,{...options,signal:current.abort.signal,cache:'no-store'});
            const data=await response.json();
            if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'暂时无法准备下载，请稍后重试');
            return data;
        };
        document.addEventListener('click',async event=>{
            const link=event.target.closest?.('a[data-svc-download]');if(!link)return;
            event.preventDefault();
            if(dialog.open)return;
            const track=link.closest('.svc-task-audio');
            const current={opener:link,url:new URL(link.dataset.svcDownload,document.baseURI),data:null,busy:false,abort:new AbortController()};
            state=current;
            q('input[value="wav"]').checked=true;
            dialog.querySelectorAll('input[type="radio"]').forEach(el=>{el.disabled=false;});
            text(q('.svc-export-track'),track?.dataset.svcTrackTitle||(link.classList.contains('svc-focus-download')?document.getElementById('svc-focus-title')?.textContent:link.textContent)||'');
            text(q('.svc-export-source'),'');text(q('.svc-export-filename output'),'');
            text(q('.svc-export-status'),'正在读取文件信息…');
            for(const format of ['wav','mp3'])text(q(`[data-export-size="${format}"]`),'正在读取…');
            q('.svc-export-confirm').disabled=true;
            dialog.showModal();q('[data-export-close]').focus();
            try{
                const data=await read(current,current.url);
                if(state!==current||current.abort.signal.aborted)return;
                current.data=data;
                text(q('.svc-export-source'),`音色模型：${data.model_name||'独立音频处理'} · 参考音频：${data.source_name||'旧任务未保存原文件名'}`);
                text(q('.svc-export-status'),'MP3 未生成时显示预估大小，生成后显示实际大小。');render();
            }catch(error){if(!current.abort.signal.aborted&&state===current)text(q('.svc-export-status'),error.message);}
        });
        dialog.addEventListener('change',event=>{if(event.target.matches('input[name="svc-export-format"]'))render();});
        dialog.addEventListener('click',async event=>{
            if(event.target.closest('[data-export-close]')){dialog.close();return;}
            if(!event.target.closest('.svc-export-confirm')||!state?.data||state.busy)return;
            const current=state,format=selected();current.busy=true;render();
            try{
                if(current.data.formats[format].status!=='ready'){
                    text(q('.svc-export-status'),`正在生成 ${format.toUpperCase()}，完成后自动开始下载…`);
                    const url=new URL(current.url);url.searchParams.set('format',format);url.searchParams.set('version',current.data.version);
                    current.data=await read(current,url,{method:'POST'});
                    while(current.data.formats[format].status!=='ready'){
                        const option=current.data.formats[format];
                        if(option.status==='error')throw new Error(option.message||'导出失败，请重试');
                        await new Promise(resolve=>setTimeout(resolve,750));
                        if(current.abort.signal.aborted||state!==current)return;
                        const data=await read(current,current.url);
                        if(data.version!==current.data.version)throw new Error('音轨已更新，请关闭后重新打开下载弹窗');
                        current.data=data;render();
                    }
                }
                if(state!==current||current.abort.signal.aborted)return;
                const url=new URL(current.url);url.pathname+='/file';url.searchParams.set('format',format);url.searchParams.set('version',current.data.version);
                const anchor=document.createElement('a');anchor.href=url;anchor.download=current.data.formats[format].filename;
                anchor.hidden=true;dialog.appendChild(anchor);anchor.click();anchor.remove();
                text(q('.svc-export-status'),'下载已开始 · '+size(current.data.formats[format].bytes));
            }catch(error){if(!current.abort.signal.aborted&&state===current)text(q('.svc-export-status'),error.message);}
            finally{if(state===current){current.busy=false;render();}}
        });
        dialog.addEventListener('close',()=>{const old=state;state=null;old?.abort.abort();if(old?.opener.isConnected)old.opener.focus();});
    }
"""

CSS = r"""
.svc-download-dialog { width: min(620px,calc(100vw - 32px)); max-height: calc(100dvh - 32px); overflow: auto; box-sizing: border-box; margin: auto; padding: 24px; border: 1px solid var(--wb-line,#2b384b); border-radius: 12px; background: var(--wb-surface,#101722); color: var(--wb-text,#f5f7fa); }
.svc-download-dialog::backdrop { background: rgba(0,0,0,.65); }
.svc-download-dialog header,.svc-export-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.svc-export-actions { margin-top: 20px; }
.svc-download-dialog h2 { margin: 0; font-size: 20px; }
.svc-download-dialog p { line-height: 1.7; overflow-wrap: anywhere; }
.svc-export-track { margin: 16px 0 6px; font-size: 14px; font-weight: 600; }
.svc-export-source { margin: 0 0 20px; color: var(--wb-muted); font-size: 12px; }
.svc-download-dialog fieldset { margin: 0; padding: 0; border: 0; }
.svc-download-dialog legend { margin-bottom: 10px; font-size: 13px; }
.svc-download-dialog fieldset label { display: grid; grid-template-columns: 18px minmax(0,1fr) auto; align-items: center; gap: 12px; margin-bottom: 10px; padding: 14px; border: 1px solid var(--wb-line); border-radius: 8px; cursor: pointer; font-size: 14px; }
.svc-download-dialog fieldset label:has(input:checked) { border-color: var(--wb-accent); background: var(--wb-active); }
.svc-download-dialog small { display: block; margin-top: 4px; color: var(--wb-muted); font-size: 12px; }
.svc-export-filename { margin-top: 16px; font-size: 12px; line-height: 1.7; }
.svc-export-filename > span { color: var(--wb-muted); }
.svc-export-filename output { display: block; margin-top: 5px; overflow-wrap: anywhere; }
.svc-export-status { min-height: 24px; margin: 14px 0; color: var(--wb-muted); font-size: 12px; }
.svc-download-dialog button { min-height: 44px; padding: 8px 16px; border: 1px solid var(--wb-line); border-radius: 6px; background: var(--wb-input); color: var(--wb-text); cursor: pointer; font: inherit; font-size: 13px; }
.svc-download-dialog .svc-export-confirm { border-color: var(--wb-accent); color: var(--wb-accent); background: var(--wb-active); }
.svc-download-dialog button:disabled { opacity: .55; cursor: wait; }
.svc-download-dialog :is(button,input):focus-visible { outline: 2px solid var(--wb-accent); outline-offset: 3px; }
.svc-download-dialog [data-export-size] { font-variant-numeric: tabular-nums; white-space: nowrap; }
"""
