"""Offline single-track waveform view, injected into the existing workbench.

Coarse signed envelopes are cached by Python; fine zoom reads bounded WAV
byte ranges. No AudioContext, full-song decode, or third-party asset is used.
"""

JS = r"""
    {
        const MAX_BYTES = 8 * 1024 * 1024;
        const dialog = document.createElement('dialog');
        dialog.className = 'svc-focus-view';
        dialog.setAttribute('aria-labelledby', 'svc-focus-title');
        dialog.innerHTML = `
            <header class="svc-focus-header">
                <button type="button" data-action="close">← 返回音轨列表</button>
                <div><h2 id="svc-focus-title">单轨波形</h2><p class="svc-focus-description">固定 dBFS 参考 · 横向缩放时间，纵向放大仅改变显示</p></div>
                <a class="svc-focus-download" aria-haspopup="dialog">下载当前音轨</a>
            </header>
            <div class="svc-focus-toolbar" role="toolbar" aria-label="波形视图工具">
                <div class="svc-focus-transport" role="group" aria-label="播放控制">
                    <button type="button" data-action="play" class="svc-focus-play" aria-label="播放" aria-pressed="false" title="播放／暂停（空格）">
                        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg><span>播放</span>
                    </button>
                    <button type="button" data-action="stop" title="停止并回到音轨开头">
                        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6h12v12H6z"/></svg><span>停止</span>
                    </button>
                    <span class="svc-focus-position" aria-label="播放时间">0:00.0 / --:--</span>
                </div>
                <button type="button" data-action="in">时间放大 ＋</button>
                <button type="button" data-action="out">时间缩小 −</button>
                <button type="button" data-action="fit">适应全曲</button>
                <button type="button" data-action="selection">放大选区</button>
                <button type="button" data-action="clear">清除选区</button>
                <label>拖动方式 <select class="svc-focus-mode"><option value="select">选择片段</option><option value="pan">平移时间轴</option></select></label>
                <label>显示放大 <input class="svc-focus-gain" type="range" min="-12" max="36" step="1" value="0" aria-label="纵向显示放大 dB"><output class="svc-focus-gain-label">+0 dB</output></label>
                <label><input class="svc-focus-follow" type="checkbox" checked> 跟随播放</label>
            </div>
            <div class="svc-focus-info"><span class="svc-focus-file"></span><span class="svc-focus-resolution" role="status"></span></div>
            <div class="svc-focus-loading" role="status" aria-live="polite" aria-atomic="true" hidden>
                <span class="svc-focus-spinner" aria-hidden="true"></span>
                <span class="svc-focus-loading-text">正在加载单轨波形…</span>
            </div>
            <div class="svc-focus-workspace">
                <div class="svc-focus-plot" tabindex="0" role="img" aria-label="单轨波形图；点击定位，拖动选区，滚轮缩放，方向键平移，空格播放">
                    <canvas class="svc-focus-wave"></canvas><canvas class="svc-focus-overlay"></canvas>
                </div>
                <aside class="svc-focus-levels" aria-label="播放位置附近的窗口电平">
                    <h3>窗口电平 · dBFS</h3><div class="svc-focus-meter-wrap">
                    <div class="svc-focus-meter"><i></i><b></b></div>
                    <div class="svc-focus-meter-scale" aria-hidden="true">
                        <span style="top:0%">0</span><span style="top:10%">−6</span><span style="top:20%">−12</span>
                        <span style="top:40%">−24</span><span style="top:60%">−36</span>
                        <span style="top:80%">−48</span><span style="top:100%">−60</span>
                    </div></div>
                    <p class="svc-focus-peak">峰值 —</p><p class="svc-focus-rms">RMS —</p>
                    <small class="svc-focus-window"></small>
                </aside>
            </div>
            <div class="svc-focus-navigation">
                <button type="button" data-action="left" aria-label="向前平移">←</button>
                <input class="svc-focus-pan" type="range" min="0" max="1" step="0.00001" value="0" aria-label="移动波形视窗">
                <button type="button" data-action="right" aria-label="向后平移">→</button>
                <span class="svc-focus-range"></span>
            </div>
            <div class="svc-focus-readout"><span class="svc-focus-cursor">光标 —</span><span class="svc-focus-selection">未选择片段</span></div>
            <audio class="svc-focus-audio" controls controlslist="nodownload" preload="metadata" aria-label="单轨试听"></audio>
            <p class="svc-focus-help">点击波形定位；拖动选择片段后可放大选区。滚轮缩放时间，Shift＋滚轮平移；也可使用按钮和时间滑块。dBFS 以数字满刻度为 0，峰值与 RMS 分别显示。此页面不修改音频。</p>`;
        (document.querySelector('.gradio-container') || document.body).appendChild(dialog);
        const q = (selector) => dialog.querySelector(selector);
        const plot = q('.svc-focus-plot');
        const wave = q('.svc-focus-wave');
        const overlay = q('.svc-focus-overlay');
        const audio = q('.svc-focus-audio');
        const gain = q('.svc-focus-gain');
        const pan = q('.svc-focus-pan');
        const resolution = q('.svc-focus-resolution');
        const loading = q('.svc-focus-loading');
        const showLoading = (message, failed = false) => {
            loading.dataset.state = failed ? 'error' : 'loading';
            q('.svc-focus-loading-text').textContent = message;
            loading.hidden = false;
            plot.setAttribute('aria-busy', String(!failed));
        };
        const hideLoading = () => {
            loading.hidden = true;
            plot.setAttribute('aria-busy', 'false');
        };
        let state = null, animation = 0, drawFrame = 0, lastMeter = 0;
        const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
        const db = (amplitude) => amplitude > 0 ? 20 * Math.log10(amplitude) : -Infinity;
        const dbText = (value) => Number.isFinite(value) ? `${value >= 0 ? '+' : ''}${value.toFixed(1)} dBFS` : '−∞ dBFS';
        const timeText = (seconds, digits = 3) => {
            const factor=10**digits;
            const n = Math.round(Math.max(0, Number(seconds) || 0)*factor)/factor;
            return `${Math.floor(n / 60)}:${(n % 60).toFixed(digits).padStart(digits ? digits + 3 : 2, '0')}`;
        };
        const geometry = () => {
            const width = Math.max(100, plot.clientWidth), height = Math.max(150, plot.clientHeight);
            return {width, height, left:62, right:14, top:32, bottom:20,
                w:Math.max(1,width-76), h:Math.max(1,height-52)};
        };
        const timeAt = (x) => state ? state.start + clamp((x - geometry().left) / geometry().w, 0, 1) * state.span : 0;
        const xAt = (seconds) => geometry().left + (seconds - state.start) / state.span * geometry().w;
        const setText = (element, text) => { if (element.textContent !== text) element.textContent = text; };
        const syncTransport = () => {
            const playing = !audio.paused && !audio.ended;
            const button = q('[data-action="play"]'), label = playing ? '暂停' : '播放';
            setText(button.querySelector('span'), label);
            button.setAttribute('aria-label', label);
            button.setAttribute('aria-pressed', String(playing));
            button.querySelector('path').setAttribute('d', playing ? 'M6 5h4v14H6zM14 5h4v14h-4z' : 'M8 5v14l11-7z');
            const duration = Number.isFinite(audio.duration) ? audio.duration : state?.data?.duration;
            const position = audio.readyState >= 1 ? audio.currentTime : state?.initialTime || 0;
            setText(q('.svc-focus-position'), `${timeText(position,1)} / ${duration > 0 ? timeText(duration,1) : '--:--'}`);
        };
        const togglePlayback = () => {
            if (!state) return;
            const current = state;
            if (!audio.paused && !audio.ended) audio.pause();
            else audio.play().catch(() => {
                if (dialog.open && state === current) setText(resolution,'暂时无法播放，请稍后再次点击播放，或关闭后重新打开音轨。');
                syncTransport();
            });
        };
        for (const name of ['play','pause','ended','timeupdate','loadedmetadata','durationchange','emptied']) {
            audio.addEventListener(name,syncTransport);
        }
        const pauseOthers = () => document.querySelectorAll('audio').forEach((item) => { if (item !== audio) item.pause(); });
        audio.addEventListener('play', () => { pauseOthers(); cancelAnimationFrame(animation); animation = requestAnimationFrame(tick); });
        document.addEventListener('play', (event) => { if (event.target !== audio && dialog.open) audio.pause(); }, true);
        const canvasContext = (canvas) => {
            const g = geometry(), ratio = Math.min(window.devicePixelRatio || 1, 2);
            if (canvas.width !== Math.round(g.width * ratio) || canvas.height !== Math.round(g.height * ratio)) {
                canvas.width = Math.round(g.width * ratio); canvas.height = Math.round(g.height * ratio);
            }
            const ctx = canvas.getContext('2d'); ctx.setTransform(ratio,0,0,ratio,0,0);
            ctx.clearRect(0,0,g.width,g.height); return ctx;
        };
        const color = (name, fallback) => getComputedStyle(dialog).getPropertyValue(name).trim() || fallback;
        const sampleMetrics = (seconds) => {
            if (!state?.data) return null;
            const d = state.data, raw = state.raw;
            if (raw && seconds >= raw.start / d.sample_rate && seconds < raw.end / d.sample_rate) {
                const center = Math.round(seconds * d.sample_rate) - raw.start;
                const half = Math.max(1, Math.round(d.sample_rate * .005));
                const a = Math.max(0,center-half), b = Math.min(raw.samples[0].length,center+half);
                let peak = 0, energy = 0;
                for (const channel of raw.samples) for (let i=a;i<b;i++) { peak=Math.max(peak,Math.abs(channel[i]));energy+=channel[i]*channel[i]; }
                return {peak, rms:Math.sqrt(energy/Math.max(1,(b-a)*raw.samples.length)),window:(b-a)/d.sample_rate,exact:true};
            }
            const index=clamp(Math.floor(seconds/d.duration*d.count),0,d.count-1);
            let peak=0,energy=0;
            for(let c=0;c<Math.min(d.channels,2);c++) {
                peak=Math.max(peak,Math.abs(d.low[c][index]),Math.abs(d.high[c][index]));energy+=d.rms[c][index]**2;
            }
            return {peak,rms:Math.sqrt(energy/Math.min(d.channels,2)),window:d.duration/d.count,exact:false};
        };
        const showMetrics = (seconds) => {
            const metrics=sampleMetrics(seconds);if(!metrics)return;
            setText(q('.svc-focus-peak'),'峰值 '+dbText(db(metrics.peak)));
            setText(q('.svc-focus-rms'),'RMS '+dbText(db(metrics.rms)));
            setText(q('.svc-focus-window'),`${metrics.exact ? '采样片段' : '缓存包络'} · 约 ${(metrics.window*1000).toFixed(1)} ms`);
            q('.svc-focus-meter i').style.clipPath=`inset(${(1-clamp((db(metrics.peak)+60)/60,0,1))*100}% 0 0)`;
            q('.svc-focus-meter b').style.bottom=clamp((db(metrics.rms)+60)/60,0,1)*100+'%';
            q('.svc-focus-meter').classList.toggle('svc-level-over',metrics.peak>1);
        };
        const paintOverlay = () => {
            if(!dialog.open||!state?.data)return;
            const ctx=canvasContext(overlay),g=geometry();
            ctx.save();ctx.beginPath();ctx.rect(g.left,g.top,g.w,g.h);ctx.clip();
            if(state.selection) {
                const [a,b]=state.selection;ctx.fillStyle='rgba(85,150,255,.18)';ctx.fillRect(xAt(a),g.top,xAt(b)-xAt(a),g.h);
            }
            if(Number.isFinite(state.hover)) {
                ctx.strokeStyle=color('--wb-muted','#b5c1d0');ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(xAt(state.hover),g.top);ctx.lineTo(xAt(state.hover),g.top+g.h);ctx.stroke();ctx.setLineDash([]);
            }
            const playhead=Number.isFinite(audio.currentTime)?audio.currentTime:state.initialTime;
            ctx.strokeStyle=color('--wb-warning','#f2ba7d');ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(xAt(playhead),g.top);ctx.lineTo(xAt(playhead),g.top+g.h);ctx.stroke();ctx.restore();
            const selected=state.selection;
            setText(q('.svc-focus-selection'),selected?`选区 ${timeText(selected[0])} — ${timeText(selected[1])} · ${(selected[1]-selected[0]).toFixed(4)} 秒`:'未选择片段');
        };
        const paint = () => {
            if(!dialog.open||!state?.data)return;
            const ctx=canvasContext(wave),g=geometry(),d=state.data;
            const channels=Math.min(d.channels,2), lane=g.h/channels, multiplier=10**(Number(gain.value)/20);
            const grid=color('--wb-line','#2b384b'),text=color('--wb-muted','#b5c1d0');
            ctx.font='11px ui-monospace, monospace';ctx.textBaseline='middle';
            ctx.fillStyle=color('--wb-input','#080c11');ctx.fillRect(g.left,g.top,g.w,g.h);
            const ideal=state.span/Math.max(2,Math.floor(g.w/125));
            const unit=10**Math.floor(Math.log10(Math.max(ideal,1e-8)));
            const tick=[1,2,5,10].find(n=>n*unit>=ideal)*unit;
            for(let t=Math.ceil(state.start/tick)*tick;t<=state.start+state.span;t+=tick) {
                const x=xAt(t);ctx.strokeStyle=grid;ctx.beginPath();ctx.moveTo(x,g.top);ctx.lineTo(x,g.top+g.h);ctx.stroke();
                ctx.fillStyle=text;ctx.textAlign='center';ctx.fillText(timeText(t,state.span<2?4:state.span<20?2:1),x,14);
            }
            ctx.save();ctx.beginPath();ctx.rect(g.left,g.top,g.w,g.h);ctx.clip();
            const raw=state.raw, useRaw=raw&&raw.start<=Math.floor(state.start*d.sample_rate)&&raw.end>=Math.min(d.frames,Math.ceil((state.start+state.span)*d.sample_rate));
            const pixels=Math.max(1,Math.floor(g.w));
            for(let c=0;c<channels;c++) {
                const mid=g.top+lane*(c+.5),amp=lane*.43;
                ctx.save();ctx.beginPath();ctx.rect(g.left,g.top+lane*c,g.w,lane);ctx.clip();
                ctx.strokeStyle=grid;ctx.lineWidth=1;
                for(const value of [-1,-.5,0,.5,1]) {const y=mid-value*amp;ctx.beginPath();ctx.moveTo(g.left,y);ctx.lineTo(g.left+g.w,y);ctx.stroke();}
                ctx.strokeStyle=color('--wb-accent','#9ac2ff');ctx.lineWidth=1;ctx.beginPath();
                if(useRaw && state.span*d.sample_rate<pixels*1.5) {
                    const channel=raw.samples[c];
                    const a=Math.max(0,Math.floor(state.start*d.sample_rate)-raw.start),b=Math.min(channel.length,Math.ceil((state.start+state.span)*d.sample_rate)-raw.start);
                    for(let i=a;i<b;i++) {const x=xAt((raw.start+i)/d.sample_rate),y=mid-channel[i]*multiplier*amp;if(i===a)ctx.moveTo(x,y);else ctx.lineTo(x,y);}
                } else {
                    for(let x=0;x<pixels;x++) {
                        const ta=state.start+x/pixels*state.span,tb=state.start+(x+1)/pixels*state.span;
                        let low=Infinity,high=-Infinity;
                        if(useRaw) {
                            const samples=raw.samples[c],a=Math.max(0,Math.floor(ta*d.sample_rate)-raw.start),b=Math.min(samples.length,Math.max(a+1,Math.ceil(tb*d.sample_rate)-raw.start));
                            for(let i=a;i<b;i++){low=Math.min(low,samples[i]);high=Math.max(high,samples[i]);}
                        } else {
                            const a=clamp(Math.floor(ta/d.duration*d.count),0,d.count-1),b=clamp(Math.ceil(tb/d.duration*d.count),a+1,d.count);
                            for(let i=a;i<b;i++){low=Math.min(low,d.low[c][i]);high=Math.max(high,d.high[c][i]);}
                        }
                        if(Number.isFinite(low)){ctx.moveTo(g.left+x,mid-high*multiplier*amp);ctx.lineTo(g.left+x,mid-low*multiplier*amp);}
                    }
                }
                ctx.stroke();
                ctx.restore();
            }
            ctx.restore();
            for(let c=0;c<channels;c++) {
                const mid=g.top+lane*(c+.5),amp=lane*.43;
                ctx.fillStyle=text;ctx.textAlign='right';
                for(const value of [-1,-.5,0,.5,1]) {
                    const level=value===0?'−∞':(20*Math.log10(Math.abs(value)/multiplier)).toFixed(1);
                    ctx.fillText(level,g.left-9,mid-value*amp);
                }
                ctx.textAlign='left';ctx.fillText(channels===1?'Mono':c?'R':'L',5,mid-amp-10);
            }
            ctx.fillStyle=text;ctx.textAlign='left';ctx.fillText('dBFS',4,g.height-10);
            setText(q('.svc-focus-range'),`${timeText(state.start)} — ${timeText(state.start+state.span)} / ${timeText(d.duration)}`);
            pan.disabled=state.span>=d.duration*.999999;pan.value=String(state.start/Math.max(d.duration-state.span,1e-9));
            setText(q('.svc-focus-gain-label'),`${Number(gain.value)>=0?'+':''}${gain.value} dB`);
            paintOverlay();
            if (!loading.hidden) hideLoading();
        };
        const schedulePaint = () => { cancelAnimationFrame(drawFrame);drawFrame=requestAnimationFrame(paint); };
        const fetchPCM = async (current,first,last,controller) => {
            const d=current.data,p=d.pcm;
            first=clamp(Math.floor(first),0,d.frames-1);last=clamp(Math.ceil(last),first+1,d.frames);
            const bytes=(last-first)*p.frame_bytes;
            if(bytes>MAX_BYTES)throw new Error('请继续放大时间轴以读取精细波形');
            const start=p.offset+first*p.frame_bytes;
            const response=await fetch(current.url,{headers:{Range:`bytes=${start}-${start+bytes-1}`},signal:controller.signal});
            if(response.status!==206){await response.body?.cancel();throw new Error('服务未返回分段数据，保留缓存包络');}
            const buffer=await response.arrayBuffer();if(buffer.byteLength!==bytes)throw new Error('音频片段长度不完整');
            const view=new DataView(buffer),count=last-first,channels=Math.min(d.channels,2),samples=Array.from({length:channels},()=>new Float32Array(count));
            const size=p.bits/8;
            for(let i=0;i<count;i++)for(let c=0;c<channels;c++) {
                const at=i*p.frame_bytes+c*size;let value;
                if(p.format===3)value=p.bits===32?view.getFloat32(at,true):view.getFloat64(at,true);
                else if(p.bits===8)value=(view.getUint8(at)-128)/128;
                else if(p.bits===16)value=view.getInt16(at,true)/32768;
                else if(p.bits===24){let v=view.getUint8(at)|(view.getUint8(at+1)<<8)|(view.getUint8(at+2)<<16);if(v&0x800000)v-=0x1000000;value=v/8388608;}
                else value=view.getInt32(at,true)/2147483648;
                if(!Number.isFinite(value))throw new Error('音频包含无效采样值');samples[c][i]=value;
            }
            return {start:first,end:last,samples};
        };
        const refine = async () => {
            const current=state;if(!current?.data)return;
            current.rawAbort?.abort();
            const d=current.data,p=d.pcm;
            if(!p){setText(resolution,'缓存包络 · 此格式不提供采样级缩放');return;}
            const first=Math.floor(current.start*d.sample_rate),last=Math.min(d.frames,Math.ceil((current.start+current.span)*d.sample_rate));
            if(current.raw&&current.raw.start<=first&&current.raw.end>=last){setText(resolution,'精细波形 · 按采样数据绘制');return;}
            if((last-first)*p.frame_bytes>MAX_BYTES){setText(resolution,'概览包络 · 继续放大可查看精细采样');return;}
            const controller=new AbortController();current.rawAbort=controller;
            setText(resolution,'正在读取当前片段的精细波形…');
            try {
                const raw=await fetchPCM(current,first,last,controller);
                if(state!==current||controller.signal.aborted)return;
                current.raw=raw;setText(resolution,'精细波形 · 按采样数据绘制');schedulePaint();
            } catch(error){if(!controller.signal.aborted&&state===current)setText(resolution,error.message);}
        };
        const setView = (start,span) => {
            if(!state?.data)return;
            const d=state.data,min=d.pcm?Math.max(8/d.sample_rate,.0001):d.duration/d.count*8;
            state.span=clamp(span,Math.min(min,d.duration),d.duration);state.start=clamp(start,0,Math.max(0,d.duration-state.span));
            state.rawAbort?.abort();clearTimeout(state.refineTimer);
            state.refineTimer=setTimeout(refine,120);schedulePaint();
        };
        const zoom = (factor,anchor) => {
            if(!state?.data)return;
            const center=anchor??(audio.currentTime>=state.start&&audio.currentTime<=state.start+state.span?audio.currentTime:state.start+state.span/2);
            const fraction=clamp((center-state.start)/state.span,0,1),span=state.span*factor;
            setView(center-fraction*span,span);
        };
        const seek = (seconds) => {
            if(!state?.data)return;
            state.initialTime=clamp(seconds,0,state.data.duration);
            if(audio.readyState>=1)audio.currentTime=state.initialTime;
            paintOverlay();showMetrics(state.initialTime);
            syncTransport();
        };
        const tick = (now) => {
            if(!dialog.open||!state?.data)return;
            const position=audio.currentTime;
            if(q('.svc-focus-follow').checked&&!audio.paused&&state.span<state.data.duration&&(position<state.start||position>state.start+state.span))setView(position-state.span*.2,state.span);
            paintOverlay();
            if(now-lastMeter>120){showMetrics(position);lastMeter=now;}
            if(!audio.paused)animation=requestAnimationFrame(tick);
        };
        audio.addEventListener('loadedmetadata',()=>{if(state)seek(state.initialTime);});
        audio.addEventListener('timeupdate',()=>{if(audio.paused){paintOverlay();showMetrics(audio.currentTime);}});
        audio.addEventListener('error',()=>{if(dialog.open)setText(resolution,'音频加载失败，可关闭后重新打开音轨。');});
        dialog.addEventListener('click',(event)=>{
            const action=event.target.closest('[data-action]')?.dataset.action;
            if(action==='close'){dialog.close();return;}
            if(action==='play'){togglePlayback();return;}
            if(action==='stop'){
                audio.pause();
                if(state){
                    state.initialTime=0;
                    if(audio.readyState>=1)audio.currentTime=0;
                    if(state.data&&q('.svc-focus-follow').checked)setView(0,state.span);
                    paintOverlay();showMetrics(0);
                }
                syncTransport();return;
            }
            if(!state?.data)return;
            if(action==='in')zoom(.5);if(action==='out')zoom(2);
            if(action==='fit')setView(0,state.data.duration);
            if(action==='selection'&&state.selection){seek(state.selection[0]);setView(state.selection[0],state.selection[1]-state.selection[0]);}
            if(action==='clear'){state.selection=null;paintOverlay();}
            if(action==='left')setView(state.start-state.span*.5,state.span);
            if(action==='right')setView(state.start+state.span*.5,state.span);
        });
        gain.addEventListener('input',schedulePaint);
        pan.addEventListener('input',()=>{if(state?.data)setView(Number(pan.value)*(state.data.duration-state.span),state.span);});
        plot.addEventListener('wheel',(event)=>{
            if(!state?.data)return;event.preventDefault();
            if(event.shiftKey)setView(state.start+Math.sign(event.deltaY||event.deltaX)*state.span*.15,state.span);
            else zoom(event.deltaY>0?1.3:1/1.3,timeAt(event.clientX-plot.getBoundingClientRect().left));
        },{passive:false});
        plot.addEventListener('pointerdown',(event)=>{
            if(!state?.data||event.button!==0)return;
            const x=event.clientX-plot.getBoundingClientRect().left;
            state.drag={x,time:timeAt(x),start:state.start,span:state.span,mode:q('.svc-focus-mode').value};
            plot.setPointerCapture(event.pointerId);plot.focus();
        });
        plot.addEventListener('pointermove',(event)=>{
            if(!state?.data)return;
            const x=event.clientX-plot.getBoundingClientRect().left,seconds=timeAt(x);
            state.hover=seconds;
            setText(q('.svc-focus-cursor'),'光标 '+timeText(seconds,4));showMetrics(seconds);
            if(state.drag){
                if(state.drag.mode==='pan')setView(state.drag.start-(x-state.drag.x)/geometry().w*state.drag.span,state.drag.span);
                else state.selection=[Math.min(state.drag.time,seconds),Math.max(state.drag.time,seconds)];
            }
            paintOverlay();
        });
        const finishDrag=(event)=>{
            if(!state?.drag)return;
            const drag=state.drag;state.drag=null;
            const x=event.clientX-plot.getBoundingClientRect().left;
            if(Math.abs(x-drag.x)<4){state.selection=null;seek(drag.time);}
            if(plot.hasPointerCapture(event.pointerId))plot.releasePointerCapture(event.pointerId);
            paintOverlay();
        };
        plot.addEventListener('pointerup',finishDrag);plot.addEventListener('pointercancel',()=>{if(state)state.drag=null;});
        plot.addEventListener('pointerleave',()=>{if(state&&!state.drag){state.hover=null;paintOverlay();}});
        dialog.addEventListener('keydown',(event)=>{
            if(event.target.matches('input,select,button,a,audio'))return;
            if(event.code==='Space'){event.preventDefault();togglePlayback();}
            if(event.key==='+'||event.key==='='){event.preventDefault();zoom(.5);}
            if(event.key==='-'){event.preventDefault();zoom(2);}
            if(event.key==='0'&&state?.data)setView(0,state.data.duration);
            if(event.key==='ArrowLeft'&&state?.data){event.preventDefault();setView(state.start-state.span*.2,state.span);}
            if(event.key==='ArrowRight'&&state?.data){event.preventDefault();setView(state.start+state.span*.2,state.span);}
        });
        new ResizeObserver(()=>{if(dialog.open)schedulePaint();}).observe(plot);
        dialog.addEventListener('close',()=>{
            cancelAnimationFrame(animation);cancelAnimationFrame(drawFrame);
            hideLoading();
            if(state){
                state.rawAbort?.abort();state.abort.abort();clearTimeout(state.refineTimer);
                const position=audio.readyState>=1?audio.currentTime:state.initialTime;
                if(state.original.isConnected&&Number.isFinite(position)) {
                    if(state.original.readyState>=1)state.original.currentTime=position;
                    else state.original.dataset.svcPendingSeek=String(position);
                }
                state.button.isConnected&&state.button.focus();
            }
            audio.pause();audio.removeAttribute('src');audio.load();state=null;
        });
        // A server restart can send new player HTML into an older open tab.
        // Leave its refresh hint visible unless this viewer actually exists.
        const markViewerReady=(root)=>{
            for(const track of matching(root,'.svc-task-audio'))track.dataset.svcViewerReady='true';
        };
        markViewerReady(document);
        for(const detail of taskDetails)new MutationObserver(mutations=>{
            for(const mutation of mutations)for(const node of mutation.addedNodes)if(node.isConnected)markViewerReady(node);
        }).observe(detail,{childList:true,subtree:true});
        document.addEventListener('click',async(event)=>{
            const button=event.target.closest?.('.svc-track-open');if(!button)return;
            event.preventDefault();
            const track=button.closest('.svc-task-audio'),original=track.querySelector(taskAudioSelector),details=track.querySelector('[data-svc-wave-detail]')?.dataset.svcWaveDetail;
            if(dialog.open)dialog.close();
            const current={button,original,url:original.getAttribute('src'),initialTime:Number(original.dataset.svcPendingSeek??original.currentTime)||0,
                data:null,raw:null,span:0,start:0,selection:null,hover:null,abort:new AbortController()};
            state=current;original.pause();gain.value='0';q('.svc-focus-follow').checked=true;
            q('#svc-focus-title').textContent=track.dataset.svcTrackTitle||'单轨波形';
            const download=track.querySelector('[data-svc-download]');
            q('.svc-focus-download').href=download.href;
            q('.svc-focus-download').dataset.svcDownload=download.dataset.svcDownload;
            setText(resolution,'正在加载单轨波形…');
            showLoading('正在加载单轨波形…');
            setText(q('.svc-focus-file'),'');setText(q('.svc-focus-range'),'');setText(q('.svc-focus-cursor'),'光标 —');
            setText(q('.svc-focus-selection'),'未选择片段');setText(q('.svc-focus-peak'),'峰值 —');setText(q('.svc-focus-rms'),'RMS —');
            canvasContext(wave);canvasContext(overlay);audio.src=current.url;dialog.showModal();q('[data-action="close"]').focus();
            syncTransport();
            try {
                if(!details)throw new Error('暂时没有详细波形，请返回列表后重新选择任务；仍可在此播放音频。');
                let d=null;
                for(let n=0;n<60;n++){
                    const response=await fetch(details,{signal:current.abort.signal});
                    if(response.ok){d=await response.json();break;}
                    if(response.status!==404)throw new Error('单轨波形不可用');
                    await new Promise(resolve=>setTimeout(resolve,1000));if(current.abort.signal.aborted)return;
                }
                if(state!==current||current.abort.signal.aborted)return;
                if(!d||d.scale!=='absolute'||!Number.isFinite(d.duration)||d.duration<=0||!Number.isInteger(d.frames)||d.frames<1||!Number.isInteger(d.count)||d.count<1||d.count>8192||!Number.isInteger(d.channels)||d.channels<1||d.channels>32||!Number.isFinite(d.sample_rate)||d.sample_rate<1)throw new Error('单轨波形元数据无效');
                for(const name of ['low','high','rms'])if(!Array.isArray(d[name])||d[name].length<Math.min(d.channels,2)||d[name].slice(0,2).some(v=>!Array.isArray(v)||v.length!==d.count||!v.every(Number.isFinite)))throw new Error('单轨波形数据无效');
                if(d.pcm&&(!Number.isInteger(d.pcm.offset)||d.pcm.offset<0||![1,3].includes(d.pcm.format)||![8,16,24,32,64].includes(d.pcm.bits)||d.pcm.frame_bytes!==d.channels*d.pcm.bits/8))d.pcm=null;
                current.data=d;current.span=d.duration;
                if(audio.readyState>=1)seek(current.initialTime);
                setText(q('.svc-focus-file'),`${d.sample_rate} Hz · ${d.channels} 声道 · 整轨峰值 ${dbText(d.peak_dbfs)} · 整轨 RMS ${dbText(d.rms_dbfs)}`);
                setView(0,d.duration);showMetrics(current.initialTime);
                if(!audio.paused){cancelAnimationFrame(animation);animation=requestAnimationFrame(tick);}
            } catch(error){if(!current.abort.signal.aborted&&state===current){
                setText(resolution,error.message);
                showLoading(error.message, true);
            }}
        });
    }
"""

CSS = r"""
.svc-focus-view { position: fixed; inset: 0; width: 100vw; height: 100dvh; max-width: none; max-height: none; box-sizing: border-box; margin: 0; padding: 20px 24px; border: 0; background: var(--wb-bg, #050709); color: var(--wb-text, #f5f7fa); overflow: auto; }
.svc-focus-view[open] { display: flex; flex-direction: column; gap: 14px; }
.svc-focus-view::backdrop { background: rgba(0,0,0,.8); }
.svc-focus-loading { position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%); z-index: 2; display: flex; flex-direction: column; align-items: center; gap: 16px; width: max-content; max-width: min(420px,calc(100vw - 48px)); box-sizing: border-box; padding: 24px 32px; border: 1px solid var(--wb-line); border-radius: 10px; background: var(--wb-surface, #101722); color: var(--wb-text, #f5f7fa); box-shadow: 0 12px 40px rgba(0,0,0,.3); font-size: 14px; line-height: 1.6; text-align: center; pointer-events: none; }
.svc-focus-loading[hidden] { display: none; }
.svc-focus-spinner { width: 36px; height: 36px; flex-shrink: 0; box-sizing: border-box; border: 3px solid var(--wb-line, #2b384b); border-top-color: var(--wb-accent, #9ac2ff); border-radius: 50%; animation: svc-focus-spin .8s linear infinite; }
.svc-focus-loading[data-state="error"] .svc-focus-spinner { display: none; }
@keyframes svc-focus-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .svc-focus-spinner { animation: none; } }
.svc-focus-header { display: flex; gap: 20px; align-items: center; justify-content: space-between; }
.svc-focus-header h2 { margin: 0 0 4px; font-size: 20px; line-height: 1.4; }
.svc-focus-description, .svc-focus-help { margin: 0; color: var(--wb-muted); font-size: 12px; line-height: 1.6; }
.svc-focus-view button, .svc-focus-view select { min-height: 42px; border: 1px solid var(--wb-line); border-radius: 6px; padding: 8px 12px; background: var(--wb-surface); color: var(--wb-text); font: inherit; font-size: 13px; cursor: pointer; }
.svc-focus-view button:hover { border-color: var(--wb-accent); background: var(--wb-active); }
.svc-focus-view a { color: var(--wb-accent); }
.svc-focus-view :is(button,a,input,select):focus-visible, .svc-focus-plot:focus-visible { outline: 2px solid var(--wb-accent); outline-offset: 3px; }
.svc-focus-toolbar { position: sticky; top: 0; z-index: 3; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; padding: 12px; border: 1px solid var(--wb-line); border-radius: 8px; background: var(--wb-surface); }
.svc-focus-transport { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding-right: 12px; border-right: 1px solid var(--wb-line); }
.svc-focus-transport button { display: inline-flex; align-items: center; gap: 7px; }
.svc-focus-transport svg { width: 16px; height: 16px; fill: currentColor; }
.svc-focus-view .svc-focus-play { min-width: 88px; border-color: var(--wb-accent); color: var(--wb-accent); background: var(--wb-active); font-weight: 650; }
.svc-focus-position { min-width: 145px; color: var(--wb-muted); font: 12px/1.5 ui-monospace, monospace; font-variant-numeric: tabular-nums; }
.svc-focus-toolbar label { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; }
.svc-focus-toolbar input[type="range"] { width: 120px; }
.svc-focus-gain-label { min-width: 48px; font-variant-numeric: tabular-nums; }
.svc-focus-info, .svc-focus-readout { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 5px 18px; font-size: 12px; color: var(--wb-muted); font-variant-numeric: tabular-nums; }
.svc-focus-workspace { display: grid; grid-template-columns: minmax(0,1fr) 110px; gap: 16px; flex: 1 0 320px; min-height: 320px; }
.svc-focus-plot { position: relative; min-width: 0; min-height: 320px; border: 1px solid var(--wb-line); border-radius: 8px; overflow: hidden; touch-action: none; cursor: crosshair; background: var(--wb-surface); }
.svc-focus-plot canvas { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.svc-focus-levels { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 12px 4px; border: 1px solid var(--wb-line); border-radius: 8px; background: var(--wb-surface); font-size: 11px; font-variant-numeric: tabular-nums; }
.svc-focus-levels h3 { margin: 0; font-size: 12px; }
.svc-focus-levels p { margin: 0; text-align: center; }
.svc-focus-levels small { text-align: center; line-height: 1.5; color: var(--wb-muted); }
.svc-focus-meter-wrap { position: relative; flex: 1; width: 72px; min-height: 120px; margin: 6px 0; }
.svc-focus-meter { position: absolute; inset: 0 auto 0 0; width: 25px; border: 1px solid var(--wb-line); background: var(--wb-input); overflow: hidden; }
.svc-focus-meter i { display: block; position: absolute; inset: 0; clip-path: inset(100% 0 0); background: linear-gradient(to top,#51ae82 0%,#d8c064 80%,#d47575 98%); }
.svc-focus-meter b { position: absolute; left: 0; right: 0; bottom: 0; height: 2px; background: var(--wb-text); }
.svc-focus-meter.svc-level-over { outline: 2px solid #d47575; }
.svc-focus-meter-scale { position: absolute; inset: 0 0 0 32px; color: var(--wb-muted); font: 11px/1 ui-monospace, monospace; }
.svc-focus-meter-scale span { position: absolute; transform: translateY(-50%); }
.svc-focus-navigation { display: flex; align-items: center; gap: 10px; }
.svc-focus-navigation input { flex: 1; min-width: 60px; }
.svc-focus-range { min-width: 210px; text-align: right; font: 12px/1.5 ui-monospace, monospace; }
.svc-focus-audio { width: 100%; min-height: 44px; }
.dark .svc-focus-audio { color-scheme: dark; }
@media (max-width: 767px) {
 .svc-focus-view { padding: 12px; gap: 10px; }
 .svc-focus-header { flex-wrap: wrap; gap: 10px; }
 .svc-focus-header > div { flex: 1 1 200px; min-width: 0; }
 .svc-focus-header h2 { font-size: 17px; overflow-wrap: anywhere; }
 .svc-focus-workspace { grid-template-columns: minmax(0,1fr); }
 .svc-focus-levels { flex-direction: row; flex-wrap: wrap; justify-content: center; gap: 8px 12px; }
 .svc-focus-meter-wrap { display: none; }
 .svc-focus-navigation { flex-wrap: wrap; }
 .svc-focus-range { flex-basis: 100%; min-width: 0; text-align: left; }
}
"""
