"""Shared, offline Gradio styling for the local voice workbench.

Layout is inspired by ClearML's list/workspace/details composition; this is a
Gradio theme, not a ClearML component implementation.
"""

from html import escape

import gradio as gr
from .svc.waveform_viewer import JS as WAVEFORM_VIEWER_JS, CSS as WAVEFORM_VIEWER_CSS
from .svc.track_groups import JS as TRACK_GROUPS_JS, CSS as TRACK_GROUPS_CSS
from .svc.download_dialog import JS as DOWNLOAD_DIALOG_JS, CSS as DOWNLOAD_DIALOG_CSS


theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    font=["system-ui", "Noto Sans SC", "Microsoft YaHei", "sans-serif"],
    font_mono=["ui-monospace", "Consolas", "monospace"],
    radius_size="sm",
).set(
    body_background_fill="#f4f6f9",
    body_background_fill_dark="#050709",
    body_text_color="#172435",
    body_text_color_dark="#f5f7fa",
    body_text_color_subdued="#516176",
    body_text_color_subdued_dark="#d2d9e3",
    block_background_fill="#ffffff",
    block_background_fill_dark="#0d1117",
    block_border_color="#d6deea",
    block_border_color_dark="#2b384b",
    block_label_text_color="#516176",
    block_label_text_color_dark="#e3e8ef",
    block_info_text_color_dark="#d2d9e3",
    block_label_background_fill_dark="#0d1117",
    block_title_background_fill_dark="#0d1117",
    block_title_text_color_dark="#f5f7fa",
    background_fill_primary_dark="#0d1117",
    background_fill_secondary_dark="#080c11",
    table_text_color_dark="#f5f7fa",
    table_even_background_fill_dark="#080c11",
    table_odd_background_fill_dark="#0d1117",
    checkbox_label_text_color_dark="#e3e8ef",
    checkbox_background_color_dark="#080c11",
    checkbox_border_color_dark="#aebdce",
    checkbox_border_color_hover_dark="#f5f7fa",
    checkbox_border_color_focus_dark="#9ac2ff",
    checkbox_border_width_dark="2px",
    checkbox_background_color_selected_dark="#2161d4",
    checkbox_border_color_selected_dark="#9ac2ff",
    checkbox_label_background_fill_selected_dark="#203753",
    checkbox_label_text_color_selected_dark="#ffffff",
    input_background_fill="#f8fafc",
    input_background_fill_dark="#080c11",
    input_border_color="#9aabc1",
    input_border_color_dark="#52627a",
    input_placeholder_color="#64748b",
    input_placeholder_color_dark="#bfc9d6",
    button_primary_background_fill="#2161d4",
    button_primary_background_fill_dark="#2161d4",
    button_primary_background_fill_hover="#174cac",
    button_primary_background_fill_hover_dark="#2b6cdd",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#edf2f8",
    button_secondary_background_fill_dark="#1b2738",
    button_secondary_text_color="#263954",
    button_secondary_text_color_dark="#f0f4fa",
    border_color_primary="#d6deea",
    border_color_primary_dark="#2b384b",
)

js = """
() => {
    const query = new URLSearchParams(window.location.search);
    let saved = null;
    try { saved = localStorage.getItem('gpt-sovits-theme'); } catch (_) {}
    const requested = query.get('__theme') || saved;
    const preference = ['light', 'dark'].includes(requested) ? requested : 'dark';
    const apply = (mode) => {
        document.documentElement.classList.toggle('dark', mode === 'dark');
        document.body.classList.toggle('dark', mode === 'dark');
        const button = document.getElementById('workbench-theme');
        if (button) button.textContent = mode === 'dark' ? '切换浅色' : '切换深色';
    };
    apply(preference);
    document.querySelectorAll('[data-workbench-service-port]').forEach((link) => {
        const url = new URL(window.location.href);
        url.port = link.dataset.workbenchServicePort;
        url.pathname = '/';
        url.search = '';
        url.hash = '';
        link.href = url.toString();
    });
    const button = document.getElementById('workbench-theme');
    if (button) button.onclick = () => {
        const mode = document.body.classList.contains('dark') ? 'light' : 'dark';
        apply(mode);
        try { localStorage.setItem('gpt-sovits-theme', mode); } catch (_) {}
    };
    document.addEventListener('keydown', (event) => {
        if (event.isComposing || event.repeat) return;
        let id = null;
        const svc = document.getElementById('svc-review-accept-next');
        const reviewingSvc = svc && svc.getClientRects().length;
        if (event.altKey && event.key === 'ArrowLeft') id = reviewingSvc ? 'svc-review-prev' : 'review-prev';
        if (event.altKey && event.key === 'ArrowRight') id = reviewingSvc ? 'svc-review-next' : 'review-next';
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            id = reviewingSvc ? (event.shiftKey ? 'svc-review-reject-next' : 'svc-review-accept-next') : 'review-save-next';
        }
        if (reviewingSvc && event.altKey && event.key.toLowerCase() === 'z') id = 'svc-review-undo';
        const control = id && document.getElementById(id);
        const hiddenUndo = id === 'svc-review-undo' && reviewingSvc;
        if (control && (hiddenUndo || control.getClientRects().length) && !control.disabled) {
            event.preventDefault();
            control.click();
        }
    });
    // The two Gradio players stay independent until the user requests a switch.
    // Switch at the audible player's timestamp and attenuate only the louder file.
    let lastPreviewSide = 'b';
    const previewPlayers = () => ({
        a: document.querySelector('#svc-preview-a audio'),
        b: document.querySelector('#svc-preview-b audio'),
        status: document.getElementById('svc-ab-status')
    });
    const matchPreviewVolume = (a, b, status) => {
        const checkbox = document.querySelector('#svc-ab-match input[type="checkbox"], input#svc-ab-match');
        const enabled = checkbox ? checkbox.checked : true;
        const rmsA = Number(status?.dataset.aRms || 0);
        const rmsB = Number(status?.dataset.bRms || 0);
        const shared = enabled && rmsA > 0 && rmsB > 0 ? Math.min(rmsA, rmsB) : 0;
        a.volume = shared ? Math.min(1, shared / rmsA) : 1;
        b.volume = shared ? Math.min(1, shared / rmsB) : 1;
    };
    document.addEventListener('click', (event) => {
        if (!(event.target instanceof Element) || !event.target.closest('#svc-ab-switch')) return;
        const {a, b, status} = previewPlayers();
        if (!a || !b || !a.src || !b.src) {
            if (status) status.textContent = '请先选择两个有试听文件的检查点。';
            return;
        }
        const sourceSide = !a.paused ? 'a' : !b.paused ? 'b' : lastPreviewSide;
        const source = sourceSide === 'a' ? a : b;
        const target = sourceSide === 'a' ? b : a;
        matchPreviewVolume(a, b, status);
        const position = Number.isFinite(source.currentTime) ? source.currentTime : 0;
        source.pause();
        target.currentTime = Number.isFinite(target.duration) ? Math.min(position, Math.max(0, target.duration - .02)) : position;
        lastPreviewSide = sourceSide === 'a' ? 'b' : 'a';
        target.play().then(() => {
            if (status) status.textContent = `正在播放 ${lastPreviewSide.toUpperCase()} · ${target.currentTime.toFixed(1)} 秒`;
        }).catch(() => {
            if (status) status.textContent = '浏览器未能播放，请先点击播放器中的播放按钮。';
        });
    });
    document.addEventListener('play', (event) => {
        const {a, b, status} = previewPlayers();
        if (event.target !== a && event.target !== b) return;
        matchPreviewVolume(a, b, status);
        if (event.target === a) { b.pause(); lastPreviewSide = 'a'; }
        else { a.pause(); lastPreviewSide = 'b'; }
        if (status) status.textContent = `正在播放 ${lastPreviewSide.toUpperCase()} · 点击按钮可同位置切换`;
    }, true);
    // Render cached peak envelopes; full audio remains on-demand. The browser
    // never decodes all stems just to draw their waveforms.
    const taskAudioSelector = 'audio[data-svc-task-audio]';
    const waveformRequests = new WeakMap();
    const trackLevelData = new WeakMap();
    const levelText = (amplitude) => amplitude > 0 ? (20 * Math.log10(amplitude)).toFixed(1) : '−∞';
    const syncTrackLevel = (track, position) => {
        const d = trackLevelData.get(track)?.data;
        if (!d) return;
        const index = Math.max(0, Math.min(d.count - 1, Math.floor(position / d.duration * d.count)));
        let peak = 0, energy = 0;
        for (let c = 0; c < d.channels; c++) {
            peak = Math.max(peak, Math.abs(d.low[c][index]), Math.abs(d.high[c][index]));
            energy += d.rms[c][index] ** 2;
        }
        const rms = Math.sqrt(energy / d.channels);
        const meter = track.querySelector('.svc-track-meter');
        if (!meter) return;
        const fraction = (value) => Math.max(0, Math.min(1, (value > 0 ? 20 * Math.log10(value) + 60 : 0) / 60));
        meter.querySelector('i').style.clipPath = `inset(${(1 - fraction(peak)) * 100}% 0 0)`;
        meter.querySelector('b').style.bottom = `${fraction(rms) * 100}%`;
        meter.classList.toggle('svc-level-over', peak > 1);
        for (const [selector,value] of [['.svc-track-peak',peak],['.svc-track-rms',rms]]) {
            const output = track.querySelector(selector), text = levelText(value);
            if (output.textContent !== text) output.textContent = text;
        }
    };
    const loadTrackLevel = async (track) => {
        if (!track || trackLevelData.has(track)) return;
        const url = track.querySelector('[data-svc-wave-detail]')?.dataset.svcWaveDetail;
        if (!url) return;
        const controller = new AbortController(), entry = {controller, data:null};
        trackLevelData.set(track, entry);
        track.querySelector('.svc-track-window').textContent = '正在加载电平…';
        try {
            let d = null;
            for (let attempt = 0; attempt < 60; attempt++) {
                if (!track.isConnected || controller.signal.aborted) return;
                const response = await fetch(url, {signal:controller.signal,cache:'no-store'});
                if (response.ok) {d = await response.json(); break;}
                if (response.status !== 404) throw new Error('电平暂不可用');
                await new Promise(resolve => setTimeout(resolve,1000));
            }
            if (!track.isConnected || controller.signal.aborted) return;
            if (!d || d.scale !== 'absolute' || !(d.duration > 0) || !Number.isInteger(d.count) || d.count < 1 || d.count > 8192 ||
                !Number.isInteger(d.channels) || d.channels < 1 || d.channels > 32) throw new Error('电平数据无效');
            for (const key of ['low','high','rms']) if (!Array.isArray(d[key]) || d[key].length !== d.channels ||
                d[key].some(v => !Array.isArray(v) || v.length !== d.count || !v.every(Number.isFinite))) throw new Error('电平数据无效');
            entry.data = d;
            track.querySelector('.svc-track-window').textContent = `约 ${(d.duration / d.count * 1000).toFixed(0)} ms 窗口`;
            syncTrackLevel(track, Number(track.querySelector(taskAudioSelector).currentTime) || 0);
        } catch (_) {
            if (!controller.signal.aborted && track.isConnected) {
                trackLevelData.delete(track);
                track.querySelector('.svc-track-window').textContent = '电平暂不可用';
            }
        }
    };
    const matching = (root, selector) => root.matches?.(selector) ? [root] : root.querySelectorAll?.(selector) || [];
    const clockTime = (seconds) => {
        const value = Math.max(0, Math.floor(Number(seconds) || 0));
        return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`;
    };
    const syncTaskWaveform = (track) => {
        const audio = track?.querySelector(taskAudioSelector);
        if (!audio) return;
        const duration = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : Number(track.dataset.duration || 0);
        const position = Number(audio.dataset.svcPendingSeek ?? audio.currentTime) || 0;
        const fraction = duration > 0 ? Math.min(1, Math.max(0, position / duration)) : 0;
        const seek = track.querySelector('.svc-wave-seek');
        seek.value = String(Math.round(fraction * 10000));
        seek.disabled = duration <= 0;
        seek.setAttribute('aria-valuetext', `${clockTime(position)} / ${clockTime(duration)}`);
        track.querySelector('.svc-wave-clip').setAttribute('width', String(fraction * 768));
        const elapsed = track.querySelector('.svc-wave-elapsed');
        const total = track.querySelector('.svc-wave-duration');
        const elapsedText = clockTime(position);
        const totalText = duration > 0 ? clockTime(duration) : '--:--';
        if (elapsed.textContent !== elapsedText) elapsed.textContent = elapsedText;
        if (total.textContent !== totalText) total.textContent = totalText;
        syncTrackLevel(track, position);
    };
    const loadTaskWaveforms = (root) => {
        for (const track of matching(root, '.svc-task-audio')) {
            if (waveformRequests.has(track)) continue;
            const controller = new AbortController();
            waveformRequests.set(track, controller);
            const waveform = track.querySelector('[data-svc-waveform]');
            const status = waveform.querySelector('.svc-wave-status');
            const url = waveform.dataset.svcWaveform;
            (async () => {
                try {
                    if (!url) throw new Error('No waveform');
                    let data = null;
                    // New/old tasks can need a first-time background peak scan.
                    // Wait for its tiny JSON file, without blocking UI events.
                    for (let attempt = 0; attempt < 60; attempt++) {
                        if (controller.signal.aborted || !track.isConnected) return;
                        const response = await fetch(url, {signal: controller.signal, cache: 'no-store'});
                        if (response.ok) { data = await response.json(); break; }
                        if (response.status !== 404) throw new Error('Waveform unavailable');
                        await new Promise((resolve) => setTimeout(resolve, attempt ? 1000 : 250));
                    }
                    if (controller.signal.aborted || !track.isConnected) return;
                    if (!data || !Array.isArray(data.peaks) || !data.peaks.length || data.peaks.length > 2048 ||
                        data.scale !== 'absolute' || !data.peaks.every((peak) => Number.isFinite(peak) && peak >= 0) ||
                        !Number.isFinite(data.duration) || data.duration <= 0) throw new Error('Invalid waveform');
                    const step = 768 / data.peaks.length;
                    const path = data.peaks.map((peak, index) => {
                        const center = ((index + .5) * step).toFixed(2);
                        const height = Math.min(1, peak) * 46;
                        return `M${center},${(50 - height).toFixed(2)}V${(50 + height).toFixed(2)}`;
                    }).join('');
                    waveform.querySelectorAll('.svc-wave-base, .svc-wave-played').forEach((item) => {
                        item.setAttribute('d', path);
                        item.setAttribute('stroke-width', String(Math.min(4, Math.max(.5, step * .65))));
                    });
                    track.dataset.duration = String(data.duration);
                    if (/^[a-f0-9]{64}$/.test(data.sample_hash || '')) track.dataset.svcSampleHash = data.sample_hash;
                    scheduleAudioGroups();
                    const db = (value) => Number.isFinite(value) ? `${value.toFixed(1)} dBFS` : '−∞ dBFS';
                    const levels = track.querySelector('.svc-wave-levels');
                    if (levels) {
                        levels.textContent = `整轨峰值 ${db(data.peak_dbfs)} · RMS ${db(data.rms_dbfs)}${data.peak_dbfs > 0 ? ' · 超过 0 dBFS，播放可能失真' : ''}`;
                        levels.classList.toggle('svc-level-warning', data.peak_dbfs > 0);
                    }
                    status.hidden = true;
                    syncTaskWaveform(track);
                    if (!track.querySelector(taskAudioSelector).paused) loadTrackLevel(track);
                } catch (error) {
                    if (controller.signal.aborted || !track.isConnected) return;
                    status.textContent = '波形暂不可用，仍可播放音频';
                } finally {
                    if (track.isConnected) waveform.setAttribute('aria-busy', 'false');
                }
            })();
        }
    };
    const releaseTaskAudio = (root) => {
        for (const track of matching(root, '.svc-task-audio')) {
            waveformRequests.get(track)?.abort();
            trackLevelData.get(track)?.controller.abort();
        }
        const players = matching(root, taskAudioSelector);
        for (const audio of players) {
            audio.pause();
            audio.removeAttribute('src');
            audio.load();
        }
    };
    let levelAnimation = 0;
    const animateTrackLevel = (audio) => {
        cancelAnimationFrame(levelAnimation);
        const draw = () => {
            if (!audio.isConnected || audio.paused || audio.ended) return;
            syncTrackLevel(audio.closest('.svc-task-audio'), audio.currentTime);
            levelAnimation = requestAnimationFrame(draw);
        };
        levelAnimation = requestAnimationFrame(draw);
    };
    document.addEventListener('play', (event) => {
        if (!event.target.matches?.(taskAudioSelector)) return;
        loadTrackLevel(event.target.closest('.svc-task-audio'));
        animateTrackLevel(event.target);
        document.querySelectorAll(taskAudioSelector).forEach((audio) => {
            if (audio !== event.target) audio.pause();
        });
    }, true);
    for (const eventName of ['timeupdate', 'loadedmetadata', 'durationchange', 'seeking', 'ended']) {
        document.addEventListener(eventName, (event) => {
            const audio = event.target;
            if (!audio.matches?.(taskAudioSelector)) return;
            if (eventName === 'loadedmetadata' && audio.dataset.svcPendingSeek !== undefined) {
                const requested = Number(audio.dataset.svcPendingSeek);
                delete audio.dataset.svcPendingSeek;
                if (Number.isFinite(audio.duration)) audio.currentTime = Math.min(requested, audio.duration);
            }
            syncTaskWaveform(audio.closest('.svc-task-audio'));
        }, true);
    }
    document.addEventListener('input', (event) => {
        if (!event.target.matches?.('.svc-wave-seek')) return;
        const track = event.target.closest('.svc-task-audio');
        loadTrackLevel(track);
        const audio = track.querySelector(taskAudioSelector);
        const duration = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : Number(track.dataset.duration || 0);
        if (!(duration > 0)) return;
        const position = Number(event.target.value) / 10000 * duration;
        if (audio.readyState >= 1 && Number.isFinite(audio.duration)) {
            delete audio.dataset.svcPendingSeek;
            audio.currentTime = Math.min(position, audio.duration);
        } else {
            const loading = audio.dataset.svcPendingSeek !== undefined;
            audio.dataset.svcPendingSeek = String(position);
            if (!loading) { audio.preload = 'metadata'; audio.load(); }
        }
        syncTaskWaveform(track);
    });
    document.addEventListener('error', (event) => {
        if (!event.target.matches?.(taskAudioSelector)) return;
        const track = event.target.closest('.svc-task-audio');
        track?.querySelectorAll('.svc-audio-error, .svc-audio-reload').forEach((item) => { item.hidden = false; });
    }, true);
    document.addEventListener('click', (event) => {
        if (!(event.target instanceof Element)) return;
        const reload = event.target.closest('.svc-audio-reload');
        if (reload) {
            const track = reload.closest('.svc-task-audio');
            track.querySelectorAll('.svc-audio-error, .svc-audio-reload').forEach((item) => { item.hidden = true; });
            track.querySelector('audio').load();
            return;
        }
        const row = event.target.closest('#svc-task-table tbody tr');
        if (!row) return;
        const identity = row.querySelector('td:last-child')?.textContent.trim();
        const current = document.querySelector('[data-svc-selected-task]')?.dataset.svcSelectedTask;
        if (identity && identity !== current) {
            document.querySelectorAll(taskAudioSelector).forEach((audio) => audio.pause());
        }
    }, true);
    const taskDetails = document.querySelectorAll('#svc-task-detail, #pitch-upload-task-detail');
    for (const taskDetail of taskDetails) new MutationObserver((mutations) => {
        // Playback meters change text frequently. Recount only when players
        // or download lists change, not on every dB/time readout update.
        const groupContent = '.svc-task-audio, .svc-download-panel';
        if (mutations.some(mutation => [...mutation.addedNodes,...mutation.removedNodes].some(node =>
            node.nodeType === 1 && (node.matches(groupContent) || node.querySelector(groupContent))))) scheduleAudioGroups();
        for (const mutation of mutations) {
            for (const removed of mutation.removedNodes) {
                if (!removed.isConnected) releaseTaskAudio(removed);
            }
            for (const added of mutation.addedNodes) {
                if (added.isConnected) loadTaskWaveforms(added);
            }
        }
    }).observe(taskDetail, {childList: true, subtree: true});
    /* SVC_AUDIO_GROUPS */
    /* SVC_DOWNLOAD_DIALOG */
    loadTaskWaveforms(document);
    /* SVC_WAVEFORM_VIEWER */
    // Gradio 4's native accordion exposes its state as .open. Give it proper
    // disclosure semantics; CSS supplies a conspicuous expand/collapse label.
    let disclosureId = 0;
    const syncDisclosure = (button) => {
        const content = button.nextElementSibling;
        if (!content) return;
        if (!content.id) content.id = `wb-disclosure-${++disclosureId}`;
        button.setAttribute('aria-expanded', String(button.classList.contains('open')));
        button.setAttribute('aria-controls', content.id);
        button.querySelector('.icon')?.setAttribute('aria-hidden', 'true');
    };
    const initDisclosures = (root) => {
        for (const button of matching(root, 'button.label-wrap')) syncDisclosure(button);
    };
    const workbench = document.getElementById('workbench-nav');
    if (workbench) {
        initDisclosures(workbench);
        new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                if (mutation.type === 'attributes' && mutation.target.matches?.('button.label-wrap')) syncDisclosure(mutation.target);
                else if (mutation.type === 'childList') {
                    for (const added of mutation.addedNodes) if (added.nodeType === 1) initDisclosures(added);
                }
            }
        }).observe(workbench, {subtree: true, childList: true, attributes: true, attributeFilter: ['class']});
    }
    // The same semantic tabs become vertical navigation on desktop.
    const nav = document.querySelector('#workbench-nav > .tab-nav');
    if (nav) {
        nav.addEventListener('click', (event) => {
            if (event.target.closest('[role="tab"]')) {
                document.querySelectorAll(taskAudioSelector).forEach((audio) => audio.pause());
            }
        });
        const desktop = window.matchMedia('(min-width: 768px)');
        const orientation = () => nav.setAttribute('aria-orientation', desktop.matches ? 'vertical' : 'horizontal');
        orientation();
        desktop.addEventListener('change', orientation);
        nav.addEventListener('keydown', (event) => {
            if (!['ArrowUp', 'ArrowDown'].includes(event.key)) return;
            const tabs = [...nav.querySelectorAll('[role="tab"]')];
            const current = tabs.indexOf(document.activeElement);
            if (current < 0) return;
            event.preventDefault();
            const next = tabs[(current + (event.key === 'ArrowDown' ? 1 : -1) + tabs.length) % tabs.length];
            next.click();
            next.focus();
        });
    }
    // Keep the selected SVC task visible after Gradio refreshes the table.
    const syncSelectedTask = () => {
        const selected = document.querySelector('[data-svc-selected-task]')?.dataset.svcSelectedTask || '';
        document.querySelectorAll('#svc-task-table tbody tr').forEach((row) => {
            const identity = row.querySelector('td:last-child')?.textContent.trim() || '';
            const active = Boolean(selected) && identity === selected;
            row.classList.toggle('svc-task-selected', active);
            row.setAttribute('aria-selected', String(active));
            if (!row.hasAttribute('tabindex')) row.tabIndex = 0;
        });
    };
    let taskSelectionPending = false;
    const taskSelectionObserver = new MutationObserver(() => {
        if (taskSelectionPending) return;
        taskSelectionPending = true;
        requestAnimationFrame(() => {
            taskSelectionPending = false;
            syncSelectedTask();
        });
    });
    for (const id of ['svc-task-table', 'svc-task-selection-marker']) {
        const target = document.getElementById(id);
        if (target) taskSelectionObserver.observe(target, {childList: true, subtree: true, characterData: true});
    }
    syncSelectedTask();
    document.addEventListener('keydown', (event) => {
        const row = event.target instanceof Element ? event.target.closest('#svc-task-table tbody tr') : null;
        if (!row || !['Enter', ' '].includes(event.key)) return;
        event.preventDefault();
        row.querySelector('td')?.click();
    });
}
"""

css = """
.svc-input-invalid textarea,
.svc-input-invalid input {
    border-color: var(--error-border-color, #ef4444) !important;
    box-shadow: 0 0 0 1px var(--error-border-color, #ef4444) !important;
}

.svc-input-invalid .info {
    color: var(--error-text-color, #ef4444);
}

.gradio-container {
    --wb-bg: #f4f6f9;
    --wb-surface: #fff;
    --wb-rail: #edf1f7;
    --wb-input: #f8fafc;
    --wb-line: #d6deea;
    --wb-text: #172435;
    --wb-muted: #516176;
    --wb-accent: #215dba;
    --wb-active: #dce9fc;
    --wb-warning: #8d4b0a;
    max-width: none !important;
    padding: 0 !important;
    font-family: system-ui, "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
    background: var(--wb-bg);
    color: var(--wb-text);
    min-height: 100dvh;
}
.dark .gradio-container, .gradio-container.dark {
    --wb-bg: #050709;
    --wb-surface: #0d1117;
    --wb-rail: #080c11;
    --wb-input: #080c11;
    --wb-line: #2b384b;
    --wb-text: #f5f7fa;
    --wb-muted: #b5c1d0;
    --wb-accent: #9ac2ff;
    --wb-active: #203753;
    --wb-warning: #f2ba7d;
}
.gradio-container > .main { padding: 0 !important; }
.gradio-container > .main > .wrap > .contain > .gap { gap: 0; }
.gradio-container .prose { color: var(--wb-text); font-size: 14px; line-height: 1.7; }
.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3 {
    color: var(--wb-text); line-height: 1.35;
}
.gradio-container button { cursor: pointer; min-height: 32px; transition: background-color 140ms ease; }
.gradio-container button:disabled { cursor: not-allowed; }
/* Keep unselected radio circles and checkbox squares visible on dark surfaces. */
.dark .gradio-container input[type="radio"],
.dark .gradio-container input[type="checkbox"],
.gradio-container.dark input[type="radio"],
.gradio-container.dark input[type="checkbox"] {
    border: 2px solid #aebdce !important;
}
.dark .gradio-container input[type="radio"]:checked,
.dark .gradio-container input[type="checkbox"]:checked,
.gradio-container.dark input[type="radio"]:checked,
.gradio-container.dark input[type="checkbox"]:checked {
    border-color: #9ac2ff !important;
}
/* Action buttons must not stretch to the height or width of adjacent forms. */
.gradio-container button.primary, .gradio-container button.secondary, .gradio-container button.stop {
    flex: 0 0 auto !important;
    align-self: center !important;
    width: auto !important;
    min-width: 0 !important;
    height: 40px !important;
    min-height: 40px !important;
    padding: 0 16px !important;
    font-size: 14px !important;
    white-space: nowrap;
}
.compact-actions { justify-content: flex-start !important; align-items: center !important; gap: 8px !important; }
.workbench-panel > button { align-self: flex-start !important; }
.training-workspace { width: 100%; box-sizing: border-box; }
.integrated-synthesis { gap: 24px !important; width: 100%; }
.integrated-synthesis .inference-sidebar { flex-basis: 340px !important; }
.inference-status { display: flex; flex-wrap: wrap; gap: 8px 18px; padding: 12px 16px; border: 1px solid var(--wb-line); background: var(--wb-surface); border-radius: 6px; }
.inference-status strong { color: var(--wb-text); font-size: 14px; }
.inference-status span { color: var(--wb-muted); font-size: 13px; }
.inference-loaded-models { display: grid; gap: 8px; padding: 13px 16px; border: 1px solid var(--wb-line); border-radius: 6px; background: var(--wb-surface); }
.inference-loaded-models strong { color: var(--wb-text); font-size: 14px; }
.inference-loaded-models p { margin: 0; color: var(--wb-muted); font-size: 13px; }
.inference-model-line { display: grid; grid-template-columns: 58px minmax(0, 1fr); align-items: baseline; gap: 10px; font-size: 13px; }
.inference-model-line span { color: var(--wb-muted); font-weight: 650; }
.inference-model-line code { color: var(--wb-text); white-space: normal; overflow-wrap: anywhere; font-size: 13px; }
.inference-loaded-models .inference-model-pending { color: var(--wb-warning); }
#tts-result-audio .controls button.playback.icon { display: none !important; }
.tts-history-picker { align-items: end !important; gap: 10px !important; }
.tts-history-picker > .form { min-width: 0; }
.tts-history-card { border: 1px solid var(--wb-line); border-radius: 7px; padding: 16px; background: var(--wb-surface); color: var(--wb-text); }
.tts-history-heading { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 4px 16px; margin-bottom: 12px; }
.tts-history-heading strong { font-size: 14px; }
.tts-history-heading span, .tts-history-text span, .tts-history-settings dt { color: var(--wb-muted); font-size: 12px; }
.tts-history-text p { margin: 5px 0 14px; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.6; }
.tts-history-settings { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 20px; margin: 0; }
.tts-history-settings > div { min-width: 0; }
.tts-history-settings dd { margin: 3px 0 0; font-size: 13px; overflow-wrap: anywhere; }
.tts-history-model { grid-column: 1 / -1; }
@media (max-width: 700px) { .tts-history-settings { grid-template-columns: minmax(0, 1fr); } }
.feedback-error { border-left: 3px solid var(--wb-warning) !important; }
.feedback-error strong { color: var(--wb-warning) !important; }
.feedback-error p, .feedback-error span { overflow-wrap: anywhere; }
.workspace-hint { color: var(--wb-muted); font-size: 12px; }
.workspace-hint .prose { color: var(--wb-muted); font-size: 12px; }
.workspace-note .prose { color: var(--wb-muted); font-size: 13px; }

.gradio-container :focus-visible { outline: 2px solid var(--wb-accent) !important; outline-offset: 3px; }
.gradio-container input, .gradio-container textarea { font-size: 14px !important; line-height: 1.6; }
/* Gradio positions the single-select arrow over the full-width input. Keep
   selected labels inside their own space in every workspace. */
.gradio-container .wrap-inner > .secondary-wrap { min-width: 0 !important; }
.gradio-container .secondary-wrap > input[role="listbox"] {
    box-sizing: border-box;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.gradio-container .secondary-wrap > input[role="listbox"]:has(+ div.icon-wrap) {
    padding-right: calc(2 * var(--size-5) + 2 * var(--spacing-sm)) !important;
}
.gradio-container .wrap-inner:has(input[role="listbox"]) > .token { max-width: 100%; }
.gradio-container ::selection { background: #3567a7; color: white; }
.gradio-container .block { box-shadow: none !important; }
#app-header { border-bottom: 1px solid var(--wb-line); padding: 12px 24px; background: var(--wb-surface); }
.header-main { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px; }
.header-brand { display: flex; align-items: baseline; gap: 14px; }
.header-brand h1 { margin: 0; font-size: 21px; font-weight: 700; letter-spacing: -.4px; color: var(--wb-text); }
.header-brand span { font-size: 13px; color: var(--wb-muted); }
.header-links { display: flex; align-items: center; flex-wrap: wrap; gap: 16px; }
.header-links a, .header-links button { font: inherit; font-size: 13px; color: var(--wb-muted); }
.header-links a { text-decoration: none; }
.header-links a:hover { color: var(--wb-accent); }
.header-links button { border: 1px solid var(--wb-line); padding: 6px 12px; border-radius: 5px; background: var(--wb-rail); }
#app-header details { color: var(--wb-muted); font-size: 12px; margin-top: 8px; }
#app-header summary { cursor: pointer; }
#app-header details p { max-width: 110ch; margin: 6px 0 0; line-height: 1.5; }
.workflow-intro.block { padding: 9px 24px !important; border-bottom: 1px solid var(--wb-line); border-radius: 0; background: var(--wb-rail); }
.workflow-intro .workflow-intro { padding: 0 !important; border: 0 !important; background: transparent; font-size: 13px; }
.workflow-intro p { margin: 0; color: var(--wb-muted); }
#workbench-nav { display: grid; grid-template-columns: 184px minmax(0, 1fr); gap: 0 !important; align-items: start; }
#workbench-nav > .tab-nav {
    grid-column: 1; grid-row: 1;
    display: flex; flex-direction: column; align-items: stretch;
    position: sticky; top: 0; gap: 6px;
    min-height: calc(100dvh - 112px);
    padding: 24px 12px;
    background: var(--wb-rail);
    border: 0; border-right: 1px solid var(--wb-line);
}
#workbench-nav > .tab-nav button {
    width: 100%; text-align: left; white-space: normal;
    padding: 12px 14px; border: 1px solid transparent !important;
    border-radius: 6px; color: var(--wb-muted); font-size: 14px; font-weight: 600;
}
#workbench-nav > .tab-nav button[aria-selected="true"] {
    background: var(--wb-active); color: var(--wb-accent); border-color: var(--wb-line) !important;
}
#workbench-nav > .tabitem {
    grid-column: 2; grid-row: 1; min-width: 0; width: 100%; max-width: 1536px;
    justify-self: center; box-sizing: border-box; padding: 28px 32px 40px !important; border: 0 !important;
}
.service-link { margin: 0 0 16px; color: var(--wb-muted); font-size: 14px; line-height: 1.6; }
.service-link a { color: var(--wb-accent); text-decoration: underline; }
.workspace-heading { margin: 0 0 20px; }
.workspace-heading h2 { margin: 0 0 8px; font-size: 26px; font-weight: 650; letter-spacing: -.4px; color: var(--wb-text); }
.workspace-heading p { margin: 0; color: var(--wb-muted); font-size: 14px; line-height: 1.6; }
.panel-heading { margin: 0; font-size: 16px; font-weight: 650; color: var(--wb-text); }
.workbench-panel .block:has(.panel-heading) { background: transparent !important; border: 0 !important; padding: 0 !important; }
.workbench-panel { gap: 16px !important; border: 1px solid var(--wb-line) !important; border-radius: 8px; padding: 20px; background: var(--wb-surface); }
.workbench-panel .prose h3 { margin: 0 0 4px; font-size: 17px; font-weight: 650; }
/* Accordion headers look and behave like full-width disclosure controls. */
#workbench-nav .block:has(> button.label-wrap) { border: 1px solid var(--wb-line) !important; border-radius: 8px !important; padding: 6px !important; background: var(--wb-rail) !important; }
#workbench-nav button.label-wrap { display: grid !important; grid-template-columns: minmax(0, 1fr) auto 20px; align-items: center; gap: 12px; min-height: 48px; width: 100%; padding: 12px 14px !important; border: 1px solid transparent !important; border-radius: 6px !important; background: var(--wb-surface) !important; color: var(--wb-text) !important; text-align: left; cursor: pointer; transition: border-color 120ms, background-color 120ms; }
#workbench-nav button.label-wrap > span:first-child { grid-column: 1; grid-row: 1; min-width: 0; font-size: 14px; font-weight: 650; line-height: 1.5; overflow-wrap: anywhere; }
#workbench-nav button.label-wrap::after { content: '点击展开'; grid-column: 2; grid-row: 1; color: var(--wb-accent); font-size: 12px; font-weight: 500; white-space: nowrap; }
#workbench-nav button.label-wrap.open::after { content: '收起'; }
#workbench-nav button.label-wrap .icon { grid-column: 3; grid-row: 1; color: var(--wb-accent); font-size: 14px; transform: rotate(-90deg) !important; }
#workbench-nav button.label-wrap.open .icon { transform: rotate(0) !important; }
#workbench-nav button.label-wrap.open { border-color: var(--wb-line) !important; margin-bottom: 0 !important; }
#workbench-nav button.label-wrap:hover { border-color: var(--wb-accent) !important; background: var(--wb-active) !important; }
#workbench-nav button.label-wrap:focus-visible { outline: 2px solid var(--wb-accent); outline-offset: 2px; }
#workbench-nav button.label-wrap + div { padding: 14px 8px 8px; }
/* Gradio Groups paint their gaps with the border color by default. */
#svc-task-detail > .styler, #svc-task-detail .gr-group, #svc-task-detail .gr-group > .styler { background: transparent !important; border: 0 !important; gap: 16px !important; --layout-gap: 16px; --form-gap-width: 16px; --block-radius: 8px !important; }
#svc-task-detail .svc-pitch-report { padding: 4px 6px; line-height: 1.7; }
#svc-task-detail .svc-pitch-report + .svc-pitch-report { margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--wb-line); }
.svc-wave-times { font-variant-numeric: tabular-nums; }
.dataset-layout { gap: 24px !important; }
.workspace-tabs > .tab-nav { margin-bottom: 20px; gap: 8px; }
.workspace-tabs > .tab-nav button { padding: 10px 16px; font-size: 14px; }
.workspace-tabs > .tabitem { padding: 0 !important; border: 0 !important; }
.svc-identity { align-items: center !important; gap: 16px !important; }
.svc-identity > .form, .svc-identity > .styler { min-width: 0; }
.svc-task-toolbar { align-items: center !important; gap: 12px !important; }
.svc-task-toolbar > .form { flex: 1 1 320px; }
.svc-task-table { overflow-x: auto; }
.svc-task-table table { min-width: 1260px; }
.svc-task-table :is(th,td):nth-child(4) { min-width: 180px; }
.svc-task-table :is(th,td):nth-child(5) { min-width: 150px; }
.svc-task-table tbody tr { cursor: pointer; }
.svc-task-table tbody tr:hover { background: color-mix(in srgb, var(--wb-accent, #2161d4) 11%, transparent) !important; }
.svc-task-table tbody tr.svc-task-selected td { background: var(--wb-active) !important; box-shadow: inset 0 2px var(--wb-accent), inset 0 -2px var(--wb-accent); }
.svc-task-table tbody tr.svc-task-selected td:first-child { box-shadow: inset 0 2px var(--wb-accent), inset 0 -2px var(--wb-accent), inset 2px 0 var(--wb-accent); }
.svc-task-table tbody tr.svc-task-selected td:last-child { box-shadow: inset 0 2px var(--wb-accent), inset 0 -2px var(--wb-accent), inset -2px 0 var(--wb-accent); }
.svc-task-table tbody tr.svc-task-row-attention { animation: svc-task-row-pulse 0.75s ease-in-out 2; }
#svc-task-selection-marker { display: none !important; }
.svc-task-pages { align-items: center !important; gap: 12px !important; }
.svc-task-pages .prose { align-self: center; }
.svc-task-detail { margin-top: 16px; }
.svc-task-audio { padding: 12px 0; }
.svc-track-heading { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 8px; }
.svc-track-open { display: inline-flex; flex-wrap: wrap; align-items: baseline; gap: 5px; border: 0; padding: 4px 0; background: transparent; color: var(--wb-text); text-align: left; font: inherit; font-weight: 650; cursor: pointer; min-height: 40px; }
.svc-track-open span { color: var(--wb-accent); font-size: 12px; font-weight: 500; }
.svc-track-open:hover { color: var(--wb-accent); text-decoration: underline; }
.svc-track-open:focus-visible { outline: 2px solid var(--wb-accent); outline-offset: 3px; }
.svc-wave-shell { display: grid; grid-template-columns: minmax(0, 1fr) 156px; gap: 14px; align-items: stretch; }
.svc-track-level { display: flex; gap: 10px; min-width: 0; padding: 8px; border: 1px solid var(--wb-line); border-radius: 6px; background: var(--wb-input); }
.svc-track-meter { position: relative; flex: 0 0 16px; overflow: hidden; border: 1px solid var(--wb-line); background: repeating-linear-gradient(to top, transparent 0 19%, var(--wb-line) 19% 20%); }
.svc-track-meter i { position: absolute; inset: 0; clip-path: inset(100% 0 0); background: linear-gradient(to top,#51ae82 0%,#d8c064 80%,#d47575 98%); }
.svc-track-meter b { position: absolute; left: 0; right: 0; height: 2px; bottom: 0; background: var(--wb-text); }
.svc-track-meter.svc-level-over { outline: 1px solid #d47575; }
.svc-track-level-text { display: flex; flex-direction: column; justify-content: space-between; gap: 3px; font: 12px/1.4 ui-monospace, monospace; font-variant-numeric: tabular-nums; white-space: nowrap; }
.svc-track-level-text span { display: flex; justify-content: space-between; gap: 8px; }
.svc-track-level-text small { color: var(--wb-muted); font-size: 10px; line-height: 1.4; }
.svc-wave-levels.svc-level-warning { color: var(--wb-warning, #dca367); }
.svc-track-level-text output { color: var(--wb-text); }
.svc-amplitude-legend { padding: 10px 14px; margin: 8px 0; border: 1px solid var(--wb-line); border-radius: 6px; color: var(--wb-muted); font-size: 12px; line-height: 1.7; }
.svc-amplitude-legend strong { color: var(--wb-text); }
.svc-audio-upgrade { margin: 4px 0; color: var(--wb-warning, #dca367); font-size: 12px; }
.svc-task-audio[data-svc-viewer-ready] .svc-audio-upgrade { display: none; }
.svc-wave-metrics { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px 16px; margin: 5px 0; color: var(--wb-muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.svc-task-audio audio { display: block; width: 100%; min-height: 44px; }
.svc-waveform { position: relative; min-width: 0; height: 112px; border-radius: 6px; background: var(--wb-input); }
.svc-waveform svg { display: block; width: 100%; height: 100%; }
.svc-waveform path { fill: none; stroke-linecap: round; }
.svc-wave-grid { stroke: var(--wb-line); stroke-width: .7; }
.svc-wave-base { stroke: var(--wb-muted); opacity: .65; }
.svc-wave-played { stroke: var(--wb-accent); }
.svc-wave-seek { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: pointer; }
.svc-wave-seek:disabled { cursor: default; }
.svc-waveform:focus-within { outline: 2px solid var(--wb-accent); outline-offset: 3px; }
.svc-wave-status { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: var(--wb-muted); pointer-events: none; }
.svc-wave-status[hidden] { display: none; }
.svc-wave-times { display: flex; justify-content: space-between; color: var(--wb-muted); font-variant-numeric: tabular-nums; margin: 4px 0 8px; }
.svc-task-audio audio:focus-visible, .svc-task-audio a:focus-visible, .svc-audio-reload:focus-visible { outline: 2px solid var(--wb-accent); outline-offset: 3px; }
.dark .svc-task-audio audio { color-scheme: dark; }
.svc-audio-error { color: var(--wb-muted); }
.svc-audio-reload { margin: 8px 0; padding: 8px 12px; border: 1px solid var(--wb-line); border-radius: 6px; }
.svc-track-downloads { display: flex; flex-wrap: wrap; gap: 8px 24px; padding-left: 20px; }
.svc-task-attention { animation: svc-task-pulse 0.75s ease-in-out 2; }
@keyframes svc-task-pulse {
    0%, 100% { border-color: var(--wb-line); box-shadow: none; }
    50% { border-color: #6da8ff; box-shadow: 0 0 0 5px rgba(80, 145, 255, .28); }
}
@keyframes svc-task-row-pulse {
    0%, 100% { outline: 0 solid transparent; }
    50% { outline: 3px solid #6da8ff; outline-offset: -3px; }
}
#svc-submit-dialog {
    position: fixed !important;
    inset: auto !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%);
    z-index: 80;
    width: min(480px, calc(100vw - 32px));
    padding: 24px;
    border: 1px solid #526b8f;
    border-radius: 12px;
    background: var(--wb-surface);
    box-shadow: 0 0 0 100vmax rgba(2, 6, 12, .76), 0 24px 64px rgba(0, 0, 0, .35);
}
.svc-model-selector { gap: 16px !important; }
.svc-layout { gap: 24px !important; align-items: start !important; }
.svc-source, .svc-review { min-width: 0 !important; }
.svc-source { flex: 0 0 340px !important; }
.svc-review { flex: 1 1 440px !important; }
.svc-review .review-actions { margin-top: 4px; }
.svc-review .prose p { overflow-wrap: anywhere; }
.svc-conversion-layout { display: grid !important; grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr); gap: 24px !important; align-items: start !important; }
.svc-conversion-input, .svc-conversion-settings { min-width: 0 !important; width: 100% !important; box-sizing: border-box; }
.svc-conversion-layout .prose p { overflow-wrap: anywhere; }
@media (max-width: 1099px) {
    .svc-conversion-layout { grid-template-columns: minmax(0, 1fr); }
}
.svc-preview-comparison { flex-wrap: wrap !important; gap: 12px !important; }
.svc-preview-comparison > * { flex: 1 1 220px !important; min-width: 0 !important; }
.svc-source-distribution { max-height: 280px; overflow-y: auto; font-size: 13px; }
.svc-source-distribution table { width: 100%; table-layout: fixed; border-collapse: collapse; }
.svc-source-distribution caption { text-align: left; color: var(--wb-muted); padding-bottom: 8px; }
.svc-source-distribution th, .svc-source-distribution td { padding: 8px 4px; border-bottom: 1px solid var(--wb-line); text-align: left; overflow-wrap: anywhere; vertical-align: top; }
.svc-source-distribution th:first-child { width: 34%; }
.svc-source-distribution small { display: block; color: var(--wb-muted); }
.svc-source-bar { display: flex; height: 5px; min-width: 1px; margin-top: 6px; border-radius: 3px; overflow: hidden; }
.svc-train-share { background: var(--wb-accent); }
.svc-val-share { background: var(--wb-warning); }
.gradio-container .tabitem > .gap { gap: 16px; }
.gradio-container input[type="number"], .review-table td { font-variant-numeric: tabular-nums; }
.task-rail { flex: 0 0 230px !important; min-width: 0 !important; }
.task-rail .task-list { max-height: 620px; overflow: auto; }
.task-list .wrap { flex-direction: column !important; gap: 7px !important; }
.task-list label { width: 100%; min-height: 60px; padding: 12px !important; border: 1px solid var(--wb-line) !important; border-radius: 5px !important; background: var(--wb-input) !important; color: var(--wb-text) !important; white-space: normal; overflow-wrap: anywhere; }
.task-list label.selected { background: var(--wb-active) !important; }
.task-list input[type="radio"] { flex-shrink: 0; }
.dataset-main { min-width: 0 !important; }
.task-status { padding: 13px 16px; border: 1px solid var(--wb-line); border-radius: 7px; background: var(--wb-surface); }
.task-status strong { color: var(--wb-text); font-size: 16px; }
.task-status p { margin: 5px 0 0; color: var(--wb-muted); font-size: 13px; }
.task-status .status-label { display: inline-block; margin-left: 12px; color: var(--wb-accent); font-size: 12px; font-weight: 650; }
.task-status progress { display: block; width: 100%; height: 5px; margin-top: 10px; accent-color: #76a8ef; }
.review-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid var(--wb-line); border-radius: 6px; background: var(--wb-surface); }
.review-metrics div { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 10px 14px; border-right: 1px solid var(--wb-line); }
.review-metrics div:last-child { border-right: 0; }
.review-metrics span { display: block; color: var(--wb-muted); font-size: 12px; }
.review-metrics strong { display: block; margin: 0; color: var(--wb-text); font-size: 18px; line-height: 1.3; font-weight: 650; }
.clip-inspector { flex: 0 0 230px !important; min-width: 0 !important; }
.clip-details { display: grid; grid-template-columns: 65px minmax(0, 1fr); gap: 10px; margin: 6px 0; color: var(--wb-muted); font-size: 13px; }
.clip-details dt { margin: 0; font-size: 12px; }
.clip-details dd { margin: 0; color: var(--wb-text); overflow-wrap: anywhere; }
.quality-note { border-left: 3px solid var(--wb-line); padding: 6px 10px; margin-top: 10px; color: var(--wb-muted); font-size: 13px; line-height: 1.6; }
.quality-note.has-flags { border-color: var(--wb-warning); color: var(--wb-warning); }
.review-controls { align-items: end !important; }
.review-controls > button { align-self: end !important; }
.empty-workspace { padding: 24px; color: var(--wb-muted); text-align: left; border: 1px dashed var(--wb-line); border-radius: 6px; line-height: 1.7; }
.empty-workspace strong { display: block; color: var(--wb-text); margin-bottom: 6px; }
.review-table { overflow: hidden; }
@media (max-width: 767px) {
    .review-table .table-wrap { overflow-x: auto !important; }
    .review-table table { min-width: 720px; }
}
.review-actions { padding-top: 12px; border-top: 1px solid var(--wb-line); }
.process-log textarea { font-family: ui-monospace, Consolas, monospace !important; font-size: 12px !important; }
.standalone-heading { margin: 18px 24px !important; }
.standalone-panel { margin: 18px 24px !important; padding: 18px !important; }
.synthesis-layout { padding: 0 24px 24px; gap: 20px !important; }
.inference-sidebar { flex: 0 0 310px !important; min-width: 0 !important; }
.inference-main { min-width: 0 !important; }
footer { display: none !important; }
@media (max-width: 1199px) {
    #workbench-nav > .tabitem { padding: 24px !important; }
    .svc-source { flex: 1 1 280px !important; }
    .svc-review { flex: 2 1 400px !important; }
    .dataset-layout { flex-wrap: wrap !important; }
    .task-rail { flex: 1 1 100% !important; }
    .task-rail .task-list { max-height: 180px; }
    .clip-inspector { flex: 1 1 230px !important; }
}
@media (max-width: 767px) {
    .gradio-container button.primary, .gradio-container button.secondary, .gradio-container button.stop { height: 44px !important; min-height: 44px !important; }
    #app-header { padding: 14px 16px; }
    .header-brand { flex-wrap: wrap; gap: 5px 14px; }
    .header-links { gap: 12px; }
    .workflow-intro { padding: 10px 16px !important; }
    #workbench-nav { display: block; }
    #workbench-nav > .tab-nav { position: static; flex-direction: row; flex-wrap: wrap; min-height: 0; padding: 10px; border-right: 0; border-bottom: 1px solid var(--wb-line); }
    #workbench-nav > .tab-nav button { width: auto; flex: 1 1 120px; text-align: center; padding: 10px; }
    #workbench-nav > .tabitem { padding: 14px !important; }
    .workbench-panel { padding: 16px; gap: 14px !important; }
    .svc-source, .svc-review { flex: 1 1 100% !important; }
    .svc-layout, .svc-conversion-layout, .integrated-synthesis, .dataset-layout { gap: 16px !important; }
    .workspace-tabs > .tab-nav { flex-wrap: wrap; margin-bottom: 16px; }
    .workspace-tabs > .tab-nav button { flex: 1 1 auto; min-height: 44px; padding: 8px; }
    .review-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .review-metrics div:nth-child(2) { border-right: 0; }
    .review-metrics div:nth-child(-n+2) { border-bottom: 1px solid var(--wb-line); }
    .task-rail, .clip-inspector, .inference-sidebar { flex: 1 1 100% !important; }
    .workspace-heading h2 { font-size: 21px; }
    .standalone-heading, .standalone-panel { margin: 14px !important; }
    .synthesis-layout { padding: 0 14px 20px; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
}
"""


js = js.replace('/* SVC_WAVEFORM_VIEWER */', WAVEFORM_VIEWER_JS)
js = js.replace('/* SVC_AUDIO_GROUPS */', TRACK_GROUPS_JS)
js = js.replace('/* SVC_DOWNLOAD_DIALOG */', DOWNLOAD_DIALOG_JS)
css += WAVEFORM_VIEWER_CSS
css += TRACK_GROUPS_CSS
css += DOWNLOAD_DIALOG_CSS


def workspace_header(title, description):
    return f'<div class="workspace-heading"><h2>{escape(title)}</h2><p>{escape(description)}</p></div>'


top_html = """
<header id="app-header">
    <div class="header-main">
        <div class="header-brand"><h1>GPT-SoVITS</h1><span>本地语音工作台</span></div>
        <nav class="header-links" aria-label="工作台资源">
            <a href="https://github.com/RVC-Boss/GPT-SoVITS" target="_blank" rel="noopener noreferrer">GitHub</a>
            <a href="https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e" target="_blank" rel="noopener noreferrer">使用文档</a>
            <button type="button" id="workbench-theme">切换浅色</button>
        </nav>
    </div>
    <details><summary>使用与许可</summary><p>{}</p></details>
</header>
"""
