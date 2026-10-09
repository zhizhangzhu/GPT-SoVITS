"""Shared, offline Gradio styling for the local voice workbench.

Layout is inspired by ClearML's list/workspace/details composition; this is a
Gradio theme, not a ClearML component implementation.
"""

from html import escape

import gradio as gr


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
    const preference = query.get('__theme') || saved || 'dark';
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
        let id = null;
        if (event.altKey && event.key === 'ArrowLeft') id = 'review-prev';
        if (event.altKey && event.key === 'ArrowRight') id = 'review-next';
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') id = 'review-save-next';
        const control = id && document.getElementById(id);
        if (control && control.getClientRects().length && !control.disabled) {
            event.preventDefault();
            control.click();
        }
    });
    // The same semantic tabs become vertical navigation on desktop.
    const nav = document.querySelector('#workbench-nav > .tab-nav');
    if (nav) {
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
}
"""

css = """
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
    --wb-muted: #d2d9e3;
    --wb-accent: #9ac2ff;
    --wb-active: #203753;
    --wb-warning: #f2ba7d;
}
.gradio-container > .main { padding: 0 !important; }
.gradio-container > .main > .wrap > .contain > .gap { gap: 0; }
.gradio-container .prose { color: var(--wb-text); font-size: 14px; line-height: 1.65; }
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
    height: 36px !important;
    min-height: 36px !important;
    padding: 0 14px !important;
    font-size: 13px !important;
    white-space: nowrap;
}
.compact-actions { justify-content: flex-start !important; align-items: center !important; gap: 8px !important; }
.workbench-panel > button { align-self: flex-start !important; }
.training-workspace { max-width: 1800px; width: 100%; box-sizing: border-box; }
.integrated-synthesis { gap: 18px !important; max-width: 1760px; width: 100%; }
.integrated-synthesis .inference-sidebar { flex-basis: 340px !important; }
.inference-status { display: flex; flex-wrap: wrap; gap: 8px 18px; padding: 12px 16px; border: 1px solid var(--wb-line); background: var(--wb-surface); border-radius: 6px; }
.inference-status strong { color: var(--wb-text); font-size: 14px; }
.inference-status span { color: var(--wb-muted); font-size: 13px; }
.workspace-hint { color: var(--wb-muted); font-size: 12px; }

.gradio-container :focus-visible { outline: 2px solid var(--wb-accent) !important; outline-offset: 3px; }
.gradio-container input, .gradio-container textarea { font-size: 14px !important; line-height: 1.6; }
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
#workbench-nav { display: grid; grid-template-columns: 178px minmax(0, 1fr); gap: 0 !important; align-items: start; }
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
#workbench-nav > .tabitem { grid-column: 2; grid-row: 1; min-width: 0; padding: 22px !important; border: 0 !important; }
.service-link { margin: 0 0 16px; color: var(--wb-muted); font-size: 14px; line-height: 1.6; }
.service-link a { color: var(--wb-accent); text-decoration: underline; }
.workspace-heading { margin: 0 0 14px; }
.workspace-heading h2 { margin: 0 0 6px; font-size: 23px; font-weight: 650; color: var(--wb-text); }
.workspace-heading p { margin: 0; color: var(--wb-muted); font-size: 14px; line-height: 1.6; }
.panel-heading { margin: 0; font-size: 16px; font-weight: 650; color: var(--wb-text); }
.workbench-panel .block:has(.panel-heading) { background: transparent !important; border: 0 !important; padding: 0 !important; }
.workbench-panel { gap: 12px !important; border: 1px solid var(--wb-line) !important; border-radius: 7px; padding: 16px; background: var(--wb-surface); }
.dataset-layout { gap: 18px !important; }
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
