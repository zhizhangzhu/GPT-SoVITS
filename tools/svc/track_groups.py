"""Progressive disclosure and exact sample deduplication for task audio."""

JS = r"""
    let audioGroupsFrame = 0;
    const refreshAudioGroups = () => {
        for (const root of taskDetails) {
            const tracks = Array.from(root.querySelectorAll('.svc-task-audio'));
            // Keep an audible player visible if matching metadata arrives
            // during playback. This never replaces a player's DOM or source.
            const playing = track => !track.querySelector(taskAudioSelector)?.paused;
            const ordered = [...tracks.filter(playing), ...tracks.filter(track => !playing(track))];
            const parents = ordered.map((_,i) => i), identities = new Map();
            const find = i => {
                while (parents[i] !== i) { parents[i] = parents[parents[i]]; i = parents[i]; }
                return i;
            };
            ordered.forEach((track,i) => {
                const source = track.querySelector(taskAudioSelector)?.getAttribute('src');
                const hash = track.dataset.svcSampleHash;
                for (const identity of [source ? 'url:'+source : '', hash ? 'samples:'+hash : '']) {
                    if (!identity) continue;
                    if (identities.has(identity)) {
                        const a = find(i), b = find(identities.get(identity));
                        parents[Math.max(a,b)] = Math.min(a,b);
                    } else identities.set(identity,i);
                }
            });
            const aliases = new Map(), duplicateKeys = new Set();
            ordered.forEach((track,i) => {
                const canonical = ordered[find(i)], duplicate = canonical !== track;
                const wrapper = track.closest('.svc-result-audio');
                if (duplicate) {
                    const key = canonical.dataset.svcTrackKey;
                    track.dataset.svcDuplicateOf = key;
                    if (wrapper) wrapper.dataset.svcDuplicateOf = key;
                    duplicateKeys.add(track.dataset.svcTrackKey);
                    if (!aliases.has(canonical)) aliases.set(canonical,[]);
                    aliases.get(canonical).push(track.dataset.svcTrackTitle);
                } else {
                    track.removeAttribute('data-svc-duplicate-of');
                    wrapper?.removeAttribute('data-svc-duplicate-of');
                }
            });
            for (const track of tracks) {
                const note = track.querySelector('.svc-track-aliases');
                if (!note) continue;
                const names = [...new Set(aliases.get(track) || [])];
                const text = names.length ? '同一音频的其他名称：'+names.join('、') : '';
                if (note.textContent !== text) note.textContent = text;
                note.hidden = !names.length;
            }
            for (const group of root.querySelectorAll('.svc-track-group')) {
                const count = Array.from(group.querySelectorAll('.svc-task-audio')).filter(track => !track.hasAttribute('data-svc-duplicate-of')).length;
                if (group.dataset.svcGroupCount !== String(count)) group.dataset.svcGroupCount = String(count);
                const header = group.querySelector('button.label-wrap');
                if (!header) continue;
                let badge = header.querySelector('.svc-group-count');
                if (!badge) {
                    badge = document.createElement('span'); badge.className = 'svc-group-count';
                    header.insertBefore(badge,header.querySelector('.icon'));
                }
                const text = count+' 条音轨';
                if (badge.textContent !== text) badge.textContent = text;
            }
            const summary = root.querySelector('.svc-audio-dedup-summary');
            if (summary) {
                const text = duplicateKeys.size ? `已合并 ${duplicateKeys.size} 处相同音频的重复展示，其他名称列在对应音轨下。` : '';
                if (summary.textContent !== text) summary.textContent = text;
                summary.hidden = !duplicateKeys.size;
            }
            for (const item of root.querySelectorAll('[data-svc-download-key]')) item.hidden = duplicateKeys.has(item.dataset.svcDownloadKey);
            for (const group of root.querySelectorAll('.svc-download-group')) {
                group.hidden = !Array.from(group.querySelectorAll('[data-svc-download-key]')).some(item => !item.hidden);
            }
            const count = root.querySelector('.svc-download-count');
            if (count) {
                const text = (tracks.length-duplicateKeys.size)+' 条';
                if (count.textContent !== text) count.textContent = text;
            }
        }
    };
    const scheduleAudioGroups = () => {
        if (audioGroupsFrame) return;
        audioGroupsFrame = requestAnimationFrame(() => { audioGroupsFrame = 0; refreshAudioGroups(); });
    };
    scheduleAudioGroups();
"""

CSS = r"""
.svc-track-group[data-svc-group-count="0"] { display: none !important; }
.svc-track-group-main { margin-top: 12px; }
.svc-track-group-sub { margin-top: 8px; }
.svc-group-count { flex-shrink: 0; margin-left: 10px; color: var(--wb-muted); font-size: 12px; font-weight: 400; }
#workbench-nav .svc-track-group > button.label-wrap { grid-template-columns: minmax(0,1fr) auto auto 20px; }
#workbench-nav .svc-track-group > button.label-wrap > .svc-group-count { grid-column: 2; grid-row: 1; margin-left: 0; }
#workbench-nav .svc-track-group > button.label-wrap::after { grid-column: 3; }
#workbench-nav .svc-track-group > button.label-wrap .icon { grid-column: 4; }
.svc-task-audio[data-svc-duplicate-of], .svc-result-audio[data-svc-duplicate-of] { display: none !important; }
.svc-track-aliases { margin: 0 0 10px; color: var(--wb-muted); font-size: 12px; line-height: 1.7; }
.svc-audio-dedup-summary { padding: 8px 0; color: var(--wb-muted); font-size: 12px; line-height: 1.7; }
.svc-track-aliases[hidden], .svc-audio-dedup-summary[hidden], .svc-download-panel [hidden] { display: none !important; }
.svc-download-panel { margin-top: 16px; padding: 12px 16px; border: 1px solid var(--wb-line); border-radius: 8px; background: var(--wb-surface); }
.svc-download-panel > summary { cursor: pointer; min-height: 32px; color: var(--wb-text); font-size: 14px; font-weight: 600; line-height: 2; }
.svc-download-panel > summary::marker { color: var(--wb-accent); }
.svc-download-panel > summary:focus-visible { outline: 2px solid var(--wb-accent); outline-offset: 4px; }
.svc-download-count { margin-left: 8px; color: var(--wb-muted); font-size: 12px; font-weight: 400; }
.svc-download-group { padding-top: 12px; }
.svc-download-group h4 { margin: 0 0 8px; font-size: 13px; color: var(--wb-muted); }
.svc-download-group .svc-track-downloads { margin: 0; gap: 8px 20px; }
"""
