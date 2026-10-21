"""Source-aware training and validation split planning."""
from collections import defaultdict
from itertools import combinations


def plan(rows, selected_sources=None):
    if len(rows) < 2:
        raise ValueError("训练前至少需要两个已审核入库片段")
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_id"]].append(row)
    totals = {source: sum(float(row["duration"]) for row in clips) for source, clips in groups.items()}
    sources = sorted(groups)
    total_seconds = sum(totals.values())
    if total_seconds <= 0:
        raise ValueError("入库片段的有效时长必须大于 0")
    warning = ""
    if len(sources) == 1:
        if selected_sources:
            raise ValueError("只有一份源素材时按片段划分验证集，无需选择验证源")
        count = min(3, max(1, round(len(rows) * .25)), len(rows) - 1)
        positions = {round((index + .5) * len(rows) / count - .5) for index in range(count)}
        validation = [row for index, row in enumerate(rows) if index in positions]
        training = [row for index, row in enumerate(rows) if index not in positions]
        selected = sources
        warning = "仅一份源素材：验证集来自同首歌曲，不能评价跨歌曲表现"
    else:
        if selected_sources is None:
            target = .2 * total_seconds
            pool = sorted(sources, key=lambda source: (abs(totals[source] - target), source))[:30]
            pool = sorted(set(pool) | set(sorted(sources, key=lambda source: totals[source])[:10]))
            options = (choice for count in range(1, min(3, len(pool), len(sources) - 1) + 1)
                       for choice in combinations(pool, count))
            selected = min(options, key=lambda choice: (
                abs(sum(totals[source] for source in choice) / total_seconds - .2)
                + 4 * max(0, sum(totals[source] for source in choice) / total_seconds - .35)
                + .01 * (len(choice) - 1), choice))
        else:
            selected = tuple(dict.fromkeys(selected_sources))
            if not selected or len(selected) >= len(sources) or any(source not in groups for source in selected):
                raise ValueError("请选择至少一份验证源，并为训练保留至少一份源素材")
        selected = set(selected)
        training = [row for row in rows if row["source_id"] not in selected]
        validation = [row for row in rows if row["source_id"] in selected]
    if not training or not validation:
        raise ValueError("训练集与验证集都需要至少一个片段")
    seconds = sum(float(row["duration"]) for row in validation)
    ratio = seconds / total_seconds if total_seconds else 0
    if len(sources) > 1 and ratio > .4:
        warning = f"验证集占总时长 {ratio:.0%}，训练可用时长偏少；可调整验证源"
    source_rows = [{"id": source, "name": groups[source][0].get("source_name") or "未命名源素材",
                    "clips": len(groups[source]), "seconds": totals[source], "validation": source in selected}
                   for source in sources]
    return {"training": training, "validation": validation, "selected_sources": sorted(selected),
            "source_rows": source_rows, "train_seconds": total_seconds - seconds, "validation_seconds": seconds,
            "warning": warning}
