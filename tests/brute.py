"""小规模暴力枚举参照，仅用于测试比对（会枚举完整分组方案）。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from app.solver import _sweep_groups


def brute_solve(
    times: Sequence[int],
    detectors: Sequence[int],
    weights: Sequence[int],
    window: int,
    families: Optional[Sequence[Sequence[int]]] = None,
) -> dict:
    n = len(times)
    groups = _sweep_groups(times, detectors, window)

    fams_of: List[Tuple[int, ...]] = [() for _ in range(n)]
    for fi, members in enumerate(families or []):
        for j in members:
            fams_of[j] = fams_of[j] + (fi,)

    # 每个候选事件占用的家族位；同一事件含同一家族两成员的永远非法
    group_fam_mask: List[Dict[Tuple[int, ...], Optional[int]]] = []
    for i in range(n):
        d: Dict[Tuple[int, ...], Optional[int]] = {}
        for g in groups[i]:
            fm = 0
            legal = True
            for j in g:
                for f in fams_of[j]:
                    if (fm >> f) & 1:
                        legal = False
                    fm |= 1 << f
            d[g] = fm if legal else None
        group_fam_mask.append(d)

    best_score = 0
    best_events = 0
    packings: List[Tuple[Tuple[int, ...], ...]] = []

    def rec(i: int, used_mask: int, fam_used: int,
            chosen: List[Tuple[int, ...]], score: int) -> None:
        nonlocal best_score, best_events, packings
        while i < n and (used_mask & (1 << i)):
            i += 1
        if i == n:
            events = len(chosen)
            if score > best_score:
                best_score, best_events = score, events
                packings = [tuple(chosen)]
            elif score == best_score:
                if events < best_events:
                    best_events = events
                    packings = [tuple(chosen)]
                elif events == best_events:
                    packings.append(tuple(chosen))
            return
        # i 作为噪声（噪声不占用家族）
        rec(i + 1, used_mask, fam_used, chosen, score)
        # 以 i 为锚点建事件
        for g in groups[i]:
            fm = group_fam_mask[i][g]
            if fm is None or fam_used & fm:
                continue
            gm = 0
            gain = 0
            for j in g:
                gm |= 1 << j
                gain += weights[j]
            if used_mask & gm:
                continue
            chosen.append(g)
            rec(i + 1, used_mask | gm, fam_used | fm, chosen, score + gain)
            chosen.pop()

    rec(0, 0, 0, [], 0)

    count = len(packings)
    pair_count: Dict[Tuple[int, int], int] = {}
    member_count: Dict[int, int] = {}
    canonical = None
    for chosen in packings:
        seq = tuple(sorted(tuple(sorted(g)) for g in chosen))
        if canonical is None or seq < canonical:
            canonical = seq
        for g in chosen:
            for j in g:
                member_count[j] = member_count.get(j, 0) + 1
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    key = (g[a], g[b])
                    pair_count[key] = pair_count.get(key, 0) + 1

    return {
        "best_score": best_score,
        "best_events": best_events,
        "total_count": count,
        "canonical": canonical or (),
        "pair_count": pair_count,
        "member_count": member_count,
    }
