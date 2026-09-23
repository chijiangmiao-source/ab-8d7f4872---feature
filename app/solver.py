"""最优命中分组求解器。

目标层级：
  1. 最大化已分组命中的可信度之和；
  2. 在满足 1 的前提下最少化事件数；
  3. 规范解按成员标识排序后的事件序列做词典序裁决。

互斥候选（alternatives）
------------------------
脉冲形状拟合器可能把同一段波形输出为跨探测器、跨时刻的多个候选命中；
同一候选家族内的成员彼此互斥：同一家族至多一个成员可被分组，作为
噪声的成员不占用家族。命中按 (时刻, 标识) 排序后，任一切分处同时
含有已出现与未出现成员的“活跃家族”不超过 6 个。求解器把窗口占用
掩码与活跃家族的已用位（至多 6 位）联合为扫描线状态，直接在合法
解释上沿用 (可信度, 事件数, 规范序列) 的目标顺序 —— 不求原结果再
删除冲突，也不枚举完整分组。

家族“已用”的判定有两个互补来源：

  * 已处理成员被分组 —— 由跨切分处的家族已用位承载；
  * 未处理成员被早先锚点的事件分组 —— 该成员必落在当前闭窗内
    （t_member <= t_anchor + W <= t_i + W），由窗口占用掩码直接读出。

复杂度保证
----------
约定：任意长度为 ``W`` 的闭窗内至多 10 个命中；任一切分处的活跃
家族至多 6 个。按 (时刻, 标识) 排序后，处理第 i 个命中时，所有
“已被早先事件占用、尚未关窗”的命中都落在 ``[t_i, t_i+W]`` 内，
至多 10 个。因此 DP 状态是至多 10 位窗口掩码 × 至多 6 位家族已用
位（每步至多 1024 * 64 个状态），事件转移枚举空闲位的子掩码
（三态 3^(k-1)，k≤10），整体为扫描线动态规划，不枚举任何完整
分组方案。方案计数为 Python 任意精度整数。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


class SolveError(ValueError):
    """输入数据违反约束（如符合窗内命中数超限、家族前沿超限）。"""

    def __init__(self, message: str, path: str = "hits") -> None:
        super().__init__(message)
        self.path = path


# 任一切分处允许同时活跃（跨越该切分处）的互斥家族上限
MAX_ACTIVE_FAMILIES = 6


def _sweep_groups(
    times: Sequence[int],
    detectors: Sequence[int],
    window: int,
) -> List[List[Tuple[int, ...]]]:
    """返回每个锚点 i 可以创建的事件。

    事件以其最小下标成员为锚点；成员下标升序、探测器互不相同，
    且最大时刻与最小时刻之差不超过 window。
    """
    n = len(times)
    groups: List[List[Tuple[int, ...]]] = [[] for _ in range(n)]
    for i in range(n):
        ti = times[i]
        det_used = {detectors[i]}
        chosen: List[int] = [i]
        open_hits: List[int] = []
        for j in range(i + 1, n):
            if times[j] - ti > window:
                break
            open_hits.append(j)

        def gen(pos: int) -> None:
            if pos == len(open_hits):
                if len(chosen) >= 2:
                    groups[i].append(tuple(chosen))
                return
            # 不选 open_hits[pos]
            gen(pos + 1)
            # 选（探测器须不冲突）
            j = open_hits[pos]
            d = detectors[j]
            if d not in det_used:
                det_used.add(d)
                chosen.append(j)
                gen(pos + 1)
                chosen.pop()
                det_used.remove(d)

        gen(0)
    return groups


def _open_members(times: Sequence[int], window: int) -> List[List[int]]:
    """open(i)：时刻落在 [t_i, t_i+W] 闭窗内的下标（含 i），至多 10 个。"""
    n = len(times)
    result: List[List[int]] = []
    for i in range(n):
        members = [i]
        for j in range(i + 1, n):
            if times[j] - times[i] > window:
                break
            members.append(j)
        result.append(members)
    return result


# (最优可信度和, 最少事件数, 最优方案数)
_Cell = Tuple[int, int, int]


def _better(a: _Cell, b: Optional[_Cell]) -> bool:
    if b is None:
        return True
    if a[0] != b[0]:
        return a[0] > b[0]
    return a[1] < b[1]


def _prepare_families(
    families: Optional[Sequence[Sequence[int]]],
    opens: Sequence[Sequence[int]],
    n: int,
):
    """构建互斥家族的扫描线结构（无家族时返回全空结构）。

    返回 ``(fams_of, active_pos, remap_table, consume_bits, famwin)``：

    - ``fams_of[j]``：命中 j 所属家族下标元组（可属于多个家族）；
    - ``active_pos[c]``：切分处 c 的活跃家族 -> 已用位位号；
    - ``remap_table[i][bits]``：切分处 i 的已用位映射到切分处 i+1；
    - ``consume_bits[i]``：命中 i 被分组时需在切分处 i+1 置位的家族位；
    - ``famwin[i]``：家族 -> 其落在 open(i) 内成员的掩码（全局位）。
    """
    if not families:
        return (
            [() for _ in range(n)],
            [{} for _ in range(n + 1)],
            [[0] for _ in range(n)],
            [0] * n,
            [{} for _ in range(n)],
        )

    fams_of: List[Tuple[int, ...]] = [() for _ in range(n)]
    spans: List[Tuple[int, int]] = []
    for fi, members in enumerate(families):
        pts: List[int] = []
        for j in members:
            if not isinstance(j, int) or isinstance(j, bool) or not 0 <= j < n:
                raise SolveError(
                    "alternative family member index out of range",
                    path="alternatives",
                )
            if j not in pts:
                pts.append(j)
        pts.sort()
        if len(pts) < 2:
            raise SolveError(
                "alternative family must contain at least 2 distinct hits",
                path="alternatives",
            )
        for j in pts:
            fams_of[j] = fams_of[j] + (fi,)
        spans.append((pts[0], pts[-1]))

    # 活跃家族：在切分处 c 同时含有已出现（下标 < c）与未出现（>= c）成员
    active: List[List[int]] = []
    active_pos: List[Dict[int, int]] = []
    for c in range(n + 1):
        fams = [fi for fi, (lo, hi) in enumerate(spans) if lo < c <= hi]
        fams.sort(key=lambda fi: (spans[fi][0], spans[fi][1], fi))
        if len(fams) > MAX_ACTIVE_FAMILIES:
            raise SolveError(
                "more than 6 alternative families span a single cut",
                path="alternatives",
            )
        active.append(fams)
        active_pos.append({f: p for p, f in enumerate(fams)})

    remap_table: List[List[int]] = []
    for c in range(n):
        dst = active_pos[c + 1]
        table: List[int] = []
        for bits in range(1 << len(active[c])):
            out = 0
            for p, f in enumerate(active[c]):
                if (bits >> p) & 1:
                    q = dst.get(f)
                    if q is not None:
                        out |= 1 << q
            table.append(out)
        remap_table.append(table)

    consume_bits: List[int] = []
    for i in range(n):
        bits = 0
        dst = active_pos[i + 1]
        for f in fams_of[i]:
            q = dst.get(f)
            if q is not None:
                bits |= 1 << q
        consume_bits.append(bits)

    famwin: List[Dict[int, int]] = []
    for i in range(n):
        win: Dict[int, int] = {}
        for j in opens[i]:
            for f in fams_of[j]:
                win[f] = win.get(f, 0) | (1 << j)
        famwin.append(win)

    return fams_of, active_pos, remap_table, consume_bits, famwin


def solve(
    times: Sequence[int],
    detectors: Sequence[int],
    weights: Sequence[int],
    window: int,
    id_order: Optional[Sequence[str]] = None,
    families: Optional[Sequence[Sequence[int]]] = None,
) -> dict:
    """计算最优分组、规范解、任意精度方案数与成对归属。

    调用前输入已按 (时刻, 标识) 稳定排序；id_order 给出排序后的标识，
    用于规范解的词典序比较（若为 None 则按下标比较）。families 给出
    互斥候选家族（成员为排序后下标）；同一家族至多一个成员可被分组，
    未提供时与原始语义完全一致。
    """
    n = len(times)
    if n == 0:
        raise SolveError("no hits")

    groups = _sweep_groups(times, detectors, window)
    opens = _open_members(times, window)
    for members in opens:
        if len(members) > 10:
            raise SolveError(
                "coincidence window contains more than 10 hits"
            )

    fams_of, active_pos, remap_table, consume_bits, famwin = (
        _prepare_families(families, opens, n)
    )

    # 候选事件预处理：同一事件含同一家族两个成员的永远非法，直接剔除
    kept_groups: List[List[Tuple[int, ...]]] = []
    group_masks: List[List[int]] = []
    group_gains: List[List[int]] = []
    group_fams: List[Dict[int, Tuple[int, ...]]] = []
    open_masks: List[int] = []
    for i in range(n):
        kept: List[Tuple[int, ...]] = []
        ms: List[int] = []
        gains: List[int] = []
        gf: Dict[int, Tuple[int, ...]] = {}
        for g in groups[i]:
            m = 0
            gain = 0
            fams: List[int] = []
            seen_fams = set()
            legal = True
            for j in g:
                m |= 1 << j
                gain += weights[j]
                for f in fams_of[j]:
                    if f in seen_fams:
                        legal = False
                        break
                    seen_fams.add(f)
                    fams.append(f)
                if not legal:
                    break
            if not legal:
                continue
            kept.append(g)
            ms.append(m)
            gains.append(gain)
            gf[m] = tuple(fams)
        kept_groups.append(kept)
        group_masks.append(ms)
        group_gains.append(gains)
        group_fams.append(gf)
        om = 0
        for j in opens[i]:
            om |= 1 << j
        open_masks.append(om)

    # gm -> gain 映射，配合子掩码枚举使用
    gain_maps = [
        dict(zip(group_masks[i], group_gains[i])) for i in range(n)
    ]

    def iter_event_masks(mask: int, i: int):
        """枚举在待决议占用 mask 下、以 i 为锚点可创建的事件 (gm, gain)。

        仅枚举“未被占用的开放命中（锚点除外）”的子掩码，
        三态计数 3^(k-1)（k≤10）。
        """
        bit_i = 1 << i
        free = open_masks[i] & ~mask & ~bit_i
        s = free
        while True:
            gm = s | bit_i
            gain = gain_maps[i].get(gm)
            if gain is not None:
                yield gm, gain
            if s == 0:
                break
            s = (s - 1) & free

    # 事件转移缓存：同一 (i, mask) 在多个家族已用位状态下重复出现，
    # 子掩码枚举与“窗内家族占用”检查只与 (i, mask) 有关，缓存之。
    # 条目为 (gm, gain, next_mask, positions)；positions 是仍需对照
    # 家族已用位的位号元组。条目总数超上限时整体清空，防止内存膨胀。
    event_cache: Dict[Tuple[int, int], list] = {}
    event_cache_entries = 0
    _EVENT_CACHE_CAP = 1 << 20

    def events_for(i: int, mask: int):
        """(i, mask) 下通过窗内家族占用检查的锚点事件列表。"""
        nonlocal event_cache_entries
        key = (i, mask)
        got = event_cache.get(key)
        if got is not None:
            return got
        bit_i = 1 << i
        apos = active_pos[i]
        fwin = famwin[i]
        out = []
        for gm, gain in iter_event_masks(mask, i):
            positions = []
            legal = True
            for f in group_fams[i][gm]:
                if fwin.get(f, 0) & mask:
                    legal = False
                    break
                p = apos.get(f)
                if p is not None:
                    positions.append(p)
            if not legal:
                continue
            out.append((gm, gain, (mask | gm) & ~bit_i, tuple(positions)))
        if event_cache_entries + len(out) > _EVENT_CACHE_CAP:
            event_cache.clear()
            event_cache_entries = 0
        event_cache[key] = out
        event_cache_entries += len(out)
        return out

    def iter_transitions(i: int, mask: int, fambits: int):
        """状态 (mask, fambits) 下处理命中 i 的全部合法转移。

        产出 (gain, events_added, next_mask, next_fambits, gm)；
        gm 为 None 表示噪声转移。窗口占用与活跃家族已用状态在此
        联合维护。
        """
        bit_i = 1 << i
        base_fam = remap_table[i][fambits]
        # 噪声转移（i 已被早先事件占用时为强制转移；被占用即已被
        # 分组，需占用其家族）
        nf = base_fam
        if mask & bit_i:
            nf |= consume_bits[i]
        yield 0, 0, mask & ~bit_i, nf, None
        if mask & bit_i:
            return
        # 以 i 为锚点创建事件：涉及的家族须未被占用（已用位或窗内
        # 已被早先事件分组的成员均视为占用，后者已在缓存中过滤）
        nf_event = base_fam | consume_bits[i]
        for gm, gain, nm, positions in events_for(i, mask):
            legal = True
            for p in positions:
                if (fambits >> p) & 1:
                    legal = False
                    break
            if legal:
                yield gain, 1, nm, nf_event, gm

    # ---------- 前向 DP ----------
    # F[i][(mask, fambits)]：处理完前 i 个命中后，待决议占用掩码为
    # mask、活跃家族已用位为 fambits 时的
    # (已锁定可信度和, 已锁定事件数, 方案数)。
    forward: List[Dict[Tuple[int, int], _Cell]] = [{(0, 0): (0, 0, 1)}]
    for i in range(n):
        cur = forward[i]
        nxt: Dict[Tuple[int, int], _Cell] = {}
        for (mask, fambits), (sc, ev, cnt) in cur.items():
            for gain, dev, nm, nf, _gm in iter_transitions(i, mask, fambits):
                cell = (sc + gain, ev + dev, cnt)
                key = (nm, nf)
                old = nxt.get(key)
                if old is None:
                    nxt[key] = cell
                elif _better(cell, old):
                    nxt[key] = cell
                elif not _better(old, cell):
                    nxt[key] = (old[0], old[1], old[2] + cnt)
        forward.append(nxt)

    best_score, best_events, total_count = forward[n][(0, 0)]

    # ---------- 后向 DP ----------
    # B[i][(mask, fambits)]：从步骤 i 的状态出发，后缀可达的
    # (可信度和, 事件数, 方案数) 最优值。只需前向可达的状态。
    backward: List[Dict[Tuple[int, int], _Cell]] = [{} for _ in range(n + 1)]
    backward[n] = {(0, 0): (0, 0, 1)}
    for i in range(n - 1, -1, -1):
        table: Dict[Tuple[int, int], _Cell] = {}
        b_next = backward[i + 1]
        for mask, fambits in forward[i]:
            best: Optional[_Cell] = None
            for gain, dev, nm, nf, _gm in iter_transitions(i, mask, fambits):
                suffix = b_next.get((nm, nf))
                if suffix is None:
                    continue
                cell = (suffix[0] + gain, suffix[1] + dev, suffix[2])
                if best is None or _better(cell, best):
                    best = cell
                elif not _better(best, cell):
                    best = (best[0], best[1], best[2] + cell[2])
            if best is not None:
                table[(mask, fambits)] = best
        backward[i] = table

    # ---------- 规范解（词典序裁决） ----------
    # 对每个可达状态 (i, mask, fambits) 求后缀的“成员标识排序后的事件
    # 序列”的词典序最小值（仅限达到该状态最优 (可信度, 事件数) 的
    # 转移）。事件在锚点处加入，与后缀已排序序列做单点插入后比较。
    def _key(j: int):
        return id_order[j] if id_order is not None else j

    group_by_mask: List[Dict[int, Tuple[int, ...]]] = [
        {m: g for g, m in zip(kept_groups[i], group_masks[i])}
        for i in range(n)
    ]
    gm_to_keys: List[Dict[int, Tuple[object, ...]]] = [
        {
            m: tuple(_key(j) for j in g)
            for g, m in zip(kept_groups[i], group_masks[i])
        }
        for i in range(n)
    ]

    memo: Dict[Tuple[int, int, int], Tuple[Tuple[object, ...], ...]] = {}

    def canonical_suffix(
        i: int, mask: int, fambits: int
    ) -> Tuple[Tuple[object, ...], ...]:
        if i == n:
            return ()
        key = (i, mask, fambits)
        if key in memo:
            return memo[key]
        target = backward[i][(mask, fambits)]
        best_seq: Optional[Tuple[Tuple[object, ...], ...]] = None
        for gain, dev, nm, nf, gm in iter_transitions(i, mask, fambits):
            suf = backward[i + 1].get((nm, nf))
            if suf is None:
                continue
            if gain + suf[0] != target[0] or dev + suf[1] != target[1]:
                continue
            suffix_seq = canonical_suffix(i + 1, nm, nf)
            if gm is None:
                cand = suffix_seq
            else:
                inserted = gm_to_keys[i][gm]
                cand_list = list(suffix_seq)
                lo, hi = 0, len(cand_list)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cand_list[mid] < inserted:
                        lo = mid + 1
                    else:
                        hi = mid
                cand_list.insert(lo, inserted)
                cand = tuple(cand_list)
            if best_seq is None or cand < best_seq:
                best_seq = cand
        assert best_seq is not None
        memo[key] = best_seq
        return best_seq

    canonical_seq = canonical_suffix(0, 0, 0)
    # 将比较键还原为下标元组
    if id_order is not None:
        key_to_index = {id_order[j]: j for j in range(n)}
        canonical_groups = [
            tuple(key_to_index[x] for x in gt) for gt in canonical_seq
        ]
    else:
        canonical_groups = [tuple(gt) for gt in canonical_seq]

    # ---------- 逐事件 / 逐对在最优解中的出现次数 ----------
    member_count: Dict[int, int] = {}
    pair_count: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        f_table = forward[i]
        b_next = backward[i + 1]
        for (fmask, ffam), (fsc, fev, fcnt) in f_table.items():
            for gain, dev, nm, nf, gm in iter_transitions(i, fmask, ffam):
                if gm is None:
                    continue
                suf = b_next.get((nm, nf))
                if suf is None:
                    continue
                if fsc + gain + suf[0] != best_score:
                    continue
                if fev + dev + suf[1] != best_events:
                    continue
                g = group_by_mask[i][gm]
                ways = fcnt * suf[2]
                for j in g:
                    member_count[j] = member_count.get(j, 0) + ways
                for a_idx in range(len(g)):
                    for b_idx in range(a_idx + 1, len(g)):
                        a, b = g[a_idx], g[b_idx]
                        key = (a, b)
                        pair_count[key] = pair_count.get(key, 0) + ways

    return {
        "best_score": best_score,
        "best_events": best_events,
        "total_count": total_count,
        "canonical": canonical_groups,
        "member_count": member_count,
        "pair_count": pair_count,
        "groups": groups,
    }
