"""最优命中分组求解器。

目标层级：
  1. 最大化已分组命中的可信度之和；
  2. 在满足 1 的前提下最少化事件数；
  3. 规范解按成员标识排序后的事件序列做词典序裁决。

互斥候选家族
------------
可选 ``families`` 给出若干家族（每个家族为下标元组）：同一家族至多一个
成员可被**分组**，作为噪声的成员不占用家族；一个命中可同时属于多个家族。
求解器在 DP 中联合维护窗口占用与活跃家族的已用状态，只在合法解释中做
最优裁决，不会先求无约束结果再事后删除冲突。

复杂度保证
----------
约定：任意长度为 ``W`` 的闭窗内至多 10 个命中。按 (时刻, 标识) 排序后，
处理第 i 个命中时，所有“已被早先事件占用、尚未关窗”的命中都落在
``[t_i, t_i+W]`` 内，至多 10 个。窗口占用部分因此是至多 10 位的位掩码
（每步至多 1024 态）。家族掩码只保留“已出现且尚有未处理成员”的家族；
任一切分处这类家族（同时含已出现与未出现成员）至多 6 个，故家族部分
至多 2^6=64 态。整体为 O(n · 1024 · 64 · 2^9) 上界内的扫描线动态规划，
不枚举任何完整分组方案。方案计数为 Python 任意精度整数。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


class SolveError(ValueError):
    """输入数据违反约束（如符合窗内命中数超限、互斥家族前沿超限）。"""

    def __init__(self, message: str, path: str = "hits") -> None:
        super().__init__(message)
        self.path = path


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
    用于规范解的词典序比较（若为 None 则按下标比较）。``families`` 为
    互斥候选家族（成员下标元组的列表，每个家族 2–6 个互异下标），为
    None 或空时语义与无家族完全一致。
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

    # ---------- 家族元数据 ----------
    # hit_fmask[j]：命中 j 所属家族的位（一个命中可属于多个家族）。
    hit_fmask: List[int] = [0] * n
    expire_fmask: List[int] = [0] * n
    nfam = len(families) if families else 0
    if families:
        fam_min = [n] * nfam
        fam_max = [-1] * nfam
        for f, members in enumerate(families):
            if not (2 <= len(members) <= 6):
                raise SolveError(
                    "family must contain 2 to 6 members", "alternatives"
                )
            if len(set(members)) != len(members):
                raise SolveError(
                    "family members must be distinct", "alternatives"
                )
            fbit = 1 << f
            for j in members:
                if not (0 <= j < n):
                    raise SolveError(
                        "family references an unknown hit", "alternatives"
                    )
                hit_fmask[j] |= fbit
                if j < fam_min[f]:
                    fam_min[f] = j
                if j > fam_max[f]:
                    fam_max[f] = j
        # 前沿：切分 c（前 c 个命中已出现）处，同时含已出现/未出现成员的
        # 家族数 = 满足 min < c <= max 的家族数，处处不得超过 6。
        diff = [0] * (n + 2)
        for f in range(nfam):
            diff[fam_min[f] + 1] += 1
            diff[fam_max[f] + 1] -= 1
        active = 0
        for c in range(1, n + 1):
            active += diff[c]
            if active > 6:
                raise SolveError(
                    "more than 6 families are open at a sweep cut",
                    "alternatives",
                )
        for f in range(nfam):
            expire_fmask[fam_max[f]] |= 1 << f

    # ---------- 候选事件（含家族合法性） ----------
    # 同一事件内不得出现同一家族的两个成员（二者互斥，不能同时为真）。
    # gain_maps[i]：事件掩码 gm -> (可信度增益, 事件占用的家族位 gf)。
    open_masks: List[int] = []
    gain_maps: List[Dict[int, Tuple[int, int]]] = []
    group_by_mask: List[Dict[int, Tuple[int, ...]]] = []
    for i in range(n):
        gm_info: Dict[int, Tuple[int, int]] = {}
        by_mask: Dict[int, Tuple[int, ...]] = {}
        for g in groups[i]:
            gf = 0
            ok = True
            gain = 0
            m = 0
            for j in g:
                fb = hit_fmask[j]
                if gf & fb:
                    ok = False
                    break
                gf |= fb
                m |= 1 << j
                gain += weights[j]
            if ok:
                gm_info[m] = (gain, gf)
                by_mask[m] = g
        gain_maps.append(gm_info)
        group_by_mask.append(by_mask)
        om = 0
        for j in opens[i]:
            om |= 1 << j
        open_masks.append(om)

    # 状态为 (窗口占用掩码 wmask, 活跃家族已用掩码 fmask)，用两级字典
    # F[i][wmask][fmask] = cell。可创建事件集合只取决于 (i, wmask)，故
    # 每个 wmask 只做一次子掩码枚举，四个 DP 阶段共用。
    # 事件转移描述：(目标窗口掩码 wm2, 可信度增益, 家族位 gf, 事件掩码 gm)。
    event_cache: List[Dict[int, Tuple[Tuple[int, int, int, int], ...]]] = [
        {} for _ in range(n)
    ]

    def event_options(i: int, mask: int) -> Tuple[Tuple[int, int, int, int], ...]:
        """以 i 为锚点在窗口占用 mask 下可创建的事件。

        返回 ``(wm2, gain, gf, gm)``：``wm2`` 为转移后窗口掩码（已清 i
        位），``gf`` 为事件占用的家族位（调用方按 ``gf & fmask`` 过滤），
        ``gm`` 为事件命中掩码。调用方需保证 i 未被占用。枚举 3^(k-1)
        （k≤10），每个 (i, mask) 仅一次。
        """
        cache = event_cache[i]
        opts = cache.get(mask)
        if opts is None:
            bit_i = 1 << i
            free = open_masks[i] & ~mask & ~bit_i
            base = mask & ~bit_i
            info = gain_maps[i]
            out: List[Tuple[int, int, int, int]] = []
            s = free
            while True:
                gm = s | bit_i
                item = info.get(gm)
                if item is not None:
                    out.append((base | s, item[0], item[1], gm))
                if s == 0:
                    break
                s = (s - 1) & free
            opts = tuple(out)
            cache[mask] = opts
        return opts

    # ---------- 前向 DP ----------
    # F[i][wmask][fmask]：处理完前 i 个命中后，窗口占用 wmask、家族已用
    # fmask 时的 (已锁定可信度和, 已锁定事件数, 方案数)。噪声不占用家族；
    # 事件转移同时占用窗口位与家族位；处理完 i 后过期最后成员已决的家族位。
    forward: List[Dict[int, Dict[int, _Cell]]] = [{0: {0: (0, 0, 1)}}]
    for i in range(n):
        cur = forward[i]
        nxt: Dict[int, Dict[int, _Cell]] = {}
        bit_i = 1 << i
        exp = expire_fmask[i]

        def merge(dst: Dict[int, _Cell], fm2: int,
                  nsc: int, nev: int, cnt: int) -> None:
            old = dst.get(fm2)
            if old is None:
                dst[fm2] = (nsc, nev, cnt)
            elif nsc > old[0] or (nsc == old[0] and nev < old[1]):
                dst[fm2] = (nsc, nev, cnt)
            elif nsc == old[0] and nev == old[1]:
                dst[fm2] = (nsc, nev, old[2] + cnt)

        for wmask, fdict in cur.items():
            # 选择一：i 作为噪声（若 i 已被早先事件占用则强制此路）。
            # 噪声不占用家族，仅过期已完结家族。
            wn = wmask & ~bit_i
            d_noise = nxt.get(wn)
            if d_noise is None:
                d_noise = {}
                nxt[wn] = d_noise
            for fm, (sc, ev, cnt) in fdict.items():
                merge(d_noise, fm & ~exp, sc, ev, cnt)
            if wmask & bit_i:
                continue
            # 选择二：以 i 为锚点创建事件（联合占用窗口与家族）
            for wm2, gain, gf, _gm in event_options(i, wmask):
                d_ev = nxt.get(wm2)
                if d_ev is None:
                    # 新桶：所有在 i 处过期的家族都以 i 为成员，故其位
                    # 必在 gf 中、在允许的 fm 中必为 0；fm -> fm2 单射，
                    # 桶内无碰撞，可用 C 级推导整体构造。
                    nxt[wm2] = {
                        (fm | gf) & ~exp: (sc + gain, ev + 1, cnt)
                        for fm, (sc, ev, cnt) in fdict.items()
                        if not (gf & fm)
                    }
                else:
                    for fm, (sc, ev, cnt) in fdict.items():
                        if gf & fm:
                            continue
                        merge(d_ev, (fm | gf) & ~exp,
                              sc + gain, ev + 1, cnt)
        forward.append(nxt)

    best_score, best_events, total_count = forward[n][0][0]

    # ---------- 后向 DP ----------
    # B[i][wmask][fmask]：从步骤 i 的状态出发，后缀可达的
    # (可信度和, 事件数, 方案数) 最优值。只需前向可达的状态。
    backward: List[Dict[int, Dict[int, _Cell]]] = [
        {} for _ in range(n + 1)
    ]
    backward[n] = {0: {0: (0, 0, 1)}}
    for i in range(n - 1, -1, -1):
        table: Dict[int, Dict[int, _Cell]] = {}
        bit_i = 1 << i
        b_next = backward[i + 1]
        exp = expire_fmask[i]
        for wmask, fdict in forward[i].items():
            wn = wmask & ~bit_i
            occupied = bool(wmask & bit_i)
            d_noise = b_next.get(wn)
            opts = () if occupied else event_options(i, wmask)
            out: Dict[int, _Cell] = {}
            for fm in fdict:
                best: Optional[_Cell] = None
                # 选择一：噪声（被占用时为强制转移）
                if d_noise is not None:
                    best = d_noise.get(fm & ~exp)
                if not occupied:
                    # 选择二：以 i 为锚点创建事件
                    for wm2, gain, gf, _gm in opts:
                        if gf & fm:
                            continue
                        d2 = b_next.get(wm2)
                        if d2 is None:
                            continue
                        suffix = d2.get((fm | gf) & ~exp)
                        if suffix is None:
                            continue
                        cell = (suffix[0] + gain, suffix[1] + 1, suffix[2])
                        if best is None:
                            best = cell
                        elif cell[0] > best[0] or (
                            cell[0] == best[0] and cell[1] < best[1]
                        ):
                            best = cell
                        elif cell[0] == best[0] and cell[1] == best[1]:
                            best = (best[0], best[1], best[2] + cell[2])
                if best is not None:
                    out[fm] = best
            if out:
                table[wmask] = out
        backward[i] = table

    # ---------- 规范解（词典序裁决） ----------
    # 对每个可达状态 (i, mask, fm) 求后缀的“成员标识排序后的事件序列”
    # 的词典序最小值（仅限达到该状态最优 (可信度, 事件数) 的合法转移）。
    # 事件在锚点处加入，与后缀已排序序列做单点插入后比较。
    def _key(j: int):
        return id_order[j] if id_order is not None else j

    gm_to_keys: List[Dict[int, Tuple[object, ...]]] = [
        {
            m: tuple(_key(j) for j in g)
            for m, g in group_by_mask[i].items()
        }
        for i in range(n)
    ]

    # memo[i][(wmask, fmask)] -> 规范后缀序列
    memo: List[Dict[Tuple[int, int], Tuple[Tuple[object, ...], ...]]] = [
        {} for _ in range(n + 1)
    ]

    def canonical_suffix(
        i: int, wmask: int, fm: int
    ) -> Tuple[Tuple[object, ...], ...]:
        if i == n:
            return ()
        memo_i = memo[i]
        state = (wmask, fm)
        cached = memo_i.get(state)
        if cached is not None:
            return cached
        target = backward[i][wmask][fm]
        bit_i = 1 << i
        exp = expire_fmask[i]
        best_seq: Optional[Tuple[Tuple[object, ...], ...]] = None

        def consider(
            gain: int,
            events_added: int,
            wm2: int,
            fm2: int,
            inserted: Optional[Tuple[object, ...]],
        ) -> None:
            nonlocal best_seq
            d2 = backward[i + 1].get(wm2)
            if d2 is None:
                return
            suf = d2.get(fm2)
            if suf is None:
                return
            if gain + suf[0] != target[0] or events_added + suf[1] != target[1]:
                return
            suffix_seq = canonical_suffix(i + 1, wm2, fm2)
            if inserted is None:
                cand = suffix_seq
            else:
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

        # 噪声（被占用时为强制转移）
        consider(0, 0, wmask & ~bit_i, fm & ~exp, None)
        if not (wmask & bit_i):
            for wm2, gain, gf, gm in event_options(i, wmask):
                if gf & fm:
                    continue
                consider(gain, 1, wm2, (fm | gf) & ~exp, gm_to_keys[i][gm])

        assert best_seq is not None
        memo_i[state] = best_seq
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
        b_next = backward[i + 1]
        exp = expire_fmask[i]
        bit_i = 1 << i
        for wmask, fdict in forward[i].items():
            if wmask & bit_i:
                continue
            for fm, (fsc, fev, fcnt) in fdict.items():
                for wm2, gain, gf, gm in event_options(i, wmask):
                    if gf & fm:
                        continue
                    d2 = b_next.get(wm2)
                    if d2 is None:
                        continue
                    suf = d2.get((fm | gf) & ~exp)
                    if suf is None:
                        continue
                    if fsc + gain + suf[0] != best_score:
                        continue
                    if fev + 1 + suf[1] != best_events:
                        continue
                    g = group_by_mask[i][gm]
                    ways = fcnt * suf[2]
                    for j in g:
                        member_count[j] = member_count.get(j, 0) + ways
                    for a_idx in range(len(g)):
                        for b_idx in range(a_idx + 1, len(g)):
                            a, b = g[a_idx], g[b_idx]
                            pair_key = (a, b)
                            pair_count[pair_key] = pair_count.get(pair_key, 0) + ways

    return {
        "best_score": best_score,
        "best_events": best_events,
        "total_count": total_count,
        "canonical": canonical_groups,
        "member_count": member_count,
        "pair_count": pair_count,
        "groups": groups,
    }
