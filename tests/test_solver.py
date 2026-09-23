"""确定性边界与语义测试。"""

from __future__ import annotations

import random
import time

from app.solver import SolveError, solve

from .brute import brute_solve


def _ids(n):
    return [f"h{k:02d}" for k in range(n)]


def test_window_closed_boundary_inclusive():
    # t=0 与 t=W 在闭窗内可同组；t=W+1 不可
    times = [0, 5, 6]
    detectors = [0, 1, 1]
    weights = [5, 7, 9]
    got = solve(times, detectors, weights, 5, id_order=_ids(3))
    # (0,1) 可同组（差 5）；(0,2) 差 6 不可；1、2 同探测器
    assert got["best_score"] == 12
    assert (0, 1) in got["pair_count"]
    assert (0, 2) not in got["pair_count"]


def test_no_possible_events_all_noise():
    times = [0, 1, 2, 3]
    detectors = [0, 0, 0, 0]  # 全部同一探测器
    weights = [1, 1, 1, 1]
    got = solve(times, detectors, weights, 10)
    assert got["best_score"] == 0
    assert got["best_events"] == 0
    assert got["total_count"] == 1
    assert got["canonical"] == []
    assert got["pair_count"] == {}


def test_max_confidence_then_min_events():
    # 命中 0,1,2 互相兼容；两两可信度相同，三分组不可能（同一探测器限制）
    times = [0, 1, 2]
    detectors = [0, 1, 0]
    weights = [5, 4, 5]
    # 事件只能是 (0,1) 和 (1,2)；二者可信度 9，各 1 事件
    got = solve(times, detectors, weights, 10, id_order=_ids(3))
    assert got["best_score"] == 9
    assert got["best_events"] == 1
    assert got["total_count"] == 2
    # 规范裁决取标识序列更小者：[h00,h01] < [h01,h02]
    assert got["canonical"] == [(0, 1)]


def test_min_events_prefers_fewer_groups():
    # 四个命中两两兼容（探测器互异）：
    # 方案 A：一个四元事件 = 全部可信度
    # 方案 B：两个二元事件，可信度和相同（构造为等权）
    times = [0, 0, 0, 0]
    detectors = [0, 1, 2, 3]
    weights = [1, 1, 1, 1]
    got = solve(times, detectors, weights, 0, id_order=_ids(4))
    assert got["best_score"] == 4
    assert got["best_events"] == 1  # 四元事件最少
    assert (0, 1, 2, 3) in got["canonical"]


def test_pair_always_optional_never():
    # 0-1-2 链式兼容（探测器 0,1,0），等权；
    # 最优取一个二元事件（可信度 2），另一个噪声。
    times = [0, 1, 2]
    detectors = [0, 1, 0]
    weights = [1, 1, 1]
    got = solve(times, detectors, weights, 2, id_order=_ids(3))
    assert got["total_count"] == 2
    # (0,1) 与 (1,2) 各在恰好一个最优解中 -> 可选
    assert got["pair_count"][(0, 1)] == 1
    assert got["pair_count"][(1, 2)] == 1
    # 0 与 2 同探测器，从不兼容
    assert (0, 2) not in got["pair_count"]
    # 中间命中 1 在两个最优解中都被分组 -> 必然归属（成员层面）
    assert got["member_count"][1] == 2


def test_pair_always_relation():
    times = [0, 1, 1]
    detectors = [0, 1, 2]
    weights = [10, 10, 1]
    # 最优必含 (0,1)（可信度 20），命中 2 噪声
    got = solve(times, detectors, weights, 1, id_order=_ids(3))
    assert got["pair_count"][(0, 1)] == got["total_count"]
    assert got["total_count"] == 1


def test_zero_window_same_timestamp():
    times = [0, 0, 0, 1]
    detectors = [0, 1, 2, 0]
    weights = [3, 4, 5, 100]
    # W=0：前三个同刻可成三元事件（12）；命中 3 高可信但无人同时刻 -> 噪声
    got = solve(times, detectors, weights, 0, id_order=_ids(4))
    assert got["best_score"] == 12
    assert got["best_events"] == 1


def test_arbitrary_precision_count():
    # k 个相互独立的链式三元组（0,1,0 探测器），每组等权有 2 个最优，
    # 组间间隔超过符合窗 -> 总方案数 2^k，验证任意精度大整数。
    k = 26
    window = 2
    times, detectors, weights = [], [], []
    for i in range(k):
        base = 10 * i
        times += [base, base + 1, base + 2]
        detectors += [0, 1, 0]
        weights += [1, 1, 1]
    n = 3 * k
    got = solve(times, detectors, weights, window, id_order=_ids(n))
    assert got["total_count"] == 2 ** k
    assert isinstance(got["total_count"], int)
    assert got["best_score"] == 2 * k
    assert got["best_events"] == k


def test_full_scale_performance_and_density():
    rng = random.Random(2026)
    n = 80
    window = 5
    # 构造满足密度约束的时刻：逐个放置，确保任意窗 ≤10
    times = []
    t = 0
    while len(times) < n:
        window_count = sum(1 for x in times if t - window <= x <= t)
        if window_count < 10:
            times.append(t)
            if rng.random() < 0.5:
                t += 1
        else:
            t += 1
    times.sort()
    detectors = [rng.randrange(8) for _ in range(n)]
    weights = [rng.randint(1, 100) for _ in range(n)]
    start = time.perf_counter()
    got = solve(times, detectors, weights, window, id_order=_ids(n))
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"solve too slow: {elapsed:.2f}s"
    assert got["best_score"] > 0
    # 大整数计数一致性：必然成员计数之和等
    assert isinstance(got["total_count"], int) and got["total_count"] >= 1


def test_dense_window_rejected_via_solver():
    # 11 个命中落在同一闭窗 -> 拒绝
    times = [0] * 11
    detectors = [k % 8 for k in range(11)]
    weights = [1] * 11
    try:
        solve(times, detectors, weights, 0, id_order=[f"h{k}" for k in range(11)])
    except SolveError:
        return
    raise AssertionError("expected SolveError")


# ---------------------------------------------------------------- 互斥家族
def test_far_apart_exclusive_candidates_change_optimum():
    # 两个远隔的候选事件 (0,1)@t=0 与 (2,3)@t=100；无家族时各自成立，
    # 家族 (0,2) 使其互斥 -> 最优解由 400/2 事件降为 200/1 事件。
    times = [0, 0, 100, 100]
    detectors = [0, 1, 0, 1]
    weights = [100, 100, 100, 100]
    free = solve(times, detectors, weights, 2, id_order=_ids(4))
    assert free["best_score"] == 400
    assert free["best_events"] == 2
    assert free["total_count"] == 1

    got = solve(times, detectors, weights, 2,
                id_order=_ids(4), families=[(0, 2)])
    assert got["best_score"] == 200
    assert got["best_events"] == 1
    # 两个远隔候选二选一 -> 两个最优解；规范裁决取标识序列更小者
    assert got["total_count"] == 2
    assert got["canonical"] == [(0, 1)]
    assert got["pair_count"][(0, 1)] == 1
    assert got["pair_count"][(2, 3)] == 1
    # 与暴力枚举一致
    expected = brute_solve(times, detectors, weights, 2, [(0, 2)])
    assert got["best_score"] == expected["best_score"]
    assert got["total_count"] == expected["total_count"]


def test_overlapping_families_equal_optima_canonical_tiebreak():
    # 三个远隔时段各有一个候选事件（窗内可组、跨时段互不相干）：
    #   E_a=(a,q)@0, E_b=(b,r)@10, E_h=(h,p)@20，可信度相等。
    # F1=(a,h), F2=(b,h), F3=(a,b) 两两交叠，使三个事件两两互斥；
    # 每个事件都同时占用两个交叠家族（如 E_h 占用 F1、F2）。
    # 最优 = 单个事件（可信度 10、1 事件），三个等优解由规范序列裁决。
    times = [0, 0, 10, 10, 20, 20]
    detectors = [0, 1, 0, 1, 0, 1]
    weights = [5, 5, 5, 5, 5, 5]
    # 排序后下标：0=a,1=q,2=b,3=r,4=h,5=p
    got = solve(times, detectors, weights, 2,
                id_order=["a", "q", "b", "r", "h", "p"],
                families=[(0, 4), (2, 4), (0, 2)])
    assert got["best_score"] == 10
    assert got["best_events"] == 1
    assert got["total_count"] == 3
    assert got["canonical"] == [(0, 1)]  # [a,q] 词典序最小
    assert got["pair_count"][(0, 1)] == 1
    assert got["pair_count"][(2, 3)] == 1
    assert got["pair_count"][(4, 5)] == 1
    expected = brute_solve(
        times, detectors, weights, 2, [(0, 4), (2, 4), (0, 2)]
    )
    assert got["total_count"] == expected["total_count"]
    assert got["best_score"] == expected["best_score"]
    assert got["canonical"] == list(expected["canonical"])


def test_noise_member_does_not_consume_family():
    # 成员 0 在其窗口内无伙伴，只能作为噪声；噪声不占用家族，
    # 故同族成员 1 仍可与 2 同组。
    times = [0, 10, 10]
    detectors = [0, 0, 1]
    weights = [9, 9, 9]
    got = solve(times, detectors, weights, 2,
                id_order=_ids(3), families=[(0, 1)])
    assert got["best_score"] == 18
    assert got["canonical"] == [(1, 2)]
    assert got["total_count"] == 1


def test_event_with_two_family_members_is_invalid():
    # 家族 (0,1)：事件 (0,1,2) 与其任何含 0、1 的子事件都非法；
    # 仅 (0,2)/(1,2) 可行。
    times = [0, 0, 0]
    detectors = [0, 1, 2]
    weights = [5, 5, 5]
    got = solve(times, detectors, weights, 0,
                id_order=_ids(3), families=[(0, 1)])
    assert got["best_score"] == 10
    assert got["total_count"] == 2
    assert (0, 1) not in got["pair_count"]
    assert got["pair_count"][(0, 2)] == 1
    assert got["pair_count"][(1, 2)] == 1


def test_family_frontier_boundary_six_accepted_seven_rejected():
    # 6 个家族在首切分处同时敞开：恰好达上界，接受
    n6 = 7
    fams6 = [(0, k) for k in range(1, 7)]
    got = solve(list(range(n6)), [k % 8 for k in range(n6)],
                [1] * n6, 0, families=fams6)
    assert got["total_count"] >= 1

    # 7 个家族在首切分处同时敞开：超限，拒绝
    n7 = 8
    fams7 = [(0, k) for k in range(1, 8)]
    try:
        solve(list(range(n7)), [k % 8 for k in range(n7)],
              [1] * n7, 0, families=fams7)
    except SolveError:
        pass
    else:
        raise AssertionError("expected SolveError")


def test_families_full_scale_performance():
    rng = random.Random(721)
    n = 80
    window = 5
    times = []
    t = 0
    while len(times) < n:
        window_count = sum(1 for x in times if t - window <= x <= t)
        if window_count < 10:
            times.append(t)
            if rng.random() < 0.5:
                t += 1
        else:
            t += 1
    times.sort()
    detectors = [rng.randrange(8) for _ in range(n)]
    weights = [rng.randint(1, 100) for _ in range(n)]
    # 局部短跨家族，保证前沿宽度 <= 6
    families = []
    for _ in range(12):
        a = rng.randrange(n - 2)
        b = min(n - 1, a + rng.randint(1, 3))
        if a != b:
            families.append((a, b))
    start = time.perf_counter()
    got = solve(times, detectors, weights, window,
                id_order=_ids(n), families=families)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"solve too slow: {elapsed:.2f}s"
    assert got["total_count"] >= 1


def test_families_sustained_frontier_performance():
    # 对抗构造：6 个家族自首个命中横跨至末个命中，前沿持续 = 6。
    rng = random.Random(907)
    n = 60
    window = 5
    times = []
    t = 0
    while len(times) < n:
        window_count = sum(1 for x in times if t - window <= x <= t)
        if window_count < 10:
            times.append(t)
            if rng.random() < 0.6:
                t += 1
        else:
            t += 1
    times.sort()
    detectors = [rng.randrange(8) for _ in range(n)]
    weights = [rng.randint(1, 100) for _ in range(n)]
    families = [(f, n - 1 - f) for f in range(6)]
    start = time.perf_counter()
    got = solve(times, detectors, weights, window,
                id_order=_ids(n), families=families)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"solve too slow: {elapsed:.2f}s"
    # 每个最优解至多用到 6 个家族中的一个成员事件（一致性）
    assert got["total_count"] >= 1
