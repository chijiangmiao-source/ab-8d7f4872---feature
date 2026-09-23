"""verify 单次服务：构建检查 + 测试 + 算法核对 + HTTP 冒烟，按结果退出。

核对项：多解计数（任意精度）、规范裁决、成对归属（必然/可选/从不）、
符合窗闭区间边界。环境变量 BASE_URL 指向待测 HTTP 服务。
"""

from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8080")

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- 构建检查
def build_check() -> None:
    for py in list(ROOT.glob("app/**/*.py")) + [ROOT / "verify.py"]:
        try:
            py_compile.compile(str(py), doraise=True)
        except py_compile.PyCompileError as exc:
            check(f"compile {py}", False, str(exc))
            return
    check("build: all modules compile", True)
    try:
        import app.web  # noqa: F401
        import app.solver  # noqa: F401
        check("build: imports succeed", True)
    except Exception as exc:  # noqa: BLE001
        check("build: imports succeed", False, repr(exc))


# ---------------------------------------------------------------- 算法核对
def algorithm_checks() -> None:
    from app.solver import SolveError, solve

    # 1) 窗口边界：闭区间，t 差恰好为 W 可同组
    r = solve([0, 5, 6], [0, 1, 1], [5, 7, 9], 5,
              id_order=["a", "b", "c"])
    check("boundary: hit at t+W is compatible",
          r["best_score"] == 12 and (0, 1) in r["pair_count"])
    r0 = solve([0, 5, 6], [0, 1, 1], [5, 7, 9], 4,
               id_order=["a", "b", "c"])
    check("boundary: hit at t+W+1 is excluded",
          r0["best_score"] == 0)

    # 2) 多解计数：三个独立链式三元组，每组 2 个等优选择 -> 2^3（任意精度）
    k = 20
    times, dets, wts = [], [], []
    for i in range(k):
        b = 10 * i
        times += [b, b + 1, b + 2]
        dets += [0, 1, 0]
        wts += [1, 1, 1]
    ids = [f"h{x:02d}" for x in range(3 * k)]
    r = solve(times, dets, wts, 2, id_order=ids)
    check("count: arbitrary-precision multi-solution count",
          r["total_count"] == 2 ** k, f"got {r['total_count']}")

    # 3) 规范裁决：两个等优事件，取成员标识序列词典序更小者
    r = solve([0, 1, 2], [0, 1, 0], [5, 4, 5], 10,
              id_order=["h00", "h01", "h02"])
    check("canonical: lexicographic adjudication",
          r["canonical"] == [(0, 1)], str(r["canonical"]))

    # 4) 成对归属：可选（链中两选其一）与必然（中心命中必入组）
    r = solve([0, 1, 2], [0, 1, 0], [1, 1, 1], 2,
              id_order=["a", "b", "c"])
    total = r["total_count"]
    check("attribution: optional pair appears in some optima",
          0 < r["pair_count"].get((0, 1), 0) < total)
    check("attribution: central member always grouped",
          r["member_count"].get(1) == total)
    # 从不归属：同探测器的 0、2
    check("attribution: same-detector pair never grouped",
          (0, 2) not in r["pair_count"])
    # 必然归属：权重悬殊使 (0,1) 出现在全部最优解
    r2 = solve([0, 1, 1], [0, 1, 2], [10, 10, 1], 1,
               id_order=["a", "b", "c"])
    check("attribution: always pair count equals total",
          r2["pair_count"].get((0, 1)) == r2["total_count"] == 1)

    # 5) 互斥候选：远隔家族成员改变最优解释（不同时刻、跨切分处）
    r = solve([0, 1, 50, 51], [0, 1, 0, 1], [5, 7, 9, 4], 2,
              id_order=["a", "b", "c", "d"], families=[[1, 2]])
    check("alternatives: distant candidates reshape optimum",
          r["best_score"] == 13 and r["best_events"] == 1
          and r["total_count"] == 1 and r["canonical"] == [(2, 3)])

    # 6) 多家族交叠的同优裁决：F={0,2}、G={1,3}，4 个同优解取最小序列
    r = solve([0, 0, 0, 0], [0, 1, 2, 3], [1, 1, 1, 1], 0,
              id_order=["a", "b", "c", "d"], families=[[0, 2], [1, 3]])
    check("alternatives: overlapping families tie adjudication",
          r["best_score"] == 2 and r["best_events"] == 1
          and r["total_count"] == 4 and r["canonical"] == [(0, 1)])

    # 7) 家族前沿边界：6 个活跃家族可解，7 个按 alternatives 拒绝
    ft = list(range(14))
    fids = [f"h{x:02d}" for x in range(14)]
    r = solve(ft, [0] * 14, [1] * 14, 0, id_order=fids,
              families=[[k, k + 7] for k in range(6)])
    check("alternatives: frontier of 6 families accepted",
          r["best_score"] == 0)
    try:
        solve(ft, [0] * 14, [1] * 14, 0, id_order=fids,
              families=[[k, k + 7] for k in range(7)])
        check("alternatives: frontier of 7 families rejected", False)
    except SolveError as exc:
        check("alternatives: frontier of 7 families rejected",
              getattr(exc, "path", None) == "alternatives")


# ---------------------------------------------------------------- 测试套件
def run_tests() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=ROOT,
    )
    check("test suite (pytest) passes", proc.returncode == 0)


# ---------------------------------------------------------------- HTTP 冒烟
def _request(method: str, path: str, body=None, timeout: float = 5.0):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers,
                                 method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def http_smoke() -> None:
    deadline = time.time() + 30
    last_err = None
    while time.time() < deadline:
        try:
            status, body = _request("GET", "/health")
            if status == 200 and body.get("status") == "ok":
                break
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = exc
        time.sleep(0.5)
    else:
        check("http: GET /health 200", False, repr(last_err))
        return
    check("http: GET /health 200", True)

    payload = {
        "window": 2,
        "detectors": ["A", "B", "C"],
        "hits": [
            {"id": "h1", "detector": "A", "time": 0, "confidence": 5},
            {"id": "h2", "detector": "B", "time": 1, "confidence": 7},
            {"id": "h3", "detector": "C", "time": 1, "confidence": 3},
            {"id": "h4", "detector": "A", "time": 9, "confidence": 4},
        ],
    }
    status, body = _request("POST", "/audit", payload)
    ok = (
        status == 200
        and body["optimal_confidence"] == 15
        and body["event_count"] == 1
        and body["canonical_groups"] == [["h1", "h2", "h3"]]
        and body["solution_count"] == "1"
    )
    check("http: POST /audit optimal grouping", ok, json.dumps(body))
    rel = {(p["a"], p["b"]): p["relation"] for p in body["pair_relations"]}
    check("http: pair relations always/never",
          rel.get(("h1", "h2")) == "always"
          and ("h3", "h4") not in rel
          and ("h1", "h4") not in rel)

    # 窗口边界经 HTTP 再验一次
    edge = dict(payload, window=1)
    status, body = _request("POST", "/audit", edge)
    check("http: closed window boundary over API",
          status == 200 and body["optimal_confidence"] == 15)

    # 字段错误：按路径返回且不得夹带结果
    bad = {"window": -1, "detectors": ["A"], "hits": []}
    try:
        status, body = _request("POST", "/audit", bad)
        check("http: invalid request rejected", False, f"status {status}")
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode())
        paths = {e["path"] for e in body.get("errors", [])}
        check("http: field errors by path without results",
              exc.code == 400
              and {"window", "detectors", "hits"} <= paths
              and "optimal_confidence" not in body)

    # 互斥候选：远隔家族改变最优解
    alt = {
        "window": 2,
        "detectors": ["A", "B"],
        "hits": [
            {"id": "h1", "detector": "A", "time": 0, "confidence": 5},
            {"id": "h2", "detector": "B", "time": 1, "confidence": 7},
            {"id": "h3", "detector": "A", "time": 50, "confidence": 9},
            {"id": "h4", "detector": "B", "time": 51, "confidence": 4},
        ],
    }
    status, body = _request("POST", "/audit", alt)
    base_conf = body["optimal_confidence"]
    alt2 = dict(alt, alternatives=[{"family": "seg1", "hits": ["h2", "h3"]}])
    status, body = _request("POST", "/audit", alt2)
    check("http: alternatives reshape optimum",
          base_conf == 25 and status == 200
          and body["optimal_confidence"] == 13
          and body["event_count"] == 1
          and body["canonical_groups"] == [["h3", "h4"]]
          and body["solution_count"] == "1")

    # 悬空引用 / 重复家族：按具体路径拒绝且不夹带结果
    bad_alt = dict(alt, alternatives=[
        {"family": "s1", "hits": ["h1", "ghost"]},
        {"family": "s1", "hits": ["h1", "h2"]},
    ])
    try:
        _request("POST", "/audit", bad_alt)
        check("http: invalid alternatives rejected by path", False)
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode())
        paths = {e["path"] for e in body.get("errors", [])}
        check("http: invalid alternatives rejected by path",
              exc.code == 400
              and "alternatives[0].hits[1]" in paths
              and "alternatives[1].family" in paths
              and "optimal_confidence" not in body)

    # 家族前沿超限：按 alternatives 路径拒绝
    dense = {
        "window": 0,
        "detectors": ["A", "B"],
        "hits": [
            {"id": f"h{k:02d}", "detector": "AB"[k % 2], "time": k,
             "confidence": 1}
            for k in range(16)
        ],
        "alternatives": [
            {"family": f"f{k}", "hits": [f"h{k:02d}", f"h{k + 8:02d}"]}
            for k in range(7)
        ],
    }
    try:
        _request("POST", "/audit", dense)
        check("http: family frontier overflow rejected", False)
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode())
        check("http: family frontier overflow rejected",
              exc.code == 400
              and any(e["path"] == "alternatives"
                      for e in body.get("errors", []))
              and "optimal_confidence" not in body)


def main() -> int:
    print(f"== verify against {BASE_URL} ==")
    build_check()
    algorithm_checks()
    run_tests()
    http_smoke()
    if failures:
        print(f"\nverify FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("\nverify OK: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
