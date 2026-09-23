"""/audit 与 /health 的 HTTP 层测试。"""

from __future__ import annotations

import json

import pytest

from app.web import app


@pytest.fixture()
def client():
    app.testing = True
    return app.test_client()


def _payload(**over):
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
    payload.update(over)
    return payload


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_audit_basic(client):
    resp = client.post("/audit", json=_payload())
    assert resp.status_code == 200
    data = resp.get_json()
    # h1,h2,h3 在窗内（0..2），三元事件可信度 15
    assert data["optimal_confidence"] == 15
    assert data["event_count"] == 1
    assert data["canonical_groups"] == [["h1", "h2", "h3"]]
    assert data["solution_count"] == "1"
    rel = {(r["a"], r["b"]): r["relation"] for r in data["pair_relations"]}
    assert rel[("h1", "h2")] == "always"
    assert rel[("h2", "h3")] == "always"
    # h1 与 h4 同探测器，不出现在可同组对中
    assert ("h1", "h4") not in rel
    assert ("h3", "h4") not in rel  # 时刻差超窗


def test_audit_window_boundary(client):
    # 窗 =1：h1(t0) 与 h2/h3(t1) 之差恰为 1，闭窗边界包含 -> 三元组仍可行
    resp = client.post("/audit", json=_payload(window=1))
    assert resp.get_json()["optimal_confidence"] == 15
    # 窗 =0：仅同时刻命中可组 -> h2+h3=10
    resp0 = client.post("/audit", json=_payload(window=0))
    assert resp0.get_json()["optimal_confidence"] == 10


def test_audit_missing_fields_no_result(client):
    resp = client.post("/audit", json={"detectors": ["A"], "hits": []})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "errors" in data
    assert "optimal_confidence" not in data
    paths = {e["path"] for e in data["errors"]}
    assert "window" in paths
    assert "detectors" in paths  # 只有 1 个探测器
    assert "hits" in paths  # 不足 4 个


def test_audit_field_errors_by_path(client):
    payload = _payload()
    payload["hits"] = [
        {"id": "x", "detector": "A", "time": 0, "confidence": 1},
        {"id": "x", "detector": "Z", "time": "q", "confidence": -2},
        {"id": "", "detector": "B", "time": 0, "confidence": 0},
        {"id": "ok", "detector": "B", "time": 0},
    ]
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 400
    paths = {e["path"] for e in resp.get_json()["errors"]}
    assert "hits[1].id" in paths          # 重复 id
    assert "hits[1].detector" in paths    # 未声明探测器
    assert "hits[1].time" in paths        # 非整数
    assert "hits[1].confidence" in paths  # 非正
    assert "hits[2].id" in paths          # 空 id
    assert "hits[2].confidence" in paths  # 零
    assert "hits[3].confidence" in paths  # 缺字段


def test_audit_bool_rejected_as_integer(client):
    payload = _payload()
    payload["window"] = True
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 400
    assert any(e["path"] == "window" for e in resp.get_json()["errors"])


def test_audit_too_many_hits_in_window(client):
    payload = _payload()
    payload["hits"] = [
        {"id": f"h{k}", "detector": "ABC"[k % 3], "time": 0,
         "confidence": 1}
        for k in range(11)
    ]
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["errors"]
    assert "optimal_confidence" not in data


def test_audit_non_json_body(client):
    resp = client.post("/audit", data="not json", content_type="text/plain")
    assert resp.status_code == 400
    assert resp.get_json()["errors"][0]["path"] == "$"


def test_audit_solution_count_is_string(client):
    resp = client.post("/audit", json=_payload())
    data = resp.get_json()
    # 任意精度以字符串传输
    assert isinstance(data["solution_count"], str)
    int(data["solution_count"])


def _distant_payload(**over):
    payload = {
        "window": 2,
        "detectors": ["A", "B"],
        "hits": [
            {"id": "h1", "detector": "A", "time": 0, "confidence": 5},
            {"id": "h2", "detector": "B", "time": 1, "confidence": 7},
            {"id": "h3", "detector": "A", "time": 50, "confidence": 9},
            {"id": "h4", "detector": "B", "time": 51, "confidence": 4},
        ],
    }
    payload.update(over)
    return payload


def test_audit_alternatives_distant_candidates_change_optimum(client):
    # 无互斥约束：两段远隔波形分别成事件，可信度 25
    base = client.post("/audit", json=_distant_payload()).get_json()
    assert base["optimal_confidence"] == 25
    assert base["event_count"] == 2

    # h2 与 h3 是同一段波形的互斥候选 -> 只能保留一段
    resp = client.post("/audit", json=_distant_payload(
        alternatives=[{"family": "seg1", "hits": ["h2", "h3"]}],
    ))
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["optimal_confidence"] == 13
    assert data["event_count"] == 1
    assert data["canonical_groups"] == [["h3", "h4"]]
    assert data["solution_count"] == "1"
    rel = {(r["a"], r["b"]): r["relation"] for r in data["pair_relations"]}
    assert rel[("h3", "h4")] == "always"
    assert rel[("h1", "h2")] == "never"  # 受限后的最优解中不再同组


def test_audit_alternatives_overlapping_families_tie(client):
    # 两家族交叠、等权同刻 -> 4 个同优解，规范序列取最小
    payload = _payload(window=0, detectors=["A", "B", "C", "D"])
    payload["hits"] = [
        {"id": f"h{k}", "detector": det, "time": 0, "confidence": 1}
        for k, det in enumerate("ABCD")
    ]
    payload["alternatives"] = [
        {"family": "F", "hits": ["h0", "h2"]},
        {"family": "G", "hits": ["h1", "h3"]},
    ]
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["optimal_confidence"] == 2
    assert data["event_count"] == 1
    assert data["solution_count"] == "4"
    assert data["canonical_groups"] == [["h0", "h1"]]
    rel = {(r["a"], r["b"]): r["relation"] for r in data["pair_relations"]}
    assert rel[("h0", "h1")] == "optional"
    assert rel[("h0", "h2")] == "never"  # 同族，永不同组


def test_audit_alternatives_field_errors_by_path(client):
    payload = _payload()
    payload["alternatives"] = [
        {"family": "seg1", "hits": ["h1", "h2"]},
        {"family": "seg1", "hits": ["h3", "h4"]},     # 重复家族
        {"family": "seg2", "hits": ["h1", "ghost"]},  # 悬空引用
        {"family": "seg3", "hits": ["h1"]},           # 成员数不足
        {"family": "seg4", "hits": ["h1", "h1"]},     # 族内重复
        {"family": "", "hits": ["h1", "h2"]},         # 空家族标识
        {"hits": ["h1", "h2"]},                       # 缺 family
        {"family": "seg5"},                           # 缺 hits
        "junk",                                       # 非对象
    ]
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert "optimal_confidence" not in data
    paths = {e["path"] for e in data["errors"]}
    assert "alternatives[1].family" in paths
    assert "alternatives[2].hits[1]" in paths
    assert "alternatives[3].hits" in paths
    assert "alternatives[4].hits[1]" in paths
    assert "alternatives[5].family" in paths
    assert "alternatives[6].family" in paths
    assert "alternatives[7].hits" in paths
    assert "alternatives[8]" in paths


def test_audit_alternatives_not_an_array(client):
    resp = client.post("/audit", json=_payload(alternatives="nope"))
    assert resp.status_code == 400
    assert any(e["path"] == "alternatives"
               for e in resp.get_json()["errors"])
    assert "optimal_confidence" not in resp.get_json()


def test_audit_alternatives_frontier_overflow(client):
    # 7 个家族同时跨越同一切分处 -> 拒绝；6 个 -> 接受
    hits = [
        {"id": f"h{k:02d}", "detector": "AB"[k % 2], "time": k,
         "confidence": 1}
        for k in range(16)
    ]

    def with_families(count):
        return {
            "window": 0,
            "detectors": ["A", "B"],
            "hits": hits,
            "alternatives": [
                {"family": f"f{k}",
                 "hits": [f"h{k:02d}", f"h{k + 8:02d}"]}
                for k in range(count)
            ],
        }

    resp6 = client.post("/audit", json=with_families(6))
    assert resp6.status_code == 200
    resp7 = client.post("/audit", json=with_families(7))
    assert resp7.status_code == 400
    data = resp7.get_json()
    assert "optimal_confidence" not in data
    assert any(e["path"] == "alternatives" for e in data["errors"])


def test_audit_alternatives_absent_and_empty_compatible(client):
    # 未提供 alternatives 与提供空列表的响应完全一致（兼容回归）
    base = client.post("/audit", json=_payload())
    empty = client.post("/audit", json=_payload(alternatives=[]))
    assert base.status_code == empty.status_code == 200
    assert base.get_json() == empty.get_json()
