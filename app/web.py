"""HTTP 服务：POST /audit 做最优命中分组，GET /health 健康检查。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request

from .solver import SolveError, solve

app = Flask(__name__)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_ascii_str(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0 and value.isascii()


def _validate(payload: Any) -> Tuple[List[Dict[str, str]], Dict[str, Any] | None]:
    """返回 (按路径记录的字段错误, 清洗后的输入)。"""
    errors: List[Dict[str, str]] = []
    if not isinstance(payload, dict):
        errors.append({"path": "$", "message": "request body must be a JSON object"})
        return errors, None
    window = payload.get("window")
    if "window" not in payload:
        errors.append({"path": "window", "message": "window is required"})
    elif not _is_int(window) or window < 0:
        errors.append(
            {"path": "window", "message": "window must be a non-negative integer"}
        )

    detectors = payload.get("detectors")
    det_values: List[Any] = []
    if "detectors" not in payload:
        errors.append({"path": "detectors", "message": "detectors is required"})
    elif not isinstance(detectors, list):
        errors.append(
            {"path": "detectors", "message": "detectors must be an array"}
        )
    else:
        if not (2 <= len(detectors) <= 8):
            errors.append(
                {"path": "detectors", "message": "detectors must contain 2 to 8 items"}
            )
        seen = set()
        for k, det in enumerate(detectors):
            if not (isinstance(det, str) or _is_int(det)) or isinstance(det, bool):
                errors.append({
                    "path": f"detectors[{k}]",
                    "message": "detector must be an integer or a string",
                })
                continue
            if det in seen:
                errors.append({
                    "path": f"detectors[{k}]",
                    "message": "duplicate detector",
                })
            seen.add(det)
            det_values.append(det)

    hits = payload.get("hits")
    clean_hits: List[Dict[str, Any]] = []
    if "hits" not in payload:
        errors.append({"path": "hits", "message": "hits is required"})
    elif not isinstance(hits, list):
        errors.append({"path": "hits", "message": "hits must be an array"})
    else:
        if not (4 <= len(hits) <= 80):
            errors.append(
                {"path": "hits", "message": "hits must contain 4 to 80 items"}
            )
        seen_ids = set()
        for k, hit in enumerate(hits):
            if not isinstance(hit, dict):
                errors.append({
                    "path": f"hits[{k}]",
                    "message": "hit must be an object",
                })
                continue
            ident = hit.get("id")
            if "id" not in hit:
                errors.append({"path": f"hits[{k}].id", "message": "id is required"})
            elif not _is_ascii_str(ident):
                errors.append({
                    "path": f"hits[{k}].id",
                    "message": "id must be a non-empty ASCII string",
                })
            elif ident in seen_ids:
                errors.append({
                    "path": f"hits[{k}].id",
                    "message": "duplicate hit id",
                })
            else:
                seen_ids.add(ident)

            detector = hit.get("detector")
            if "detector" not in hit:
                errors.append({
                    "path": f"hits[{k}].detector",
                    "message": "detector is required",
                })
            elif detectors is not None and isinstance(detectors, list):
                if detector not in det_values:
                    errors.append({
                        "path": f"hits[{k}].detector",
                        "message": "detector must be one of the declared detectors",
                    })

            time_v = hit.get("time")
            if "time" not in hit:
                errors.append({
                    "path": f"hits[{k}].time",
                    "message": "time is required",
                })
            elif not _is_int(time_v):
                errors.append({
                    "path": f"hits[{k}].time",
                    "message": "time must be an integer",
                })

            conf = hit.get("confidence")
            if "confidence" not in hit:
                errors.append({
                    "path": f"hits[{k}].confidence",
                    "message": "confidence is required",
                })
            elif not _is_int(conf) or conf <= 0:
                errors.append({
                    "path": f"hits[{k}].confidence",
                    "message": "confidence must be a positive integer",
                })

            clean_hits.append(hit)

    # 互斥候选家族（可选；缺省时语义与既有完全一致）
    clean_alts: List[Dict[str, Any]] = []
    # 悬空引用检查所需的已声明命中标识
    known_ids = {
        h.get("id") for h in clean_hits if isinstance(h, dict)
    }
    if "alternatives" in payload:
        alternatives = payload.get("alternatives")
        if not isinstance(alternatives, list):
            errors.append({
                "path": "alternatives",
                "message": "alternatives must be an array",
            })
        else:
            seen_families = set()
            for k, alt in enumerate(alternatives):
                if not isinstance(alt, dict):
                    errors.append({
                        "path": f"alternatives[{k}]",
                        "message": "alternative must be an object",
                    })
                    continue
                family = alt.get("family")
                if "family" not in alt:
                    errors.append({
                        "path": f"alternatives[{k}].family",
                        "message": "family is required",
                    })
                elif not _is_ascii_str(family):
                    errors.append({
                        "path": f"alternatives[{k}].family",
                        "message": "family must be a non-empty ASCII string",
                    })
                elif family in seen_families:
                    errors.append({
                        "path": f"alternatives[{k}].family",
                        "message": "duplicate family",
                    })
                else:
                    seen_families.add(family)

                members = alt.get("members")
                member_ids: List[Any] = []
                if "members" not in alt:
                    errors.append({
                        "path": f"alternatives[{k}].members",
                        "message": "members is required",
                    })
                elif not isinstance(members, list):
                    errors.append({
                        "path": f"alternatives[{k}].members",
                        "message": "members must be an array",
                    })
                elif not (2 <= len(members) <= 6):
                    errors.append({
                        "path": f"alternatives[{k}].members",
                        "message": "members must contain 2 to 6 items",
                    })
                else:
                    seen_members = set()
                    for p, member in enumerate(members):
                        if not _is_ascii_str(member):
                            errors.append({
                                "path": f"alternatives[{k}].members[{p}]",
                                "message": "member must be a non-empty ASCII string",
                            })
                            continue
                        if member in seen_members:
                            errors.append({
                                "path": f"alternatives[{k}].members[{p}]",
                                "message": "duplicate member in family",
                            })
                        else:
                            seen_members.add(member)
                            member_ids.append(member)
                # 悬空引用：成员必须是已声明的命中标识
                for member in member_ids:
                    if member not in known_ids:
                        errors.append({
                            "path": f"alternatives[{k}].members",
                            "message": f"member {member!r} does not match any hit id",
                        })
                clean_alts.append({"family": family, "members": member_ids})

    if errors:
        return errors, None

    clean = {
        "window": window,
        "detectors": det_values,
        "hits": clean_hits,
        "alternatives": clean_alts,
    }
    return [], clean


def _run(clean: Dict[str, Any]) -> Dict[str, Any]:
    window = clean["window"]
    det_index = {d: k for k, d in enumerate(clean["detectors"])}
    raw = clean["hits"]
    # 按 (时刻, 标识) 排序
    order = sorted(range(len(raw)), key=lambda k: (raw[k]["time"], raw[k]["id"]))
    sorted_hits = [raw[k] for k in order]

    times = [h["time"] for h in sorted_hits]
    detectors = [det_index[h["detector"]] for h in sorted_hits]
    weights = [h["confidence"] for h in sorted_hits]
    ids = [h["id"] for h in sorted_hits]

    # 互斥候选家族：标识 -> 排序后的下标（家族顺序沿用请求顺序）
    id_to_pos = {h["id"]: k for k, h in enumerate(sorted_hits)}
    families = [
        tuple(id_to_pos[m] for m in alt["members"])
        for alt in clean.get("alternatives", [])
    ]
    # 每个命中所属家族位（用于在归属关系中排除同家族命中对）
    hit_fmask_web = [0] * len(sorted_hits)
    for f, members in enumerate(families):
        for pos in members:
            hit_fmask_web[pos] |= 1 << f

    result = solve(
        times, detectors, weights, window,
        id_order=ids, families=families,
    )

    canonical_groups = [
        sorted(ids[j] for j in group) for group in result["canonical"]
    ]
    canonical_groups.sort()

    total = result["total_count"]
    pair_relations = []
    n = len(ids)
    # 仅枚举“在全部约束下可同组”的命中对：探测器不同、时刻差不超过符合窗，
    # 且不同属任何互斥家族（同家族成员绝无合法事件可同时包含二者）。
    for a in range(n):
        for b in range(a + 1, n):
            if detectors[a] == detectors[b]:
                continue
            if times[b] - times[a] > window:
                continue
            if hit_fmask_web[a] & hit_fmask_web[b]:
                continue
            ways = result["pair_count"].get((a, b), 0)
            if ways == 0:
                relation = "never"
            elif ways == total:
                relation = "always"
            else:
                relation = "optional"
            pair_relations.append({
                "a": ids[a], "b": ids[b], "relation": relation,
            })

    return {
        "optimal_confidence": result["best_score"],
        "event_count": result["best_events"],
        "solution_count": str(total),
        "canonical_groups": canonical_groups,
        "pair_relations": pair_relations,
    }


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.post("/audit")
def audit() -> Any:
    payload = request.get_json(silent=True)
    errors, clean = _validate(payload)
    if errors:
        return jsonify({"errors": errors}), 400
    try:
        return jsonify(_run(clean))
    except SolveError as exc:
        return jsonify({
            "errors": [{"path": exc.path, "message": str(exc)}]
        }), 400
