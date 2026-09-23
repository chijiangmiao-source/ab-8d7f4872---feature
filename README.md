# 脉冲中子命中分组审计服务

将 2–8 个探测器、4–80 个命中（唯一 ASCII 标识、所属探测器、整数时刻、
正整数可信度）在给定非负符合窗（闭区间）下做最优事件分组。

## 优化目标（按优先级）

1. **最大化已分组命中的可信度之和**；
2. 在其前提下 **最少化事件数**；
3. 多解时以「成员标识排序后的事件序列」做 **词典序规范裁决**。

约束：一个事件至少两个命中、同一探测器至多一个、最晚与最早时刻差 ≤ W；
每个命中只能属于一个事件或作为噪声；任意长度 W 的闭窗内至多 10 个命中。

## 接口

- `GET /health` → `200 {"status":"ok"}`
- `POST /audit`
  ```json
  {
    "window": 2,
    "detectors": ["A", "B", "C"],
    "hits": [
      {"id": "h1", "detector": "A", "time": 0, "confidence": 5},
      {"id": "h2", "detector": "B", "time": 1, "confidence": 7}
    ]
  }
  ```
  成功 `200`：
  ```json
  {
    "optimal_confidence": 12,
    "event_count": 1,
    "solution_count": "1",
    "canonical_groups": [["h1", "h2"]],
    "pair_relations": [{"a": "h1", "b": "h2", "relation": "always"}]
  }
  ```
  失败 `400`（仅含按路径的字段错误，不夹带任何结果）：
  ```json
  {"errors": [{"path": "hits[1].time", "message": "..."}]}
  ```

- `POST /audit` 可选字段 `alternatives`：脉冲形状拟合器对同一段波形给出的
  **互斥候选命中家族**。每项含唯一 ASCII `family` 标识与 2–6 个互异命中
  标识 `members`；同一家族至多一个成员可被分组，**作为噪声的成员不占用
  家族**，一个命中可同时属于多个家族。未提供 `alternatives`（或为空数组）
  时响应与无家族语义完全一致。
  ```json
  {
    "window": 2,
    "detectors": ["A", "B"],
    "hits": [
      {"id": "h1", "detector": "A", "time": 0, "confidence": 5},
      {"id": "h2", "detector": "B", "time": 1, "confidence": 7},
      {"id": "h3", "detector": "A", "time": 9, "confidence": 6},
      {"id": "h4", "detector": "B", "time": 9, "confidence": 8}
    ],
    "alternatives": [
      {"family": "shape-7", "members": ["h1", "h3"]}
    ]
  }
  ```
  按 `(时刻, 标识)` 排序后，**任一切分处同时含已出现与未出现成员的家族
  不得超过 6 个**，否则以路径 `alternatives` 返回 400。悬空成员引用、
  重复家族、家族内重复成员、成员数越界等均按具体路径拒绝（如
  `alternatives[2].members[1]`）。同一事件不可能同时包含同家族的两个
  成员，这样的命中对也不会出现在 `pair_relations` 中。

成对归属 `relation`：

- `always`（必然）：该可同组命中对出现在 **所有** 最优解中；
- `optional`（可选）：出现在部分而非全部最优解中；
- `never`（从不）：可同组（探测器不同、时刻差 ≤ W）但不在任何最优解中同组。

`solution_count` 为任意精度整数的十进制字符串。

## 算法（不枚举完整方案）

按 `(时刻, 标识)` 排序后做扫描线动态规划。处理命中 i 时，所有“已被早先
事件占用、尚未关窗”的命中都落在 `[t_i, t_i+W]` 闭窗内，至多 10 个，
因此窗口占用是 ≤10 位的位掩码（≤1024 态）。互斥家族联合维护为“活跃
家族已用掩码”：家族位只在其首成员至末成员之间存活（末成员决议后过期），
任一切分处敞开家族 ≤6 个，故家族部分 ≤64 态，组合状态
`F[i][窗口掩码][家族掩码]` 至多 1024×64。事件转移枚举空闲位的子掩码
（三态 3^(k-1)，k≤10），同事件含同家族两成员的候选预先剔除；候选事件
集合只取决于 `(i, 窗口掩码)`，子掩码枚举做一次后由前向/后向/规范/计数
四个阶段复用。由前向/后向 DP 得到：

- 最大可信度和、最少事件数（词典型比较）；
- 任意精度最优方案计数（Python 大整数）；
- 词典序最小的规范解（按状态记忆化的后缀裁决）；
- 每个候选事件在**受限后全部最优解**中的出现次数 = 前向方案数 × 后向
  方案数，进而得到每对命中的必然 / 可选 / 从不归属。

## 运行

```bash
# 宿主机端口可配置（默认 8080）
HOST_PORT=9000 docker compose up --build app

# verify 单次服务：测试 + 构建检查 + 算法核对 + HTTP 冒烟，按结果退出
docker compose --profile verify run --rm verify
```

本地开发：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
.venv/bin/gunicorn -b 127.0.0.1:8080 app.web:app
BASE_URL=http://127.0.0.1:8080 .venv/bin/python verify.py
```
