"""倍速播放方案核心逻辑（纯标准库，不依赖 FastAPI，便于单测）。

把按时间排列的词序列按**同一说话人连续区间**合并为语速块（speaker run），
每块从 100%、125%、150% 三档语速中选一档：块内首词 ``start_ms`` 至末词
``end_ms`` 整段变速（块内词间停顿一并等比压缩），**块间静音保持原时长**。
块播放时长按 ``ceil(原时长 × 100 / 语速)`` 向上取整。总播放时长 = 块间
静音之和（常量）+ 各块播放时长，要求不超过目标最长播放时长。

目标按字典序三级裁决：

1. 最小化总变速代价 ``Σ 原时长 × (语速 - 100)²``——100% 代价 0，
   125% 代价 ``625×原时长``，150% 代价 ``2500×原时长``；
2. 同代价取总播放时长最长（块间静音为常量，等价于各块播放时长之和最长）；
3. 再取语速序列字典序最小（语速数值按块顺序逐位比较，100 < 125 < 150）。

算法（折半枚举 Meet-in-the-Middle；块数 k ≤ 20、每块 3 档 =>
每半至多 3¹⁰ = 59049 个组合，复杂度与原时长量级无关、内存有硬上界）：

* **退化直出**：全程 100% 即满足目标时，代价 0、总时长最长、语速序列
  字典序最小，直接返回；全程 150% 仍超预算（或目标短于块间静音）则无解；
* **折半枚举**：在中点 m=k//2 处分为左右两半，各枚举全部 3^|half| 个
  （代价, 播放时长）组合。右半按代价聚合出
  ``frontier = [(代价, 该代价下最小播放时长)]`` 并做支配剪枝（代价升序、
  最小播放时长严格递减），另以 ``{代价: (最小, 最大播放时长)}`` 备查；
* **最小代价 C\***：对每个左半点 (cL, wL)，在右半前沿上二分求播放时长
  ``<= 预算-wL`` 的最小右半代价（递减序列取第一个达标位置），
  ``cL + cR`` 的最小值即 C*；
* **最长总时长 W\***：在 ``cL + cR = C*`` 的全部配对中取播放时长之和
  最大者（右半恰花 cR 时取其最大播放时长）；
* **贪心重建**：增量预建每个后缀起点的左尾 ``[t,m)`` 与右尾
  ``[max(t,m),k)`` 点表并转为 ``(代价, 播放时长)`` 元组集合（由相邻表
  前拼一个块生成，总体仅 3^⌈k/2⌉ 量级）。逐块按 100%→125%→150% 试档，
  用集合直查判定“左尾 + 右尾是否存在代价恰为 C\*-已选代价、播放时长恰为
  W\*-已选时长且不超预算的配对”。第一位能取 100% 就不取更高速档，
  归纳可得语速序列字典序最小。
"""

from bisect import bisect_left
from math import ceil

from .errors import DomainError
from .segmentation import validate_words

MAX_PLAYBACK_BLOCKS = 20

SPEEDS = (100, 125, 150)


def _fail(code, message, **extra):
    raise DomainError(code, message, **extra)


def _merge_blocks(words):
    """把同一说话人的连续词合并为块，返回 [(start_word, end_word), ...]。"""
    blocks = []
    start = 0
    for i in range(1, len(words) + 1):
        if i == len(words) or words[i].speaker != words[i - 1].speaker:
            blocks.append((start, i - 1))
            start = i
    return blocks


def _enumerate(options, lo, hi):
    """枚举块区间 [lo, hi) 全部语速组合。

    返回 (代价列表, 播放时长列表)，按下标对齐（每个组合一项）。
    """
    costs = [0]
    weights = [0]
    for i in range(lo, hi):
        new_costs = []
        new_weights = []
        for cost, weight in zip(costs, weights):
            for w, c in options[i]:
                new_costs.append(cost + c)
                new_weights.append(weight + w)
        costs, weights = new_costs, new_weights
    return costs, weights


def plan_playback(words, target_ms):
    """求最优倍速播放方案，返回 (blocks, total_duration_ms, total_original_ms)。

    参数 words 中每个对象需具备属性：start_ms / end_ms / text / speaker
    （沿用同一套词序与时间校验）；target_ms 为目标最长播放总时长（非负整数）。

    blocks 每项为 dict：index / speaker / start_word / end_word /
    original_start_ms / original_end_ms / original_duration_ms /
    speed_percent / playback_duration_ms / playback_start_ms /
    playback_end_ms；播放时刻从 0 起依次累加“块播放时长 + 块间静音”，
    块间静音保持原时长且不并入任一块。块数超过 20 抛出
    DomainError(code="too_many_playback_chunks")；无可行方案抛出
    DomainError(code="infeasible_playback_plan")（均为 HTTP 422）。
    """
    validate_words(words)
    ranges = _merge_blocks(words)
    block_count = len(ranges)
    if block_count > MAX_PLAYBACK_BLOCKS:
        _fail(
            "too_many_playback_chunks",
            f"同一说话人连续区间共 {block_count} 块，超过语速块上限 "
            f"{MAX_PLAYBACK_BLOCKS} 块",
            block_count=block_count,
            max_blocks=MAX_PLAYBACK_BLOCKS,
        )

    # ---- 块原时长与块间静音（末词 end_ms - 首词 start_ms 整段变速）----
    durations = [words[e].end_ms - words[s].start_ms for s, e in ranges]
    silences = [
        words[ranges[b + 1][0]].start_ms - words[ranges[b][1]].end_ms
        for b in range(block_count - 1)
    ]
    silence_total = sum(silences)
    budget = target_ms - silence_total  # 各块播放时长之和的上限

    # 每块每档：(播放时长, 代价)。三档播放时长随语速非升、代价非降。
    options = [
        [(ceil(d * 100 / speed), d * (speed - 100) ** 2) for speed in SPEEDS]
        for d in durations
    ]

    # ---- 退化情形 ----
    all_slow_weight = sum(opt[0][0] for opt in options)   # 全程 100%
    all_fast_weight = sum(opt[2][0] for opt in options)   # 全程 150%
    if all_slow_weight <= budget:
        return _build_output(
            words, ranges, durations, silences, silence_total,
            [100] * block_count, block_count,
        )
    if budget < 0 or all_fast_weight > budget:
        # 连全程 150% 的最紧播放时长之和都超过预算（或目标短于块间静音）
        _fail(
            "infeasible_playback_plan",
            "不存在总播放时长不超过目标的语速方案",
            block_count=block_count,
            target_ms=target_ms,
            minimum_duration_ms=silence_total + all_fast_weight,
        )

    # ---- 折半枚举：左半 [0, mid)、右半 [mid, k) ----
    mid = block_count // 2
    left_costs, left_weights = _enumerate(options, 0, mid)
    right_costs, right_weights = _enumerate(options, mid, block_count)

    # 右半按代价聚合：{代价: [最小播放时长, 最大播放时长]}
    range_by_cost = {}
    for c, w in zip(right_costs, right_weights):
        if c in range_by_cost:
            lo_w, hi_w = range_by_cost[c]
            range_by_cost[c] = [min(lo_w, w), max(hi_w, w)]
        else:
            range_by_cost[c] = [w, w]

    # 支配前沿：代价升序、最小播放时长严格递减（更贵且不更短的点对
    # “给定剩余预算找最小可行代价”毫无用处）
    frontier_costs = []
    frontier_neg_min = []
    best_weight = None
    for c in sorted(range_by_cost):
        min_w = range_by_cost[c][0]
        if best_weight is None or min_w < best_weight:
            frontier_costs.append(c)
            frontier_neg_min.append(-min_w)  # 递增，供 bisect_left
            best_weight = min_w

    def min_right_cost_within(room):
        """右半播放时长 <= room 时的最小代价；不可达返回 None。

        frontier 最小播放时长严格递减：第一个 <= room 的位置即答案，
        等价于在递增值 -min_weight 上二分第一个 >= -room。
        """
        idx = bisect_left(frontier_neg_min, -room)
        if idx == len(frontier_costs):
            return None
        return frontier_costs[idx]

    # ---- 最小代价 C* ----
    min_cost = None
    for cL, wL in zip(left_costs, left_weights):
        cR = min_right_cost_within(budget - wL)
        if cR is not None:
            total = cL + cR
            if min_cost is None or total < min_cost:
                min_cost = total

    # ---- 同代价最长总时长 W*：总价恰为 C* 的配对取播放时长之和最大 ----
    max_weight = -1
    for cL, wL in zip(left_costs, left_weights):
        entry = range_by_cost.get(min_cost - cL)
        if entry is None:
            continue
        min_w, max_w = entry
        if min_w <= budget - wL:
            max_weight = max(max_weight, wL + max_w)

    # ---- 各后缀起点的点表，供贪心重建拼对 ----
    # 前 t 块固定后，未选块在中点处拆为两段：左尾 [t,mid) 与右尾
    # [max(t,mid), k)（t >= mid 时左尾为空；t < mid 时右尾为整个右半）。
    # 点表均由相邻表“前拼一个块”增量构造，避免重复枚举：
    # 左尾表从空表 [mid,mid) 起向左长，右尾表从空表 [k,k) 起向左长。
    def _prepend(options_i, costs, weights):
        nc, nw = [], []
        for c, w in zip(costs, weights):
            for w_opt, c_opt in options_i:
                nc.append(c + c_opt)
                nw.append(w + w_opt)
        return nc, nw

    empty = ([0], [0])
    left_tail = [None] * (block_count + 1)
    left_tail[mid] = empty
    for t in range(mid - 1, -1, -1):
        left_tail[t] = _prepend(options[t], *left_tail[t + 1])
    for t in range(mid + 1, block_count + 1):
        left_tail[t] = empty
    right_side = [None] * (block_count + 1)
    for t in range(mid):
        right_side[t] = (right_costs, right_weights)  # 整个右半 [mid, k)
    right_side[block_count] = empty
    for t in range(block_count - 1, mid - 1, -1):
        right_side[t] = _prepend(options[t], *right_side[t + 1])

    # 两侧点表各建 (代价, 播放时长) 元组集合：重建查询为“是否存在
    # cL+cR=need_cost、wL+wR=need_weight 且不超预算”的精确配对，
    # 元组集合上可 O(1) 判定（每侧至多 3^10 个不同点）。
    rs_pairs = [None] * (block_count + 1)
    lt_pairs = [None] * (block_count + 1)
    for t in range(block_count + 1):
        rs_costs, rs_weights = right_side[t][0], right_side[t][1]
        lt_costs, lt_weights = left_tail[t][0], left_tail[t][1]
        rs_pairs[t] = set(zip(rs_costs, rs_weights))
        lt_pairs[t] = set(zip(lt_costs, lt_weights))

    def completion_exists(t, prefix_cost, prefix_weight):
        """前 t 块已固定（已花 prefix_cost/占 prefix_weight）时，未选块是否
        能恰好补到总价 min_cost、总播放 max_weight 且不超预算。"""
        need_cost = min_cost - prefix_cost
        need_weight = max_weight - prefix_weight
        cap = budget - prefix_weight
        if need_cost < 0 or need_weight < 0 or cap < 0:
            return False

        left_pairs, right_pairs = lt_pairs[t], rs_pairs[t]
        # 在较小的一侧枚举 (c, w)，另一侧直接查 (need_c-c, need_w-w)
        outer, inner = (
            (left_pairs, right_pairs)
            if len(left_pairs) <= len(right_pairs)
            else (right_pairs, left_pairs)
        )
        for c_out, w_out in outer:
            c_in, w_in = need_cost - c_out, need_weight - w_out
            if c_in < 0 or w_in < 0 or w_out + w_in > cap:
                continue
            if (c_in, w_in) in inner:
                return True
        return False

    # ---- 贪心重建：逐块取存在最优补全的最低档（100→125→150）----
    speed_indices = []
    prefix_cost = 0
    prefix_weight = 0
    for i in range(block_count):
        for speed_idx, (w, c) in enumerate(options[i]):
            # 试选本档后，从 i+1 起的后缀须能补到 (C*, W*)；
            # 注意补全判定不依赖本块之前各块的具体档位，仅依赖累计量。
            if completion_exists(
                i + 1, prefix_cost + c, prefix_weight + w
            ):
                speed_indices.append(speed_idx)
                prefix_cost += c
                prefix_weight += w
                break
        else:  # pragma: no cover - C*、W* 的存在性已保证必有补全
            _fail(
                "infeasible_playback_plan",
                "不存在总播放时长不超过目标的语速方案",
                block_count=block_count,
                target_ms=target_ms,
            )
    speeds = [SPEEDS[idx] for idx in speed_indices]

    return _build_output(
        words, ranges, durations, silences, silence_total,
        speeds, block_count,
    )


def _build_output(
    words, ranges, durations, silences, silence_total, speeds, block_count
):
    """由语速序列组装播放块与总时长；播放时刻从 0 累加（块间静音保持原时长）。"""
    blocks_out = []
    cursor = 0
    for i, (s, e) in enumerate(ranges):
        play_duration = ceil(durations[i] * 100 / speeds[i])
        blocks_out.append(
            {
                "index": i,
                "speaker": words[s].speaker,
                "start_word": s,
                "end_word": e,
                "original_start_ms": words[s].start_ms,
                "original_end_ms": words[e].end_ms,
                "original_duration_ms": durations[i],
                "speed_percent": speeds[i],
                "playback_duration_ms": play_duration,
                "playback_start_ms": cursor,
                "playback_end_ms": cursor + play_duration,
            }
        )
        cursor += play_duration
        if i < block_count - 1:
            cursor += silences[i]
    return blocks_out, cursor, silence_total + sum(durations)
