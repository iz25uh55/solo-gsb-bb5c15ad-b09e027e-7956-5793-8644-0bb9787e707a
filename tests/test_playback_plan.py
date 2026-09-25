"""倍速播放方案核心逻辑单元测试 + 暴力枚举对拍（纯标准库）。

暴力枚举每块 3 档语速的全部组合（至多 3^k），按
（总播放时长不超目标可行；总代价升序、总播放时长降序、语速序列字典序升序）
取最优，与 plan_playback 对比。
"""

import itertools
import random
import unittest
from math import ceil
from types import SimpleNamespace

from app.errors import DomainError
from app.playback_plan import SPEEDS, plan_playback


def W(start, end, text="词", speaker="A"):
    return SimpleNamespace(start_ms=start, end_ms=end, text=text, speaker=speaker)


def brute_force(words, target_ms):
    """独立实现：枚举全部语速组合，返回 (语速序列, 块列表, 总时长) 或 None。"""
    # 同一说话人连续词合块
    ranges = []
    start = 0
    for i in range(1, len(words) + 1):
        if i == len(words) or words[i].speaker != words[i - 1].speaker:
            ranges.append((start, i - 1))
            start = i
    durations = [words[e].end_ms - words[s].start_ms for s, e in ranges]
    silences = [
        words[ranges[b + 1][0]].start_ms - words[ranges[b][1]].end_ms
        for b in range(len(ranges) - 1)
    ]
    silence_total = sum(silences)
    budget = target_ms - silence_total

    best_key = None
    best_seq = None
    for seq in itertools.product(SPEEDS, repeat=len(ranges)):
        weight = sum(ceil(d * 100 / sp) for d, sp in zip(durations, seq))
        if weight > budget:
            continue
        cost = sum(d * (sp - 100) ** 2 for d, sp in zip(durations, seq))
        total = silence_total + weight
        key = (cost, -total, seq)
        if best_key is None or key < best_key:
            best_key = key
            best_seq = seq
    if best_seq is None:
        return None
    return best_seq, ranges, durations, silences, silence_total


class BasicPlanTests(unittest.TestCase):
    def setUp(self):
        # 块 0：A=[0,1] 首词 0 至末词 1（0～1000ms）；块间静音 500ms；
        # 块 1：B=[2]（1500～2500ms，1000ms）
        self.words = [
            W(0, 600, "甲", "A"),
            W(600, 1000, "乙", "A"),
            W(1500, 2500, "丙", "B"),
        ]

    def test_all_hundred_percent_when_target_loose(self):
        blocks, total, original = plan_playback(self.words, 10**9)
        self.assertEqual(original, 2500)
        self.assertEqual(total, 2500)
        self.assertEqual([b["speed_percent"] for b in blocks], [100, 100])
        # 块内首词 start_ms 至末词 end_ms 整段；播放时刻含块间静音
        self.assertEqual(
            blocks[0],
            {
                "index": 0,
                "speaker": "A",
                "start_word": 0,
                "end_word": 1,
                "original_start_ms": 0,
                "original_end_ms": 1000,
                "original_duration_ms": 1000,
                "speed_percent": 100,
                "playback_duration_ms": 1000,
                "playback_start_ms": 0,
                "playback_end_ms": 1000,
            },
        )
        self.assertEqual(
            (
                blocks[1]["original_start_ms"],
                blocks[1]["original_end_ms"],
                blocks[1]["playback_start_ms"],
                blocks[1]["playback_end_ms"],
            ),
            (1500, 2500, 1500, 2500),
        )

    def test_target_forces_speedup(self):
        # 原总时长 2500（块时长各 1000、静音 500）。目标 2300 =>
        # 块时长预算 1800：(100,125) 权重 1000+800=1800、代价 625000；
        # (125,100) 代价相同，时长也相同（1800）=> 字典序取 (100,125)
        blocks, total, _ = plan_playback(self.words, 2300)
        self.assertEqual([b["speed_percent"] for b in blocks], [100, 125])
        self.assertEqual(total, 2300)
        self.assertEqual(blocks[1]["playback_start_ms"], 1500)  # 1000 + 静音 500
        self.assertEqual(blocks[1]["playback_end_ms"], 2300)

    def test_lex_tie_broken_toward_earlier_blocks(self):
        # 目标 2099 => 块时长预算 1599。(125,150) 与 (150,125) 权重均
        # 800+667=1467、代价均 3,125,000（更便宜的 (125,125) 权重 1600
        # 差 1ms 而不可行）；同代价同总时长，字典序取 (125,150)。
        blocks, total, _ = plan_playback(self.words, 2099)
        self.assertEqual([b["speed_percent"] for b in blocks], [125, 150])
        self.assertEqual(total, 1967)  # 500 静音 + 1467 块播放

    def test_equal_cost_prefers_longer_total(self):
        # 块 0=1600ms、块 1=400ms（端点相接，无块间静音）。目标 1599：
        # (125,150) 权重 1280+267=1547，(150,125) 权重 1067+320=1387，
        # 代价均为 1600×625 + 400×2500 = 2,000,000（更便宜的 (125,125)
        # 权重 1600、(125,100) 权重 1680 均超预算）；同代价取总时长更长者。
        words = [W(0, 1600, "甲", "A"), W(1600, 2000, "乙", "B")]
        blocks, total, _ = plan_playback(words, 1599)
        self.assertEqual([b["speed_percent"] for b in blocks], [125, 150])
        self.assertEqual(total, 1547)

    def test_infeasible_when_target_below_fastest(self):
        # 最紧方案：两块均 150%，权重 ceil(200000/150)=667×2=1334，加静音 500
        with self.assertRaises(DomainError) as ctx:
            plan_playback(self.words, 1833)
        err = ctx.exception
        self.assertEqual(err.code, "infeasible_playback_plan")
        self.assertEqual(err.extra["block_count"], 2)
        self.assertEqual(err.extra["target_ms"], 1833)
        self.assertEqual(err.extra["minimum_duration_ms"], 1834)
        # 恰好可行
        blocks, total, _ = plan_playback(self.words, 1834)
        self.assertEqual([b["speed_percent"] for b in blocks], [150, 150])
        self.assertEqual(total, 1834)

    def test_target_shorter_than_interblock_silence(self):
        # 块间静音 500ms：target=499 时预算为负，立即不可行
        with self.assertRaises(DomainError) as ctx:
            plan_playback(self.words, 499)
        self.assertEqual(ctx.exception.code, "infeasible_playback_plan")
        self.assertEqual(ctx.exception.extra["minimum_duration_ms"], 1834)

    def test_zero_duration_blocks(self):
        # 零时长词/块：任何语速播放时长均为 0、代价均为 0，全部取字典序 100
        words = [W(0, 0, "甲", "A"), W(0, 0, "乙", "B")]
        blocks, total, original = plan_playback(words, 0)
        self.assertEqual([b["speed_percent"] for b in blocks], [100, 100])
        self.assertEqual((total, original), (0, 0))

    def test_single_block(self):
        words = [W(100, 1100, "甲", "A"), W(1100, 2100, "乙", "A")]
        blocks, total, original = plan_playback(words, 1500)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["original_duration_ms"], 2000)
        self.assertEqual(blocks[0]["speed_percent"], 150)  # ceil(200000/150)=1334
        self.assertEqual(blocks[0]["playback_duration_ms"], 1334)
        self.assertEqual(total, 1334)
        self.assertEqual(original, 2000)

    def test_silence_kept_original_duration(self):
        # 大块间静音 1000ms 不被压缩：目标 2600 => 块时长预算 1600，
        # 最优 (125,125)：两块各播 800，代价 1,250,000。
        words = [W(0, 1000, "甲", "A"), W(2000, 3000, "乙", "B")]
        blocks, total, _ = plan_playback(words, 2600)
        self.assertEqual([b["speed_percent"] for b in blocks], [125, 125])
        # 块 0 播 800（0～800），静音保持 1000，块 1 播 800（1800～2600）
        self.assertEqual(blocks[0]["playback_start_ms"], 0)
        self.assertEqual(blocks[0]["playback_end_ms"], 800)
        self.assertEqual(blocks[1]["playback_start_ms"], 1800)
        self.assertEqual(blocks[1]["playback_end_ms"], 2600)
        self.assertEqual(total, 2600)

    def test_silence_makes_target_infeasible(self):
        # 目标短于“块间静音 + 最紧块播放”时不可行：静音 1000，
        # 两块 1000ms 全程 150% 共 1334 => 下限 2334
        words = [W(0, 1000, "甲", "A"), W(2000, 3000, "乙", "B")]
        with self.assertRaises(DomainError) as ctx:
            plan_playback(words, 2333)
        self.assertEqual(ctx.exception.code, "infeasible_playback_plan")
        self.assertEqual(ctx.exception.extra["minimum_duration_ms"], 2334)


class ChunkLimitTests(unittest.TestCase):
    def test_more_than_twenty_blocks_rejected(self):
        words = [
            W(i * 100, i * 100 + 100, "词", "A" if i % 2 == 0 else "B")
            for i in range(21)
        ]
        with self.assertRaises(DomainError) as ctx:
            plan_playback(words, 10**9)
        err = ctx.exception
        self.assertEqual(err.code, "too_many_playback_chunks")
        self.assertEqual(err.extra["block_count"], 21)
        self.assertEqual(err.extra["max_blocks"], 20)

    def test_twenty_blocks_accepted(self):
        words = [
            W(i * 100, i * 100 + 100, "词", "A" if i % 2 == 0 else "B")
            for i in range(20)
        ]
        blocks, total, original = plan_playback(words, 10**9)
        self.assertEqual(len(blocks), 20)
        self.assertEqual(original, 2000)
        self.assertEqual(total, 2000)

    def test_many_words_same_speaker_one_block(self):
        words = [W(i * 100, i * 100 + 100, "词", "A") for i in range(100)]
        blocks, _, _ = plan_playback(words, 10**9)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["start_word"], 0)
        self.assertEqual(blocks[0]["end_word"], 99)
        self.assertEqual(blocks[0]["original_duration_ms"], 10000)

    def test_word_validation_reused(self):
        with self.assertRaises(DomainError) as ctx:
            plan_playback([], 0)
        self.assertEqual(ctx.exception.code, "empty_words")
        with self.assertRaises(DomainError) as ctx:
            plan_playback([W(1000, 2000), W(900, 1500)], 0)
        self.assertEqual(ctx.exception.code, "time_out_of_order")
        with self.assertRaises(DomainError) as ctx:
            plan_playback([W(0, 2000), W(1500, 3000)], 0)
        self.assertEqual(ctx.exception.code, "overlapping_time_range")


class DifferentialTests(unittest.TestCase):
    def test_matches_brute_force(self):
        random.seed(20260925)
        feasible = 0
        infeasible = 0
        for trial in range(4000):
            # 随机词序列：2～7 个说话人区间，每区间 1～3 词，含零时长与停顿
            words = []
            t = 0
            run_count = random.randint(2, 7)
            for run in range(run_count):
                speaker = chr(ord("A") + run)
                for _ in range(random.randint(1, 3)):
                    t += random.choice([0, 0, 0, 200, 600])  # 词间/块间停顿
                    dur = random.choice([0, 1, 100, 333, 1000])
                    words.append(W(t, t + dur, "词", speaker))
                    t += dur
            original_total = words[-1].end_ms - words[0].start_ms
            target = random.choice(
                [0, original_total // 2, original_total, original_total + 1, 10**9]
            )

            expected = brute_force(words, target)
            if expected is None:
                infeasible += 1
                with self.assertRaises(DomainError) as ctx:
                    plan_playback(words, target)
                self.assertEqual(
                    ctx.exception.code, "infeasible_playback_plan", f"trial {trial}"
                )
                continue
            feasible += 1
            exp_seq, ranges, durations, silences, silence_total = expected
            blocks, total, original = plan_playback(words, target)
            self.assertEqual(
                [b["speed_percent"] for b in blocks], list(exp_seq), f"trial {trial}"
            )
            # 自洽校验
            cursor = 0
            for i, block in enumerate(blocks):
                s, e = ranges[i]
                self.assertEqual(block["start_word"], s)
                self.assertEqual(block["end_word"], e)
                self.assertEqual(block["speaker"], words[s].speaker)
                self.assertEqual(
                    block["original_duration_ms"],
                    words[e].end_ms - words[s].start_ms,
                )
                self.assertEqual(
                    block["playback_duration_ms"],
                    ceil(durations[i] * 100 / exp_seq[i]),
                )
                self.assertEqual(block["playback_start_ms"], cursor)
                cursor += block["playback_duration_ms"]
                self.assertEqual(block["playback_end_ms"], cursor)
                if i < len(ranges) - 1:
                    cursor += silences[i]  # 块间静音保持原时长
            self.assertEqual(cursor, total)
            self.assertEqual(total, silence_total + sum(
                ceil(d * 100 / sp) for d, sp in zip(durations, exp_seq)
            ))
            self.assertLessEqual(total, target)
            self.assertEqual(original, original_total)
        self.assertGreater(feasible, 1500)
        self.assertGreater(infeasible, 50)


if __name__ == "__main__":
    unittest.main()
