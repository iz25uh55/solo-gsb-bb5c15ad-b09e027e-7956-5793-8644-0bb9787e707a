"""HTTP 接口层测试。

依赖 fastapi + httpx（TestClient 基于 httpx）。
未安装时自动跳过；核心算法测试见 test_segmentation.py（纯标准库即可运行）。
"""

import unittest

try:
    from fastapi.testclient import TestClient
    from app.main import app
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False


@unittest.skipUnless(_HAS_DEPS, "fastapi/httpx 未安装，跳过接口层测试")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_basic_segmentation(self):
        resp = self.client.post(
            "/segments",
            json={
                "words": [
                    {"start_ms": 0, "end_ms": 2000, "text": "第一句。", "speaker": "A"},
                    {"start_ms": 2000, "end_ms": 4200, "text": "第二句！", "speaker": "B"},
                    {"start_ms": 4200, "end_ms": 6200, "text": "第三句？", "speaker": "B"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_score"], 0)
        self.assertEqual(len(data["segments"]), 2)  # B 的两句合段更优（末段零分）
        self.assertIn("1\n00:00:00,000 --> 00:00:02,000", data["srt"])

    def test_overlap_rejected(self):
        resp = self.client.post(
            "/segments",
            json={
                "words": [
                    {"start_ms": 0, "end_ms": 2000, "text": "甲", "speaker": "A"},
                    {"start_ms": 1500, "end_ms": 3000, "text": "乙", "speaker": "A"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "overlapping_time_range")

    def test_out_of_order_rejected(self):
        resp = self.client.post(
            "/segments",
            json={
                "words": [
                    {"start_ms": 2000, "end_ms": 3000, "text": "甲", "speaker": "A"},
                    {"start_ms": 1000, "end_ms": 1500, "text": "乙", "speaker": "A"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "time_out_of_order")

    def test_infeasible_returns_earliest_prefix(self):
        resp = self.client.post(
            "/segments",
            json={
                "words": [
                    {"start_ms": 0, "end_ms": 800, "text": "短", "speaker": "A"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 422)
        err = resp.json()["error"]
        self.assertEqual(err["code"], "infeasible_segmentation")
        self.assertEqual(err["last_word_index"], 0)

    def test_invalid_schema(self):
        resp = self.client.post("/segments", json={"words": []})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")


@unittest.skipUnless(_HAS_DEPS, "fastapi/httpx 未安装，跳过接口层测试")
class AdBreakApiTests(unittest.TestCase):
    WORDS = [
        {"start_ms": 0, "end_ms": 500, "text": "甲", "speaker": "A"},
        {"start_ms": 1300, "end_ms": 1800, "text": "乙", "speaker": "A"},
        {"start_ms": 2700, "end_ms": 3200, "text": "丙", "speaker": "B"},
        {"start_ms": 3700, "end_ms": 4200, "text": "丁", "speaker": "B"},
    ]
    # 三处静音：[500,1300]=800ms（中点 900）、[1800,2700]=900ms（中点 2250）、
    # [3200,3700]=500ms（中点 3450）

    def setUp(self):
        self.client = TestClient(app)

    def post(self, **overrides):
        payload = {
            "words": self.WORDS,
            "break_count": 2,
            "min_silence_ms": 500,
            "min_spacing_ms": 0,
        }
        payload.update(overrides)
        return self.client.post("/ad-breaks", json=payload)

    def test_basic_ad_breaks(self):
        resp = self.post()
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # 总静音最大：900 + 800（中点间隔 1350 与 1200 均不受 min_spacing=0 限制）
        self.assertEqual(data["total_silence_ms"], 1700)
        self.assertEqual(
            data["breaks"],
            [
                {
                    "index": 0,
                    "time_ms": 900,
                    "silence_start_ms": 500,
                    "silence_end_ms": 1300,
                    "silence_ms": 800,
                },
                {
                    "index": 1,
                    "time_ms": 2250,
                    "silence_start_ms": 1800,
                    "silence_end_ms": 2700,
                    "silence_ms": 900,
                },
            ],
        )

    def test_spacing_forces_spread(self):
        # 中点两两相距 1350/1200/2550：间隔 >= 2000 时只能取首尾
        resp = self.post(min_spacing_ms=2000)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual([b["time_ms"] for b in data["breaks"]], [900, 3450])
        self.assertEqual(data["total_silence_ms"], 1300)

    def test_min_silence_filters_candidates(self):
        # 阈值 850 只剩 900ms 一处候选，选 2 个无解
        resp = self.post(min_silence_ms=850)
        self.assertEqual(resp.status_code, 422)
        err = resp.json()["error"]
        self.assertEqual(err["code"], "infeasible_ad_breaks")
        self.assertEqual(err["candidate_count"], 1)

    def test_infeasible_spacing(self):
        resp = self.post(min_spacing_ms=100_000)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "infeasible_ad_breaks")

    def test_break_count_out_of_range(self):
        for bad in (0, 6):
            resp = self.post(break_count=bad)
            self.assertEqual(resp.status_code, 422)
            self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_negative_or_zero_thresholds_rejected(self):
        for field, bad in (("min_silence_ms", 0), ("min_spacing_ms", -1)):
            resp = self.post(**{field: bad})
            self.assertEqual(resp.status_code, 422)
            self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_word_validation_reused(self):
        words = [dict(self.WORDS[1]), dict(self.WORDS[0])]  # 时间倒序
        resp = self.post(words=words, break_count=1)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "time_out_of_order")

    def test_segments_still_works(self):
        resp = self.client.post("/segments", json={"words": self.WORDS})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("segments", resp.json())


@unittest.skipUnless(_HAS_DEPS, "fastapi/httpx 未安装，跳过接口层测试")
class SpeakerReconcileApiTests(unittest.TestCase):
    WORDS = [
        {"start_ms": 0, "end_ms": 500, "text": "甲", "speaker": "A"},
        {"start_ms": 500, "end_ms": 1000, "text": "乙", "speaker": "A"},
        {"start_ms": 1000, "end_ms": 1500, "text": "丙", "speaker": "B"},
        {"start_ms": 1500, "end_ms": 2000, "text": "丁", "speaker": "A"},
    ]

    def setUp(self):
        self.client = TestClient(app)

    def post(self, **overrides):
        payload = {"words": self.WORDS, "locks": [], "switch_cost": 5}
        payload.update(overrides)
        return self.client.post("/speaker-reconcile", json=payload)

    def test_basic_reconcile_smooths(self):
        # A A B A，switch_cost=5：保留原标注需 2 次换人（成本 10），
        # 词 2 改为 A 仅 1 次改动（成本 1）
        resp = self.post()
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual([w["speaker"] for w in data["words"]], ["A"] * 4)
        self.assertEqual(data["changed_word_indices"], [2])
        self.assertEqual(data["total_cost"], 1)
        # 词序、文本、时间不变
        self.assertEqual(
            [(w["start_ms"], w["end_ms"], w["text"]) for w in data["words"]],
            [(0, 500, "甲"), (500, 1000, "乙"), (1000, 1500, "丙"), (1500, 2000, "丁")],
        )

    def test_lock_respected(self):
        # 锁定词 2 为 B，switch_cost=2：AABB 与 BBBB 同为成本 3，取字典序小的 AABB
        resp = self.post(locks=[{"word_index": 2, "speaker": "B"}], switch_cost=2)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual([w["speaker"] for w in data["words"]], ["A", "A", "B", "B"])
        self.assertEqual(data["changed_word_indices"], [3])
        self.assertEqual(data["total_cost"], 3)

    def test_lock_can_introduce_new_speaker(self):
        resp = self.post(locks=[{"word_index": 0, "speaker": "C"}], switch_cost=0)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["words"][0]["speaker"], "C")
        self.assertEqual(data["changed_word_indices"], [0])
        self.assertEqual(data["total_cost"], 1)

    def test_locks_may_be_omitted(self):
        resp = self.client.post(
            "/speaker-reconcile",
            json={"words": self.WORDS, "switch_cost": 5},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["changed_word_indices"], [2])

    def test_lock_index_out_of_range(self):
        for bad in (4, -1):
            resp = self.post(locks=[{"word_index": bad, "speaker": "A"}])
            self.assertEqual(resp.status_code, 422)
            err = resp.json()["error"]
            self.assertEqual(err["code"], "lock_index_out_of_range")
            self.assertEqual(err["word_index"], bad)

    def test_duplicate_lock(self):
        resp = self.post(
            locks=[{"word_index": 1, "speaker": "A"}, {"word_index": 1, "speaker": "B"}]
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "duplicate_lock")

    def test_too_many_speakers(self):
        words = [
            {"start_ms": i * 500, "end_ms": (i + 1) * 500, "text": "字", "speaker": s}
            for i, s in enumerate("ABCDEFGH")
        ]
        resp = self.post(words=words, locks=[{"word_index": 0, "speaker": "Z"}])
        self.assertEqual(resp.status_code, 422)
        err = resp.json()["error"]
        self.assertEqual(err["code"], "too_many_speakers")
        self.assertEqual(err["speaker_count"], 9)

    def test_switch_cost_out_of_range(self):
        for bad in (-1, 11):
            resp = self.post(switch_cost=bad)
            self.assertEqual(resp.status_code, 422)
            self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_lock_schema_invalid(self):
        resp = self.post(locks=[{"word_index": 0}])  # 缺 speaker
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_word_validation_reused(self):
        words = [dict(self.WORDS[1]), dict(self.WORDS[0])]  # 时间倒序
        resp = self.post(words=words)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "time_out_of_order")


@unittest.skipUnless(_HAS_DEPS, "fastapi/httpx 未安装，跳过接口层测试")
class ChapterAlignApiTests(unittest.TestCase):
    WORDS = [
        {"start_ms": 0, "end_ms": 500, "text": "大家好", "speaker": "A"},
        {"start_ms": 600, "end_ms": 1200, "text": "今天聊", "speaker": "A"},
        {"start_ms": 1300, "end_ms": 2000, "text": "动态规划", "speaker": "A"},
        {"start_ms": 2200, "end_ms": 2800, "text": "明天见", "speaker": "B"},
    ]

    def setUp(self):
        self.client = TestClient(app)

    def post(self, **overrides):
        payload = {
            "words": self.WORDS,
            "chapters": [
                {"title": "开场", "cue": "大家", "approx_ms": 0},
                {"title": "主题", "cue": "动态", "approx_ms": 1000},
            ],
            "min_gap_ms": 500,
        }
        payload.update(overrides)
        return self.client.post("/chapter-align", json=payload)

    def test_basic_alignment(self):
        resp = self.post()
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_deviation_ms"], 300)
        self.assertEqual(
            data["chapters"],
            [
                {
                    "index": 0,
                    "title": "开场",
                    "time_ms": 0,
                    "word_index": 0,
                    "char_start": 0,
                    "char_end": 2,
                },
                {
                    "index": 1,
                    "title": "主题",
                    "time_ms": 1300,
                    "word_index": 2,
                    "char_start": 0,
                    "char_end": 2,
                },
            ],
        )

    def test_cross_word_hit(self):
        words = [
            {"start_ms": 0, "end_ms": 500, "text": "动态", "speaker": "A"},
            {"start_ms": 500, "end_ms": 1000, "text": "规划", "speaker": "A"},
        ]
        resp = self.post(
            words=words,
            chapters=[{"title": "t", "cue": "动态规划", "approx_ms": 100}],
            min_gap_ms=1,
        )
        self.assertEqual(resp.status_code, 200)
        ch = resp.json()["chapters"][0]
        self.assertEqual(ch["time_ms"], 0)  # 候选时刻取命中首字符所在词
        self.assertEqual(ch["word_index"], 0)
        self.assertEqual([ch["char_start"], ch["char_end"]], [0, 2])  # 截断到词末

    def test_speaker_boundary_blocks_hit(self):
        words = [
            {"start_ms": 0, "end_ms": 100, "text": "ab", "speaker": "A"},
            {"start_ms": 100, "end_ms": 200, "text": "cd", "speaker": "B"},
        ]
        resp = self.post(
            words=words,
            chapters=[{"title": "t", "cue": "bc", "approx_ms": 0}],
            min_gap_ms=1,
        )
        self.assertEqual(resp.status_code, 422)
        err = resp.json()["error"]
        self.assertEqual(err["code"], "infeasible_chapter_alignment")
        self.assertEqual(err["chapter_index"], 0)

    def test_gap_infeasible(self):
        words = [
            {"start_ms": 0, "end_ms": 100, "text": "甲", "speaker": "A"},
            {"start_ms": 200, "end_ms": 300, "text": "乙", "speaker": "A"},
        ]
        resp = self.post(
            words=words,
            chapters=[
                {"title": "a", "cue": "甲", "approx_ms": 0},
                {"title": "b", "cue": "乙", "approx_ms": 0},
            ],
            min_gap_ms=1000,
        )
        self.assertEqual(resp.status_code, 422)
        err = resp.json()["error"]
        self.assertEqual(err["code"], "infeasible_chapter_alignment")
        self.assertEqual(err["chapter_count"], 2)
        self.assertEqual(err["min_gap_ms"], 1000)

    def test_min_gap_must_be_positive(self):
        for bad in (0, -1):
            resp = self.post(min_gap_ms=bad)
            self.assertEqual(resp.status_code, 422)
            self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_chapter_fields_required(self):
        # 空 chapters、空 cue、负 approx_ms 均为 422
        for chapters in (
            [],
            [{"title": "t", "cue": "", "approx_ms": 0}],
            [{"title": "t", "cue": "甲", "approx_ms": -1}],
            [{"title": "", "cue": "甲", "approx_ms": 0}],
        ):
            resp = self.post(chapters=chapters)
            self.assertEqual(resp.status_code, 422)
            self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_word_validation_reused(self):
        words = [dict(self.WORDS[1]), dict(self.WORDS[0])]  # 时间倒序
        resp = self.post(words=words)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "time_out_of_order")

    def test_segments_still_works(self):
        resp = self.client.post("/segments", json={"words": self.WORDS})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("segments", resp.json())


@unittest.skipUnless(_HAS_DEPS, "fastapi/httpx 未安装，跳过接口层测试")
class DuplicatePassagesApiTests(unittest.TestCase):
    WORDS = [
        {"start_ms": 0, "end_ms": 800, "text": "欢迎收听", "speaker": "A"},
        {"start_ms": 800, "end_ms": 1600, "text": "小宇宙", "speaker": "A"},
        {"start_ms": 1600, "end_ms": 2400, "text": "播客", "speaker": "A"},
        {"start_ms": 2400, "end_ms": 4000, "text": "今天聊动态规划。", "speaker": "B"},
        {"start_ms": 5000, "end_ms": 5800, "text": "欢迎收听", "speaker": "A"},
        {"start_ms": 5800, "end_ms": 6600, "text": "小宇宙", "speaker": "A"},
        {"start_ms": 6600, "end_ms": 7400, "text": "播客", "speaker": "A"},
    ]

    def setUp(self):
        self.client = TestClient(app)

    def post(self, **overrides):
        payload = {"words": self.WORDS, "min_words": 2, "min_gap_ms": 1000}
        payload.update(overrides)
        return self.client.post("/duplicate-passages", json=payload)

    def test_basic_match(self):
        resp = self.post()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json()["match"],
            {
                "word_count": 3,
                "text": "欢迎收听小宇宙播客",
                "first": {
                    "start_word": 0,
                    "end_word": 2,
                    "start_ms": 0,
                    "end_ms": 2400,
                },
                "second": {
                    "start_word": 4,
                    "end_word": 6,
                    "start_ms": 5000,
                    "end_ms": 7400,
                },
            },
        )

    def test_match_null_when_below_min_words(self):
        resp = self.post(min_words=4)  # 最长重复为 3 词
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json()["match"])

    def test_match_null_when_gap_too_large(self):
        resp = self.post(min_gap_ms=6000)  # 两处首词仅相隔 5000ms
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json()["match"])

    def test_speaker_must_match(self):
        words = [dict(w) for w in self.WORDS]
        words[4]["speaker"] = "B"  # 破坏第二处的符号序列
        resp = self.post(words=words)
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json()["match"])

    def test_min_words_out_of_range(self):
        for bad in (1, 21):
            resp = self.post(min_words=bad)
            self.assertEqual(resp.status_code, 422)
            self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_negative_min_gap_rejected(self):
        resp = self.post(min_gap_ms=-1)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_words_over_2000_rejected(self):
        words = [
            {"start_ms": i * 500, "end_ms": i * 500 + 400, "text": "字", "speaker": "A"}
            for i in range(2001)
        ]
        resp = self.post(words=words)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")
        # 2000 词恰好在上限内
        resp = self.post(words=words[:2000])
        self.assertEqual(resp.status_code, 200)

    def test_word_validation_reused(self):
        words = [dict(self.WORDS[1]), dict(self.WORDS[0])]  # 时间倒序
        resp = self.post(words=words)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "time_out_of_order")

    def test_segments_still_works(self):
        resp = self.client.post("/segments", json={"words": self.WORDS})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("segments", resp.json())


@unittest.skipUnless(_HAS_DEPS, "fastapi/httpx 未安装，跳过接口层测试")
class RedactPreviewApiTests(unittest.TestCase):
    WORDS = [
        {"start_ms": 0, "end_ms": 800, "text": "记得使用", "speaker": "A"},
        {"start_ms": 800, "end_ms": 1400, "text": "优惠码", "speaker": "A"},
        {"start_ms": 1400, "end_ms": 2000, "text": "八八八", "speaker": "A"},
        {"start_ms": 2200, "end_ms": 3400, "text": "别错过", "speaker": "B"},
    ]

    def setUp(self):
        self.client = TestClient(app)

    def post(self, **overrides):
        payload = {"words": self.WORDS, "phrases": ["使用优惠码", "八八八"]}
        payload.update(overrides)
        return self.client.post("/redact-preview", json=payload)

    def test_basic_redact(self):
        resp = self.post()
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(
            [w["text"] for w in data["words"]],
            ["记得██", "███", "███", "别错过"],
        )
        # 词序、时间、说话人保持不变
        self.assertEqual(
            [(w["start_ms"], w["end_ms"], w["speaker"]) for w in data["words"]],
            [(0, 800, "A"), (800, 1400, "A"), (1400, 2000, "A"), (2200, 3400, "B")],
        )
        self.assertEqual(
            data["hits"],
            [
                {
                    "index": 0,
                    "phrase": "使用优惠码",
                    "start_word": 0,
                    "end_word": 1,
                    "start_char": 2,
                    "end_char": 3,
                },
                {
                    "index": 1,
                    "phrase": "八八八",
                    "start_word": 2,
                    "end_word": 2,
                    "start_char": 0,
                    "end_char": 3,
                },
            ],
        )

    def test_hit_cannot_cross_speaker(self):
        resp = self.post(phrases=["码八"])  # 词 1 末与词 2 首同属 A，可跨词命中
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["hits"]), 1)
        resp = self.post(phrases=["八别"])  # 跨说话人边界，不得命中
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["hits"], [])

    def test_no_hit_returns_original_words(self):
        resp = self.post(phrases=["不存在"])
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["hits"], [])
        self.assertEqual(data["words"], self.WORDS)

    def test_overlap_resolution(self):
        words = [{"start_ms": 0, "end_ms": 100, "text": "aaaaa", "speaker": "A"}]
        resp = self.post(words=words, phrases=["aa"])
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["words"][0]["text"], "████a")
        self.assertEqual(
            [(h["start_char"], h["end_char"]) for h in data["hits"]], [(0, 2), (2, 4)]
        )
        # 同起点短语较长优先
        resp = self.post(words=words, phrases=["aa", "aaaa"])
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["words"][0]["text"], "████a")
        self.assertEqual([h["phrase"] for h in data["hits"]], ["aaaa"])

    def test_phrases_count_out_of_range(self):
        for bad in ([], ["甲"] * 51):
            resp = self.post(phrases=bad)
            self.assertEqual(resp.status_code, 422)
            self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_empty_phrase_rejected(self):
        resp = self.post(phrases=["甲", ""])
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_words_over_2000_rejected(self):
        words = [
            {"start_ms": i * 500, "end_ms": i * 500 + 400, "text": "字", "speaker": "A"}
            for i in range(2001)
        ]
        resp = self.post(words=words)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")
        # 2000 词恰好在上限内
        resp = self.post(words=words[:2000])
        self.assertEqual(resp.status_code, 200)

    def test_word_validation_reused(self):
        words = [dict(self.WORDS[1]), dict(self.WORDS[0])]  # 时间倒序
        resp = self.post(words=words)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "time_out_of_order")

    def test_segments_still_works(self):
        resp = self.client.post("/segments", json={"words": self.WORDS})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("segments", resp.json())


@unittest.skipUnless(_HAS_DEPS, "fastapi/httpx 未安装，跳过接口层测试")
class TrailerPlanApiTests(unittest.TestCase):
    WORDS = [
        {"start_ms": 0, "end_ms": 800, "text": "甲", "speaker": "A"},
        {"start_ms": 800, "end_ms": 1600, "text": "乙", "speaker": "A"},
        {"start_ms": 1600, "end_ms": 2400, "text": "丙", "speaker": "B"},
        {"start_ms": 2400, "end_ms": 3200, "text": "丁", "speaker": "B"},
        {"start_ms": 3200, "end_ms": 4000, "text": "戊", "speaker": "A"},
    ]
    CANDIDATES = [
        {"start_word": 0, "end_word": 1, "score": 5},   # A，1600ms
        {"start_word": 2, "end_word": 3, "score": 7},   # B，1600ms
        {"start_word": 4, "end_word": 4, "score": 3},   # A，800ms
    ]

    def setUp(self):
        self.client = TestClient(app)

    def post(self, **overrides):
        payload = {
            "words": self.WORDS,
            "candidates": self.CANDIDATES,
            "budget_ms": 2400,
            "required_speakers": ["A", "B"],
        }
        payload.update(overrides)
        return self.client.post("/trailer-plan", json=payload)

    def test_basic_trailer_plan(self):
        resp = self.post()
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_score"], 10)
        self.assertEqual(data["total_duration_ms"], 2400)
        self.assertEqual(
            data["clips"],
            [
                {
                    "index": 0,
                    "candidate_index": 1,
                    "speaker": "B",
                    "score": 7,
                    "start_word": 2,
                    "end_word": 3,
                    "source_start_ms": 1600,
                    "source_end_ms": 3200,
                    "duration_ms": 1600,
                    "trailer_start_ms": 0,
                    "trailer_end_ms": 1600,
                },
                {
                    "index": 1,
                    "candidate_index": 2,
                    "speaker": "A",
                    "score": 3,
                    "start_word": 4,
                    "end_word": 4,
                    "source_start_ms": 3200,
                    "source_end_ms": 4000,
                    "duration_ms": 800,
                    "trailer_start_ms": 1600,
                    "trailer_end_ms": 2400,
                },
            ],
        )

    def test_infeasible_returns_422(self):
        resp = self.post(budget_ms=2399)  # 覆盖 A+B 至少需 2400ms
        self.assertEqual(resp.status_code, 422)
        err = resp.json()["error"]
        self.assertEqual(err["code"], "infeasible_trailer_plan")
        self.assertEqual(err["candidate_count"], 3)
        self.assertEqual(err["budget_ms"], 2399)

    def test_required_speaker_absent_infeasible(self):
        resp = self.post(required_speakers=["C"])
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "infeasible_trailer_plan")

    def test_candidate_index_out_of_range(self):
        for bad in (
            {"start_word": -1, "end_word": 0, "score": 1},
            {"start_word": 0, "end_word": 5, "score": 1},
        ):
            resp = self.post(candidates=[bad])
            self.assertEqual(resp.status_code, 422)
            err = resp.json()["error"]
            self.assertEqual(err["code"], "candidate_index_out_of_range")
            self.assertEqual(err["candidate_index"], 0)

    def test_candidate_range_inverted(self):
        resp = self.post(candidates=[{"start_word": 2, "end_word": 1, "score": 1}])
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "candidate_range_inverted")

    def test_candidate_speaker_mixed(self):
        resp = self.post(candidates=[{"start_word": 1, "end_word": 2, "score": 1}])
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "candidate_speaker_mixed")

    def test_candidate_count_out_of_range(self):
        for bad in (
            [],
            [{"start_word": 0, "end_word": 0, "score": 1}] * 21,
        ):
            resp = self.post(candidates=bad)
            self.assertEqual(resp.status_code, 422)
            self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_negative_score_rejected(self):
        resp = self.post(candidates=[{"start_word": 0, "end_word": 0, "score": -1}])
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_budget_and_required_speakers_schema(self):
        for overrides in (
            {"budget_ms": -1},
            {"required_speakers": []},
            {"required_speakers": ["A", "B", "C", "D"]},
            {"required_speakers": ["A", ""]},
        ):
            resp = self.post(**overrides)
            self.assertEqual(resp.status_code, 422)
            self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_word_validation_reused(self):
        words = [dict(self.WORDS[1]), dict(self.WORDS[0])]  # 时间倒序
        resp = self.post(words=words)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "time_out_of_order")

    def test_segments_still_works(self):
        resp = self.client.post("/segments", json={"words": self.WORDS})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("segments", resp.json())


@unittest.skipUnless(_HAS_DEPS, "fastapi/httpx 未安装，跳过接口层测试")
class PlaybackPlanApiTests(unittest.TestCase):
    WORDS = [
        {"start_ms": 0, "end_ms": 1000, "text": "开场白", "speaker": "A"},
        {"start_ms": 1000, "end_ms": 2000, "text": "欢迎收听", "speaker": "A"},
        {"start_ms": 3000, "end_ms": 4000, "text": "今天聊聊", "speaker": "B"},
        {"start_ms": 4000, "end_ms": 5000, "text": "动态规划", "speaker": "B"},
        {"start_ms": 5500, "end_ms": 6500, "text": "下期再见", "speaker": "A"},
    ]
    # 块：A 2000ms、B 2000ms、A 1000ms；块间静音 1000ms 与 500ms

    def setUp(self):
        self.client = TestClient(app)

    def post(self, **overrides):
        payload = {"words": self.WORDS, "max_duration_ms": 5700}
        payload.update(overrides)
        return self.client.post("/playback-plan", json=payload)

    def test_basic_playback_plan(self):
        resp = self.post()
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_cost"], 2_500_000)
        self.assertEqual(data["total_duration_ms"], 5700)
        self.assertEqual(
            data["blocks"],
            [
                {
                    "index": 0,
                    "start_word": 0,
                    "end_word": 1,
                    "speaker": "A",
                    "source_start_ms": 0,
                    "source_end_ms": 2000,
                    "duration_ms": 2000,
                    "speed": 125,
                    "playback_duration_ms": 1600,
                    "playback_start_ms": 0,
                    "playback_end_ms": 1600,
                },
                {
                    "index": 1,
                    "start_word": 2,
                    "end_word": 3,
                    "speaker": "B",
                    "source_start_ms": 3000,
                    "source_end_ms": 5000,
                    "duration_ms": 2000,
                    "speed": 125,
                    "playback_duration_ms": 1600,
                    "playback_start_ms": 2600,
                    "playback_end_ms": 4200,
                },
                {
                    "index": 2,
                    "start_word": 4,
                    "end_word": 4,
                    "speaker": "A",
                    "source_start_ms": 5500,
                    "source_end_ms": 6500,
                    "duration_ms": 1000,
                    "speed": 100,
                    "playback_duration_ms": 1000,
                    "playback_start_ms": 4700,
                    "playback_end_ms": 5700,
                },
            ],
        )

    def test_infeasible_returns_422(self):
        resp = self.post(max_duration_ms=4834)  # 全 150% 也需 4835ms
        self.assertEqual(resp.status_code, 422)
        err = resp.json()["error"]
        self.assertEqual(err["code"], "infeasible_playback_plan")
        self.assertEqual(err["block_count"], 3)
        self.assertEqual(err["max_duration_ms"], 4834)
        # 放宽到 4835 即可行
        resp = self.post(max_duration_ms=4835)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["total_duration_ms"], 4835)

    def test_too_many_blocks(self):
        words = [
            {
                "start_ms": i * 500,
                "end_ms": i * 500 + 400,
                "text": "词",
                "speaker": "AB"[i % 2],
            }
            for i in range(21)
        ]
        resp = self.post(words=words)
        self.assertEqual(resp.status_code, 422)
        err = resp.json()["error"]
        self.assertEqual(err["code"], "too_many_blocks")
        self.assertEqual(err["block_count"], 21)
        # 20 块恰好在上限内
        resp = self.post(words=words[:20])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["blocks"]), 20)

    def test_max_duration_schema(self):
        for bad in (-1, None):
            resp = self.post(max_duration_ms=bad)
            self.assertEqual(resp.status_code, 422)
            self.assertEqual(resp.json()["error"]["code"], "invalid_request")
        # 缺省 max_duration_ms
        resp = self.client.post("/playback-plan", json={"words": self.WORDS})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_word_validation_reused(self):
        words = [dict(self.WORDS[1]), dict(self.WORDS[0])]  # 时间倒序
        resp = self.post(words=words)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "time_out_of_order")

    def test_segments_still_works(self):
        resp = self.client.post("/segments", json={"words": self.WORDS})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("segments", resp.json())


if __name__ == "__main__":
    unittest.main()
