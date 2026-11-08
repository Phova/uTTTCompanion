"""Candidate collection preserves real turns and source-defined categories."""
import json
from pathlib import Path
import tempfile
import unittest

from deploy.collect_chat_candidates import (
    Rejected, candidate_from_row, collect_candidates, iter_records, parse_belle,
)
from deploy.prepare_recovery_curriculum import parse_conversation


def belle(index=0, answer=None):
    return {"instruction": f"Human: 我的名字是小羽{index}。\nAssistant: 你好！\nHuman: 我刚才叫什么名字？\nAssistant: ",
            "input": "", "output": answer or f"你叫小羽{index}。", "id": index, "copyright": "fixture copyright"}


def messages(source="everyday-conversations", user="How are you?", answer="I'm doing well."):
    return {"source": source, "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]}


class CandidateCollectionTests(unittest.TestCase):
    def setUp(self):
        self.cache = Path(__file__).resolve().parents[1] / ".cache"
        self.cache.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=self.cache)
        self.root = Path(self.tmp.name)
        assert self.root.resolve().is_relative_to(self.cache.resolve())

    def tearDown(self):
        assert self.root.resolve().is_relative_to(self.cache.resolve())
        self.tmp.cleanup()

    def write(self, name, rows, array=False):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            if array:
                json.dump(rows, stream, ensure_ascii=False, indent=2)
            else:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def test_belle_uses_original_turns_and_final_output_with_one_eos_per_answer(self):
        row = candidate_from_row("belle", belle(), "belle.json")
        self.assertEqual(parse_conversation(row["text"]), [("我的名字是小羽0。", "你好！"), ("我刚才叫什么名字？", "你叫小羽0。")])
        self.assertEqual(row["text"].count("<|eos|>"), 2)
        self.assertEqual(row["turn_count"], 2)
        self.assertEqual((row["category"], row["language"]), ("chat", "zh"))
        self.assertEqual(row["copyright"], "fixture copyright")
        alternative = candidate_from_row("belle", belle(answer="你的名字是小羽0。"), "other-copy.json")
        self.assertEqual(row["family"], alternative["family"])
        self.assertNotEqual(row["text"], alternative["text"])

    def test_belle_rejects_missing_and_ambiguous_roles_rather_than_repairing_them(self):
        invalid = [dict(belle(), input="ambiguous extra context"), dict(belle(), output=""),
                   dict(belle(), instruction="Human: hi\nAssistant: "),
                   dict(belle(), instruction="Human: hi\nHuman: again\nAssistant: hello\nAssistant: "),
                   dict(belle(), instruction=belle()["instruction"] + "existing final answer"),
                   dict(belle(), output="Human: invented next turn\nAssistant: invented answer")]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(Rejected):
                parse_belle(item)

    def test_only_explicit_smol_dialogue_sources_are_chat(self):
        for source in ("everyday-conversations", "systemchats-30k"):
            result = candidate_from_row("smol", messages(source), "smol.parquet")
            self.assertEqual(result["category"], "chat")
        for source in ("smol-magpie-ultra-short", "smol-constraints", "openhermes-50k", "self-oss-instruct", ""):
            with self.subTest(source=source), self.assertRaisesRegex(Rejected, "explicit_dialogue"):
                candidate_from_row("smol", messages(source), "smol.parquet")
        code = candidate_from_row("coding", messages("self-oss-instruct", "Define add(a, b)", "def add(a, b):\n    return a + b"), "code.parquet")
        self.assertEqual(code["category"], "coding")
        with self.assertRaisesRegex(Rejected, "source_mismatch"):
            candidate_from_row("coding", messages("smol-magpie-ultra"), "code.parquet")

    def test_system_context_is_retained_and_invalid_message_order_is_rejected(self):
        row = messages("systemchats-30k")
        row["messages"].insert(0, {"role": "system", "content": "Answer in one short sentence."})
        result = candidate_from_row("smol", row, "system.parquet")
        self.assertTrue(parse_conversation(result["text"])[0][0].startswith("Answer in one short sentence.\n\n"))
        for sequence in ([{"role": "assistant", "content": "hello"}],
                         [{"role": "user", "content": "hello"}],
                         [{"role": "user", "content": "hello"}, {"role": "tool", "content": "result"}],
                         [{"role": "user", "content": "hello"}, {"role": "user", "content": "again"}]):
            with self.assertRaises(Rejected):
                candidate_from_row("everyday", {"messages": sequence}, "everyday.parquet")

    def test_chat_quota_excludes_code_and_specialist_questions_but_keeps_everyday_support(self):
        for question in ("请写Python代码，实现一个排序函数。", "请给出药品的处方剂量。", "应该接种哪种疫苗？"):
            with self.subTest(question=question), self.assertRaisesRegex(Rejected, "chat_(coding_question|specialist_domain)"):
                candidate_from_row("everyday", messages(user=question, answer="This is a source answer."), "everyday.parquet")
        for question in ("我有点焦虑，想找人聊聊。", "今天锻炼后有点累，想吃一顿简单的饭。"):
            result = candidate_from_row("everyday", messages(user=question, answer="先坐下来休息一下，慢慢聊。"), "everyday.parquet")
            self.assertEqual(result["category"], "chat")

    def test_belle_focus_keeps_daily_turns_and_rejects_long_form_document_tasks(self):
        for question in ("今天工作很累，想休息一下。", "给朋友买生日礼物，有什么建议？", "晚饭想做得简单一点。"):
            row = belle()
            row["instruction"] = f"Human: {question}\nAssistant: 我们可以先聊聊你的想法。\nHuman: 你还有别的建议吗？\nAssistant: "
            row["output"] = "可以先从一个容易做到的小步骤开始。"
            result = candidate_from_row("belle", row, "belle.json")
            self.assertEqual(result["chat_focus"], "belle_every_turn_daily_heuristic_v3")
            self.assertTrue(result["daily_topic_matches"])
        for question in ("写一篇关于朋友的长篇文章。", "请提取下文文章的总结。", "帮我创作一首关于音乐的诗。", "请写一本小说。"):
            row = belle()
            row["instruction"] = f"Human: {question}\nAssistant: 可以。\nHuman: 继续。\nAssistant: "
            row["output"] = "这是一个完整的回答。"
            with self.subTest(question=question), self.assertRaisesRegex(Rejected, "long_form"):
                candidate_from_row("belle", row, "belle.json")
        unrelated = belle()
        unrelated["instruction"] = "Human: 请解释量子场论。\nAssistant: 这是物理学理论。\nHuman: 还有呢？\nAssistant: "
        with self.assertRaisesRegex(Rejected, "technical_topic|non_daily_turn"):
            candidate_from_row("belle", unrelated, "belle.json")

    def test_belle_rejects_mixed_poem_or_product_turns_even_with_later_gift_chat(self):
        for question in ("给我一首五言绝句。", "为一个新产品起品牌名字。", "帮我想一句口号。", "机器学习和AI有什么区别？"):
            row = {"instruction": f"Human: {question}\nAssistant: 这是前一轮的回答。\nHuman: 给朋友送礼物有什么建议？\nAssistant:",
                   "input": "", "output": "可以先了解朋友喜欢什么。"}
            with self.subTest(question=question), self.assertRaises(Rejected):
                candidate_from_row("belle", row, "belle.json")
        finance = {"instruction": "解释黄金和金本位。", "output": "这是一条财经说明。", "human_verified": True,
                   "task_type": {"major": "通用问答"}}
        with self.assertRaisesRegex(Rejected, "specialist_domain"):
            candidate_from_row("coig", finance, "coig.jsonl")

    def test_belle_accepts_greeting_then_daily_feeling_and_short_followup_only(self):
        row = {"instruction": "Human: hello\nAssistant: 你好！\nHuman: 今天工作很累，还有点焦虑。\nAssistant: 先休息一下吧。\nHuman: 能具体说说吗？\nAssistant:",
               "input": "", "output": "可以先喝杯水，再把最着急的一件事写下来。"}
        result = candidate_from_row("belle", row, "belle.json")
        self.assertEqual(result["daily_turn_classification"], ["opening", "daily_topic", "short_followup"])
        self.assertEqual(len(parse_conversation(result["text"])), 3)
        for source in ("everyday", "smol"):
            with self.assertRaisesRegex(Rejected, "technical_topic"):
                candidate_from_row(source, messages(user="What is machine learning?"), "chat.parquet")

    def test_incomplete_promises_and_coig_non_qa_metadata_are_rejected(self):
        for answer in ("文章长度超过50汉字，因此仅展示部分内容。", "由于篇幅限制，此处省略内容。", "The rest is omitted for brevity."):
            with self.subTest(answer=answer), self.assertRaisesRegex(Rejected, "incomplete_answer"):
                candidate_from_row("everyday", messages(answer=answer), "everyday.parquet")
        base = {"instruction": "请解释春天为什么会变暖。", "output": "春天阳光照射逐渐增强。", "human_verified": True}
        for major in ("文本生成", "模仿写诗", "名词解释", "其他"):
            with self.subTest(major=major), self.assertRaisesRegex(Rejected, "general_qa"):
                candidate_from_row("coig", {**base, "task_type": {"major": major}}, "coig.jsonl")
        result = candidate_from_row("coig", {**base, "task_type": {"major": "基础试题"}}, "coig.jsonl")
        self.assertEqual(result["category"], "reasoning")

    def test_coig_requires_verified_general_content_and_preserves_copyright(self):
        base = {"instruction": "解释一下为什么一天有白天和黑夜。", "input": "给小学生讲。", "output": "地球自转让同一地方交替面向太阳。",
                "human_verified": True, "copyright": "retained", "domain": ["general"], "task_type": {"major": "问答"}}
        candidate = candidate_from_row("coig", base, "coig.jsonl")
        self.assertEqual(candidate["category"], "reasoning")
        self.assertEqual(candidate["copyright"], "retained")
        self.assertIn("给小学生讲。", candidate["text"])
        for change in ({"human_verified": False}, {"human_verified": "true"}, {"instruction": "给我一张用药处方。"},
                       {"domain": ["投资"]}, {"instruction": "请写Python代码。"}, {"task_type": {"major": "法律"}},
                       {"input": "患者需要使用药物，处方剂量是多少？"}):
            with self.subTest(change=change), self.assertRaises(Rejected):
                candidate_from_row("coig", {**base, **change}, "coig.jsonl")

    def test_heuristic_rejections_do_not_mistake_user_mentions_for_assistant_identity(self):
        for answer in ("我是由OpenAI开发的ChatGPT。", "As an AI language model developed by OpenAI, I can help.",
                       "今天天气是晴天。", "The current weather is sunny.", "我已经帮你发送邮件。", "I have already booked the hotel.",
                       "你需要先好好休息一下。你需要先好好休息一下。你需要先好好休息一下。",
                       "invalid <|assistant|> embedded role"):
            with self.subTest(answer=answer), self.assertRaises(Rejected):
                candidate_from_row("everyday", messages(answer=answer), "everyday.parquet")
        result = candidate_from_row("everyday", messages(user="What is OpenAI?", answer="OpenAI is an AI research company."), "everyday.parquet")
        self.assertEqual(result["turn_count"], 1)

    def test_rejects_assistant_future_tool_promises_but_keeps_user_agency_and_explanations(self):
        for answer in (
            "好的，我会为你搜索一下最近的摇滚音乐会，然后把结果发给你。是否需要我帮你购票？",
            "我可以帮你购票。", "需要我帮你预订酒店吗？", "我会把结果发给你。",
            "I'll search for concerts and send you the results.", "Would you like me to book the tickets?",
            "好的，我正在为您打开电影推荐系统。", "I'm currently opening the booking website.",
        ):
            with self.subTest(answer=answer), self.assertRaisesRegex(Rejected, "future_tool_action"):
                candidate_from_row("everyday", messages(answer=answer), "everyday.parquet")
        for answer in (
            "你可以先在演出官网搜索，确认时间后自行购票。", "你想看什么类型的演出？预算是多少？",
            "我可以教你如何搜索演出信息。", "你可以自行搜索最近的电影排期，再到官网购票。",
            "You can search for concerts and book tickets yourself.", "I can explain how to book tickets.",
        ):
            with self.subTest(answer=answer):
                candidate_from_row("everyday", messages(answer=answer), "everyday.parquet")

    def test_rejects_undated_recent_movie_or_trend_claims_but_keeps_mood_and_dated_facts(self):
        for answer in (
            "建议你看看最近的电影《黑寡妇》。",
            "当然可以。最近有一部叫做《速度与激情：特别行动》的电影。",
            "最近流行复古摇滚。", "I recommend the latest movie, Black Widow.",
            "Currently, sundresses and shift dresses are very popular for casual wear.",
            "Pastel colors are trending this season.",
        ):
            with self.subTest(answer=answer), self.assertRaisesRegex(Rejected, "unverified_recent_entertainment"):
                candidate_from_row("everyday", messages(answer=answer), "everyday.parquet")
        for answer in (
            "听起来你最近心情低落，我们可以先聊聊发生了什么。", "你最近想看轻松一点还是刺激一点的电影？",
            "《黑寡妇》于2021年上映，你可以按喜好选择。", "You have been feeling low recently; would you like to talk?",
        ):
            with self.subTest(answer=answer):
                candidate_from_row("everyday", messages(answer=answer), "everyday.parquet")

    def test_streams_arrays_and_jsonl_with_unicode_line_separators(self):
        rows = [dict(belle(), output="line\u2028separator"), dict(belle(2), output="a" * 70000), belle(3)]
        for array in (False, True):
            path = self.write("array.json" if array else "lines.jsonl", rows, array=array)
            self.assertEqual(list(iter_records(path)), rows)
        invalid = self.root / "bad.json"
        invalid.write_text('[{"a":1},]', encoding="utf-8")
        with self.assertRaises(ValueError):
            list(iter_records(invalid))

    def test_reservoir_is_seeded_distinct_and_reports_shortage_without_duplication(self):
        source = self.write("belle.json", [belle(i) for i in range(20)] + [belle(3)], array=True)
        first, second = self.root / "one.jsonl", self.root / "two.jsonl"
        a = collect_candidates([("belle", source)], first, limit_per_category=5, seed=42)
        b = collect_candidates([("belle", source)], second, limit_per_category=5, seed=42)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(a["selected"], {"chat": 5, "reasoning": 0, "coding": 0})
        self.assertEqual(a["eligible"], {"chat": 20})
        self.assertEqual(a["rejected"]["belle:duplicate_text"], 1)
        self.assertEqual(len({row["family"] for row in iter_records(first)}), 5)
        self.assertEqual(a["selected_multi_turn"], 5)
        self.assertTrue(first.with_suffix(".manifest.json").exists())
        self.assertEqual(a["files"][0]["sha256"], b["files"][0]["sha256"])

    def test_budget_never_truncates_or_adds_synthetic_turns_and_existing_output_is_preserved(self):
        source = self.write("data.jsonl", [belle()])
        output = self.root / "empty.jsonl"
        result = collect_candidates([("belle", source)], output, max_chars=5)
        self.assertEqual(sum(result["selected"].values()), 0)
        self.assertEqual(result["rejected"]["belle:over_character_budget"], 1)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            collect_candidates([("belle", source)], output)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
