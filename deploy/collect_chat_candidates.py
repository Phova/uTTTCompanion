"""Collect explicitly sourced chat/reasoning/code candidates without upsampling.

Filtering is conservative and heuristic. It is not a factual, medical, or code
correctness verification. The curriculum builder performs BPE-budget and split
checks after this bounded, streaming collection step.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re


PROTOCOL = "uttt-chat-candidates-v3"
SOURCES = {
    "belle": ("chat", "zh", "https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M"),
    "everyday": ("chat", "en", "https://huggingface.co/datasets/HuggingFaceTB/smoltalk"),
    "smol": ("chat", "en", "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk"),
    "coig": ("reasoning", "zh", "https://huggingface.co/datasets/m-a-p/COIG-CQIA"),
    "coding": ("coding", "en", "https://huggingface.co/datasets/HuggingFaceTB/smoltalk"),
}
ROLE = re.compile(r"\b(Human|Assistant):")
SPECIAL = re.compile(r"<\|[^<>]*\|>")
IDENTITY = re.compile(
    r"(?:我(?:是|叫|由|来自)|作为|I(?: am|'m| was developed by)|as (?:an? )?|you are|你是)"
    r"[^\n。.!?]{0,60}(?:ChatGPT|GPT-?[34]|OpenAI|Claude|Anthropic|Qwen|通义|文心|Gemini|Bard|Siri)", re.I)
WEATHER = re.compile(r"(?:今天|今日|现在|当前)[^\n。]{0,30}(?:天气(?:是|为|晴|阴|多云)|气温(?:是|为)|\d+\s*(?:摄氏|度|°[CF]))"
                     r"|(?:today|current)[^\n.!?]{0,45}(?:weather is|temperature is|\d+\s*(?:degrees|°[CF]))", re.I)
EXECUTED = re.compile(r"我(?:已经|已)(?:帮你)?(?:发送|预订|购买|删除|重启|安装|下载|上传|支付|下单|执行|运行)"
                      r"|\bI(?:'ve| have| already)?\s+(?:already\s+|successfully\s+)*(?:sent|booked|purchased|deleted|restarted|installed|downloaded|uploaded|paid|executed|run)\b", re.I)
FUTURE_TOOL_ACTION = re.compile(
    r"(?:我(?:会|将|可以|能|来|正在|已经|已)?|让我|(?:需要|要不要|是否需要)我)"
    r"(?:(?:先|马上|现在|稍后|立即|亲自|为你|为您|帮你|帮您|替你|替您|自己|去))*"
    r"(?:搜索|搜寻|查询|检索|预订|订票|购票|买票|下单|支付|转账|发送|发给|上传|下载|安装|重启|删除|执行|运行|打开|启动|(?:把|将)[^。！？!?\n]{0,16}(?:发送|发)给)"
    r"|\b(?:I(?:'ll|'m| will| can| am going to| am)|let me|would you like me to|shall I)\s+"
    r"(?:(?:help you|first|now|currently|already|personally|then)\s+)*(?:search(?:ing)?|look(?:ing)? up|book(?:ing)?|buy(?:ing)?|purchase|reserve|send(?:ing)?|email|upload|download|install|pay|run|execute|open(?:ing)?|launch(?:ing)?)\b", re.I)
RECENT_ENTERTAINMENT = re.compile(
    r"(?:最近|近期|最新)(?:的|上映的|热映的)?(?:电影|新片)"
    r"|(?:最近|近期)(?:有|上映|推出|发布)[^。！？!?\n]{0,45}(?:电影|新片)"
    r"|(?:最近|近期)(?:很|比较|最)?(?:流行|热门|热播|热映)"
    r"|\b(?:the latest|the newest|a recent|a recently released)\s+(?:movie|film)\b"
    r"|\b(?:recently|currently)\s+(?:trending|popular)\b"
    r"|\bcurrently\b[^.!?\n]{0,90}\b(?:popular|trending)\b"
    r"|\b(?:are|is)\s+trending\s+this season\b", re.I)
RECENT_EXISTENCE = re.compile(r"(?:最近|近期)(?:有|上映|推出|发布)")
USER_LOOKUP_ADVICE = re.compile(
    r"(?:你|您)(?:也)?(?:可以|可|不妨|最好|需要)[^。！？!?\n]{0,35}(?:搜索|搜寻|查询|查看|查阅|检索|购票|订票|买票)"
    r"|\byou (?:can|could|may|should)\s+[^.!?\n]{0,45}\b(?:search|look up|check|book|buy)\b", re.I)
SPECIALIST = re.compile(r"医学|医疗|临床|病症|病理|诊断|诊疗|用药|处方|药物|药品|剂量|疫苗|接种|化疗|手术|中医|证券|股票|基金|投资|理财|金融|黄金|金本位|期货|外汇|债券|诉讼|法律|法院|律师|合同法|民法|刑法"
                        r"|\b(?:medical|diagnosis|prescription|dosage|stock trading|investment advice|legal advice|lawsuit)\b", re.I)
CODE_QUESTION = re.compile(r"```|Python|JavaScript|TypeScript|SQL|C\+\+|Java|\b(?:function|programming)\b|编程|写.{0,5}(?:函数|代码)", re.I)
LONG_FORM_TASK = re.compile(
    r"(?:写|撰写|创作|生成|编写|续写|改写|仿写|模仿|润色)[^\n。！？]{0,28}(?:文章|论文|诗歌?|散文|小说|故事|剧本|作文|演讲稿|长篇|报告)"
    r"|五言|七言|律诗|绝句"
    r"|(?:讲|编)[^\n。！？]{0,8}故事"
    r"|(?:总结|摘要|概括|提取|归纳)[^\n。！？]{0,28}(?:文章|论文|文本|段落|文中|故事|正文|下文)"
    r"|(?:文章|论文|文本|段落)[^\n。！？]{0,28}(?:总结|摘要|概括|提取|归纳)"
    r"|\b(?:write|compose|generate|create|draft|continue|rewrite|summarize|summarise)\b[^\n.!?]{0,60}\b(?:essay|paper|poem|poetry|novel|story|article|manuscript)\b", re.I)
INCOMPLETE_PROMISE = re.compile(
    r"仅展示部分|只展示部分|内容省略|此处省略|由于篇幅|限于篇幅|篇幅限制|文章长度超过|由于长度|省略若干|余下省略"
    r"|due to (?:space|length)|omitted for brevity|only a portion (?:is|has been)|the rest is omitted", re.I)
DAILY_TOPIC = re.compile(
    r"工作|上班|下班|同事|朋友|家人|父母|爸爸|妈妈|孩子|伴侣|恋人|心情|难过|开心|烦恼|休息|放松|吃饭|早餐|午饭|晚饭|饮食|做饭|做菜|"
    r"睡觉|失眠|电影|音乐|旅行|旅游|运动|锻炼|焦虑|压力|累|礼物|生日|偏好|喜欢|周末|聊天|陪我|无聊|孤独|名字|自我介绍|称呼")
TECHNICAL_TOPIC = re.compile(
    r"人工智能|机器学习|深度学习|神经网络|科技产品|科技|量子|气候变化|碳中和|碳排放|大数据|区块链|物联网"
    r"|(?:产品|品牌|企业|公司|项目|网站|商店|店铺)[^\n。！？]{0,12}(?:名字|名称|命名)"
    r"|(?<![A-Za-z0-9])AI(?![A-Za-z0-9])|\b(?:artificial intelligence|machine learning|climate change|quantum|blockchain)\b", re.I)
CHAT_OPENINGS = frozenset({"你好", "您好", "嗨", "哈喽", "早上好", "下午好", "晚上好", "在吗", "可以聊聊吗", "能聊聊吗",
                           "hello", "hi", "hey", "goodmorning", "goodevening", "howareyou", "canwechat"})
CHAT_FOLLOWUPS = frozenset({"还有别的建议吗", "还有其他建议吗", "还有什么建议吗", "你还有别的建议吗", "有什么建议吗",
                            "能具体说说吗", "可以具体说说吗", "再具体一点", "能再具体一点吗", "为什么", "为什么呢", "然后呢",
                            "怎么做", "该怎么做", "怎么办", "那我该怎么办", "谢谢", "谢谢你", "好的", "好", "明白了", "了解了",
                            "有例子吗", "能举个例子吗", "可以举个例子吗", "能再说一点吗", "继续", "再来一个"})
COIG_QA_MAJOR = re.compile(r"问答|试题|考试|数学|逻辑推理|常识")
COIG_NON_QA = re.compile(r"文本生成|文本创作|模仿写诗|仿写|诗歌|名词解释|长解释")


class Rejected(ValueError):
    pass


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _message(value):
    if not isinstance(value, str) or not value.strip() or "\0" in value or "\ufffd" in value:
        raise Rejected("invalid_message")
    if SPECIAL.search(value):
        raise Rejected("embedded_special_token")
    return value.strip()


def parse_belle(row):
    """Preserve existing Human/Assistant turns; output fills only the final slot."""
    instruction, output = row.get("instruction"), row.get("output")
    if not isinstance(instruction, str) or row.get("input") not in (None, ""):
        raise Rejected("belle_ambiguous_input")
    matches = list(ROLE.finditer(instruction))
    if (len(matches) < 4 or instruction[:matches[0].start()].strip()
            or matches[-1].group(1) != "Assistant" or instruction[matches[-1].end():].strip()):
        raise Rejected("belle_incomplete_multiturn")
    turns, user = [], None
    for index, marker in enumerate(matches):
        expected = "Human" if index % 2 == 0 else "Assistant"
        if marker.group(1) != expected:
            raise Rejected("belle_role_order")
        text = (instruction[marker.end():matches[index + 1].start()]
                if index + 1 < len(matches) else output)
        text = _message(text)
        if index % 2 == 0:
            user = text
        else:
            if ROLE.search(text):
                raise Rejected("embedded_belle_role")
            turns.append((user, text))
    if len(turns) < 2:
        raise Rejected("belle_not_multiturn")
    return turns, ""


def parse_messages(row):
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise Rejected("missing_messages")
    turns, system, user = [], [], None
    for message in messages:
        if not isinstance(message, dict):
            raise Rejected("invalid_role_record")
        role, content = message.get("role"), _message(message.get("content"))
        if role == "system" and not turns and user is None:
            system.append(content)
        elif role == "user" and user is None:
            user = content
        elif role == "assistant" and user is not None:
            turns.append((user, content))
            user = None
        else:
            raise Rejected("message_role_order")
    if not turns or user is not None:
        raise Rejected("incomplete_messages")
    system_text = "\n\n".join(system)
    if system_text:
        turns[0] = (system_text + "\n\n" + turns[0][0], turns[0][1])
    return turns, system_text


def quality_filter(turns, system=""):
    if system and IDENTITY.search(system):
        raise Rejected("brand_identity")
    for _, answer in turns:
        if INCOMPLETE_PROMISE.search(answer):
            raise Rejected("incomplete_answer_promise")
        if IDENTITY.search(answer):
            raise Rejected("brand_identity")
        if WEATHER.search(answer):
            raise Rejected("unverified_current_weather")
        if EXECUTED.search(answer):
            raise Rejected("claims_executed_action")
        if FUTURE_TOOL_ACTION.search(answer):
            raise Rejected("claims_future_tool_action")
        for clause in re.split(r"[。！？!?;；\n]+", answer):
            recent = RECENT_ENTERTAINMENT.search(clause)
            lookup = USER_LOOKUP_ADVICE.search(clause)
            # Advising the user to check a current listing asserts no current
            # title/trend. A named movie or an existence claim still does.
            if recent and (not lookup or lookup.start() >= recent.start()
                           or RECENT_EXISTENCE.search(clause) or re.search(r"《[^》]+》", clause)):
                raise Rejected("unverified_recent_entertainment")
        prose = re.sub(r"```.*?```", "", answer, flags=re.S)
        parts = [re.sub(r"\s+", "", part).casefold() for part in re.split(r"[。！？!?;\n]+", prose)]
        if any(count >= 3 for part, count in Counter(part for part in parts if len(part) >= 8).items()):
            raise Rejected("repeated_sentence")
        compact = re.sub(r"\s+", "", prose)
        if re.search(r"(.{8,64}?)\1\1", compact):
            raise Rejected("repeated_span")


def candidate_from_row(kind, row, source_file, index=0, max_chars=1800):
    if kind not in SOURCES or not isinstance(row, dict):
        raise Rejected("invalid_source_record")
    category, language, source = SOURCES[kind]
    if kind == "belle":
        turns, system = parse_belle(row)
    elif kind == "coig":
        if row.get("human_verified") is not True:
            raise Rejected("coig_not_human_verified")
        task_type = row.get("task_type")
        major = task_type.get("major") if isinstance(task_type, dict) else None
        major_text = json.dumps(major, ensure_ascii=False)
        task_text = json.dumps(task_type, ensure_ascii=False)
        if COIG_NON_QA.search(task_text) or not COIG_QA_MAJOR.search(major_text):
            raise Rejected("coig_not_general_qa_or_basic_test")
        question = _message(row.get("instruction"))
        context = row.get("input")
        if context not in (None, ""):
            question += "\n\n" + _message(context)
        labels = json.dumps({key: row.get(key) for key in ("domain", "task_type")}, ensure_ascii=False)
        if SPECIALIST.search(question + " " + labels):
            raise Rejected("specialist_domain")
        if CODE_QUESTION.search(question):
            raise Rejected("coding_in_reasoning_source")
        turns, system = [(question, _message(row.get("output")))], ""
    else:
        label = str(row.get("source", "")).lower()
        if kind == "smol" and label not in {"everyday-conversations", "systemchats", "systemchats-30k"}:
            raise Rejected("smol_not_explicit_dialogue_source")
        if kind == "coding" and label and label not in {"self-oss-instruct", "self-oss-starcoder2-instruct"}:
            raise Rejected("coding_source_mismatch")
        turns, system = parse_messages(row)
    if category == "chat":
        # Focus this phase's chat quota on everyday conversation. These source
        # pools can also contain coding or specialist questions; do not count
        # those as daily chat merely because the file is called a chat dataset.
        if any(CODE_QUESTION.search(user) for user, _ in turns):
            raise Rejected("chat_coding_question")
        if any(SPECIALIST.search(user) for user, _ in turns):
            raise Rejected("chat_specialist_domain")
        if any(LONG_FORM_TASK.search(user) for user, _ in turns):
            raise Rejected("chat_long_form_or_document_task")
        if any(TECHNICAL_TOPIC.search(user) for user, _ in turns):
            raise Rejected("chat_technical_topic")
        if kind == "belle":
            daily_turns = []
            for index, (user, _) in enumerate(turns):
                short = "".join(user.split()).casefold().strip("。.!！?？~～，,；;：:")
                if DAILY_TOPIC.search(user):
                    daily_turns.append("daily_topic")
                elif short in CHAT_OPENINGS:
                    daily_turns.append("opening")
                elif index > 0 and short in CHAT_FOLLOWUPS:
                    daily_turns.append("short_followup")
                else:
                    raise Rejected("belle_non_daily_turn")
            if "daily_topic" not in daily_turns:
                raise Rejected("belle_no_daily_topic")
    text = "\n".join(f"<|user|> {user}\n<|assistant|> {answer}<|eos|>" for user, answer in turns)
    if max_chars and len(text) > max_chars:
        raise Rejected("over_character_budget")
    quality_filter(turns, system)
    # The identity uses all original user turns, before any later BPE prefix
    # selection, so that one parent conversation remains one validation family.
    prompts = [" ".join(user.split()).casefold() for user, _ in turns]
    family = "public-conversation:" + digest(json.dumps([source, prompts], ensure_ascii=False, separators=(",", ":")))
    result = {"category": category, "language": language, "source": source, "family": family,
            "source_file": str(source_file), "source_subset": row.get("source", kind),
            "upstream_row_id": row.get("id", row.get("conversation_id", index)),
            "copyright": row.get("copyright") or "See upstream dataset card and component license terms.",
            "upstream_human_verified": row.get("human_verified"),
            "quality_status": "heuristically_filtered_not_independently_verified",
            "turn_count": len(turns), "text": text}
    if category == "chat":
        result["chat_focus"] = "belle_every_turn_daily_heuristic_v3" if kind == "belle" else "explicit_dialogue_source"
        if kind == "belle":
            result["daily_topic_matches"] = sorted({topic for user, _ in turns for topic in DAILY_TOPIC.findall(user)})
            result["daily_turn_classification"] = daily_turns
    return result


def _json_array(stream):
    decoder, buffer, done, expecting = json.JSONDecoder(), "", False, "value"
    allow_empty = True
    while True:
        if not done and (not buffer.strip() or expecting == "more"):
            chunk = stream.read(65536)
            done = not chunk
            buffer += chunk
            if expecting == "more":
                expecting = "value"
        buffer = buffer.lstrip()
        if not buffer:
            if done:
                raise ValueError("Unterminated JSON array")
            continue
        if expecting == "separator":
            if buffer[0] == ",":
                buffer, expecting = buffer[1:], "value"
                continue
            if buffer[0] != "]":
                raise ValueError("Expected a comma or closing JSON array")
        if buffer[0] == "]":
            if expecting == "value" and not allow_empty:
                raise ValueError("Trailing comma in JSON array")
            if buffer[1:].strip() or stream.read().strip():
                raise ValueError("Unexpected content after JSON array")
            return
        try:
            record, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            if done:
                raise
            chunk = stream.read(65536)
            done = not chunk
            buffer += chunk
            continue
        yield record
        allow_empty = False
        buffer, expecting = buffer[end:], "separator"


def iter_records(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
        return
    with path.open(encoding="utf-8-sig") as stream:
        first = stream.read(1)
        while first and first.isspace():
            first = stream.read(1)
        if first == "[":
            yield from _json_array(stream)
        elif first:
            line = first + stream.readline()
            yield json.loads(line)
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def collect_candidates(sources, output, *, limit_per_category=20000, seed=20261108, max_chars=1800):
    if type(limit_per_category) is not int or limit_per_category < 1 or type(max_chars) is not int or max_chars < 0:
        raise ValueError("limit_per_category must be positive and max_chars nonnegative")
    output = Path(output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Candidate output/manifest already exists")
    reservoirs = {category: [] for category in ("chat", "reasoning", "coding")}
    rngs = {category: random.Random(seed + int(digest(category)[:8], 16)) for category in reservoirs}
    eligible, rejected, scanned, seen, files = Counter(), Counter(), Counter(), set(), []
    for kind, path in sources:
        path = Path(path)
        if kind not in SOURCES:
            raise ValueError("Unknown source kind: " + kind)
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        files.append({"kind": kind, "file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": checksum.hexdigest()})
        for index, raw in enumerate(iter_records(path)):
            scanned[kind] += 1
            try:
                candidate = candidate_from_row(kind, raw, str(path.resolve()), index, max_chars)
            except Rejected as error:
                rejected[kind + ":" + str(error)] += 1
                continue
            identity = digest(candidate["text"])
            if identity in seen:
                rejected[kind + ":duplicate_text"] += 1
                continue
            seen.add(identity)
            category = candidate["category"]
            eligible[category] += 1
            pool = reservoirs[category]
            if len(pool) < limit_per_category:
                pool.append(candidate)
            else:
                selected = rngs[category].randrange(eligible[category])
                if selected < limit_per_category:
                    pool[selected] = candidate
    records = [row for category in sorted(reservoirs) for row in reservoirs[category]]
    random.Random(seed).shuffle(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"protocol": PROTOCOL, "seed": seed, "limit_per_category": limit_per_category,
              "max_chars": max_chars, "files": files, "scanned": dict(scanned), "eligible": dict(eligible),
              "selected": {category: len(pool) for category, pool in reservoirs.items()},
              "selected_multi_turn": sum(row["turn_count"] > 1 for row in records),
              "rejected": dict(sorted(rejected.items())), "output_sha256": digest(output.read_text(encoding="utf-8")),
              "notes": ["No upsampling: shortages remain visible in selected counts.",
                        "Source labels identify candidate categories; heuristics do not establish answer correctness.",
                        "Full original turns and parent-family identities are retained; BPE filtering happens downstream."]}
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in SOURCES:
        parser.add_argument("--" + kind, type=Path, action="append", default=[], help="Source JSON/JSONL/parquet; may be repeated")
    parser.add_argument("--output", type=Path, required=True, help="New candidates JSONL file")
    parser.add_argument("--limit-per-category", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--max-chars", type=int, default=1800, help="Whole-conversation character budget; 0 disables it")
    args = parser.parse_args(argv)
    sources = [(kind, path) for kind in SOURCES for path in getattr(args, kind)]
    if not sources:
        parser.error("At least one source is required")
    report = collect_candidates(sources, args.output, limit_per_category=args.limit_per_category, seed=args.seed, max_chars=args.max_chars)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
