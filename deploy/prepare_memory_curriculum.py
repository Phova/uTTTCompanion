"""Prepare BPE-verified delayed recall/correction data for uTTT meta training.

All supervision is on one final assistant answer. Facts and corrections finish
more than one configured attention window before that answer, using actual
tokenizer offsets. This is a distance constraint, not proof a multilayer frozen
attention backbone cannot solve the task; evaluate dynamic versus frozen memory.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re


PROTOCOL = "delayed_nonce_recall_correction_v1"
SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|user|>", "<|assistant|>")
PAYLOAD_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

# These original neutral sentences deliberately contain neither task codes nor
# role delimiters. Each row shuffles distinct sentences, rather than repeating
# one sentence to manufacture a long context.
NEUTRAL_EN = (
    "A gardener checks the soil before deciding how much water the young plants need.",
    "The morning train reaches the small station shortly after the market opens.",
    "A library keeps its reference books close to the tables used for quiet study.",
    "Clouds move slowly across the valley while sunlight reaches the eastern hills.",
    "A careful reader compares the evidence in an article with the claims in its title.",
    "The museum places a short description beside each object in the exhibition.",
    "Several neighbors share tools when they repair the fence around their garden.",
    "A walking path follows the river until it reaches a bridge made of stone.",
    "The cook lets the bread cool on a rack before cutting it into even slices.",
    "A map can show the distance between towns without describing the road surface.",
    "During a lesson the class observes how the length of a shadow changes outdoors.",
    "The photographer waits for a clear view instead of moving the nesting birds.",
    "A notebook helps a student keep observations separate from later explanations.",
    "The concert hall opens its doors early so the audience can find their seats.",
    "An old wooden chair becomes useful again after its loose joints are repaired.",
    "The weather report describes the expected rain and the direction of the wind.",
    "A team tests a small version of a bridge before discussing a larger design.",
    "The shopkeeper arranges fresh fruit so customers can compare its size and color.",
    "A bicycle runs more smoothly when its chain is clean and properly adjusted.",
    "The afternoon ferry carries passengers from the island to the main harbor.",
    "An editor reads a paragraph aloud to notice sentences that are hard to follow.",
    "The park has benches near the entrance and drinking water beside the playground.",
    "A telescope gathers light from distant objects and directs it toward an eyepiece.",
    "The baker weighs the flour before combining it with the other ingredients.",
    "Leaves on the shaded side of a tree can look different from leaves in direct sun.",
    "A railway timetable lists arrival times separately from the names of nearby streets.",
    "The artist begins with a light sketch and adds darker lines after checking proportions.",
    "A glass of cold water develops droplets when warm moist air meets its surface.",
    "The hiking group studies the trail map before leaving the shelter in the morning.",
    "A visitor can learn about a town by walking through its older residential streets.",
    "The workshop stores sharp instruments in a closed drawer after the lesson ends.",
    "A good explanation connects an observation with the steps needed to understand it.",
    "The river carries small pieces of sand downstream after heavy rain in the mountains.",
    "A volunteer records the kinds of birds seen beside the lake during the same hour.",
    "The classroom window allows fresh air to enter without disturbing the papers on the desk.",
    "A simple experiment changes one condition while keeping the other conditions stable.",
    "The theater staff check the lights before the performers begin their rehearsal.",
    "A farmer rotates crops to help maintain the condition of the soil across seasons.",
    "The bus stops near a public square where several walking routes begin.",
    "A carpenter measures a board twice before making a cut along the marked line.",
    "The sea appears darker when a cloud blocks the sunlight over the water.",
    "A shared calendar helps a small group notice when two activities overlap.",
    "The science exhibit lets visitors compare the shape of several mineral samples.",
    "A shallow bowl loses heat faster when more of the warm liquid is exposed to air.",
    "The university garden provides a quiet place for students to read between classes.",
    "A clear table heading explains what the numbers in the column are intended to measure.",
    "The path becomes narrower near the rocks but widens again beside the meadow.",
    "A repair guide explains the order in which the parts should be removed and replaced.",
)
NEUTRAL_ZH = (
    "园丁会先观察土壤的湿润程度，再决定是否给新栽的植物浇水。",
    "清晨的列车到达小站时，附近的集市刚刚开始营业。",
    "图书馆把常用参考书放在阅览桌附近，方便读者查找资料。",
    "云层缓慢地经过山谷，阳光照亮了东边的山坡。",
    "阅读文章时，可以把作者提供的证据与文章标题中的判断进行比较。",
    "博物馆在每件展品旁边放置简短说明，介绍它的用途和年代。",
    "邻居们修理花园围栏时，会互相借用合适的工具。",
    "沿河的小路经过一片草地，最后通向一座石桥。",
    "面包出炉后需要放在架子上稍微冷却，然后再切成整齐的薄片。",
    "地图可以显示城镇之间的距离，但不一定说明路面的实际情况。",
    "课堂上的观察活动记录了同一物体的影子在不同时间的长度变化。",
    "摄影师等待视野变得清晰，而没有移动正在筑巢的鸟。",
    "使用笔记本记录观察时，可以把直接看到的现象与之后的解释分开。",
    "音乐厅提前开放入口，让观众有足够时间找到自己的座位。",
    "旧木椅松动的连接处修好以后，又能继续在日常生活中使用。",
    "天气预报介绍预计的降雨情况，也会说明风向的变化。",
    "设计团队先测试一个小型桥梁模型，再讨论更大的结构方案。",
    "商店把新鲜水果分开放置，便于顾客比较它们的大小和颜色。",
    "自行车链条经过清洁和适当调整后，骑行通常会更加顺畅。",
    "下午的渡船把岛上的乘客送往对岸的主要港口。",
    "编辑把段落朗读出来，有时能发现阅读时不易察觉的表达问题。",
    "公园入口附近设有长椅，儿童活动区旁边提供饮用水。",
    "望远镜收集远处物体发出的光，并把光线引向便于观察的位置。",
    "制作点心之前，烘焙者会先称量面粉，再混合其他配料。",
    "同一棵树上，阴影里的叶片与阳光直射的叶片可能呈现不同状态。",
    "铁路时刻表主要列出列车的到达时间，而不是附近街道的分布。",
    "画家先画浅色草图，确认比例合适之后，再添加更清楚的线条。",
    "温暖潮湿的空气遇到冷水杯的表面时，杯壁可能出现细小水滴。",
    "徒步小组在早晨离开休息点以前，会共同查看路线图。",
    "游客沿着旧居民区的街道步行，可以了解城镇过去的生活环境。",
    "实践课结束后，工作人员会把锋利的工具收回有盖的储物盒。",
    "清楚的说明需要连接观察结果与理解这个结果所必需的步骤。",
    "山区降下大雨之后，河水会把少量沙粒带到下游。",
    "志愿者在固定时段记录湖边出现的鸟类，以便比较不同日期的观察。",
    "教室的窗户可以引入新鲜空气，同时避免桌上的纸张被风吹散。",
    "简单的实验通常只改变一个条件，并尽量保持其他条件稳定。",
    "剧场工作人员在演员开始排练之前，会逐一检查照明设备。",
    "农民根据季节安排不同作物，有助于维持土壤的良好状态。",
    "公共汽车在广场附近停靠，乘客可以从那里选择不同的步行路线。",
    "木匠在沿着标记切割木板之前，会再次核对测量的长度。",
    "当云层遮挡水面上方的阳光时，海水看起来可能会更暗。",
    "共同使用的日历可以帮助小组发现两项活动在时间上是否重叠。",
    "科学展览允许参观者比较几种矿物样品的形状与表面特征。",
    "盛有热水的浅碗会让更多水面接触空气，因此散热速度可能更快。",
    "大学里的小花园为学生提供了课间阅读和短暂休息的地方。",
    "表格中清楚的列标题能够说明每一列数字实际测量的内容。",
    "步道在岩石附近变得狭窄，经过草甸时又恢复了较宽的路面。",
    "维修说明会按照顺序介绍各个部件应当如何拆下以及重新安装。",
)

FACT_TEMPLATES = {
    ("train", "en", "recall"): (
        "Profile {name}: the assigned access code is {answer}. Remember this exact code.",
        "Please retain one fact about {name}. Their reference code is {answer}.",
        "For {name}, store the private task code {answer}; it will be requested later.",
    ),
    ("train", "en", "correction"): (
        "Profile {name}: the old access code was {old}. Correction: replace it with {answer}. Only the replacement is current.",
        "Record {old} as the earlier code for {name}. Update: their code is now {answer}; the earlier value is obsolete.",
        "For {name}, an initial note gave {old}. A later correction gives {answer}. Retain the corrected code.",
    ),
    ("heldout", "en", "recall"): (
        "An entry for {name} associates that person with the code {answer}. Keep the association available.",
        "Here is a fact for later retrieval: {name} has the lookup code {answer}.",
    ),
    ("heldout", "en", "correction"): (
        "The entry for {name} first listed {old}. That entry has been superseded: the valid code is {answer}.",
        "A note about {name} needs revision. Instead of the previously recorded {old}, use {answer} from now on.",
    ),
    ("train", "zh", "recall"): (
        "请记住用户{name}的一条信息：其访问代码是{answer}。稍后需要准确取回这个代码。",
        "用户{name}的资料中登记了参考代码{answer}，请保留这一对应关系。",
        "为用户{name}保存任务代码{answer}，后续问题会询问这个值。",
    ),
    ("train", "zh", "correction"): (
        "用户{name}的旧访问代码是{old}。现更正为{answer}，今后只采用更正后的代码。",
        "先前记录的用户{name}代码为{old}。更新：当前代码应为{answer}，原来的值已经作废。",
        "关于用户{name}，初始笔记给出了{old}；后续修订明确改成{answer}。请保留修订后的值。",
    ),
    ("heldout", "zh", "recall"): (
        "这里有一项供之后查询的资料：{name}对应的检索代码为{answer}。",
        "资料卡将{name}与代码{answer}关联起来，请在后续对话中保留这项信息。",
    ),
    ("heldout", "zh", "correction"): (
        "资料卡曾把{name}的代码写作{old}。该记录已被替代，目前有效的代码为{answer}。",
        "需要修订{name}的一项记录：此前的{old}不再适用，从现在起应使用{answer}。",
    ),
}
QUESTION_TEMPLATES = {
    ("train", "en"): ("What is the current code for {name}? Output only the code.",
                         "Retrieve {name}'s valid code from the earlier information. Reply with the code alone."),
    ("heldout", "en"): ("Which code should be used for {name} now? Give just its exact value.",
                           "Find the latest code associated with {name}. Return that value without explanation."),
    ("train", "zh"): ("用户{name}当前的代码是什么？只输出代码。", "请取回之前记录的{name}有效代码，只回答代码本身。"),
    ("heldout", "zh"): ("现在应该为{name}采用哪一个代码？请仅给出准确的值。", "查询资料中{name}最后有效的代码，不要附加说明。"),
}


def clean_filler(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Filler sentences must be nonempty strings")
    if "<|" in text or "|>" in text or "\x00" in text or "\ufffd" in text:
        raise ValueError("Filler may not contain role/token delimiters, NUL or replacement characters")
    return " ".join(text.split())


def load_filler(path=None, source=None):
    if path is None:
        return {language: [(clean_filler(text), "builtin:neutral-sentences-v1") for text in texts]
                for language, texts in (("en", NEUTRAL_EN), ("zh", NEUTRAL_ZH))}
    if not source or not isinstance(source, str):
        raise ValueError("External filler requires an explicit --filler-source provenance label")
    pools, seen = {"en": [], "zh": []}, set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line) if line.lstrip().startswith("{") else {"text": line}
            text = clean_filler(row.get("text") or row.get("content"))
            if text in seen:
                continue
            if not 20 <= len(text) <= 400:
                raise ValueError("External filler rows must be individual natural sentences of 20 to 400 characters")
            seen.add(text)
            language = "zh" if re.search(r"[\u3400-\u9fff]", text) else "en"
            pools[language].append((text, str(row.get("source") or source)))
    if any(len(pool) < 12 for pool in pools.values()):
        raise ValueError("External filler requires at least 12 distinct English and 12 distinct Chinese sentences")
    return pools


def load_tokenizer(path):
    from tokenizers import Tokenizer
    raw = Path(path).read_bytes()
    if json.loads(raw).get("model", {}).get("type") != "BPE":
        raise ValueError("An actual BPE tokenizer JSON is required")
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    for expected, token in enumerate(SPECIAL_TOKENS):
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"Tokenizer must assign {token} ID {expected}")
    return tokenizer, hashlib.sha256(raw).hexdigest()


def measure(tokenizer, text, fact_end_character, chunk_size):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = [1] + encoded.ids + [2]  # Exactly the dataset's BOS/append-EOS policy.
    if ids.count(4) != 1 or ids.count(5) != 1 or not text.endswith("<|eos|>"):
        raise ValueError("Memory rows must contain one user, one final assistant and a real answer EOS")
    assistant = ids.index(5)
    answer_start = assistant + 1
    # Offsets are in the original input text. Include any token that overlaps
    # the fact/correction boundary, giving a conservative completed-fact index.
    fact_end = 1 + max(index + 1 for index, (start, end) in enumerate(encoded.offsets)
                       if end > 0 and start < fact_end_character)
    return {"tokens_with_dataset_bos_eos": len(ids), "fact_end_token_exclusive": fact_end,
            "answer_start_token": answer_start, "fact_to_answer_gap_tokens": answer_start - fact_end,
            "fact_last_chunk": (fact_end - 1) // chunk_size, "answer_chunk": answer_start // chunk_size,
            "final_assistant_tokens": len(ids) - answer_start - 1,
            "user_roles": ids.count(4), "assistant_roles": ids.count(5),
            "answer_eos_present": ids[-2:] == [2, 2]}


def fresh_payload(rng, used, prefix, length=10):
    while True:
        value = prefix + "".join(rng.choice(PAYLOAD_ALPHABET) for _ in range(length))
        if value not in used:
            used.add(value)
            return value


def build_row(tokenizer, pools, split, index, rng, used_names, used_payloads, seq_len, window, chunk_size):
    language = "zh" if index % 2 else "en"
    task = "correction" if (index // 2) % 2 else "recall"
    name = fresh_payload(rng, used_names, "person-", length=8)
    answer = fresh_payload(rng, used_payloads, "KEY-", length=rng.choice((8, 10, 12)))
    old = fresh_payload(rng, used_payloads, "KEY-", length=10) if task == "correction" else None
    facts = rng.choice(FACT_TEMPLATES[(split, language, task)]).format(name=name, answer=answer, old=old)
    question = rng.choice(QUESTION_TEMPLATES[(split, language)]).format(name=name)
    prefix = "<|user|> " + facts + "\n\n"
    suffix = "\n\n" + question + "\n<|assistant|> " + answer + "<|eos|>"
    for value in (answer, old):
        if value is not None and value.casefold() in question.casefold():
            raise ValueError("Answer payload must not appear in the retrieval question")
    order = list(pools[language])
    rng.shuffle(order)
    target_length = rng.randint(max(window + chunk_size + 32, seq_len - 128), seq_len)
    sentences, sources, best = [], set(), None
    for sentence, source in order:
        if any(value is not None and value.casefold() in sentence.casefold() for value in (answer, old)):
            continue
        candidate_sentences = [*sentences, sentence]
        filler = "\n".join(candidate_sentences)
        text = prefix + filler + suffix
        metrics = measure(tokenizer, text, len("<|user|> ") + len(facts), chunk_size)
        if metrics["tokens_with_dataset_bos_eos"] > seq_len:
            continue
        sentences = candidate_sentences
        sources.add(source)
        if (len(sentences) >= 8 and metrics["fact_to_answer_gap_tokens"] > window
                and metrics["answer_chunk"] > metrics["fact_last_chunk"] and metrics["answer_eos_present"]):
            best = (text, metrics, filler, sorted(sources), len(sentences))
            if metrics["tokens_with_dataset_bos_eos"] >= target_length:
                break
    if best is None:
        raise ValueError("Could not fit a delayed answer with sufficient distinct filler; increase seq_len or provide more filler sentences")
    text, metrics, filler, sources, sentence_count = best
    prompt = facts + "\n\n" + filler + "\n\n" + question
    metadata = {"task": task, "language": language, "entity": name, "answer": answer,
                "obsolete_answer": old, "filler_sentences": sentence_count, **metrics}
    return {"text": text, "source": PROTOCOL, "filler_sources": sources, "memory_task": metadata}, {
        "prompt": prompt, "accepted": [answer], "mode": "chat", "memory_task": metadata,
    }


def digest_values(values):
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def ranges(values):
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"file": path.name, "rows": len(rows), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(output, tokenizer_path, train_samples=2048, heldout_samples=256, seq_len=1024,
            sliding_window=512, chunk_size=128, seed=42, filler_path=None, filler_source=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new curriculum directory: {output}")
    if min(train_samples, heldout_samples, chunk_size) < 1 or sliding_window < 1:
        raise ValueError("Sample counts, chunk_size and sliding_window must be positive")
    if seq_len < sliding_window + chunk_size + 32:
        raise ValueError("seq_len must leave room beyond the attention window for a later chunk, question and answer")
    tokenizer, tokenizer_hash = load_tokenizer(tokenizer_path)
    pools = load_filler(filler_path, filler_source)
    used_names, used_payloads, splits, tasks = set(), set(), {}, []
    identifiers = {}
    for split, count in (("train", train_samples), ("heldout", heldout_samples)):
        rng = random.Random(f"{PROTOCOL}:{seed}:{split}")
        previous_names, previous_payloads = set(used_names), set(used_payloads)
        rows = []
        for index in range(count):
            row, task = build_row(tokenizer, pools, split, index, rng, used_names, used_payloads,
                                  seq_len, sliding_window, chunk_size)
            rows.append(row)
            if split == "heldout":
                tasks.append(task)
        splits[split] = rows
        identifiers[split] = {"entities": used_names - previous_names, "payloads": used_payloads - previous_payloads}
    intersections = {key: len(identifiers["train"][key] & identifiers["heldout"][key]) for key in ("entities", "payloads")}
    if any(intersections.values()):
        raise ValueError("Train and heldout identifiers overlap")
    # Generate and validate every row before creating the output directory.
    output.mkdir(parents=True, exist_ok=False)
    files = {split: write_jsonl(output / f"{split}.jsonl", rows) for split, rows in splits.items()}
    with (output / "eval_tasks.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(tasks, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    files["eval_tasks"] = {"file": "eval_tasks.json", "rows": len(tasks),
                           "sha256": hashlib.sha256((output / "eval_tasks.json").read_bytes()).hexdigest()}
    stats = {}
    for split, rows in splits.items():
        metrics = [row["memory_task"] for row in rows]
        stats[split] = {
            "tasks": dict(Counter(item["task"] for item in metrics)),
            "languages": dict(Counter(item["language"] for item in metrics)),
            "tokens_with_dataset_bos_eos": ranges([item["tokens_with_dataset_bos_eos"] for item in metrics]),
            "fact_to_answer_gap_tokens": ranges([item["fact_to_answer_gap_tokens"] for item in metrics]),
            "distinct_filler_sentences_per_row": ranges([item["filler_sentences"] for item in metrics]),
            "entities": len(identifiers[split]["entities"]), "payloads": len(identifiers[split]["payloads"]),
            "entity_set_sha256": digest_values(identifiers[split]["entities"]),
            "payload_set_sha256": digest_values(identifiers[split]["payloads"]),
        }
    manifest = {
        "protocol": PROTOCOL, "seed": seed, "tokenizer_sha256": tokenizer_hash,
        "vocab_size": tokenizer.get_vocab_size(), "seq_len": seq_len,
        "sliding_window": sliding_window, "chunk_size": chunk_size,
        "strict_minimum_fact_to_answer_gap": sliding_window + 1,
        "identifier_intersections": intersections, "files": files, "statistics": stats,
        "filler_sources": sorted({source for pool in pools.values() for _, source in pool}),
        "filler_pool_sha256": digest_values(f"{language}:{source}:{text}" for language, pool in pools.items() for text, source in pool),
        "supervision": "One final assistant answer; user facts, corrections, filler and question are not CE targets.",
        "notes": ["train.jsonl is the only training input; heldout.jsonl and eval_tasks.json must remain held out.",
                  "Lengths include BOS and the dataset-appended EOS in addition to the answer's explicit real EOS.",
                  "Every final answer occurs after its fact/correction by more than sliding_window actual BPE tokens.",
                  "Multiple attention layers can propagate information beyond a single layer's window; distance alone is not proof of memory use.",
                  "Compare exact CE and generation with dynamic writes versus frozen memory on the same heldout records.",
                  "These synthetic random-code tasks target delayed retrieval/correction, not general chat quality."],
    }
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--heldout-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--sliding-window", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filler-path", type=Path)
    parser.add_argument("--filler-source")
    args = parser.parse_args()
    result = prepare(args.output, args.tokenizer, args.train_samples, args.heldout_samples,
                     args.seq_len, args.sliding_window, args.chunk_size, args.seed,
                     args.filler_path, args.filler_source)
    print("MEMORY_CURRICULUM_PREPARED " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
