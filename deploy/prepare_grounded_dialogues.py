"""Small, grounded dialogue lessons with explicit facts and complete role pairs.

These template families teach constraints, corrections and response ownership.
They are not a broad conversation corpus or independently judged chat quality.
Two languages share a semantic family and must stay in the same data split.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

try:
    from .prepare_recovery_curriculum import parse_conversation
except ImportError:
    from prepare_recovery_curriculum import parse_conversation


DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
ZH_DAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
SOURCE = "uttt:grounded-dialogues-v1"


def identity(skill, args):
    return hashlib.sha256(json.dumps([skill, args], ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def specifications():
    result = {skill: [] for skill in ("available_day", "correction", "budget_choice", "listed_ingredients")}
    for days in itertools.combinations(range(7), 3):
        for free in days:
            result["available_day"].append({"days": list(days), "free": free})
    for before, after in itertools.permutations(range(7), 2):
        for hour in (9, 11, 14, 16, 18):
            result["correction"].append({"before": before, "after": after, "hour": hour})
    for budget in (20, 30, 40, 50, 60, 80):
        for cheap in (5, 10, 15):
            for extra in (5, 10, 20):
                for position in range(3):
                    costs = [budget + extra, budget + extra + 10]
                    costs.insert(position, cheap)
                    result["budget_choice"].append({"budget": budget, "costs": costs, "choice": position})
    ingredients = ("辣椒", "蘑菇", "洋葱", "香菜", "黄瓜", "番茄", "土豆", "胡萝卜")
    for avoid in ingredients:
        for other in ingredients:
            if other == avoid:
                continue
            for position in (0, 1):
                result["listed_ingredients"].append({"avoid": avoid, "other": other, "choice": position})
    return result


def lesson(skill, args, language):
    zh = language == "zh"
    if skill == "available_day":
        days = [ZH_DAYS[d] if zh else DAYS[d] for d in args["days"]]
        free = ZH_DAYS[args["free"]] if zh else DAYS[args["free"]]
        busy = [d for d in days if d != free]
        first = (f"我只考虑{'、'.join(days)}见朋友，但{'、'.join(busy)}都要工作，只有{free}空闲。请记住这次安排。" if zh else
                 f"I can meet a friend only on {', '.join(days)}. I work on {' and '.join(busy)}; only {free} is free. Keep this arrangement in mind.")
        ack = f"明白，{free}空闲，另外两天要工作。" if zh else f"Understood: {free} is free; you work on the other two days."
        question = "那我应该选哪天见朋友？只推荐一天。" if zh else "Which day should I choose to meet my friend? Recommend only one day."
        answer = f"选{free}，这是你空闲的一天。" if zh else f"Choose {free}, the day you are free."
        expected = free
    elif skill == "correction":
        before = ZH_DAYS[args["before"]] if zh else DAYS[args["before"]]
        after = ZH_DAYS[args["after"]] if zh else DAYS[args["after"]]
        hour = f"{args['hour']:02d}:00"
        first = f"我原来约在{before}{hour}取书。" if zh else f"My original book pickup is {before} at {hour}."
        ack = f"原来的取书时间是{before}{hour}。" if zh else f"Your original pickup is {before} at {hour}."
        correction = f"更正一下，改成{after}{hour}，之前的日期作废。" if zh else f"Correction: it is now {after} at {hour}. The old day no longer applies."
        corrected = f"好的，以{after}{hour}为准。" if zh else f"Got it: {after} at {hour} is the current arrangement."
        question = "我现在应该什么时候去取书？简短回答。" if zh else "When should I pick up the book now? Answer briefly."
        answer = f"{after}{hour}去取书。" if zh else f"Pick it up {after} at {hour}."
        return [(first, ack), (correction, corrected), (question, answer)], after
    elif skill == "budget_choice":
        labels = ("笔记本", "马克杯", "收纳盒") if zh else ("notebook", "mug", "storage box")
        options = "，".join(f"{label}{cost}元" for label, cost in zip(labels, args["costs"])) if zh else ", ".join(f"{label}: {cost} credits" for label, cost in zip(labels, args["costs"]))
        first = f"这次只买一件，预算不超过{args['budget']}元。可选：{options}。" if zh else f"I want one item, with a budget of at most {args['budget']} credits. Options: {options}."
        ack = "我会只从这些选项里挑一件，并遵守预算。" if zh else "I will choose one of these items and stay within your budget."
        question = "按这个预算，推荐哪一件？只说一件。" if zh else "Which item fits that budget? Recommend only one."
        expected = labels[args["choice"]]
        answer = f"推荐{expected}，价格{args['costs'][args['choice']]}元，没有超出预算。" if zh else f"Choose the {expected}; at {args['costs'][args['choice']]} credits it fits your budget."
    elif skill == "listed_ingredients":
        translations = {"辣椒": "chili", "蘑菇": "mushroom", "洋葱": "onion", "香菜": "cilantro", "黄瓜": "cucumber", "番茄": "tomato", "土豆": "potato", "胡萝卜": "carrot"}
        avoid = args["avoid"] if zh else translations[args["avoid"]]
        other = args["other"] if zh else translations[args["other"]]
        labels = ("组合甲", "组合乙") if zh else ("option A", "option B")
        contents = [avoid, avoid]
        contents[args["choice"]] = other
        first = (f"只按我列的原料挑选：{labels[0]}是米饭加{contents[0]}，{labels[1]}是米饭加{contents[1]}。我不想吃{avoid}。" if zh else
                 f"Choose using only these listed ingredients: {labels[0]} has rice and {contents[0]}; {labels[1]} has rice and {contents[1]}. I do not want {avoid}.")
        ack = f"明白，要避开你不想吃的{avoid}，只从这两个组合中选择。" if zh else f"Understood. I will avoid {avoid} and choose between these two listed options."
        question = "那选哪一个？只推荐一个，不增加其他食物。" if zh else "Which should I choose? Recommend one without adding any other food."
        expected = labels[args["choice"]]
        answer = f"选{expected}，它列出的原料是米饭和{other}，没有{avoid}。" if zh else f"Choose {expected}: its listed ingredients are rice and {other}, without {avoid}."
    else:
        raise ValueError("Unknown dialogue skill")
    return [(first, ack), (question, answer)], expected


def generate(seed=20261111, max_families_per_skill=160):
    if type(max_families_per_skill) is not int or max_families_per_skill < 1:
        raise ValueError("max_families_per_skill must be a positive integer")
    rows = []
    for skill, candidates in specifications().items():
        ordered = sorted(candidates, key=lambda args: hashlib.sha256(f"{seed}:{identity(skill,args)}".encode()).hexdigest())
        for args in ordered[:max_families_per_skill]:
            key = identity(skill, args)
            for language in ("zh", "en"):
                turns, expected = lesson(skill, args, language)
                text = "\n".join(f"<|user|> {user}\n<|assistant|> {answer}<|eos|>" for user, answer in turns)
                if parse_conversation(text) != turns:
                    raise RuntimeError("Malformed dialogue")
                rows.append({"category":"chat", "language":language, "source":SOURCE,
                             "copyright":"Programmatically authored for this repository; no external text copied.",
                             "family":"grounded:"+key, "validation_group_id":"grounded:"+key,
                             "foundation_skill":"chat:"+skill, "selection_priority":0,
                             "grounded_operation":skill, "arguments":args, "expected_choice":expected,
                             "deterministic_answer":True, "text":text})
    manifest = {"protocol":SOURCE, "seed":seed, "rows":len(rows), "semantic_families":len(rows)//2,
                "by_skill":dict(Counter(row['foundation_skill'] for row in rows)),
                "notes":["Two languages share each semantic family.", "Facts are explicit in the user messages; ingredient composition is stipulated rather than external food advice.",
                         "Grounded template practice does not establish broad or natural conversation quality."]}
    return rows, manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--max-families-per-skill',type=int,default=160)
    args=parser.parse_args()
    rows,manifest=generate(args.seed,args.max_families_per_skill)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    data=''.join(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n' for row in rows).encode()
    args.output.write_bytes(data)
    manifest['sha256']=hashlib.sha256(data).hexdigest()
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
