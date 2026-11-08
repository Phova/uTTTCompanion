"""Independent chat/reasoning/code diagnostics, without executing generated code.

Open conversation is retained for human review. Exact answers and a deliberately
restricted, bounded Python-expression interpreter supply separate functional
checks; neither valid syntax nor low repetition is called general chat quality.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import operator
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROTOCOL = "chat-curriculum-diagnostics-v1"
_MARKER = re.compile(r"<\|[^<>]*\|>")
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
            ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
            ast.Is: operator.is_, ast.IsNot: operator.is_not, ast.In: lambda x, y: x in y,
            ast.NotIn: lambda x, y: x not in y}
_CALLS = {"len": len, "sum": sum, "sorted": sorted, "min": min, "max": max, "abs": abs}
_ANNOTATION_NAMES = frozenset({"int", "float", "str", "bool", "list", "tuple", "dict"})


class RestrictedCodeError(ValueError):
    pass


def _safe_annotation(node, depth=0):
    """Inspect builtin type syntax only; annotations are never evaluated."""
    if node is None:
        return True
    if depth > 8:
        return False
    if isinstance(node, ast.Name):
        return node.id in _ANNOTATION_NAMES
    if isinstance(node, ast.Constant):
        return node.value is None
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id in {"list", "tuple", "dict"}):
        arguments = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        kind = node.value.id
        if ((kind == "list" and len(arguments) != 1) or (kind == "dict" and len(arguments) != 2)
                or not 1 <= len(arguments) <= 8):
            return False
        if (kind == "tuple" and len(arguments) == 2 and isinstance(arguments[-1], ast.Constant)
                and arguments[-1].value is Ellipsis):
            arguments = arguments[:1]
        return all(_safe_annotation(item, depth + 1) for item in arguments)
    return False


def _bounded(value, depth=0):
    if depth > 8:
        raise RestrictedCodeError("Value nesting exceeds the protocol limit")
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        if not -1e12 <= value <= 1e12:
            raise RestrictedCodeError("Numeric value exceeds the protocol limit")
        return value
    if type(value) is str:
        if len(value) > 512:
            raise RestrictedCodeError("String exceeds the protocol limit")
        return value
    if type(value) in (list, tuple, dict):
        if len(value) > 128:
            raise RestrictedCodeError("Container exceeds the protocol limit")
        for item in (list(value.keys()) + list(value.values()) if isinstance(value, dict) else value):
            _bounded(item, depth + 1)
        return value
    raise RestrictedCodeError("Unsupported value type")


def _same_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(_same_value(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(_same_value(actual[key], expected[key]) for key in actual)
    return actual == expected


def extract_python(reply):
    """Accept one code block or bare code; never execute imports or annotations."""
    if not isinstance(reply, str) or len(reply) > 16000:
        raise RestrictedCodeError("Reply is not bounded source text")
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S | re.I)
    if "```" in reply and len(blocks) != 1:
        raise RestrictedCodeError("Expected exactly one complete Python code block")
    return blocks[0].strip() if blocks else reply.strip()


def restricted_python_check(reply, function, cases):
    """Interpret a bounded pure-function subset; no exec/eval/import/attributes.

    Supported statements: return, local assignment, if. Expressions include
    basic arithmetic/boolean/comparison, slicing and six fixed pure builtins.
    Signature annotations may name known builtin types and simple container
    generics; this syntax is inspected, never evaluated or used as runtime code.
    Unsupported valid Python is a failed check in THIS subset, not a claim that
    the program would fail unrestricted execution or a general coding benchmark.
    """
    result = {"protocol": "bounded_python_ast_v1", "passed": False,
              "generated_code_executed": False, "cases_total": len(cases), "cases_passed": 0}
    try:
        source = extract_python(reply)
        tree = ast.parse(source)
        if len(list(ast.walk(tree))) > 256:
            raise RestrictedCodeError("AST exceeds the protocol limit")
        body = list(tree.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.FunctionDef) or body[0].name != function:
            raise RestrictedCodeError("Expected the requested single function definition")
        definition = body[0]
        args = definition.args
        if (definition.decorator_list or not _safe_annotation(definition.returns)
                or getattr(definition, "type_params", []) or args.defaults or args.kw_defaults
                or args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs
                or any(not _safe_annotation(arg.annotation) or arg.arg.startswith("_")
                       or arg.arg in _CALLS for arg in args.args)):
            raise RestrictedCodeError("Function signature is outside the supported subset")
        names = [arg.arg for arg in args.args]
        if len(set(names)) != len(names):
            raise RestrictedCodeError("Duplicate function arguments")
        budget = [2000]

        def expression(node, env):
            budget[0] -= 1
            if budget[0] < 0:
                raise RestrictedCodeError("Interpreter budget exhausted")
            if isinstance(node, ast.Constant):
                value = node.value
            elif isinstance(node, ast.Name) and node.id in env:
                value = env[node.id]
            elif isinstance(node, ast.List):
                value = [expression(item, env) for item in node.elts]
            elif isinstance(node, ast.Tuple):
                value = tuple(expression(item, env) for item in node.elts)
            elif isinstance(node, ast.Dict) and all(key is not None for key in node.keys):
                value = {expression(k, env): expression(v, env) for k, v in zip(node.keys, node.values)}
            elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
                left, right = expression(node.left, env), expression(node.right, env)
                if isinstance(node.op, ast.Mult):
                    for sequence, count in ((left, right), (right, left)):
                        if type(sequence) in (str, list, tuple) and type(count) is int and len(sequence) * max(count, 0) > 512:
                            raise RestrictedCodeError("Sequence multiplication exceeds the protocol limit")
                value = _BINARY[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd, ast.Not):
                item = expression(node.operand, env)
                value = -item if isinstance(node.op, ast.USub) else +item if isinstance(node.op, ast.UAdd) else not item
            elif isinstance(node, ast.BoolOp) and type(node.op) in (ast.And, ast.Or):
                value = expression(node.values[0], env)
                for item in node.values[1:]:
                    if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value):
                        break
                    value = expression(item, env)
            elif isinstance(node, ast.IfExp):
                value = expression(node.body if expression(node.test, env) else node.orelse, env)
            elif isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops):
                left, value = expression(node.left, env), True
                for op, right_node in zip(node.ops, node.comparators):
                    right = expression(right_node, env)
                    if not _COMPARE[type(op)](left, right):
                        value = False
                        break
                    left = right
            elif isinstance(node, ast.Subscript):
                index = node.slice
                if isinstance(index, ast.Slice):
                    index = slice(*(expression(item, env) if item is not None else None
                                    for item in (index.lower, index.upper, index.step)))
                else:
                    index = expression(index, env)
                value = expression(node.value, env)[index]
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CALLS and not node.keywords:
                values = [expression(item, env) for item in node.args]
                if len(values) != 1:
                    raise RestrictedCodeError("Supported builtins take one argument")
                value = _CALLS[node.func.id](*values)
            else:
                raise RestrictedCodeError("Unsupported expression: " + type(node).__name__)
            return _bounded(value)

        class Returned(Exception):
            pass

        def statements(items, env):
            for node in items:
                budget[0] -= 1
                if budget[0] < 0:
                    raise RestrictedCodeError("Interpreter budget exhausted")
                if isinstance(node, ast.Return):
                    value = expression(node.value, env) if node.value is not None else None
                    raise Returned(value)
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name.startswith("_") or name in _CALLS:
                        raise RestrictedCodeError("Private names and builtin shadowing are unsupported")
                    env[name] = expression(node.value, env)
                elif isinstance(node, ast.If):
                    statements(node.body if expression(node.test, env) else node.orelse, env)
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    continue
                else:
                    raise RestrictedCodeError("Unsupported statement: " + type(node).__name__)

        for case in cases:
            arguments = case["args"]
            if len(arguments) != len(names):
                raise RestrictedCodeError("Test arguments do not match the function signature")
            env = dict(zip(names, (_bounded(item) for item in arguments)))
            try:
                statements(definition.body, env)
                actual = None
            except Returned as returned:
                actual = returned.args[0]
            result["cases_passed"] += int(_same_value(actual, case["expected"]))
        result["passed"] = result["cases_passed"] == result["cases_total"]
    except (SyntaxError, ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, OverflowError, RecursionError) as error:
        result["error"] = str(error)
    return result


def validate_tasks(tasks):
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("At least one diagnostic task is required")
    normalized, identities = [], set()
    for original in tasks:
        row = dict(original)
        for key in ("category", "task", "language", "prompt"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("Tasks require nonblank " + key)
        if row["category"] not in ("chat", "reasoning", "coding") or _MARKER.search(row["prompt"]):
            raise ValueError("Unsupported category or embedded role marker")
        row["family"] = row.get("family") or row.get("validation_group_id") or row.get("semantic_key")
        if not isinstance(row["family"], str) or not row["family"].strip():
            raise ValueError("Every task needs an explicit family identity")
        history = row.get("history", [])
        if not isinstance(history, list) or len(history) % 2:
            raise ValueError("History requires complete user/assistant pairs")
        for index, message in enumerate(history):
            if (message.get("role") != ("user" if index % 2 == 0 else "assistant")
                    or not isinstance(message.get("content"), str) or not message["content"].strip()
                    or _MARKER.search(message["content"])):
                raise ValueError("History contains an invalid role or content")
        checks = row.get("code_check")
        if row["category"] == "coding":
            if (not isinstance(checks, dict) or checks.get("protocol") != "bounded_python_ast_v1"
                    or not isinstance(checks.get("function"), str) or not checks["function"].isidentifier()
                    or checks["function"].startswith("_") or not isinstance(checks.get("cases"), list)
                    or not 1 <= len(checks["cases"]) <= 32):
                raise ValueError("Coding tasks require bounded_python_ast_v1 cases")
            for case in checks["cases"]:
                if not isinstance(case, dict) or not isinstance(case.get("args"), list) or "expected" not in case:
                    raise ValueError("Each code case needs args and expected")
                _bounded(case["args"])
                _bounded(case["expected"])
            if "accepted" in row:
                raise ValueError("Code checks and exact text answers cannot be mixed")
        elif checks is not None:
            raise ValueError("Only coding tasks may have code_check")
        if "accepted" in row and (not isinstance(row["accepted"], list) or not row["accepted"]
                                  or not all(isinstance(value, str) for value in row["accepted"])):
            raise ValueError("Accepted answers must be a nonempty string list")
        if row["category"] == "reasoning" and "accepted" not in row:
            raise ValueError("Reasoning tasks require exact accepted answers")
        row["id"] = row.get("id") or hashlib.sha256(json.dumps(
            [row["category"], row["family"], row["prompt"], history], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if row["id"] in identities:
            raise ValueError("Diagnostic IDs are duplicated")
        identities.add(row["id"])
        normalized.append(row)
    return normalized


def summarize_results(results):
    groups = defaultdict(list)
    for row in results:
        for group in ("all", "category:" + row["category"], "task:" + row["task"], "language:" + row["language"]):
            groups[group].append(row)
    return {name: {
        "samples": len(items), "automatic_checks": sum(row["automatic_pass"] is not None for row in items),
        "automatic_passes": sum(row["automatic_pass"] is True for row in items),
        "manual_review_required": sum(row["automatic_pass"] is None for row in items),
        "eos_rate": sum(row["eos_generated"] for row in items) / len(items),
        "token_limit_rate": sum(row["stop_reason"] == "token_limit" for row in items) / len(items),
        "mean_text_4gram_repetition": sum(row["text_4gram_repetition"] for row in items) / len(items),
    } for name, items in sorted(groups.items())}


def evaluate(checkpoint, tasks, output, *, max_new_tokens=128, device=None, seed=42):
    tasks = validate_tasks(tasks)
    from deploy.evaluate_model import evaluate as evaluate_model
    prepared = []
    for row in tasks:
        task = dict(row)
        if row.get("history"):
            transcript = "".join(f"<|user|> {row['history'][i]['content'].strip()}\n<|assistant|> "
                                 f"{row['history'][i+1]['content'].strip()}<|eos|>\n"
                                 for i in range(0, len(row["history"]), 2))
            task["prompt"] = transcript + f"<|user|> {row['prompt'].strip()}\n<|assistant|>"
            task["mode"] = "continuation"
        elif row.get("mode", "chat") != "chat":
            raise ValueError("Diagnostics support chat tasks or explicit structured history")
        if "accepted" in task:
            task["match_mode"] = task.get("match_mode", "strict_text")
        prepared.append(task)
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uttt_chat_diagnostic_", dir=path.parent) as temporary:
        raw = evaluate_model(checkpoint, prepared, Path(temporary) / "raw.json", max_new_tokens,
                             device, mode="chat", seed=seed, greedy_only=True)
    results = []
    for task, generated in zip(tasks, raw["results"]):
        result = dict(generated)
        result.update({key: task[key] for key in ("id", "category", "task", "language", "family", "prompt")})
        if task.get("history"):
            result["history"] = task["history"]
            result["input_mode"] = "explicit_recent_history"
        result["automatic_pass"] = generated.get("exact_match")
        if task["category"] == "coding":
            checks = task["code_check"]
            result["code_check"] = restricted_python_check(generated["reply"], checks["function"], checks["cases"])
            result["automatic_pass"] = result["code_check"]["passed"]
        results.append(result)
    identity = hashlib.sha256(json.dumps(tasks, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = {"protocol": PROTOCOL, "checkpoint": str(checkpoint), "tasks_sha256": identity,
              "training_metrics": raw["training_metrics"], "summary": summarize_results(results), "results": results,
              "notes": ["Unpenalized greedy, no online writes; explicit text history does not establish fast-weight memory.",
                        "Open chat needs human review; observable EOS/repetition are not a chat quality score.",
                        "Coding checks interpret a bounded pure-function subset; valid unsupported Python fails this restricted protocol.",
                        "Signature annotations support known builtin scalar/container types and simple list/tuple/dict generics; annotations are never evaluated.",
                        "No generated code is executed and no imports, attributes, loops, comprehensions or recursion are supported."]}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("CHAT_DIAGNOSTIC_COMPLETE " + json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    arguments = parser.parse_args()
    with Path(arguments.tasks).open(encoding="utf-8") as stream:
        tasks = [json.loads(line) for line in stream if line.strip()]
    evaluate(arguments.checkpoint, tasks, arguments.output, max_new_tokens=arguments.max_new_tokens,
             device=arguments.device, seed=arguments.seed)
