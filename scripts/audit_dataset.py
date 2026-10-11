"""Read-only dataset inspection; this script never creates inference documents."""
import argparse
import collections
import json
import pathlib
import re


def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8-sig"))


def audit(rule_paths, question_paths):
    rules = {}
    duplicates = []
    for path in rule_paths:
        for row in read(path):
            key = str(row.get("rule_id", row.get("id", "")))
            if key in rules:
                duplicates.append(key)
            rules[key] = row.get("rule_text", row.get("text", ""))
    report = {
        "rules": len(rules),
        "empty_rule_ids": [key for key, text in rules.items() if not text.strip()],
        "duplicate_rule_ids": duplicates,
        "question_sets": [],
    }
    normalize = lambda text: re.sub(r"\s+", "", text)
    for path in question_paths:
        rows = read(path)
        counts = collections.Counter(str(row["question_id"]) for row in rows)
        item = {
            "file": str(path),
            "count": len(rows),
            "labelled": sum("answer" in row for row in rows),
            "duplicate_question_ids": [key for key, count in counts.items() if count > 1],
            "multi_rule_questions": sum(len(row.get("rule_id", [])) > 1 for row in rows),
            "negation_questions": sum(
                any(word in row["question_text"] for word in ("不是", "不属于", "错误", "不正确", "除外"))
                for row in rows
            ),
            "answer_distribution": dict(collections.Counter(row["answer"] for row in rows if "answer" in row)),
            "missing_rule_ids": [],
            "inline_reference_count": 0,
            "inline_reference_differences": [],
        }
        for row in rows:
            for key in row.get("rule_id", []):
                if str(key) not in rules:
                    item["missing_rule_ids"].append({"question_id": row["question_id"], "rule_id": key})
            for reference in row.get("refered_rules", []):
                item["inline_reference_count"] += 1
                match = re.match(r"\s*(\d+)\s*[.．、]\s*(.*)", reference, re.S)
                if match:
                    key, text = match.groups()
                    if key not in rules or normalize(text) != normalize(rules[key]):
                        item["inline_reference_differences"].append({"question_id": row["question_id"], "rule_id": key})
        report["question_sets"].append(item)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", nargs="+", required=True)
    parser.add_argument("--questions", nargs="+", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(args.rules, args.questions)
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        target = pathlib.Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    print(content)
