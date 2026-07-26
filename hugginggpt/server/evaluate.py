import json
import os
import re
import sys
from collections import defaultdict
from typing import Optional

# ── Answer extraction ────────────────────────────────────────────────────────

def extract_answer(message: str) -> Optional[str]:
    """
    Extract a single choice letter (A-F) from a free-form LLM response.
    Strategy (in priority order):
      1. Look inside an explicit "Direct Answer" / "### Answer" section.
      2. Look for bold markdown answer: **(X)** or **X** near "answer".
      3. Look for "(X)" in the first 400 chars of the message.
      4. Look for "answer is X" / "answer: X" anywhere.
      5. Return None if nothing found.
    """
    if not message or not isinstance(message, str):
        return None

    LETTER = r'[A-F]'

    # 1. Direct-answer section  -----------------------------------------------
    section_pat = re.compile(
        r'(?:direct answer|###\s*(?:direct\s+)?answer|answer to your request)'
        r'[\s\S]{0,300}',
        re.IGNORECASE
    )
    section_match = section_pat.search(message)
    if section_match:
        section = section_match.group(0)
        # Bold like **(B)** or **B**
        m = re.search(r'\*\*\(?(' + LETTER + r')\)?\**', section)
        if m:
            return m.group(1)
        # Plain like (B) or "(B) lamp"
        m = re.search(r'\((' + LETTER + r')\)', section)
        if m:
            return m.group(1)

    # 2. Bold markdown answer anywhere ----------------------------------------
    # **(B)** or **(B) some text** - pick the first one
    bold_matches = re.findall(r'\*\*\((' + LETTER + r')\)', message)
    if bold_matches:
        return bold_matches[0]

    # **B** near "answer" within 120 chars
    for m in re.finditer(r'\*\*(' + LETTER + r')\*\*', message):
        ctx = message[max(0, m.start()-120):m.end()+40].lower()
        if 'answer' in ctx or 'correct' in ctx or 'therefore' in ctx:
            return m.group(1)

    # 3. "(X)" in first 400 chars of message ----------------------------------
    head = message[:400]
    paren_matches = re.findall(r'\((' + LETTER + r')\)', head)
    if paren_matches:
        return paren_matches[0]

    # 4. "answer is X" / "answer: X" / "therefore X" -------------------------
    ans_pat = re.compile(
        r'(?:answer(?:\s+is)?|therefore|correct(?:\s+answer)?|the\s+answer)\s*[:\s]\s*\(?(' + LETTER + r')\)?',
        re.IGNORECASE
    )
    m = ans_pat.search(message)
    if m:
        return m.group(1)

    # 5. Any (X) anywhere in message ------------------------------------------
    all_parens = re.findall(r'\((' + LETTER + r')\)', message)
    if all_parens:
        return all_parens[0]

    return None


# ── Evaluation ───────────────────────────────────────────────────────────────

FAILURE_MARKERS = [
    "upstream error",
    "do request failed",
    "openai_error",
    "network error",
    "request failed after",
]


def is_api_failure(message: str) -> bool:
    """Detect entries where the LLM never produced a real answer (pipeline/API failed)."""
    if not message:
        return True
    m = message.lower()
    return any(p in m for p in FAILURE_MARKERS)


def derive_output_path(result_file: str, output_arg: Optional[str]) -> str:
    """If output path not given, turn result_<dataset>.json -> accuracy_<dataset>.json in the same dir."""
    if output_arg:
        return output_arg
    d, base = os.path.split(result_file)
    name, _ = os.path.splitext(base)
    if name.startswith("result_"):
        name = "accuracy_" + name[len("result_"):]
    else:
        name = "accuracy_" + name
    return os.path.join(d, name + ".json")


def evaluate(result_file: str, output_file: Optional[str] = None):
    with open(result_file) as f:
        data = json.load(f)

    per_task = defaultdict(lambda: {"correct": 0, "total": 0, "parsed": 0, "api_failed": 0})
    details = []
    correct_total = 0
    parsed_total = 0
    api_failed_total = 0

    for entry in data:
        sample_id = entry["id"]
        task = entry.get("task", "unknown")
        gt = entry.get("ground_truth", "").strip().upper()
        response = entry.get("response", {})
        message = response.get("message", "") if isinstance(response, dict) else ""

        # _manual_answer: injected by audit script for implicit-answer samples
        manual = response.get("_manual_answer") if isinstance(response, dict) else None

        api_failed = is_api_failure(message)
        if api_failed:
            api_failed_total += 1

        if manual:
            predicted = manual.strip().upper()
            parsed_total += 1
        elif api_failed:
            predicted = None
        else:
            predicted = extract_answer(message)
            if predicted:
                predicted = predicted.upper()
                parsed_total += 1

        is_correct = (predicted == gt) if predicted else False
        if is_correct:
            correct_total += 1

        per_task[task]["total"] += 1
        per_task[task]["correct"] += int(is_correct)
        per_task[task]["parsed"] += int(predicted is not None)
        per_task[task]["api_failed"] += int(api_failed)

        details.append({
            "id": sample_id,
            "task": task,
            "ground_truth": gt,
            "predicted": predicted,
            "correct": is_correct,
            "api_failed": api_failed,
        })

    total = len(data)
    unparsed = total - parsed_total - api_failed_total  # responded but we couldn't parse a letter

    summary = {
        "result_file": os.path.abspath(result_file),
        "total_samples": total,
        "correct": correct_total,
        "parsed": parsed_total,
        "unparsed_but_responded": unparsed,
        "api_failed": api_failed_total,
        "accuracy_overall": round(correct_total / total, 4) if total else 0.0,
        "accuracy_of_parsed": round(correct_total / parsed_total, 4) if parsed_total else 0.0,
        "accuracy_of_responded": round(
            correct_total / (total - api_failed_total), 4
        ) if (total - api_failed_total) else 0.0,
        "per_task": {
            task: {
                "total": t["total"],
                "correct": t["correct"],
                "parsed": t["parsed"],
                "api_failed": t["api_failed"],
                "accuracy": round(t["correct"] / t["total"], 4) if t["total"] else 0.0,
            }
            for task, t in sorted(per_task.items())
        },
    }

    out_path = derive_output_path(result_file, output_file)
    payload = {"summary": summary, "per_question": details}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  File: {result_file}")
    print(f"{'='*60}")
    print(f"  Total samples         : {total}")
    print(f"  API-failed responses  : {api_failed_total}  (no LLM output)")
    print(f"  Parsed answers        : {parsed_total}  "
          f"(unparsed but responded: {unparsed})")
    print(f"  Correct               : {correct_total}")
    if total:
        print(f"  Accuracy (overall)    : {correct_total/total*100:.1f}%")
    if parsed_total:
        print(f"  Accuracy (of parsed)  : {correct_total/parsed_total*100:.1f}%")
    if (total - api_failed_total) > 0:
        print(f"  Accuracy (of responded): "
              f"{correct_total/(total-api_failed_total)*100:.1f}%")

    print(f"\n{'─'*60}")
    print(f"  Per-task breakdown:")
    print(f"{'─'*60}")
    for task in sorted(per_task):
        t = per_task[task]
        acc = t["correct"] / t["total"] * 100 if t["total"] else 0
        print(f"  {task:<35} {t['correct']:>3}/{t['total']:<3}  "
              f"({acc:4.1f}%)  api_failed={t['api_failed']}")

    print(f"\n{'─'*60}")
    print(f"  Per-question results:")
    print(f"{'─'*60}")
    for d in details:
        if d["api_failed"]:
            mark = "!"  # API error, not a model wrong answer
            pred_str = "API_ERR"
        else:
            mark = "+" if d["correct"] else "-"
            pred_str = d["predicted"] if d["predicted"] else "?"
        print(f"  [{mark}] {d['id']:<50}  GT={d['ground_truth']}  Pred={pred_str}")

    print(f"\n  -> wrote {out_path}")
    print(f"{'='*60}\n")
    return summary, details


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    # Usage: python evaluate.py <result.json> [output.json] [<result2.json> ...]
    # If an arg looks like "--out=path", pair it with the preceding file.
    if not args:
        args = ["result_blink_100.json", "result_cvbench_100.json"]

    # Simple pairing: each arg is a result file; output path is auto-derived.
    # To override, pass explicit output after each input separated by :
    # e.g.  result_blink_500.json:accuracy_blink.json
    for a in args:
        if ":" in a and os.path.exists(a.split(":", 1)[0]):
            in_f, out_f = a.split(":", 1)
            evaluate(in_f, out_f)
        else:
            evaluate(a)
