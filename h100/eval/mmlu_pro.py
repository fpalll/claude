"""MMLU-Pro evaluation.

Exact settings from Qwen3.5-27B model card:
  - Score:     86.1
  - Shot:      0-shot
  - Dataset:   TIGER-Lab/MMLU-Pro  (test split, ~12k questions, options A-J)

Generation params match VLMEvalKit Qwen3-VL implementation:
  temperature=0.01, top_p=0.8, top_k=20, presence_penalty=1.5

Prompt format: question + options, then "Answer with the option letter only."
(matches VLMEvalKit vlmeval/vlm/qwen3_vl/prompt.py MCQ format)
"""
import json
import os
import re
from collections import defaultdict

from datasets import load_dataset

from .api_client import batch_run, call_api, get_client

LETTERS = list("ABCDEFGHIJ")


def _fmt_options(options: list) -> str:
    return "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))


def _fmt_prompt(ex: dict) -> str:
    return (
        f"{ex['question']}\n\n"
        f"{_fmt_options(ex['options'])}\n\n"
        "Answer with the option letter only."
    )


def _extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    text = text.strip()
    if text[:1].upper() in LETTERS:
        return text[:1].upper()
    # "The answer is (X)" / "the answer is X"
    m = re.search(r"[Tt]he answer is \(?([A-Ja-j])\)?", text)
    if m:
        return m.group(1).upper()
    # Last standalone letter A-J (case-insensitive)
    m = re.findall(r"\b([A-Ja-j])\b", text)
    return m[-1].upper() if m else None


def run_mmlu_pro(
    model: str,
    api_base: str,
    sample: int | None = None,
    output_dir: str = "results",
    thinking: bool = True,   # kept for API compat; thinking is server-side now
    workers: int = 32,
) -> float:
    print("\n[MMLU-Pro] Loading dataset (test split)...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", trust_remote_code=True)
    test_items = list(ds["test"])

    if sample:
        by_subj: dict[str, list] = defaultdict(list)
        for ex in test_items:
            by_subj[ex.get("category", "")].append(ex)
        n_per = max(1, sample // max(len(by_subj), 1))
        test_items = [ex for items in by_subj.values() for ex in items[:n_per]][:sample]
        print(f"  [DEBUG] Sampled {len(test_items)} examples across subjects")

    client = get_client(api_base)

    def eval_one(idx: int, item: dict):
        # No system prompt — matches VLMEvalKit Qwen3-VL prompt.py
        messages = [{"role": "user", "content": _fmt_prompt(item)}]
        response = call_api(client, model, messages, max_tokens=8192)
        pred = _extract_answer(response)
        gold = item["answer"]
        return {
            "question_id": item.get("question_id"),
            "subject": item.get("category", ""),
            "gold": gold,
            "pred": pred,
            "correct": pred == gold,
            "response": response,
        }

    print(f"[MMLU-Pro] Evaluating {len(test_items)} items (0-shot, {workers} workers)...")
    results = [r for r in batch_run(eval_one, test_items, workers=workers, desc="MMLU-Pro") if r]

    correct = sum(r["correct"] for r in results)
    total = len(results)
    acc = correct / total if total else 0.0

    by_subj: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        s = by_subj[r["subject"]]
        s["correct"] += r["correct"]
        s["total"] += 1

    print(f"\n=== MMLU-Pro  {model} ===")
    print(f"Overall: {acc:.4f}  ({correct}/{total})  [target: 86.1%]")
    print("Per subject:")
    for subj, s in sorted(by_subj.items()):
        print(f"  {subj:45s} {s['correct']/s['total']:.4f}  (n={s['total']})")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"mmlu_pro_{model.replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(
            {"model": model, "accuracy": acc, "correct": correct, "total": total,
             "target_score": 86.1, "by_subject": dict(by_subj), "raw": results},
            f, indent=2,
        )
    print(f"Saved → {out_path}")
    return acc
