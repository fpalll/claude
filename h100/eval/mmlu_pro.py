"""MMLU-Pro evaluation.

Target (Qwen3.5-27B model card): 86.1%  |  0-shot

Dense model alignment notes
----------------------------
Qwen3.5-27B is a dense model — all 27B parameters are active on every token.
This makes output format more sensitive to decoding settings compared to MoE models:

  - MoE (e.g. 35B-A3B, ~3B active params): tolerant of varied output format
  - Dense (27B, all params active): strict format required for stable extraction

Fixes applied for dense model alignment:
  1. temperature=0 (fully greedy) — eliminates sampling-induced format variance
  2. Explicit "ANSWER: X" sentinel — deterministic extraction anchor
  3. Case-insensitive extraction fallback — handles rare lowercase outputs
"""
import json
import os
import re
from collections import defaultdict

from datasets import load_dataset

from .api_client import ModelConfig, batch_run, call_api, get_client, get_config

LETTERS = list("ABCDEFGHIJ")


def _fmt_options(options: list) -> str:
    return "\n".join(f"({LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def _fmt_prompt(ex: dict) -> str:
    # Explicit ANSWER: sentinel gives the extraction a deterministic anchor,
    # which is especially important for dense models that can produce verbose output.
    return (
        f"{ex['question']}\n\n"
        f"{_fmt_options(ex['options'])}\n\n"
        "Reply with ONLY the letter of the correct answer in this exact format:\n"
        "ANSWER: X"
    )


def _extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    # Primary: explicit sentinel format "ANSWER: X"
    m = re.search(r"ANSWER\s*:\s*\(?([A-Ja-j])\)?", text)
    if m:
        return m.group(1).upper()
    # Fallback 1: "The answer is (X)"
    m = re.search(r"[Tt]he answer is \(?([A-Ja-j])\)?", text)
    if m:
        return m.group(1).upper()
    # Fallback 2: first character if it's a valid letter
    if text.strip()[:1].upper() in LETTERS:
        return text.strip()[:1].upper()
    # Fallback 3: last standalone letter in text (case-insensitive)
    m = re.findall(r"\b([A-Ja-j])\b", text)
    return m[-1].upper() if m else None


def run_mmlu_pro(
    model: str,
    api_base: str,
    sample: int | None = None,
    output_dir: str = "results",
    thinking: bool = True,
    workers: int = 32,
) -> float:
    config: ModelConfig = get_config(model)
    print(f"\n[MMLU-Pro] Model config: temperature={config.temperature}, top_p={config.top_p}, top_k={config.top_k}")

    print("[MMLU-Pro] Loading dataset (test split)...")
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
        messages = [{"role": "user", "content": _fmt_prompt(item)}]
        response = call_api(client, model, messages, config=config)
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

    print(f"[MMLU-Pro] Evaluating {len(test_items)} items ({workers} workers)...")
    results = [r for r in batch_run(eval_one, test_items, workers=workers, desc="MMLU-Pro") if r]

    correct = sum(r["correct"] for r in results)
    total = len(results)
    null_preds = sum(1 for r in results if r["pred"] is None)
    acc = correct / total if total else 0.0

    by_subj: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        s = by_subj[r["subject"]]
        s["correct"] += r["correct"]
        s["total"] += 1

    print(f"\n=== MMLU-Pro  {model} ===")
    print(f"Overall:          {acc:.4f}  ({correct}/{total})  [target: 86.1%]")
    print(f"Failed extraction: {null_preds}/{total}  ({100*null_preds/total:.1f}%)")
    if null_preds / max(total, 1) > 0.05:
        print("  ⚠  >5% null predictions — check debug.py output for response format issues")
    print("Per subject:")
    for subj, s in sorted(by_subj.items()):
        print(f"  {subj:45s} {s['correct']/s['total']:.4f}  (n={s['total']})")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"mmlu_pro_{model.replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(
            {"model": model,
             "config": {"temperature": config.temperature, "top_p": config.top_p, "top_k": config.top_k},
             "accuracy": acc, "correct": correct, "total": total, "null_predictions": null_preds,
             "target_score": 86.1, "by_subject": dict(by_subj), "raw": results},
            f, indent=2,
        )
    print(f"Saved → {out_path}")
    return acc
