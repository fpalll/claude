"""MMMU evaluation.

Target (Qwen3.5-27B model card): 82.3%  |  0-shot

Dense model alignment notes
----------------------------
Same pipeline as MoE models produces lower scores on dense Qwen3.5-27B because:
  - Dense models are more sensitive to decoding stochasticity
  - Output format becomes less consistent under sampling
  - Explicit answer sentinel (ANSWER: X) gives extraction a stable anchor

Fixes applied for dense model alignment:
  1. temperature=0 (auto-selected by get_config() for non-MoE models)
  2. Explicit ANSWER: sentinel in prompt
  3. Case-insensitive extraction with multiple fallback patterns
"""
import json
import os
import re
from collections import defaultdict

from datasets import load_dataset

from .api_client import ModelConfig, batch_run, call_api, get_client, get_config, make_image_content

MMMU_SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Management",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music",
    "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
]


def _get_images(example: dict):
    return [example[f"image_{i}"] for i in range(1, 8) if example.get(f"image_{i}") is not None]


def _fmt_options(options) -> str:
    return "\n".join(options) if isinstance(options, list) else str(options)


def _build_content(example: dict) -> list:
    content = [make_image_content(img) for img in _get_images(example)]
    question = example.get("question", "")
    options = _fmt_options(example.get("options", []))
    content.append({
        "type": "text",
        "text": (
            f"{question}\n\n"
            f"{options}\n\n"
            "Reply with ONLY the letter of the correct answer in this exact format:\n"
            "ANSWER: X"
        ),
    })
    return content


def _extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"ANSWER\s*:\s*\(?([A-Da-d])\)?", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"[Tt]he answer is \(?([A-Da-d])\)?", text)
    if m:
        return m.group(1).upper()
    if text.strip()[:1].upper() in "ABCD":
        return text.strip()[:1].upper()
    m = re.findall(r"\b([A-Da-d])\b", text)
    return m[-1].upper() if m else None


def run_mmmu(
    model: str,
    api_base: str,
    sample: int | None = None,
    output_dir: str = "results",
    thinking: bool = True,
    workers: int = 32,
) -> float:
    config: ModelConfig = get_config(model)
    print(f"\n[MMMU] Model config: temperature={config.temperature}, top_p={config.top_p}, top_k={config.top_k}")

    print("[MMMU] Loading dataset (30 subjects × validation split)...")
    all_items: list = []
    for subj in MMMU_SUBJECTS:
        try:
            ds = load_dataset("MMMU/MMMU", subj, split="validation", trust_remote_code=True)
            for ex in ds:
                ex = dict(ex)
                ex["_subject"] = subj
                all_items.append(ex)
        except Exception as e:
            print(f"  Warning: could not load {subj}: {e}")

    print(f"  Loaded {len(all_items)} MMMU validation examples")

    if sample:
        by_subj: dict[str, list] = defaultdict(list)
        for ex in all_items:
            by_subj[ex["_subject"]].append(ex)
        n_per = max(1, sample // len(by_subj))
        all_items = [ex for items in by_subj.values() for ex in items[:n_per]][:sample]
        print(f"  [DEBUG] Sampled {len(all_items)} examples")

    client = get_client(api_base)

    def eval_one(idx: int, item: dict):
        try:
            messages = [{"role": "user", "content": _build_content(item)}]
            response = call_api(client, model, messages, config=config)
            pred = _extract_answer(response)
            gold = item.get("answer", "")
            if isinstance(gold, list):
                gold = gold[0] if gold else ""
            return {
                "id": item.get("id"),
                "subject": item.get("_subject", ""),
                "question_type": item.get("question_type", ""),
                "gold": gold,
                "pred": pred,
                "correct": pred == gold,
                "response": response,
            }
        except Exception as e:
            return {"id": item.get("id"), "subject": item.get("_subject", ""),
                    "error": str(e), "correct": False, "pred": None}

    print(f"[MMMU] Evaluating {len(all_items)} items ({workers} workers)...")
    results = [r for r in batch_run(eval_one, all_items, workers=workers, desc="MMMU") if r]

    correct = sum(r["correct"] for r in results)
    total = len(results)
    null_preds = sum(1 for r in results if r.get("pred") is None)
    acc = correct / total if total else 0.0

    by_subj_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        s = by_subj_stats[r["subject"]]
        s["correct"] += r["correct"]
        s["total"] += 1

    print(f"\n=== MMMU  {model} ===")
    print(f"Overall:           {acc:.4f}  ({correct}/{total})  [target: 82.3%]")
    print(f"Failed extraction: {null_preds}/{total}  ({100*null_preds/total:.1f}%)")
    if null_preds / max(total, 1) > 0.05:
        print("  ⚠  >5% null predictions — check debug.py output for response format issues")
    print("Per subject:")
    for subj, s in sorted(by_subj_stats.items()):
        print(f"  {subj:45s} {s['correct']/s['total']:.4f}  (n={s['total']})")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"mmmu_{model.replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(
            {"model": model,
             "config": {"temperature": config.temperature, "top_p": config.top_p, "top_k": config.top_k},
             "accuracy": acc, "correct": correct, "total": total, "null_predictions": null_preds,
             "target_score": 82.3,
             "by_subject": {k: v for k, v in by_subj_stats.items()}, "raw": results},
            f, indent=2,
        )
    print(f"Saved → {out_path}")
    return acc
