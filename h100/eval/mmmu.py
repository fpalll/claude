"""MMMU evaluation.

Exact settings from Qwen3.5-27B model card:
  - Score:     82.3
  - Shot:      0-shot
  - Dataset:   MMMU/MMMU  (validation split, 900 examples across 30 subjects)

Prompt format matches VLMEvalKit vlmeval/vlm/qwen3_vl/prompt.py:
  - No system prompt
  - Images first, then question text + options
  - "Answer with the option letter only."

Generation params match VLMEvalKit vlmeval/vlm/qwen3_vl/model.py:
  temperature=0.01, top_p=0.8, top_k=20, presence_penalty=1.5
"""
import json
import os
import re
from collections import defaultdict

from datasets import load_dataset

from .api_client import batch_run, call_api, get_client, make_image_content

# All 30 MMMU subjects
MMMU_SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Management",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music",
    "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
]


def _get_images(example: dict):
    images = []
    for i in range(1, 8):
        img = example.get(f"image_{i}")
        if img is not None:
            images.append(img)
    return images


def _fmt_options(options) -> str:
    # MMMU options are already "A. text" formatted
    return "\n".join(options) if isinstance(options, list) else str(options)


def _build_content(example: dict) -> list:
    """Images first, then question + options — matches VLMEvalKit _build_mcq_prompt."""
    content = [make_image_content(img) for img in _get_images(example)]
    question = example.get("question", "")
    options = _fmt_options(example.get("options", []))
    content.append({
        "type": "text",
        "text": f"{question}\n{options}\nAnswer with the option letter only.",
    })
    return content


def _extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    text = text.strip()
    if text[:1].upper() in "ABCD":
        return text[:1].upper()
    m = re.search(r"[Tt]he answer is \(?([A-Da-d])\)?", text)
    if m:
        return m.group(1).upper()
    m = re.findall(r"\b([A-Da-d])\b", text)
    return m[-1].upper() if m else None


def run_mmmu(
    model: str,
    api_base: str,
    sample: int | None = None,
    output_dir: str = "results",
    thinking: bool = True,   # kept for API compat; thinking is server-side via --reasoning-parser
    workers: int = 32,
) -> float:
    print("\n[MMMU] Loading dataset (30 subjects × validation split)...")
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
            content = _build_content(item)
            # No system prompt — matches VLMEvalKit Qwen3-VL prompt.py
            messages = [{"role": "user", "content": content}]
            response = call_api(client, model, messages, max_tokens=8192)
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
                    "error": str(e), "correct": False}

    print(f"[MMMU] Evaluating {len(all_items)} items (0-shot, {workers} workers)...")
    results = [r for r in batch_run(eval_one, all_items, workers=workers, desc="MMMU") if r]

    correct = sum(r["correct"] for r in results)
    total = len(results)
    acc = correct / total if total else 0.0

    by_subj_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        s = by_subj_stats[r["subject"]]
        s["correct"] += r["correct"]
        s["total"] += 1

    print(f"\n=== MMMU  {model} ===")
    print(f"Overall: {acc:.4f}  ({correct}/{total})  [target: 82.3%]")
    print("Per subject:")
    for subj, s in sorted(by_subj_stats.items()):
        print(f"  {subj:45s} {s['correct']/s['total']:.4f}  (n={s['total']})")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"mmmu_{model.replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(
            {"model": model, "accuracy": acc, "correct": correct, "total": total,
             "target_score": 82.3,
             "by_subject": {k: v for k, v in by_subj_stats.items()}, "raw": results},
            f, indent=2,
        )
    print(f"Saved → {out_path}")
    return acc
