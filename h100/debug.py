"""
Quick diagnostic — run this first to see what the model actually outputs.

Usage:
  python debug.py --model Qwen/Qwen3.5-27B --api-base http://127.0.0.1:18000/v1
"""
import argparse
import json
import re
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:18000/v1")
    args = parser.parse_args()

    client = OpenAI(base_url=args.api_base, api_key="EMPTY")

    # ── 0. Show selected config ───────────────────────────────────────────────
    import sys
    sys.path.insert(0, ".")
    from eval.api_client import get_config
    cfg = get_config(args.model)
    print("=" * 60)
    print("0. MODEL CONFIG (dense vs MoE auto-detection)")
    print(f"   model      : {args.model}")
    print(f"   temperature: {cfg.temperature}  ← 0 = dense/greedy, 0.01 = MoE")
    print(f"   top_p      : {cfg.top_p}")
    print(f"   top_k      : {cfg.top_k}")
    print(f"   presence_p : {cfg.presence_penalty}")

    # ── 1. Check what model is served ────────────────────────────────────────
    print("=" * 60)
    print("1. SERVED MODELS")
    models = client.models.list()
    for m in models.data:
        print(f"   {m.id}")

    # ── 2. Send a simple MCQ and print the FULL raw response ─────────────────
    print("\n" + "=" * 60)
    print("2. RAW API RESPONSE (simple MCQ)")
    messages = [
        {
            "role": "user",
            "content": (
                "What is the capital of France?\n"
                "A. Berlin\nB. Madrid\nC. Paris\nD. Rome\n\n"
                "Answer with the option letter only."
            ),
        }
    ]
    resp = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=0.01,
        top_p=0.8,
        max_tokens=1024,
        extra_body={"top_k": 20},
    )
    choice = resp.choices[0]
    content = choice.message.content
    print(f"  finish_reason : {choice.finish_reason}")
    print(f"  content       : {repr(content)}")
    # Check for non-standard reasoning field
    raw = choice.model_dump() if hasattr(choice, "model_dump") else {}
    msg_raw = raw.get("message", {})
    if "reasoning_content" in msg_raw:
        print(f"  reasoning_content: {repr(msg_raw['reasoning_content'][:200])}")
    if "<think>" in (content or ""):
        print("  ⚠  <think> block found in content — reasoning_parser may not be stripping it")

    # ── 3. Check extraction on the response ──────────────────────────────────
    print("\n" + "=" * 60)
    print("3. ANSWER EXTRACTION TEST")

    def extract(text):
        if not text:
            return None
        t = text.strip()
        if t[:1] in "ABCDabcd":
            return t[:1].upper()
        m = re.search(r"[Tt]he answer is \(?([A-Da-d])\)?", text)
        if m:
            return m.group(1).upper()
        m = re.search(r"\b([A-Da-d])\b", text)
        return m.group(1).upper() if m else None

    extracted = extract(content)
    print(f"  extracted answer : {repr(extracted)}  (expected 'C')")
    if extracted == "C":
        print("  ✓ extraction working")
    else:
        print("  ✗ extraction FAILED — this is likely why scores are halved")

    # ── 4. Run 5 MMLU-Pro style questions and show all responses ─────────────
    print("\n" + "=" * 60)
    print("4. FIVE MMLU-PRO STYLE QUESTIONS (shows response pattern)")
    questions = [
        ("Which of the following is a measure of central tendency?", ["Range", "Mean", "Variance", "Standard deviation"], "B"),
        ("What does DNA stand for?", ["Deoxyribonucleic acid", "Diribonucleic acid", "Deoxyribonitric acid", "Dioxyribonucleic acid"], "A"),
        ("What is 17 × 13?", ["191", "201", "221", "211"], "C"),
        ("Who wrote 'Pride and Prejudice'?", ["Charlotte Brontë", "Jane Austen", "George Eliot", "Emily Brontë"], "B"),
        ("The speed of light in vacuum is approximately:", ["3×10^6 m/s", "3×10^8 m/s", "3×10^10 m/s", "3×10^12 m/s"], "B"),
    ]
    correct = 0
    none_count = 0
    for i, (q, opts, gold) in enumerate(questions):
        opt_str = "\n".join(f"{chr(65+j)}. {o}" for j, o in enumerate(opts))
        msg = [{"role": "user", "content": f"{q}\n{opt_str}\n\nAnswer with the option letter only."}]
        r = client.chat.completions.create(
            model=args.model, messages=msg,
            temperature=0.01, top_p=0.8, max_tokens=512,
            extra_body={"top_k": 20},
        )
        raw_content = r.choices[0].message.content or ""
        pred = extract(raw_content)
        is_correct = pred == gold
        if pred is None:
            none_count += 1
        if is_correct:
            correct += 1
        print(f"  Q{i+1}: raw={repr(raw_content[:80]):80s}  pred={pred}  gold={gold}  {'✓' if is_correct else '✗'}")

    print(f"\n  Score: {correct}/5  |  Failed extractions (None): {none_count}/5")
    if none_count > 1:
        print("  ⚠  Many None extractions → extraction is the problem")
    if correct < 3:
        print("  ⚠  Low accuracy even with valid extractions → model output format issue")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. LIKELY CAUSE")
    if "<think>" in (content or "") and "</think>" not in (content or ""):
        print("  → reasoning_parser NOT stripping think blocks; restart SGLang with --reasoning-parser qwen3")
    elif none_count > 1:
        print("  → answer extraction failing; model may output lowercase letters or non-standard format")
        print("    Fix: update extraction in eval/mmlu_pro.py, eval/mmmu.py, eval/videomme.py to be case-insensitive")
    elif correct < 3:
        print("  → model is giving wrong answers; check model is loaded correctly (instruct vs base)")
    else:
        print("  → single-question test passed; problem may be intermittent or benchmark-specific")
        print("    Run with --sample 20 and check the raw JSON output in results/")


if __name__ == "__main__":
    main()
