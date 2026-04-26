"""Video-MME evaluation.

Targets (Qwen3.5-27B model card):
  w/o subtitles: 82.8%  |  w/ subtitles: 87.0%  |  0-shot

Frame extraction matches VLMEvalKit Qwen3-VL model.py:
  fps=2, max nframe=128

Dense model alignment notes
----------------------------
Same issue as MMLU-Pro / MMMU: dense 27B model requires stricter decoding
and explicit answer format to produce stable single-letter outputs.
  1. temperature=0 (auto-selected via get_config() for non-MoE models)
  2. Explicit ANSWER: sentinel in prompt
  3. Case-insensitive extraction with multiple fallback patterns
"""
import io
import json
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image
from datasets import load_dataset

from .api_client import ModelConfig, batch_run, call_api, get_client, get_config, make_image_content

TARGET_FPS = 2
MAX_FRAMES = 128


def _parse_duration(duration_field) -> str:
    s = str(duration_field).lower()
    if "short" in s or "< 2" in s or "2min" in s:
        return "short"
    if "long" in s or "30min" in s or "60min" in s:
        return "long"
    return "medium"


def _extract_frames(video_bytes: bytes, fps: float = TARGET_FPS,
                    max_frames: int = MAX_FRAMES) -> list[Image.Image]:
    try:
        import decord
        from decord import VideoReader, cpu
        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0))
        video_fps = vr.get_avg_fps()
        total = len(vr)
        step = max(1, round(video_fps / fps))
        indices = list(range(0, total, step))
        if len(indices) > max_frames:
            indices = [indices[i] for i in np.linspace(0, len(indices) - 1, max_frames, dtype=int)]
        return [Image.fromarray(f) for f in vr.get_batch(indices).asnumpy()]
    except Exception:
        pass

    try:
        import av
        container = av.open(io.BytesIO(video_bytes))
        stream = container.streams.video[0]
        video_fps = float(stream.average_rate or 24)
        step = max(1, round(video_fps / fps))
        frames: list[Image.Image] = []
        for i, frame in enumerate(container.decode(video=0)):
            if i % step == 0:
                frames.append(frame.to_image())
        container.close()
        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
        return frames
    except Exception as e:
        print(f"\n  [video decode error] {e}")
        return []


def _get_video_bytes(item: dict) -> bytes | None:
    v = item.get("video")
    if isinstance(v, bytes):
        return v
    if isinstance(v, dict) and "bytes" in v:
        return v["bytes"]
    if hasattr(v, "read"):
        return v.read()
    return None


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


def _build_question_content(frame_content, question_text, options, subtitle=None) -> list:
    options_str = "\n".join(options) if isinstance(options, list) else str(options)
    text_parts = []
    if subtitle:
        text_parts.append(f"Subtitles:\n{subtitle}\n")
    text_parts.append(
        f"{question_text}\n\n"
        f"{options_str}\n\n"
        "Reply with ONLY the letter of the correct answer in this exact format:\n"
        "ANSWER: X"
    )
    return frame_content + [{"type": "text", "text": "\n".join(text_parts)}]


def _run_pass(items, model, client, config, with_subs, workers, desc) -> list:
    def eval_video(idx: int, item: dict):
        video_bytes = _get_video_bytes(item)
        if not video_bytes:
            return None
        frames = _extract_frames(video_bytes)
        if not frames:
            return None

        frame_content = [make_image_content(f) for f in frames]
        subtitle = item.get("subtitle") if with_subs else None
        questions = item.get("questions", [])
        if not questions:
            return None

        q_results = []
        for q in questions:
            options = q.get("options", [])
            gold = q.get("answer", "")
            if isinstance(gold, list):
                gold = gold[0] if gold else ""
            content = _build_question_content(frame_content, q.get("question", ""), options, subtitle)
            messages = [{"role": "user", "content": content}]
            response = call_api(client, model, messages, config=config)
            pred = _extract_answer(response)
            q_results.append({
                "video_id": item.get("video_id"),
                "question_id": q.get("question_id"),
                "duration": _parse_duration(item.get("duration_category", item.get("duration", ""))),
                "gold": gold,
                "pred": pred,
                "correct": pred == gold,
                "response": response,
                "n_frames": len(frames),
                "with_subs": with_subs,
            })
        return q_results

    nested = batch_run(eval_video, items, workers=workers, desc=desc)
    all_q: list = []
    for r in nested:
        if r:
            all_q.extend(r) if isinstance(r, list) else all_q.append(r)
    return all_q


def _stats(results: list, label: str, target: float) -> dict:
    correct = sum(r["correct"] for r in results)
    total = len(results)
    null_preds = sum(1 for r in results if r.get("pred") is None)
    acc = correct / total if total else 0.0
    by_dur: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        s = by_dur[r["duration"]]
        s["correct"] += r["correct"]
        s["total"] += 1
    avg_frames = sum(r.get("n_frames", 0) for r in results) / max(total, 1)
    print(f"\n=== Video-MME {label} ===")
    print(f"Overall:           {acc:.4f}  ({correct}/{total})  [target: {target}%]")
    print(f"Failed extraction: {null_preds}/{total}  ({100*null_preds/total:.1f}%)")
    print(f"Avg frames:        {avg_frames:.1f}  (fps={TARGET_FPS}, max={MAX_FRAMES})")
    for dur, s in sorted(by_dur.items()):
        n = s["total"]
        print(f"  {dur:50s} {s['correct']/n:.4f}  (n={n})")
    return {"accuracy": acc, "correct": correct, "total": total, "null_predictions": null_preds,
            "target_score": target, "by_duration": dict(by_dur)}


def run_videomme(
    model: str,
    api_base: str,
    sample: int | None = None,
    output_dir: str = "results",
    thinking: bool = True,
    workers: int = 8,
) -> float:
    config: ModelConfig = get_config(model)
    print(f"\n[Video-MME] Model config: temperature={config.temperature}, top_p={config.top_p}, top_k={config.top_k}")

    print("[Video-MME] Loading dataset (test split)...")
    ds = load_dataset("lmms-lab/Video-MME", split="test", trust_remote_code=True)
    items = list(ds)
    print(f"  Loaded {len(items)} videos")

    if sample:
        by_dur: dict[str, list] = defaultdict(list)
        for it in items:
            by_dur[_parse_duration(it.get("duration_category", it.get("duration", "")))].append(it)
        n_per = max(1, sample // max(len(by_dur), 1))
        sampled: list = []
        for cat_items in by_dur.values():
            sampled.extend(cat_items[:n_per])
        items = sampled[:sample]
        print(f"  [DEBUG] Sampled {len(items)} videos")

    client = get_client(api_base)

    no_sub_results = _run_pass(items, model, client, config, with_subs=False,
                               workers=workers, desc="Video-MME (no subs)")
    sub_results = _run_pass(items, model, client, config, with_subs=True,
                            workers=workers, desc="Video-MME (w/ subs)")

    no_sub_stats = _stats(no_sub_results, f"{model} (w/o subtitles)", target=82.8)
    sub_stats = _stats(sub_results, f"{model} (w/ subtitles)", target=87.0)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"videomme_{model.replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": model,
            "config": {"temperature": config.temperature, "top_p": config.top_p, "top_k": config.top_k},
            "frame_extraction": {"fps": TARGET_FPS, "max_frames": MAX_FRAMES},
            "without_subtitles": {**no_sub_stats, "raw": no_sub_results},
            "with_subtitles": {**sub_stats, "raw": sub_results},
        }, f, indent=2)
    print(f"Saved → {out_path}")

    return (no_sub_stats["accuracy"] + sub_stats["accuracy"]) / 2
