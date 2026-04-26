"""
BFCL-V4 Web Search evaluation.

Target (Qwen3.5-27B model card): 68.5%

Official spec (gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html):
  - 100 multi-hop questions (2-4 hops each)
  - Tools: duckduckgo_search + fetch_url_content
  - Scoring: exact match after lowercase + punctuation normalization
  - No SerpAPI required — uses duckduckgo-search library (free)

Dense model alignment (same fix as MMLU-Pro/MMMU):
  - temperature=0 for Qwen3.5-27B (greedy, format-stable)

Agent loop improvements (from BFCL analysis):
  1. Multi-hop enforcement via system prompt
  2. Forced tool usage (no memory-only answers)
  3. Always fetch full page after search (not snippet-only)
  4. Retry on tool failure with different keywords
  5. Strict ANSWER: sentinel for reliable extraction

Dataset loading:
  The BFCL-V4 web search dataset is not yet on HuggingFace.
  This script tries three sources in order:
    1. Local file (--dataset-path)
    2. BFCL GitHub repo (auto-download)
    3. Official HuggingFace dataset (gorilla-llm/Berkeley-Function-Calling-Leaderboard)
"""
import json
import os
import re
import string
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from .api_client import get_client, get_config, ModelConfig
from .tools.web_search import TOOL_DEFINITIONS, execute_tool

# ── System prompt ─────────────────────────────────────────────────────────────
# Incorporates multi-hop enforcement + dense model alignment
SYSTEM_PROMPT = """\
You are a web search agent. You MUST answer by searching the web — never from memory.

STRICT RULES:
1. Break the question into sub-questions and search each one separately (multi-hop).
2. For EVERY search: first call duckduckgo_search, then call fetch_url_content on the \
most relevant result URL — do NOT rely on snippets alone.
3. Keep search queries SHORT (3-6 words) for better retrieval.
4. If a tool fails (timeout, 403, 429): retry with different keywords or a different URL.
5. Do NOT answer from training knowledge — only from web results.
6. After gathering evidence from multiple searches, provide your final answer.

FINAL ANSWER FORMAT (respond with ONLY this JSON, nothing else):
{"answer": "<short precise answer>", "context": "<brief explanation>"}

If you cannot find the answer after thorough searching:
{"answer": "I do not know", "context": "<what you searched for>"}
"""

MAX_TURNS = 15          # max agent steps before forcing a final answer
MAX_TOOL_RESULT_CHARS = 6000   # truncate tool results to stay within context

# ── Dataset loading ────────────────────────────────────────────────────────────

_BFCL_V4_URLS = [
    # Try the BFCL GitHub repo directly (v4 web search file)
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/data/BFCL_v4_web_search.json",
    # Alternate naming convention
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/data/BFCL_v3_multi_turn_web_search.json",
]


def _download_dataset(cache_path: str = "data/bfcl_v4_web_search.json") -> list[dict]:
    import urllib.request

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        print(f"  [cache] Loading from {cache_path}")
        return _load_jsonl(cache_path)

    for url in _BFCL_V4_URLS:
        try:
            print(f"  Trying: {url}")
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read().decode("utf-8")
            with open(cache_path, "w") as f:
                f.write(raw)
            print(f"  Downloaded → {cache_path}")
            return _load_jsonl(cache_path)
        except Exception as e:
            print(f"  Failed: {e}")

    raise FileNotFoundError(
        "Could not download BFCL-V4 web search dataset automatically.\n"
        "Manual steps:\n"
        "  1. Go to https://github.com/ShishirPatil/gorilla/tree/main/"
        "berkeley-function-call-leaderboard/data\n"
        "  2. Download the web_search JSON file\n"
        "  3. Pass it with: --dataset-path /path/to/file.json"
    )


def _load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip malformed lines
    if not items:
        # Try loading as a single JSON array
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
    return items


def load_dataset(dataset_path: str | None = None) -> list[dict]:
    if dataset_path:
        print(f"[BFCL] Loading from {dataset_path}")
        return _load_jsonl(dataset_path)

    # Try auto-download
    print("[BFCL] Attempting to download BFCL-V4 web search dataset...")
    items = _download_dataset()

    # Normalise to {id, question, ground_truth} format
    normalised = []
    for item in items:
        q = item.get("question", item.get("query", ""))
        # Handle multi-turn format [[{role, content}]]
        if isinstance(q, list):
            for turn in q:
                if isinstance(turn, list):
                    for msg in turn:
                        if msg.get("role") == "user":
                            q = msg["content"]
                            break
                elif isinstance(turn, dict) and turn.get("role") == "user":
                    q = turn["content"]
                    break
        normalised.append({
            "id":           item.get("id", ""),
            "question":     q,
            "ground_truth": item.get("ground_truth", item.get("answer", "")),
        })
    return normalised


# ── Scoring ───────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _is_correct(pred: str | None, gold: str | None) -> bool:
    if not pred or not gold:
        return False
    gold_list = gold if isinstance(gold, list) else [gold]
    pred_n = _normalise(pred)
    return any(_normalise(g) == pred_n for g in gold_list)


# ── Answer parsing ─────────────────────────────────────────────────────────────

def _parse_final_answer(text: str | None) -> str | None:
    if not text:
        return None
    # Try JSON parse
    for pattern in [
        r'\{[^{}]*"answer"\s*:\s*"([^"]+)"[^{}]*\}',
        r'\{[^{}]*"answer"\s*:\s*([^,}]+)',
    ]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip().strip('"')
    # Try plain text fallbacks
    m = re.search(r"[Aa]nswer\s*:\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    return None


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(
    client: OpenAI,
    model: str,
    config: ModelConfig,
    question: str,
    verbose: bool = False,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_calls_made = 0
    final_content = None

    for turn in range(MAX_TURNS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=config.temperature,
                top_p=config.top_p,
                presence_penalty=config.presence_penalty,
                max_tokens=4096,
                extra_body={"top_k": config.top_k},
            )
        except Exception as e:
            return {"answer": None, "error": str(e), "turns": turn, "tool_calls": tool_calls_made}

        choice = resp.choices[0]
        msg = choice.message

        # Convert message to dict for history (handle tool_calls field)
        msg_dict: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        # Model stopped — check for final answer
        if choice.finish_reason == "stop" or not msg.tool_calls:
            final_content = msg.content or ""
            if verbose:
                print(f"    [turn {turn+1}] stop — content: {repr(final_content[:120])}")
            break

        # Execute tool calls
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                if verbose:
                    print(f"    [turn {turn+1}] tool={fn_name} args={fn_args}")

                result = execute_tool(fn_name, fn_args)
                # Truncate to avoid blowing context
                result_str = str(result)[:MAX_TOOL_RESULT_CHARS]
                tool_calls_made += 1

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
    else:
        # Hit MAX_TURNS — ask model to produce final answer now
        messages.append({
            "role": "user",
            "content": (
                "You have reached the search limit. "
                "Based on what you found, provide your final answer now in this format:\n"
                '{"answer": "<answer>", "context": "<reasoning>"}'
            ),
        })
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=512,
            )
            final_content = resp.choices[0].message.content or ""
        except Exception:
            final_content = ""

    answer = _parse_final_answer(final_content)
    return {
        "answer": answer,
        "raw_response": final_content,
        "turns": min(turn + 1, MAX_TURNS),
        "tool_calls": tool_calls_made,
    }


# ── Main evaluation ────────────────────────────────────────────────────────────

def run_bfcl_websearch(
    model: str,
    api_base: str,
    dataset_path: str | None = None,
    sample: int | None = None,
    output_dir: str = "results",
    thinking: bool = True,
    workers: int = 4,       # keep low — each job makes many sequential API calls
    verbose: bool = False,
) -> float:
    config = get_config(model)
    print(f"\n[BFCL-V4] Model config: temperature={config.temperature}, top_p={config.top_p}, top_k={config.top_k}")

    items = load_dataset(dataset_path)
    print(f"[BFCL-V4] Loaded {len(items)} questions")

    if sample:
        items = items[:sample]
        print(f"  [DEBUG] Sampled {len(items)} questions")

    client = get_client(api_base)

    def eval_one(idx: int, item: dict) -> dict:
        result = run_agent(client, model, config, item["question"], verbose=verbose)
        gold = item.get("ground_truth", "")
        pred = result.get("answer")
        correct = _is_correct(pred, gold)
        return {
            "id": item.get("id", idx),
            "question": item["question"],
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "turns": result.get("turns", 0),
            "tool_calls": result.get("tool_calls", 0),
            "raw_response": result.get("raw_response", ""),
            "error": result.get("error"),
        }

    results: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(eval_one, i, item): i for i, item in enumerate(items)}
        for fut in tqdm(as_completed(futures), total=len(items), desc="BFCL-V4"):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = {"id": idx, "correct": False, "error": str(e)}

    results = [r for r in results if r is not None]
    correct = sum(r["correct"] for r in results)
    total = len(results)
    null_preds = sum(1 for r in results if r.get("pred") is None)
    avg_turns = sum(r.get("turns", 0) for r in results) / max(total, 1)
    avg_tools = sum(r.get("tool_calls", 0) for r in results) / max(total, 1)
    acc = correct / total if total else 0.0

    print(f"\n=== BFCL-V4 Web Search  {model} ===")
    print(f"Overall:           {acc:.4f}  ({correct}/{total})  [target: 68.5%]")
    print(f"Failed extraction: {null_preds}/{total}  ({100*null_preds/total:.1f}%)")
    print(f"Avg turns/q:       {avg_turns:.1f}  (max={MAX_TURNS})")
    print(f"Avg tool calls/q:  {avg_tools:.1f}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"bfcl_websearch_{model.replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(
            {"model": model,
             "config": {"temperature": config.temperature, "top_p": config.top_p, "top_k": config.top_k},
             "accuracy": acc, "correct": correct, "total": total,
             "null_predictions": null_preds,
             "avg_turns": avg_turns, "avg_tool_calls": avg_tools,
             "target_score": 68.5, "raw": results},
            f, indent=2,
        )
    print(f"Saved → {out_path}")
    return acc
