import os
import sys
import json
import argparse
import base64
import mimetypes
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import warnings
import string
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports from refactored files
from dataset_utils import load_dataset, dump_image, MMMU_preproc
from eval_utils import build_judge, eval_single_sample, _openai_assistant_text

# Optional heavy deps (vLLM path only; imported lazily in run_inference_vllm)


def build_mmmu_prompt(line, dump_image_func, dataset, prompt_style: str = "default"):
    """Build MMMU dataset prompt with standard resolution settings.

    prompt_style:
      - default: Qwen3-VL instruct style (images then text; Options: A. ...).
      - qwen35: Qwen3.5 demo style (images first, then text; Choices: (A) ...;
                JSON answer instruction for multiple-choice).
    """
    # Standard resolution settings
    MIN_PIXELS = 1280 * 28 * 28  # ~1M pixels
    MAX_PIXELS = 5120 * 28 * 28  # ~4M pixels

    tgt_path = dump_image_func(line)
    question = line["question"]
    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }
    hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None

    if prompt_style == "qwen35":
        chunks = []
        if hint is not None:
            chunks.append(f"Hint: {hint}")
        chunks.append(question.strip())
        if len(options):
            ordered = sorted(options.items(), key=lambda kv: kv[0])
            choice_lines = [f"({k}) {item}" for k, item in ordered]
            chunks.append("Choices:\n" + "\n".join(choice_lines))
            letters = ",".join(k for k, _ in ordered)
            chunks.append(
                "Think step by step before answering.\n"
                "Please show your choice in the answer field with only the choice letter, "
                'e.g., "answer": "C".\n'
                f'The value of "answer" must be exactly one of: {letters}.'
            )
        else:
            chunks.append(
                "Solve the problem step by step. On the last line, give your final answer "
                'as: ANSWER: <answer>'
            )
        prompt = "\n\n".join(chunks)
    else:
        options_prompt = "Options:\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"
        prompt = ""
        if hint is not None:
            prompt += f"Hint: {hint}\n"
        prompt += f"Question: {question}\n"
        if len(options):
            prompt += options_prompt
            prompt += "Please select the correct answer from the options above. \n"
        prompt = prompt.rstrip()

    # Build messages in standard conversation format
    content = []
    paths = tgt_path if isinstance(tgt_path, list) else [tgt_path]
    for p in paths:
        content.append(
            {
                "type": "image",
                "image": p,
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            }
        )
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    return messages


def _encode_image_file(path: str) -> Dict[str, Any]:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def mmmu_messages_to_openai(messages: List[Dict]) -> List[Dict[str, Any]]:
    """
    Convert Qwen-style multimodal messages (local image paths) to OpenAI / SGLang
    chat.completions format (image_url + text).
    """
    out = []
    for msg in messages:
        role = msg["role"]
        parts = []
        for block in msg["content"]:
            if block["type"] == "text":
                parts.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                p = block["image"]
                if isinstance(p, list):
                    for one in p:
                        parts.append(_encode_image_file(one))
                else:
                    parts.append(_encode_image_file(p))
            else:
                raise ValueError(f"Unsupported content block type: {block.get('type')}")
        out.append({"role": role, "content": parts})
    return out


def prepare_inputs_for_vllm(messages, processor):
    """Prepare inputs for vLLM (following the examples in README.md)."""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def _normalize_openai_base_url(url: str) -> str:
    u = url.rstrip("/")
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def _subsample_dataframe_for_debug(
    data: pd.DataFrame, max_samples: Optional[int], sample_seed: int
) -> pd.DataFrame:
    """Random subset for quick debug runs (reproducible with sample_seed)."""
    if max_samples is None or max_samples <= 0:
        return data
    n = min(int(max_samples), len(data))
    if n >= len(data):
        return data
    return data.sample(n=n, random_state=sample_seed).reset_index(drop=True)


def _json_safe_value(x: Any) -> Any:
    """Best-effort JSON-serializable value for result records."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _json_safe_value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe_value(v) for v in x]
    return x


def _vllm_output_raw_dict(output) -> Dict[str, Any]:
    """Extract serializable raw generation fields from vLLM RequestOutput."""
    o0 = output.outputs[0]
    raw: Dict[str, Any] = {"text": o0.text}
    for attr in (
        "finish_reason",
        "stop_reason",
        "cumulative_logprob",
        "token_ids",
        "logprobs",
    ):
        if not hasattr(o0, attr):
            continue
        val = getattr(o0, attr)
        if val is None:
            continue
        if hasattr(val, "model_dump"):
            try:
                raw[attr] = val.model_dump(mode="json")
                continue
            except Exception:
                pass
        try:
            json.dumps(val)
            raw[attr] = val
        except (TypeError, ValueError):
            raw[attr] = str(val)
    return raw


def _openai_completion_raw_dict(completion) -> Dict[str, Any]:
    """Serializable snapshot of chat completion (full raw generation path)."""
    out: Dict[str, Any] = {}
    try:
        if hasattr(completion, "model_dump"):
            out = completion.model_dump(mode="json")
        elif hasattr(completion, "dict"):
            out = completion.dict()  # type: ignore[assignment]
    except Exception as e:
        out = {"model_dump_error": f"{type(e).__name__}: {e}"}
    if not out and completion is not None:
        try:
            ch = completion.choices[0]
            msg = ch.message
            msg_d = (
                msg.model_dump(mode="json")
                if hasattr(msg, "model_dump")
                else {"content": getattr(msg, "content", None)}
            )
            out = {
                "choices": [
                    {
                        "finish_reason": getattr(ch, "finish_reason", None),
                        "message": msg_d,
                    }
                ],
            }
            if getattr(completion, "usage", None) is not None:
                u = completion.usage
                out["usage"] = (
                    u.model_dump(mode="json") if hasattr(u, "model_dump") else str(u)
                )
        except Exception as e:
            out = {"fallback_error": f"{type(e).__name__}: {e}"}
    return _json_safe_value(out)


def _normalize_finish_reason(fr: Any) -> Optional[str]:
    """Map provider finish_reason (str / enum / int) to a lowercase token."""
    if fr is None:
        return None
    if isinstance(fr, str):
        s = fr.strip().lower()
        return s if s else None
    name = getattr(fr, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip().lower()
    s = str(fr).strip().lower()
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s if s else None


def _retry_sleep(base: float, attempt_idx: int) -> None:
    """Exponential backoff capped (~120s)."""
    sleep_s = min(base * (2 ** min(attempt_idx, 10)), 120.0)
    time.sleep(sleep_s)


def _openai_infer_acceptable(completion: Any, response_text: str) -> bool:
    """Require Chat Completions finish_reason == stop and non-empty merged assistant text."""
    if completion is None:
        return False
    try:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return False
        ch = choices[0]
        fr = _normalize_finish_reason(getattr(ch, "finish_reason", None))
        if fr != "stop":
            return False
    except (IndexError, AttributeError, TypeError):
        return False
    if response_text is None or not str(response_text).strip():
        return False
    return True


def _vllm_output_acceptable(output: Any) -> bool:
    """Require first sequence finish_reason == stop and non-empty decoded text."""
    try:
        o0 = output.outputs[0]
    except (IndexError, AttributeError, TypeError):
        return False
    text = getattr(o0, "text", None)
    if text is None or not str(text).strip():
        return False
    fr = _normalize_finish_reason(getattr(o0, "finish_reason", None))
    return fr == "stop"


def run_inference_openai(args):
    """Run inference via OpenAI-compatible HTTP API (e.g. SGLang)."""
    from openai import OpenAI

    print("\n" + "=" * 80)
    print("MMMU inference (OpenAI-compatible server, e.g. SGLang)")
    print("=" * 80 + "\n")

    base_url = _normalize_openai_base_url(args.api_base)
    client = OpenAI(
        base_url=base_url,
        api_key=(args.api_key or "EMPTY"),
        timeout=args.request_timeout,
    )

    data = load_dataset(args.dataset)
    print(f"Loaded {len(data)} samples from {args.dataset}")
    max_samples = getattr(args, "max_samples", None)
    if max_samples is not None and max_samples > 0 and max_samples < len(data):
        seed = getattr(args, "sample_seed", 0)
        data = _subsample_dataframe_for_debug(data, max_samples, seed)
        print(f"Debug subsample: using {len(data)} rows (max_samples={max_samples}, sample_seed={seed})")

    img_root = os.path.join(os.environ["LMUData"], "images", "MMMU")
    os.makedirs(img_root, exist_ok=True)

    def dump_image_func(line):
        return dump_image(line, img_root)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    cot_prompt = ""
    if args.use_cot:
        cot_prompt = (
            args.cot_prompt
            if args.cot_prompt
            else " If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."
        )
        print(f"Using CoT prompt: {cot_prompt[:50]}...")

    print(f"\nServer: {base_url}")
    print(f"Model (served name): {args.served_model_name}")
    print(
        f"Generation: max_tokens={args.max_new_tokens}, temperature={args.temperature}, "
        f"top_p={args.top_p}, top_k={args.top_k}"
    )
    if getattr(args, "no_thinking", False):
        print("Qwen3 thinking: disabled (--no-thinking → chat_template_kwargs.enable_thinking=false)")
    print(f"Concurrency: {args.num_workers} workers\n")

    samples: List[Tuple[Dict, List, List]] = []
    for _, line in tqdm(data.iterrows(), total=len(data), desc="Building prompts"):
        line_dict = line.to_dict()
        for k, v in list(line_dict.items()):
            if isinstance(v, np.integer):
                line_dict[k] = int(v)
            elif isinstance(v, np.floating):
                line_dict[k] = float(v)

        messages = build_mmmu_prompt(
            line, dump_image_func, args.dataset, getattr(args, "prompt_style", "default")
        )
        if args.use_cot and messages and messages[0]["content"]:
            last = messages[0]["content"][-1]
            if last["type"] == "text":
                last["text"] += cot_prompt

        openai_messages = mmmu_messages_to_openai(messages)
        samples.append((line_dict, messages, openai_messages))

    extra_body: Dict[str, Any] = {}
    if args.repetition_penalty is not None and args.repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = args.repetition_penalty
    if getattr(args, "no_thinking", False):
        ctk = dict(extra_body.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = ctk

    def chat_kwargs():
        kw = {
            "model": args.served_model_name,
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.presence_penalty is not None:
            kw["presence_penalty"] = args.presence_penalty
        if args.top_k > 0:
            extra_body_local = dict(extra_body)
            extra_body_local["top_k"] = args.top_k
            kw["extra_body"] = extra_body_local
        elif extra_body:
            kw["extra_body"] = dict(extra_body)
        return kw

    max_inf_retries = getattr(args, "infer_max_retries", 64)
    retry_base_sleep = getattr(args, "infer_retry_base_sleep", 1.0)

    def infer_one(
        pack: Tuple[Dict, List, List],
    ) -> Tuple[Dict, List, str, Optional[Dict[str, Any]]]:
        line_dict, messages, openai_messages = pack
        raw_generation: Optional[Dict[str, Any]] = None
        last_response = ""
        last_exc: Optional[Exception] = None
        for attempt in range(max_inf_retries):
            try:
                completion = client.chat.completions.create(
                    messages=openai_messages,
                    **chat_kwargs(),
                )
                raw_generation = _openai_completion_raw_dict(completion)
                msg = completion.choices[0].message
                msg_dict = (
                    msg.model_dump()
                    if hasattr(msg, "model_dump")
                    else {
                        k: getattr(msg, k, None)
                        for k in (
                            "content",
                            "reasoning_content",
                            "reasoning",
                            "thinking",
                            "thought",
                        )
                    }
                )
                response = _openai_assistant_text(msg_dict)
                if not response and getattr(msg, "content", None):
                    response = str(msg.content)
                last_response = response
                if _openai_infer_acceptable(completion, response):
                    if isinstance(raw_generation, dict):
                        raw_generation["infer_attempt"] = attempt + 1
                    return line_dict, messages, response, raw_generation
            except Exception as e:
                last_exc = e
                last_response = f"[ERROR] {type(e).__name__}: {e}"
                if args.debug_errors:
                    last_response += "\n" + traceback.format_exc()
                raw_generation = {
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc() if args.debug_errors else None,
                }
            _retry_sleep(retry_base_sleep, attempt)
        if isinstance(raw_generation, dict):
            raw_generation["infer_retry_exhausted"] = True
            raw_generation["infer_max_retries"] = max_inf_retries
            if last_exc is not None:
                raw_generation["last_exception"] = f"{type(last_exc).__name__}: {last_exc}"
        elif raw_generation is None:
            raw_generation = {
                "infer_retry_exhausted": True,
                "infer_max_retries": max_inf_retries,
            }
        return line_dict, messages, last_response, raw_generation

    print("Running inference...")
    start_time = time.time()
    results_rows: List[Any] = [None] * len(samples)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(infer_one, samples[i]): i for i in range(len(samples))
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
            i = futures[fut]
            results_rows[i] = fut.result()

    elapsed = time.time() - start_time
    print(f"\nInference done in {elapsed:.2f}s ({len(data) / elapsed:.2f} samples/s)\n")

    results = []
    for row in results_rows:
        line_dict, messages, response, raw_generation = row
        response_final = str(response).split("</redacted_thinking>")[-1].strip()
        index = line_dict["index"]
        results.append(
            {
                "question_id": int(index) if isinstance(index, np.integer) else index,
                "annotation": line_dict,
                "task": args.dataset,
                "result": {
                    "gen": response_final,
                    "gen_raw": response,
                    "raw_generation": raw_generation,
                },
                "messages": messages,
            }
        )

    with open(args.output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print(f"Results saved to {args.output_file} ({len(results)} samples)")


def run_inference_vllm(args):
    """Run inference on the MMMU dataset using vLLM."""
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    print("\n" + "=" * 80)
    print("MMMU Inference with vLLM (High-Speed Mode)")
    print("=" * 80 + "\n")

    data = load_dataset(args.dataset)
    print(f"Loaded {len(data)} samples from {args.dataset}")
    max_samples = getattr(args, "max_samples", None)
    if max_samples is not None and max_samples > 0 and max_samples < len(data):
        seed = getattr(args, "sample_seed", 0)
        data = _subsample_dataframe_for_debug(data, max_samples, seed)
        print(f"Debug subsample: using {len(data)} rows (max_samples={max_samples}, sample_seed={seed})")

    img_root = os.path.join(os.environ["LMUData"], "images", "MMMU")
    os.makedirs(img_root, exist_ok=True)

    def dump_image_func(line):
        return dump_image(line, img_root)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    cot_prompt = ""
    if args.use_cot:
        cot_prompt = (
            args.cot_prompt
            if args.cot_prompt
            else " If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."
        )
        print(f"Using CoT prompt: {cot_prompt[:50]}...")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        stop_token_ids=[],
    )

    print(f"\nGeneration (vLLM): max_tokens={sampling_params.max_tokens}, "
          f"temperature={sampling_params.temperature}, top_p={sampling_params.top_p}, "
          f"top_k={sampling_params.top_k}")

    print(f"Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("Processor loaded\n")

    print(f"Initializing vLLM: {args.model_path}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   Tensor parallel size: {args.tensor_parallel_size}")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.max_images_per_prompt},
        seed=123,
    )
    print("vLLM initialized\n")

    print("Preparing inputs for vLLM...")
    all_inputs = []
    all_line_dicts = []
    all_messages = []

    for _, line in tqdm(data.iterrows(), total=len(data), desc="Building prompts"):
        line_dict = line.to_dict()
        for k, v in line_dict.items():
            if isinstance(v, np.integer):
                line_dict[k] = int(v)
            elif isinstance(v, np.floating):
                line_dict[k] = float(v)

        messages = build_mmmu_prompt(
            line, dump_image_func, args.dataset, getattr(args, "prompt_style", "default")
        )

        if args.use_cot and messages and messages[0]["content"]:
            last_content = messages[0]["content"][-1]
            if last_content["type"] == "text":
                last_content["text"] += cot_prompt

        vllm_input = prepare_inputs_for_vllm(messages, processor)

        all_inputs.append(vllm_input)
        all_line_dicts.append(line_dict)
        all_messages.append(messages)

    print(f"Prepared {len(all_inputs)} inputs\n")

    max_inf_retries = getattr(args, "infer_max_retries", 64)
    retry_base_sleep = getattr(args, "infer_retry_base_sleep", 1.0)

    print("Running vLLM batch inference...")
    start_time = time.time()

    n = len(all_inputs)
    outputs: List[Any] = [None] * n
    last_attempt: Dict[int, Any] = {}
    pending = set(range(n))
    wave = 0
    while pending and wave < max_inf_retries:
        idx_list = sorted(pending)
        batch_in = [all_inputs[i] for i in idx_list]
        batch_out = llm.generate(batch_in, sampling_params=sampling_params)
        for local_j, global_i in enumerate(idx_list):
            out = batch_out[local_j]
            last_attempt[global_i] = out
            if _vllm_output_acceptable(out):
                outputs[global_i] = out
                pending.discard(global_i)
        if pending:
            _retry_sleep(retry_base_sleep, wave)
        wave += 1

    if pending:
        print(
            f"Warning: {len(pending)} sample(s) did not get finish_reason=stop after "
            f"{max_inf_retries} batch attempt(s); using last generation."
        )
        for i in pending:
            if i in last_attempt:
                outputs[i] = last_attempt[i]

    for i in range(n):
        if outputs[i] is None:
            outputs[i] = last_attempt.get(i)
    missing_out = [i for i in range(n) if outputs[i] is None]
    if missing_out:
        raise RuntimeError(
            f"vLLM inference produced no output for indices {missing_out} (check server logs)."
        )

    total_time = time.time() - start_time
    print(f"\nInference completed in {total_time:.2f} seconds")
    if len(data) > 0:
        print(f"  Average: {total_time / len(data):.2f} seconds/sample")
        print(f"  Throughput: {len(data) / total_time:.2f} samples/second\n")

    print("Saving results...")
    results = []

    for line_dict, messages, output in zip(all_line_dicts, all_messages, outputs):
        response = output.outputs[0].text
        index = line_dict["index"]

        response_final = str(response).split("</redacted_thinking>")[-1].strip()
        raw_generation = _json_safe_value(_vllm_output_raw_dict(output))
        try:
            o0 = output.outputs[0]
            fr = _normalize_finish_reason(getattr(o0, "finish_reason", None))
            if fr != "stop":
                if isinstance(raw_generation, dict):
                    raw_generation["infer_retry_exhausted"] = True
                    raw_generation["infer_max_retries"] = max_inf_retries
        except (IndexError, AttributeError, TypeError):
            pass

        result = {
            "question_id": int(index) if isinstance(index, np.integer) else index,
            "annotation": line_dict,
            "task": args.dataset,
            "result": {
                "gen": response_final,
                "gen_raw": response,
                "raw_generation": raw_generation,
            },
            "messages": messages,
        }
        results.append(result)

    with open(args.output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print(f"\nResults saved to {args.output_file}")
    print(f"Total samples processed: {len(results)}")


def run_inference(args):
    if getattr(args, "api_base", None):
        run_inference_openai(args)
    else:
        run_inference_vllm(args)


def run_evaluation(args):
    """Run evaluation on inference results."""
    # Load results
    results = []
    with open(args.input_file, "r") as f:
        for line in f:
            job = json.loads(line)
            annotation = job["annotation"]
            annotation["prediction"] = job["result"]["gen"]
            results.append(annotation)

    data = pd.DataFrame.from_records(results)
    data = data.sort_values(by="index")
    data["prediction"] = [str(x) for x in data["prediction"]]
    # If not choice label, then use lower case
    for k in list(data.keys()):
        data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

    # Load dataset
    meta = load_dataset(args.dataset)

    # Validation
    print(f"len(data): {len(data)}")
    print(f"len(meta): {len(meta)}")
    meta_q_map = {x: y for x, y in zip(meta["index"], meta["question"])}
    data_map = {x: y for x, y in zip(data["index"], data["question"])}
    for k in data_map:
        assert k in meta_q_map, (
            "eval_file should be the same as or a subset of dataset MMMU_DEV_VAL"
        )

    answer_map = {i: c for i, c in zip(meta["index"], meta["answer"])}
    data = MMMU_preproc(data)
    answer_map = {
        k: (v if v in list(string.ascii_uppercase) else "A") for k, v in answer_map.items()
    }
    data = data[data["index"].isin(answer_map)]
    data["GT"] = [answer_map[idx] for idx in data["index"]]
    items = []
    for i in range(len(data)):
        item = data.iloc[i]
        items.append(item)

    # Build judge model
    model = build_judge(
        model=getattr(args, "eval_model", "gpt-3.5-turbo-0125"),
        api_type=getattr(args, "api_type", "dash"),
    )

    # Prepare evaluation tasks
    eval_tasks = []
    for item in items:
        eval_tasks.append((model, item))

    # Run evaluation
    eval_results = []

    # Debug mode: process single-threaded with first few samples
    debug = os.environ.get("DEBUG", "").lower() == "true"
    if debug:
        print("Running in debug mode with first 5 samples...")
        for task in eval_tasks[:5]:
            try:
                result = eval_single_sample(task)
                eval_results.append(result)
            except Exception as e:
                print(f"Error processing task: {e}")
                print(f"Task details: {task}")
                raise
    else:
        # Normal mode: process all samples with threading
        nproc = getattr(args, "nproc", 4)
        with ThreadPoolExecutor(max_workers=nproc) as executor:
            for result in tqdm(
                executor.map(eval_single_sample, eval_tasks),
                total=len(eval_tasks),
                desc="Evaluating",
            ):
                eval_results.append(result)

    # Calculate overall accuracy
    accuracy = sum(r["hit"] for r in eval_results) / len(eval_results)

    # Calculate accuracy by split
    results_by_split = {}
    for result in eval_results:
        split = result.get("split", "unknown")
        if split not in results_by_split:
            results_by_split[split] = []
        results_by_split[split].append(result)

    accuracy_by_split = {}
    for split, split_results in results_by_split.items():
        split_accuracy = sum(r["hit"] for r in split_results) / len(split_results)
        accuracy_by_split[split] = split_accuracy
        print(
            f"Accuracy for {split} split: {split_accuracy:.4f} "
            f"({sum(r['hit'] for r in split_results)}/{len(split_results)})"
        )

    # Save results
    output_df = pd.DataFrame(eval_results)
    output_df.to_csv(args.output_file, index=False)

    # Save accuracy
    with open(args.output_file.replace(".csv", "_acc.json"), "w") as f:
        json.dump(
            {
                "overall_accuracy": accuracy,
                "accuracy_by_split": accuracy_by_split,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 50}")
    print("Evaluation Results:")
    print(f"{'=' * 50}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="MMMU evaluation: vLLM (local) or OpenAI-compatible API (e.g. SGLang)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    infer_parser = subparsers.add_parser(
        "infer", help="Run inference (vLLM locally or remote OpenAI-compatible server)"
    )
    infer_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local model dir (HuggingFace). Required for vLLM; not used for --api-base.",
    )
    infer_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    infer_parser.add_argument(
        "--prompt-style",
        type=str,
        default="default",
        choices=["default", "qwen35"],
        help='Prompt layout: "qwen35" matches Qwen3.5 OpenAI demo (Choices (A)/(B)/… + JSON answer hint)',
    )
    infer_parser.add_argument(
        "--data-dir", type=str, help="The absolute path of MMMU_DEV_VAL.tsv (sets LMUData)"
    )
    infer_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    infer_parser.add_argument("--use-cot", action="store_true", help="Use Chain-of-Thought prompting")
    infer_parser.add_argument(
        "--cot-prompt", type=str, default="", help="Custom Chain-of-Thought prompt"
    )

    # OpenAI-compatible server (SGLang, vLLM OpenAI server, etc.)
    infer_parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI API base URL, e.g. http://127.0.0.1:30000 (trailing /v1 added if missing)",
    )
    infer_parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key (many local servers accept any non-empty string)",
    )
    infer_parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Model name as registered on the server (required with --api-base)",
    )
    infer_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Concurrent HTTP requests when using --api-base (default: 4)",
    )
    infer_parser.add_argument(
        "--request-timeout",
        type=float,
        default=1800.0,
        help="Per-request HTTP timeout (seconds) for --api-base (default: 1800). "
        "Thinking models with large max-new-tokens often need 3600+; infer_think.sh defaults to 3600.",
    )
    infer_parser.add_argument(
        "--debug-errors",
        action="store_true",
        help="Append full tracebacks to failed API responses in output",
    )
    infer_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, randomly subsample the dataset to at most this many rows (debug). "
        "Use --sample-seed for reproducibility.",
    )
    infer_parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed for --max-samples subsampling (default: 0)",
    )
    infer_parser.add_argument(
        "--infer-max-retries",
        type=int,
        default=64,
        help="Retry OpenAI/vLLM generation until finish_reason is stop (or non-empty text for "
        "OpenAI) or this many attempts (default: 64)",
    )
    infer_parser.add_argument(
        "--infer-retry-base-sleep",
        type=float,
        default=1.0,
        help="Base seconds for exponential backoff between retries (default: 1.0)",
    )

    # vLLM specific parameters
    infer_parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size (default: number of GPUs; vLLM only)",
    )
    infer_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization 0.0-1.0 (vLLM only)",
    )
    infer_parser.add_argument(
        "--max-model-len",
        type=int,
        default=128000,
        help="Maximum model context length (vLLM only)",
    )
    infer_parser.add_argument(
        "--max-images-per-prompt",
        type=int,
        default=10,
        help="Maximum images per prompt (vLLM only)",
    )

    # Generation parameters
    infer_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32768,
        help="Maximum tokens to generate",
    )
    infer_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling",
    )
    infer_parser.add_argument("--top-p", type=float, default=0.8, help="Top-p sampling")
    infer_parser.add_argument("--top-k", type=int, default=20, help="Top-k (vLLM; sent in extra_body for API)")
    infer_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (vLLM; for API passed as extra_body when != 1.0)",
    )
    infer_parser.add_argument(
        "--presence-penalty",
        type=float,
        default=1.5,
        help="Presence penalty (OpenAI / vLLM where supported)",
    )
    infer_parser.add_argument(
        "--no-thinking",
        action="store_true",
        default=False,
        help="Disable Qwen3 extended thinking (vLLM/SGLang: "
        "extra_body.chat_template_kwargs.enable_thinking=false).",
    )

    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "--data-dir", type=str, help="The absolute path of MMMU_DEV_VAL.tsv"
    )
    eval_parser.add_argument(
        "--input-file", type=str, required=True, help="Input file with inference results"
    )
    eval_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    eval_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    eval_parser.add_argument(
        "--eval-model",
        type=str,
        default="gpt-3.5-turbo-0125",
        help="Model to use for evaluation",
    )
    eval_parser.add_argument(
        "--api-type",
        type=str,
        default="dash",
        choices=["dash", "mit"],
        help="API type for evaluation",
    )
    eval_parser.add_argument("--nproc", type=int, default=4, help="Number of threads for eval")

    args = parser.parse_args()

    if hasattr(args, "data_dir") and args.data_dir:
        os.environ["LMUData"] = args.data_dir

    if args.command == "infer":
        if args.api_base:
            if not args.served_model_name:
                infer_parser.error("--served-model-name is required when using --api-base")
        else:
            if not args.model_path:
                infer_parser.error("--model-path is required for vLLM unless --api-base is set")

        if args.tensor_parallel_size is None and not args.api_base:
            import torch

            args.tensor_parallel_size = torch.cuda.device_count()
            print(f"Auto-set tensor_parallel_size to {args.tensor_parallel_size}")

        run_inference(args)
    elif args.command == "eval":
        run_evaluation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
