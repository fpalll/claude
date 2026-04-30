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
from typing import List, Dict, Any, Tuple, Union
import warnings
import string
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports from refactored files
from dataset_utils import (
    load_videomme_dataset,
    build_videomme_prompt,
    materialize_zip_videos_in_messages,
    read_videomme_zip_video_bytes,
)
from eval_utils import build_judge, eval_single_sample, _openai_assistant_text

# vLLM / torch imported lazily in run_inference_vllm


def _videomme_zip_kwargs(args: Any) -> Dict[str, Any]:
    """Pass-through for build_videomme_prompt zip archive options."""
    vz = getattr(args, "videos_zip", None)
    sz = getattr(args, "subtitles_zip", None)
    return {
        "videos_zip": vz if vz else None,
        "subtitles_zip": sz if sz else None,
    }


def _normalize_openai_base_url(url: str) -> str:
    u = url.rstrip("/")
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def _encode_video_for_openai(src: Union[str, Dict[str, str]]) -> Dict[str, Any]:
    """Encode a local .mp4 path or zip-backed video as a data URL for OpenAI-compatible APIs."""
    if isinstance(src, dict) and "_videomme_zip" in src:
        raw = read_videomme_zip_video_bytes(src)
        mime = "video/mp4"
        b64 = base64.standard_b64encode(raw).decode("ascii")
        return {
            "type": "video_url",
            "video_url": {"url": f"data:{mime};base64,{b64}"},
        }
    path = str(src)
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "video/mp4"
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:{mime};base64,{b64}"},
    }


def videomme_messages_to_openai(messages: List[Dict]) -> List[Dict[str, Any]]:
    """
    Convert VideoMME / Qwen-style messages (video path + text blocks) to
    OpenAI-compatible chat.completions content (e.g. SGLang).
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            if isinstance(content, str):
                out.append({"role": "system", "content": content})
            else:
                out.append({"role": "system", "content": str(content)})
            continue

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        parts: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError(f"Unsupported content block: {type(block)}")
            if "text" in block and "video" not in block:
                parts.append({"type": "text", "text": block["text"]})
            elif "video" in block:
                parts.append(_encode_video_for_openai(block["video"]))
            elif block.get("type") == "text":
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image_url":
                parts.append(block)
            else:
                raise ValueError(f"Unsupported multimodal block keys: {list(block.keys())}")
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


def run_inference_openai(args):
    """Run inference via OpenAI-compatible HTTP API (e.g. SGLang)."""
    from openai import OpenAI

    print("\n" + "=" * 80)
    print("VideoMME inference (OpenAI-compatible server, e.g. SGLang)")
    print("=" * 80 + "\n")

    base_url = _normalize_openai_base_url(args.api_base)
    client = OpenAI(
        base_url=base_url,
        api_key=(args.api_key or "EMPTY"),
        timeout=args.request_timeout,
    )

    data = load_videomme_dataset(args.data_dir, duration=args.duration)
    print(f"Loaded {len(data)} samples (duration={args.duration})")

    if args.max_samples is not None and args.max_samples > 0:
        data = data[: args.max_samples]
        print(f"Testing mode: processing only first {len(data)} samples")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    sys_prompt = None
    if args.sys_prompt and os.path.exists(args.sys_prompt):
        with open(args.sys_prompt, "r") as f:
            sys_prompt = f.read().strip()
        print(f"Loaded system prompt from {args.sys_prompt}")

    print(f"\nServer: {base_url}")
    print(f"Model (served name): {args.served_model_name}")
    print(
        f"Generation: max_tokens={args.max_new_tokens}, temperature={args.temperature}, "
        f"top_p={args.top_p}, top_k={args.top_k}"
    )
    print(
        f"Video: fps={args.fps}; extra_body mm_processor_kwargs="
        f"{{fps: {args.fps}, do_sample_frames: {not args.no_video_do_sample_frames}}} "
        f"(disabled={args.no_extra_mm_processor_kwargs})"
    )
    print(f"Concurrency: {args.num_workers} workers")
    print(
        "Note: each request embeds the full video as base64; large files need enough RAM and HTTP limits.\n"
    )
    if getattr(args, "videos_zip", None):
        print(f"Videos from zip(s): {args.videos_zip} (no unpacked videos/ tree required)")
    if getattr(args, "subtitles_zip", None):
        print(f"Subtitles from zip(s): {args.subtitles_zip}")

    samples: List[Tuple[Dict, List, List]] = []
    for data_item in tqdm(data, desc="Building prompts"):
        messages, annotation = build_videomme_prompt(
            data_item,
            args.data_dir,
            use_subtitle=args.use_subtitle,
            fps=args.fps,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            total_pixels=args.total_pixels,
            sys_prompt=sys_prompt,
            prompt_style=getattr(args, "prompt_style", "default"),
            **_videomme_zip_kwargs(args),
        )
        openai_messages = videomme_messages_to_openai(messages)
        samples.append((annotation, messages, openai_messages))

    extra_body: Dict[str, Any] = {}
    if args.repetition_penalty is not None and args.repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = args.repetition_penalty

    def chat_kwargs():
        kw = {
            "model": args.served_model_name,
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.presence_penalty is not None:
            kw["presence_penalty"] = args.presence_penalty
        eb = dict(extra_body)
        if args.top_k > 0:
            eb["top_k"] = args.top_k
        if not getattr(args, "no_extra_mm_processor_kwargs", False):
            eb["mm_processor_kwargs"] = {
                "fps": args.fps,
                "do_sample_frames": not getattr(args, "no_video_do_sample_frames", False),
            }
        if eb:
            kw["extra_body"] = eb
        return kw

    def infer_one(
        pack: Tuple[Dict, List, List],
    ) -> Tuple[Dict, List, str]:
        annotation, messages, openai_messages = pack
        try:
            completion = client.chat.completions.create(
                messages=openai_messages,
                **chat_kwargs(),
            )
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
        except Exception as e:
            response = f"[ERROR] {type(e).__name__}: {e}"
            if args.debug_errors:
                response += "\n" + traceback.format_exc()
        return annotation, messages, response

    print("Running inference...")
    start_time = time.time()
    results_rows: List[Tuple[Dict, List, str]] = [None] * len(samples)  # type: ignore

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(infer_one, samples[i]): i for i in range(len(samples))
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
            i = futures[fut]
            results_rows[i] = fut.result()

    elapsed = time.time() - start_time
    n = len(data)
    print(f"\nInference done in {elapsed:.2f}s ({n / elapsed:.2f} samples/s)\n")

    results = []
    for annotation, messages, response in results_rows:
        response_final = str(response).split("</redacted_thinking>")[-1].strip()
        result = {
            "question_id": annotation["question_id"],
            "annotation": annotation,
            "task": f"VideoMME_{args.duration}_{'w_subtitle' if args.use_subtitle else 'wo_subtitle'}",
            "result": {"gen": response_final, "gen_raw": response},
            "messages": messages,
        }
        results.append(result)

    with open(args.output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print(f"Results saved to {args.output_file} ({len(results)} samples)")


def run_inference_vllm(args):
    """Run inference on the VideoMME dataset using vLLM."""
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    print("\n" + "=" * 80)
    print("VideoMME Inference with vLLM (High-Speed Mode)")
    print("=" * 80 + "\n")

    data = load_videomme_dataset(args.data_dir, duration=args.duration)
    print(f"Loaded {len(data)} samples from VideoMME (duration={args.duration})")

    if args.max_samples is not None and args.max_samples > 0:
        data = data[: args.max_samples]
        print(f"Testing mode: Processing only first {len(data)} samples")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    sys_prompt = None
    if args.sys_prompt and os.path.exists(args.sys_prompt):
        with open(args.sys_prompt, "r") as f:
            sys_prompt = f.read().strip()
        print(f"Loaded system prompt from {args.sys_prompt}")

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
    print(f"Video: fps={args.fps}, min_pixels={args.min_pixels}, max_pixels={args.max_pixels}, "
          f"min_frames={args.min_frames}, max_frames={args.max_frames}, total_pixels={args.total_pixels}")
    print(f"use_subtitle={args.use_subtitle}")
    if getattr(args, "videos_zip", None):
        print(f"Videos from zip(s): {args.videos_zip} (temp .mp4 per sample for vLLM)")
    if getattr(args, "subtitles_zip", None):
        print(f"Subtitles from zip(s): {args.subtitles_zip}")
    print()

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
        limit_mm_per_prompt={"video": args.max_videos_per_prompt},
        seed=args.seed,
    )
    print("vLLM initialized\n")

    print("Preparing inputs for vLLM...")
    all_inputs = []
    all_annotations = []
    all_messages = []
    zip_temp_files: List[str] = []

    for data_item in tqdm(data, desc="Building prompts"):
        messages, annotation = build_videomme_prompt(
            data_item,
            args.data_dir,
            use_subtitle=args.use_subtitle,
            fps=args.fps,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            total_pixels=args.total_pixels,
            sys_prompt=sys_prompt,
            prompt_style=getattr(args, "prompt_style", "default"),
            **_videomme_zip_kwargs(args),
        )

        messages_fs, tmp_paths = materialize_zip_videos_in_messages(messages)
        zip_temp_files.extend(tmp_paths)
        vllm_input = prepare_inputs_for_vllm(messages_fs, processor)

        all_inputs.append(vllm_input)
        all_annotations.append(annotation)
        all_messages.append(messages)

    print(f"Prepared {len(all_inputs)} inputs\n")

    print("Running vLLM batch inference...")
    start_time = time.time()

    try:
        outputs = llm.generate(all_inputs, sampling_params=sampling_params)
    finally:
        for p in zip_temp_files:
            try:
                os.unlink(p)
            except OSError:
                pass

    total_time = time.time() - start_time
    print(f"\nInference completed in {total_time:.2f} seconds")
    print(f"  Average: {total_time / len(data):.2f} seconds/sample")
    print(f"  Throughput: {len(data) / total_time:.2f} samples/second\n")

    print("Saving results...")
    results = []

    for annotation, messages, output in zip(all_annotations, all_messages, outputs):
        response = output.outputs[0].text

        response_final = str(response).split("</redacted_thinking>")[-1].strip()

        result = {
            "question_id": annotation["question_id"],
            "annotation": annotation,
            "task": f"VideoMME_{args.duration}_{'w_subtitle' if args.use_subtitle else 'wo_subtitle'}",
            "result": {"gen": response_final, "gen_raw": response},
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
            annotation["index"] = job["question_id"]
            annotation["category"] = annotation["domain"]
            results.append(annotation)

    data = pd.DataFrame.from_records(results)
    data = data.sort_values(by="index")
    data["prediction"] = [str(x) for x in data["prediction"]]

    # Build choices columns (A, B, C, D) from annotation
    for idx, row in data.iterrows():
        choices = row["choices"]
        for k, v in choices.items():
            data.at[idx, k] = v

    # Build judge model
    model = build_judge(
        model=getattr(args, "eval_model", "gpt-3.5-turbo-0125"),
        api_type=getattr(args, "api_type", "dash"),
    )

    # Prepare evaluation tasks
    eval_tasks = []
    for idx, item in data.iterrows():
        eval_tasks.append((model, item))

    # Run evaluation
    eval_results = []

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

    # Calculate accuracy by category
    results_by_category = {}
    for result in eval_results:
        category = result.get("domain", "unknown")
        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append(result)

    accuracy_by_category = {}
    for category, cat_results in results_by_category.items():
        cat_accuracy = sum(r["hit"] for r in cat_results) / len(cat_results)
        accuracy_by_category[category] = cat_accuracy
        print(
            f"Accuracy for {category}: {cat_accuracy:.4f} "
            f"({sum(r['hit'] for r in cat_results)}/{len(cat_results)})"
        )

    # Calculate accuracy by sub_category
    results_by_subcategory = {}
    for result in eval_results:
        sub_category = result.get("sub_category", "unknown")
        if sub_category not in results_by_subcategory:
            results_by_subcategory[sub_category] = []
        results_by_subcategory[sub_category].append(result)

    accuracy_by_subcategory = {}
    for sub_category, subcat_results in results_by_subcategory.items():
        subcat_accuracy = sum(r["hit"] for r in subcat_results) / len(subcat_results)
        accuracy_by_subcategory[sub_category] = subcat_accuracy

    # Save results
    output_df = pd.DataFrame(eval_results)
    output_df.to_csv(args.output_file, index=False)

    # Save accuracy
    with open(args.output_file.replace(".csv", "_acc.json"), "w") as f:
        json.dump(
            {
                "overall_accuracy": accuracy,
                "accuracy_by_category": accuracy_by_category,
                "accuracy_by_subcategory": accuracy_by_subcategory,
            },
            f,
            indent=2,
        )

    # Also save as TSV format (consistent with original implementation)
    tsv_file = args.output_file.replace(".csv", ".tsv")
    output_df.to_csv(tsv_file, sep="\t", index=False)

    print(f"\n{'=' * 50}")
    print("Evaluation Results:")
    print(f"{'=' * 50}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="VideoMME: vLLM (local) or OpenAI-compatible API (e.g. SGLang)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    infer_parser = subparsers.add_parser(
        "infer", help="Run inference (vLLM locally or remote OpenAI-compatible server)"
    )
    infer_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local model dir (HuggingFace). Required for vLLM; not used with --api-base.",
    )
    infer_parser.add_argument(
        "--data-dir", type=str, required=True, help="VideoMME data directory"
    )
    infer_parser.add_argument(
        "--videos-zip",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Zip containing .mp4 files named <videoID>.mp4 (any path inside the zip). "
            "Repeat for multiple archives (e.g. short/medium/long). Skips a full videos/ unpack."
        ),
    )
    infer_parser.add_argument(
        "--subtitles-zip",
        action="append",
        default=None,
        metavar="PATH",
        help="Zip containing .srt files named <videoID>.srt; use with --use-subtitle.",
    )
    infer_parser.add_argument(
        "--duration",
        type=str,
        default="short",
        choices=["short", "medium", "long"],
        help="Video duration type (short/medium/long)",
    )
    infer_parser.add_argument(
        "--use-subtitle", action="store_true", help="Use subtitles if available"
    )
    infer_parser.add_argument(
        "--prompt-style",
        type=str,
        default="default",
        choices=["default", "qwen35"],
        help='Prompt layout: "qwen35" adds Choices (A)/(B)/… + JSON answer hint (Qwen3.5 style)',
    )
    infer_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    infer_parser.add_argument(
        "--sys-prompt", type=str, default=None, help="Path to system prompt file"
    )
    infer_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (testing; default: all)",
    )

    infer_parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI API base URL, e.g. http://127.0.0.1:30000 (/v1 appended if missing)",
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
        help="Model name on the server (required with --api-base)",
    )
    infer_parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Concurrent HTTP requests for --api-base (default: 2; videos are large)",
    )
    infer_parser.add_argument(
        "--request-timeout",
        type=float,
        default=3600.0,
        help="Per-request timeout in seconds for --api-base (default: 3600)",
    )
    infer_parser.add_argument(
        "--debug-errors",
        action="store_true",
        help="Append full tracebacks to failed API responses in output",
    )
    infer_parser.add_argument(
        "--no-extra-mm-processor-kwargs",
        action="store_true",
        help="Do not send mm_processor_kwargs inside extra_body (if the server rejects it)",
    )
    infer_parser.add_argument(
        "--no-video-do-sample-frames",
        action="store_true",
        help="Set do_sample_frames=false in mm_processor_kwargs (default: true)",
    )

    # Video processing parameters (used when building prompts; vLLM uses them via qwen_vl_utils)
    infer_parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
    infer_parser.add_argument(
        "--min-pixels",
        type=int,
        default=128 * 28 * 28,
        help="Minimum pixels per frame (vLLM / subtitle alignment)",
    )
    infer_parser.add_argument(
        "--max-pixels",
        type=int,
        default=512 * 28 * 28,
        help="Maximum pixels per frame (vLLM / subtitle alignment)",
    )
    infer_parser.add_argument(
        "--min-frames", type=int, default=4, help="Minimum number of frames"
    )
    infer_parser.add_argument(
        "--max-frames", type=int, default=512, help="Maximum number of frames"
    )
    infer_parser.add_argument(
        "--total-pixels",
        type=int,
        default=24576 * 28 * 28,
        help="Total pixels across all frames (vLLM)",
    )

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
        help="GPU memory utilization (vLLM only)",
    )
    infer_parser.add_argument(
        "--max-model-len",
        type=int,
        default=128000,
        help="Maximum model context length (vLLM only)",
    )
    infer_parser.add_argument(
        "--max-videos-per-prompt",
        type=int,
        default=1,
        help="Maximum videos per prompt (vLLM only)",
    )
    infer_parser.add_argument("--seed", type=int, default=3407, help="Random seed (vLLM only)")

    infer_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32768,
        help="Maximum tokens to generate",
    )
    infer_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    infer_parser.add_argument("--top-p", type=float, default=0.8, help="Top-p")
    infer_parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k (vLLM; for API sent in extra_body when > 0)",
    )
    infer_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (vLLM; API extra_body when != 1.0)",
    )
    infer_parser.add_argument(
        "--presence-penalty",
        type=float,
        default=1.5,
        help="Presence penalty (where supported)",
    )

    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "--data-dir", type=str, required=True, help="VideoMME data directory"
    )
    eval_parser.add_argument(
        "--input-file", type=str, required=True, help="Input file with inference results"
    )
    eval_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
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
