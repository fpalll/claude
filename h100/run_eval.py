"""
Unified benchmark runner — reproduces Qwen3.5-27B HuggingFace model card scores.

Target scores (from https://huggingface.co/Qwen/Qwen3.5-27B):
  MMLU-Pro                  86.1%
  MMMU                      82.3%
  VideoMME w/o subtitles    82.8%
  VideoMME w/ subtitles     87.0%

Generation params (match VLMEvalKit Qwen3-VL model.py):
  temperature=0.01, top_p=0.8, top_k=20, presence_penalty=1.5

Thinking is handled server-side by SGLang --reasoning-parser qwen3.
No special flag needed here.

Usage
-----
# Step 1: find your model name
curl http://127.0.0.1:18000/v1/models | python3 -m json.tool

# Step 2: quick debug run (50 samples each benchmark, a few minutes)
python run_eval.py --model Qwen/Qwen3.5-27B --api-base http://127.0.0.1:18000/v1 --sample 50

# Step 3: full run
python run_eval.py --model Qwen/Qwen3.5-27B --api-base http://127.0.0.1:18000/v1
"""
import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Qwen3.5-27B HF scores on MMLU-Pro / MMMU / Video-MME",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True,
        help="Exact model name from SGLang (curl http://localhost:50038/v1/models)",
    )
    parser.add_argument(
        "--benchmark", nargs="+",
        choices=["mmlu_pro", "mmmu", "videomme", "bfcl", "all"],
        default=["all"],
        help="Benchmark(s) to run (default: all)",
    )
    parser.add_argument(
        "--bfcl-dataset", default=None,
        help="Path to BFCL-V4 web search JSON file (auto-downloaded if not provided)",
    )
    parser.add_argument(
        "--bfcl-verbose", action="store_true",
        help="Print each agent turn for BFCL (useful for debugging)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Limit to N examples per benchmark for debugging (default: full dataset)",
    )
    parser.add_argument(
        "--api-base", default="http://localhost:50038/v1",
        help="SGLang OpenAI-compatible API base (default: http://localhost:50038/v1)",
    )
    parser.add_argument(
        "--workers", type=int, default=32,
        help="Parallel API workers (default: 32; Video-MME caps at 8 automatically)",
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory for JSON result files (default: results/)",
    )
    args = parser.parse_args()

    benchmarks = args.benchmark
    if "all" in benchmarks:
        benchmarks = ["mmlu_pro", "mmmu", "videomme", "bfcl"]

    # Sanity-check SGLang connectivity
    try:
        import urllib.request
        with urllib.request.urlopen(f"{args.api_base}/models", timeout=5) as r:
            model_data = json.loads(r.read())
        served = [m["id"] for m in model_data.get("data", [])]
        if args.model not in served:
            print(f"WARNING: '{args.model}' not in SGLang model list: {served}")
            print("  Continuing — verify --model if you get API errors.\n")
        else:
            print(f"SGLang OK: '{args.model}' is available.\n")
    except Exception as e:
        print(f"WARNING: cannot reach SGLang at {args.api_base}: {e}")
        print("  Continuing anyway.\n")

    kwargs = dict(
        model=args.model,
        api_base=args.api_base,
        sample=args.sample,
        output_dir=args.output,
    )

    summary: dict[str, float] = {}

    if "mmlu_pro" in benchmarks:
        from eval.mmlu_pro import run_mmlu_pro
        summary["mmlu_pro"] = run_mmlu_pro(**kwargs, workers=args.workers)

    if "mmmu" in benchmarks:
        from eval.mmmu import run_mmmu
        summary["mmmu"] = run_mmmu(**kwargs, workers=args.workers)

    if "videomme" in benchmarks:
        from eval.videomme import run_videomme
        avg = run_videomme(**kwargs, workers=min(args.workers, 8))

    if "bfcl" in benchmarks:
        from eval.bfcl_websearch import run_bfcl_websearch
        summary["bfcl_websearch"] = run_bfcl_websearch(
            model=args.model,
            api_base=args.api_base,
            dataset_path=args.bfcl_dataset,
            sample=args.sample,
            output_dir=args.output,
            workers=min(args.workers, 4),  # BFCL is sequential per question
            verbose=args.bfcl_verbose,
        )
        # Load the detailed JSON to show both variants in summary
        safe = args.model.replace("/", "_")
        vmme_path = os.path.join(args.output, f"videomme_{safe}.json")
        try:
            with open(vmme_path) as f:
                vmme = json.load(f)
            summary["videomme_no_subs"] = vmme["without_subtitles"]["accuracy"]
            summary["videomme_w_subs"] = vmme["with_subtitles"]["accuracy"]
        except Exception:
            summary["videomme_avg"] = avg

    # Print comparison table
    TARGETS = {
        "mmlu_pro":         86.1,
        "mmmu":             82.3,
        "videomme_no_subs": 82.8,
        "videomme_w_subs":  87.0,
        "bfcl_websearch":   68.5,
    }
    print("\n" + "=" * 62)
    print(f"SUMMARY — {args.model}  (thinking=server-side via --reasoning-parser qwen3)")
    print(f"{'Benchmark':<22} {'Reproduced':>12} {'HF Target':>12} {'Delta':>8}")
    print("-" * 62)
    for bench, acc in summary.items():
        target = TARGETS.get(bench)
        if target:
            delta = acc * 100 - target
            print(f"  {bench:<20} {acc*100:>10.2f}%  {target:>10.1f}%  {delta:>+7.2f}%")
        else:
            print(f"  {bench:<20} {acc*100:>10.2f}%")
    print("=" * 62)

    os.makedirs(args.output, exist_ok=True)
    safe = args.model.replace("/", "_")
    summary_path = os.path.join(args.output, f"summary_{safe}.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "sample": args.sample,
            "scores": summary,
            "targets": TARGETS,
        }, f, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
