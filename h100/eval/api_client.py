"""SGLang OpenAI-compatible API client with vision support and concurrency.

Dense vs MoE decoding config
------------------------------
Dense models (e.g. Qwen3.5-27B) activate ALL parameters every token.
They are significantly more sensitive to decoding settings and prompt format:
  - stochastic decoding (temperature > 0) causes higher variance in output format
  - even small temperature values can make the model write explanatory text
    instead of the expected single letter
  → use temperature=0 (fully greedy) for deterministic, format-stable output

MoE models (e.g. Qwen3.5-35B-A3B) activate only ~3B params per token.
  - more tolerant of varied prompts and non-zero temperature
  - temperature=0.01 is fine

Thinking is handled server-side by --reasoning-parser qwen3 at SGLang launch.
The model produces <think>...</think> internally; SGLang strips it before returning.
"""
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from openai import OpenAI
from PIL import Image
from tqdm import tqdm


@dataclass
class ModelConfig:
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    max_tokens: int


# Dense model: fully greedy — eliminates format variance caused by sampling
DENSE_CONFIG = ModelConfig(
    temperature=0,
    top_p=1.0,
    top_k=1,
    presence_penalty=0.0,
    max_tokens=8192,
)

# MoE model: matches VLMEvalKit vlmeval/vlm/qwen3_vl/model.py
MOE_CONFIG = ModelConfig(
    temperature=0.01,
    top_p=0.8,
    top_k=20,
    presence_penalty=1.5,
    max_tokens=8192,
)


def get_config(model_name: str) -> ModelConfig:
    """Return the appropriate config based on model architecture."""
    name = model_name.lower()
    # MoE models have active-param suffixes like -A3B, -A22B
    if any(tag in name for tag in ["-a3b", "-a22b", "-a14b", "moe"]):
        return MOE_CONFIG
    # Dense models get greedy config
    return DENSE_CONFIG


def get_client(api_base: str, api_key: str = "EMPTY") -> OpenAI:
    return OpenAI(base_url=api_base, api_key=api_key)


def encode_pil_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_image_content(image: Image.Image) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_image(image)}"},
    }


def call_api(
    client: OpenAI,
    model: str,
    messages: list,
    config: ModelConfig | None = None,
    max_retries: int = 3,
) -> str | None:
    if config is None:
        config = get_config(model)

    # top_k passed via extra_body (SGLang-specific, not part of OpenAI spec)
    extra_body: dict = {"top_k": config.top_k}

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=config.temperature,
                top_p=config.top_p,
                presence_penalty=config.presence_penalty,
                max_tokens=config.max_tokens,
                extra_body=extra_body,
            )
            content = resp.choices[0].message.content or ""
            # Safety net: strip <think>...</think> if reasoning_parser didn't catch it
            if "<think>" in content and "</think>" in content:
                content = content.split("</think>", 1)[-1].strip()
            return content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n  [API error] {e}")
                return None
            time.sleep(2**attempt)
    return None


def batch_run(fn, items: list, workers: int = 32, desc: str = "Evaluating") -> list:
    """Run fn(idx, item) for each item in parallel, preserving order."""
    results: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, i, item): i for i, item in enumerate(items)}
        for fut in tqdm(as_completed(futures), total=len(items), desc=desc):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"\n  [Worker error idx={idx}] {e}")
    return results
