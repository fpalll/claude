"""SGLang OpenAI-compatible API client with vision support and concurrency.

Generation parameters match VLMEvalKit's Qwen3-VL implementation (vlmeval/vlm/qwen3_vl/model.py):
  temperature=0.01, top_p=0.8, top_k=20, presence_penalty=1.5

Thinking is enabled via SGLang's --reasoning-parser qwen3 flag (set at server launch).
The model internally produces <think>...</think> content; SGLang strips it before returning.
"""
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# Matches VLMEvalKit vlmeval/vlm/qwen3_vl/model.py generation config
GENERATION_PARAMS = {
    "temperature": 0.01,
    "top_p": 0.8,
    "presence_penalty": 1.5,
    # top_k=20 passed via extra_body (non-standard OpenAI param)
}


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
    max_tokens: int = 32768,
    max_retries: int = 3,
) -> str | None:
    # top_k=20 is passed via extra_body (SGLang-specific, not in OpenAI spec)
    # Thinking is handled server-side by --reasoning-parser qwen3; no extra flag needed here
    extra_body: dict = {"top_k": 20}

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                extra_body=extra_body,
                **GENERATION_PARAMS,
            )
            content = resp.choices[0].message.content or ""
            # Safety net: strip any <think>...</think> that leaked through
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
