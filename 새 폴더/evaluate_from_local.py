import torch
import re
import json
import time
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

choices = ["A","B","C","D","E","F","G","H","I","J"]

max_model_length = 131072
max_new_tokens = 512
BATCH_SIZE = 6        # 🔥 核心（可調 4~8）
NUM_WORKERS = 2       # 🔥 parallel

random.seed(42)

# ---------------- DATA ----------------
def load_data():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    return dataset["test"], dataset["validation"]

# ---------------- MODEL ----------------
def load_model():
    model_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        max_memory={
            0: "80GB",
            1: "80GB"
        }
    )

    model.eval()
    return model, tokenizer

# ---------------- PROMPT ----------------
SUFFIX = '\nReturn JSON only: {"answer": "C"}'

def build_prompt(example):
    prompt = f"Question:\n{example['question']}\nOptions:\n"
    for i,opt in enumerate(example["options"]):
        if opt != "N/A":
            prompt += f"{choices[i]}. {opt}\n"
    prompt += "Answer: Let's think step by step.\n"
    prompt += SUFFIX
    return prompt

# ---------------- BATCH GENERATE ----------------
def batch_generate(model, tokenizer, prompts):
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True
        )
        for p in prompts
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=65536,
            do_sample=True,
            temperature=1.0,
            top_p=0.95
        )

    results = []
    for i in range(len(prompts)):
        input_len = inputs["input_ids"][i].shape[0]
        out = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
        results.append(out)

    return results

# ---------------- EXTRACT ----------------
def extract(text):
    m = re.search(r'"answer"\s*:\s*"([A-J])"', text)
    if m:
        return m.group(1)
    return None

# ---------------- MAIN ----------------
def main():
    model, tokenizer = load_model()
    test_df, val_df = load_data()

    results = []
    correct = 0
    total = len(test_df)

    pbar = tqdm(total=total, desc="TOTAL PROGRESS")

    prompts = [build_prompt(x) for x in test_df]

    for i in range(0, total, BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]

        outputs = batch_generate(model, tokenizer, batch)

        for j, out in enumerate(outputs):
            idx = i + j
            pred = extract(out)

            if pred == test_df[idx]["answer"]:
                correct += 1

            results.append({
                "question": test_df[idx]["question"],
                "pred": pred,
                "gt": test_df[idx]["answer"],
                "output": out
            })

            pbar.update(1)

    acc = correct / total
    print(f"\nFinal Accuracy: {acc:.4f}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/qwen3.5-27b")
    args = parser.parse_args()
    main()