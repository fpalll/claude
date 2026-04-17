"""
Evaluation utilities for VideoMME benchmark.
Contains functions for answer extraction and evaluation.
"""

import os
import re
import requests
import time
import random
import string
import copy
import traceback
import pandas as pd
from PIL import Image
from typing import List, Dict, Tuple, Any


def _openai_assistant_text(message: Dict[str, Any]) -> str:
    """
    Normalize assistant text from OpenAI-compatible chat.completions.
    Reasoning / thinking servers (e.g. SGLang with Qwen3) often set content=null
    and put text in reasoning_content or similar fields.
    """
    if not message:
        return ""

    def _one_field(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            chunks: List[str] = []
            for block in value:
                if isinstance(block, dict):
                    t = block.get("text")
                    if t is not None and str(t).strip():
                        chunks.append(str(t).strip())
                elif isinstance(block, str) and block.strip():
                    chunks.append(block.strip())
            return "\n".join(chunks).strip()
        return str(value).strip()

    pieces: List[str] = []
    for key in (
        "reasoning_content",
        "reasoning",
        "thinking",
        "thought",
        "content",
    ):
        if key not in message:
            continue
        part = _one_field(message.get(key))
        if part:
            pieces.append(part)
    return "\n".join(pieces).strip()


def _eval_extraction_verbose() -> bool:
    return os.environ.get("EVAL_EXTRACTION_VERBOSE", "").lower() in (
        "1",
        "true",
        "yes",
    )


def encode_image_to_base64(image, target_size=None):
    """Encode an image to base64 string."""
    import base64
    import io
    
    if target_size is not None:
        width, height = image.size
        if width > height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)
        image = image.resize((new_width, new_height))
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

class OpenAIWrapper:
    """Wrapper for OpenAI API."""
    
    def __init__(self, model, api_base, api_key, timeout=60, retry=5, wait=5):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.fail_msg = 'Failed to obtain answer via API.'
    
    def generate(self, messages):
        """Generate a response from the API."""
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        
        # Format messages for API
        formatted_messages = []
        for msg in messages:
            if msg['type'] == 'text':
                formatted_messages.append({"role": "user", "content": [{"type": "text", "text": msg['value']}]})
            elif msg['type'] == 'image':
                # Load and encode the image
                image = Image.open(msg['value'])
                image_data = encode_image_to_base64(image)
                formatted_messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                })
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": 4096,
            "temperature": 0
        }
        
        for i in range(self.retry):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    try:
                        msg = (resp_json.get("choices") or [{}])[0].get("message") or {}
                    except (TypeError, IndexError, AttributeError):
                        time.sleep(self.wait)
                        continue
                    text = _openai_assistant_text(msg)
                    if text:
                        return text
                    print("API error: empty assistant text (retrying)")
                
                time.sleep(self.wait)
            except Exception as e:
                print(f"API error: {e}")
                time.sleep(self.wait)
        
        return self.fail_msg

class DashScopeWrapper:
    """Wrapper for DashScope API."""
    
    def __init__(self, model, api_base, api_key, timeout=60, retry=5, wait=5):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.fail_msg = 'Failed to obtain answer via API.'
    
    def generate(self, messages):
        """Generate a response from the API."""
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        
        # Format messages for API
        formatted_messages = []
        for msg in messages:
            if msg['type'] == 'text':
                formatted_messages.append({"role": "user", "content": [{"type": "text", "text": msg['value']}]})
            elif msg['type'] == 'image':
                # Load and encode the image
                image = Image.open(msg['value'])
                image_data = encode_image_to_base64(image)
                formatted_messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                })
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_completion_tokens": 4096,
            "n": 1,
            "temperature": 0,
            "stream": False
        }

        for i in range(self.retry):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    
                    # Check finish reason
                    for output in resp_json['choices']:
                        if output['finish_reason'] not in ['stop', 'function_call']:
                            print(f"DashScope finished with error: {resp_json}")
                            time.sleep(self.wait)
                            continue
                    
                    return resp_json['choices'][0]['message']['content']
                else:
                    print(f"DashScope API error: HTTP {response.status_code}")
                    try:
                        error_content = response.json()
                        print(f"Error details: {error_content}")
                    except:
                        print(f"Raw error content: {response.content.decode('utf-8', errors='replace')}")
                
                time.sleep(self.wait)
            except requests.exceptions.ConnectionError as conn_err:
                print(f"DashScope: Connection error occurred: {conn_err}")
                time.sleep(self.wait)
            except requests.exceptions.Timeout as timeout_err:
                print(f"DashScope: Timeout error occurred: {timeout_err}")
                time.sleep(self.wait)
            except requests.exceptions.RequestException as req_err:
                print(f"DashScope: Request exception occurred: {req_err}")
                time.sleep(self.wait)
            except Exception as e:
                print(f"DashScope: An error occurred: {e}")
                print(traceback.format_exc())
                time.sleep(self.wait)
        
        return self.fail_msg

def build_judge(model, api_type):
    """Build a judge model for evaluation."""
    if api_type == "mit":
        api_key = os.environ.get("MIT_SPIDER_TOKEN", "")
        api_base = os.environ.get("MIT_SPIDER_URL", "").strip()
        if not api_base:
            raise ValueError(
                "MIT_SPIDER_URL is empty. The judge needs a full chat-completions URL.\n"
                "Example for a local OpenAI-compatible server (SGLang / vLLM):\n"
                "  export MIT_SPIDER_URL='http://127.0.0.1:30000/v1/chat/completions'\n"
                "  export MIT_SPIDER_TOKEN='EMPTY'   # or your key if required\n"
                "  export MIT_SPIDER_TIMEOUT=600    # optional; HTTP read timeout in seconds (default 600)\n"
                "Then run eval with: --api-type mit --eval-model <served_model_name>"
            )
        raw_to = os.environ.get("MIT_SPIDER_TIMEOUT", "600").strip()
        try:
            timeout_sec = float(raw_to) if raw_to else 600.0
        except ValueError:
            timeout_sec = 600.0
        return OpenAIWrapper(model, api_base, api_key, timeout=timeout_sec)
    elif api_type == "dash":
        api_key = os.environ.get("CHATGPT_DASHSCOPE_API_KEY", "").strip()
        api_base = os.environ.get("DASHSCOPE_API_BASE", "").strip()
        if not api_base or not api_key:
            raise ValueError(
                "DashScope judge config is incomplete. Set both:\n"
                "  export CHATGPT_DASHSCOPE_API_KEY='...'\n"
                "  export DASHSCOPE_API_BASE='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'\n"
                "(adjust URL if your region/docs differ)\n"
                "Then run eval with: --api-type dash --eval-model <compatible model id>"
            )
        return DashScopeWrapper(model, api_base, api_key)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")

def can_infer_option(answer, choices):
    """Rule-based extraction of answer option."""
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = copy.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3:
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False

def can_infer_text(answer, choices):
    """Extract answer by matching text content."""
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False

def can_infer_relaxed(answer, choices):
    """
    Extra heuristics for instruction/CoT outputs: 'Answer: B', '**C**', final line 'D', etc.
    """
    if not answer or not choices:
        return False
    valid = {k.upper() for k in choices if k in string.ascii_uppercase}
    if not valid:
        return False
    text = str(answer).strip()
    for sep in ("</think>", "</redacted_thinking>"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1].strip()

    patterns = [
        r'"answer"\s*:\s*"([A-J])"',
        r"'answer'\s*:\s*'([A-J])'",
        r"(?i)\*\*\s*([A-J])\s*\*\*",
        r"(?i)\(\s*([A-J])\s*\)\s*$",
        r"(?i)(?:final\s+)?answer\s*[:：]\s*\*?\s*([A-J])\b",
        r"(?i)(?:the\s+correct\s+(?:option|answer)\s+is|choice\s+is|option\s+is)\s+\*?\s*([A-J])\b",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text))
        for m in reversed(matches):
            letter = m.group(1).upper()
            if letter in valid:
                return letter

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        last = re.sub(r"^[\s\*_`#]+|[\s\*_`#]+$", "", last)
        last = last.strip(" .,:;!\"'()[]{}")
        if len(last) == 1 and last.upper() in valid:
            return last.upper()
    return False


def can_infer(answer, choices):
    """Combined approach to infer answer choice."""
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    if copt:
        return copt
    ct = can_infer_text(answer, choices)
    if ct:
        return ct
    return can_infer_relaxed(answer, choices)

def build_choices(item):
    """Build choices dictionary from item."""
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret

def build_option_str(option_dict):
    """Build option string from dictionary."""
    s = 'There are several options: \n'
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s

def build_prompt(question, options, prediction):
    """Build prompt for judge model."""
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)

def extract_answer_from_item(model, item, wait=5):
    """Extract answer from model prediction using rule-based and model-based approaches."""
    # Build choices dictionary
    choices = build_choices(item)
    option_str = build_option_str(choices)
    prompt = build_prompt(item['question'], option_str, item['prediction'])
    
    # Try rule-based extraction first
    prediction = item['prediction']
    ret = can_infer(prediction, choices)
    
    if ret:
        if ret == 'Z':
            extract_flag = False
            log = f"Rule extract failed with rule result: {ret} prediction: {prediction}"
        else:
            extract_flag = True
            log = f"Rule extract success with rule result: {ret} prediction: {prediction}"
        return dict(opt=ret, log=log, extract_model='rule', extract_flag=extract_flag)
    
    # If rule-based extraction fails, use model-based extraction
    if _eval_extraction_verbose():
        print("Rule extract failed. Use model-based extraction.")
    if model is None:
        assert model is not None, "Judge model is None for VideoMME !!!"

    # Try model-based extraction with retries
    retry = 25
    while retry:
        messages_for_judge = [{"type": "text", "value": prompt}]
        ans = model.generate(messages_for_judge)
        if "Failed to obtain answer via API" in ans:
            if _eval_extraction_verbose():
                print("API failed to answer.")
        else:
            ret = can_infer(ans, choices)
            if ret and ret != "Z":
                log = f"{model.model} extract Succeed. {model.model}:{ans}\n"
                return dict(
                    opt=ret, log=log, extract_model=model.model, extract_flag=True
                )
            if _eval_extraction_verbose():
                print(
                    f"Output includes 0 / > 1 letter among candidates "
                    f"{set(choices)} and Z: {ans}"
                )
        retry -= 1
        T = random.random() * wait * 2
        time.sleep(T)
        
        if retry == 0:
            options = list(choices) + ['Z'] if 'Z' not in choices else list(choices)
            log = f'{model.model} extract failed. randomly generate one. {model.model} response:{ans}\n'
            return dict(opt=random.choice(options), log=log, extract_model=model.model, extract_flag=False)

def eval_single_sample(args):
    """Evaluate a single sample."""
    model, item = args
        
    # Extract answer using the combined approach
    result = extract_answer_from_item(model, item)
    
    # Determine if the answer is correct
    hit = 1 if result['opt'] == item['answer'] else 0
    
    return {
        "index": item['index'],
        "question_id": item['question_id'],
        "question": item['question'],
        "domain": item['category'],
        "sub_category": item['sub_category'],
        "prediction": item['prediction'],
        "extracted_answer": result['opt'],
        "extraction_method": result['extract_model'],
        "extraction_success": result['extract_flag'],
        "extraction_log": result['log'],
        "gt": item['answer'],
        "hit": hit
    }

