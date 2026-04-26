https://github.com/abysslover/qwen36_tool_calling_failure

The following are multiple choice questions (with answers) about {subject}.
Think step by step and then finish your answer with "ANSWER: [LETTER]"

[5-shot examples]

Answer the following multiple choice question...
Question:
...
Options:
...

❌ 5. Chat vs completion format
HF:
raw text prompt
sglang default:
chat template wrapping
💥 Impact
especially bad for 27B
adds tokens like <|assistant|>
✅ Fix
Use:
--chat-template none
or manually send plain prompt
