# Real-model check: Qwen/Qwen2.5-0.5B-Instruct (CPU)

KGW delta=2.0, gamma=0.25, 160 new tokens, 12 prompts, z threshold 4. Runtime 225 s.

| text | mean tokens | mean z | detected |
|---|---|---|---|
| watermarked | 160 | 8.72 | 12/12 |
| unwatermarked | 160 | 0.46 | 0/12 |
| watermarked -> LLM paraphrase | 148 | 1.75 | 2/12 |
| watermarked -> first 60 tokens | 60 | 4.35 | 6/12 |
