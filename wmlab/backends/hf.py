"""Optional Hugging Face transformers backend (``pip install torch transformers``).

Applies the same KGW / Gumbel schemes at the level of the model's own tokenizer ids, so you can
repeat the experiments on a real transformer (tested on the box with SmolLM2-135M/360M-Instruct
and Qwen2.5-0.5B-Instruct on CPU). Detection needs only the tokenizer and the key.

Word-level (n-gram) and tokenizer-level watermarks live in different id spaces: a text
watermarked by this backend must be checked with :meth:`HFWatermarker.detect`, not with
``wmlab detect`` (which uses the word tokenizer).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..hashing import context_seed, uniforms
from ..watermark import Detection, WatermarkConfig, detect_ids


def _require():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extras
        raise RuntimeError("the HF backend needs: pip install torch transformers") from e


def make_processor(cfg: WatermarkConfig, vocab_size: int, seed: int | None = None):
    """A transformers LogitsProcessor implementing the configured watermark."""
    _require()
    import torch
    from transformers import LogitsProcessor

    all_ids = np.arange(vocab_size, dtype=np.uint64)

    class WatermarkProcessor(LogitsProcessor):
        def __init__(self):
            self.used: list[set] = []
            self.gen = torch.Generator().manual_seed(seed if seed is not None else 0)

        def __call__(self, input_ids, scores):
            h = cfg.context_width
            while len(self.used) < input_ids.shape[0]:
                self.used.append(set())
            for b in range(input_ids.shape[0]):
                ctx = tuple(int(x) for x in input_ids[b, -h:].tolist())
                u = torch.from_numpy(uniforms(context_seed(cfg.key, ctx), all_ids[: scores.shape[-1]]))
                u = u.to(scores.dtype)
                if cfg.scheme == "kgw":
                    scores[b] = scores[b] + cfg.delta * (u < cfg.gamma).to(scores.dtype)
                else:
                    probs = torch.softmax(scores[b].float(), dim=-1)
                    if ctx in self.used[b]:
                        choice = int(torch.multinomial(probs, 1, generator=self.gen))
                    else:
                        self.used[b].add(ctx)
                        r = u.float().clamp(1e-12, 1 - 1e-12)
                        score = torch.where(probs > 0, torch.log(r) / probs, torch.full_like(probs, -float("inf")))
                        choice = int(torch.argmax(score))
                    forced = torch.full_like(scores[b], -float("inf"))
                    forced[choice] = 0.0
                    scores[b] = forced
            return scores

    return WatermarkProcessor()


class HFWatermarker:
    def __init__(self, model_name: str, cfg: WatermarkConfig | None = None, *, dtype: str = "float32"):
        _require()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.cfg = cfg or WatermarkConfig()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype))
        self.model.eval()

    def _inputs(self, prompt: str, chat: bool):
        if chat and getattr(self.tok, "chat_template", None):
            ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
            return ids
        return self.tok(prompt, return_tensors="pt")

    def generate(self, prompt: str, max_new_tokens: int = 200, *, watermark: bool = True, chat: bool = True,
                 temperature: float = 1.0, top_k: int = 0, seed: int | None = 0) -> dict[str, Any]:
        import torch
        from transformers import LogitsProcessorList
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self._inputs(prompt, chat)
        procs = LogitsProcessorList()
        sample = True
        if watermark:
            procs.append(make_processor(self.cfg, self.model.config.vocab_size, seed))
            sample = self.cfg.scheme == "kgw"
        kwargs: dict[str, Any] = dict(max_new_tokens=max_new_tokens, logits_processor=procs, do_sample=sample,
                                      pad_token_id=self.tok.eos_token_id)
        if sample:
            kwargs.update(temperature=temperature, top_k=top_k or None, top_p=1.0)
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        new = out[0, inputs["input_ids"].shape[1]:].tolist()
        if self.tok.eos_token_id in new:
            new = new[: new.index(self.tok.eos_token_id)]
        return {"text": self.tok.decode(new, skip_special_tokens=True), "ids": new}

    def detect(self, text: str) -> Detection:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return detect_ids(ids, self.cfg, display=self.tok.convert_ids_to_tokens(ids))

    def paraphrase(self, text: str, max_new_tokens: int = 320, seed: int = 0) -> str:
        """Unwatermarked LLM paraphrase, used as a realistic removal attack in experiments."""
        prompt = ("Paraphrase the following passage in your own words. Keep every fact, "
                  "do not add commentary, output only the paraphrase.\n\n" + text)
        return self.generate(prompt, max_new_tokens, watermark=False, chat=True, temperature=0.7, seed=seed)["text"]
