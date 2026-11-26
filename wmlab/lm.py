"""A small trigram language model that runs anywhere (CPU, no downloads).

Interpolated absolute-discounting trigram model trained on the bundled public-domain corpus. It is
not a good writer, but it has a real next-token distribution with realistic entropy, which is all
a watermark needs. That makes every experiment reproducible on a laptop in seconds. For a real
transformer use :mod:`wmlab.backends.hf` (optional, needs ``torch`` + ``transformers``).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import corpus
from .tokenize import detokenize, token_id, tokenize
from .watermark import WatermarkConfig, gumbel_choice, kgw_bias

D = 0.75  # absolute discount


@dataclass
class Generation:
    prompt: str
    tokens: list[str]
    prompt_tokens: list[str]
    watermarked: bool
    scheme: str | None
    casemap: dict[str, str] | None = None

    @property
    def text(self) -> str:
        return detokenize(self.tokens, self.casemap)

    @property
    def full_tokens(self) -> list[str]:
        return self.prompt_tokens + self.tokens


def truecase_map(text: str, min_count: int = 3, ratio: float = 0.85) -> dict[str, str]:
    """Words that are almost always capitalised in the corpus (names, places) -> their usual form."""
    import re
    forms: dict[str, Counter] = defaultdict(Counter)
    for m in re.finditer(r"(?<![.!?\"]\s)(?<!^)\b([A-Za-z]+)\b", text):
        w = m.group(1)
        forms[w.lower()][w] += 1
    out = {}
    for low, c in forms.items():
        total = sum(c.values())
        form, n = c.most_common(1)[0]
        if total >= min_count and form[0].isupper() and n / total >= ratio and len(low) > 1:
            out[low] = form
    return out


class NGramLM:
    casemap: dict[str, str] = {}

    def __init__(self, tokens: Sequence[str], top_unigrams: int = 300):
        vocab = sorted(set(tokens))
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        self.ids = np.array([token_id(w) for w in vocab], dtype=np.uint64)
        idx = [self.index[w] for w in tokens]
        uni = Counter(idx)
        self.total = len(idx)
        self.p1 = np.array([uni[i] for i in range(len(vocab))], dtype=np.float64) + 0.01
        self.p1 /= self.p1.sum()
        self.bi: dict[int, dict[int, int]] = defaultdict(dict)
        self.tri: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        for a, b in zip(idx, idx[1:]):
            d = self.bi[a]
            d[b] = d.get(b, 0) + 1
        for a, b, c in zip(idx, idx[1:], idx[2:]):
            d = self.tri[(a, b)]
            d[c] = d.get(c, 0) + 1
        self.bi_tot = {k: sum(v.values()) for k, v in self.bi.items()}
        self.tri_tot = {k: sum(v.values()) for k, v in self.tri.items()}
        self.top = np.argsort(-self.p1)[:top_unigrams]
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_text(cls, text: str, **kw) -> "NGramLM":
        lm = cls(tokenize(text), **kw)
        lm.casemap = truecase_map(text)
        return lm

    # ------------------------------------------------------------------ probabilities
    def _p2_vec(self, v: int, cand: np.ndarray) -> np.ndarray:
        p = self.p1[cand]
        conts = self.bi.get(v)
        if not conts:
            return p
        tot = self.bi_tot[v]
        c = np.array([conts.get(int(i), 0) for i in cand], dtype=np.float64)
        lam = D * len(conts) / tot
        return np.maximum(c - D, 0) / tot + lam * p

    def distribution(self, u: int, v: int) -> tuple[np.ndarray, np.ndarray]:
        """Candidate vocabulary indices and their (renormalised) probabilities for context (u, v)."""
        key = (u, v)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        cand_set = set(self.top.tolist())
        cand_set.update(self.bi.get(v, {}).keys())
        cand_set.update(self.tri.get(key, {}).keys())
        cand = np.array(sorted(cand_set), dtype=np.int64)
        p = self._p2_vec(v, cand)
        conts = self.tri.get(key)
        if conts:
            tot = self.tri_tot[key]
            c = np.array([conts.get(int(i), 0) for i in cand], dtype=np.float64)
            lam = D * len(conts) / tot
            p = np.maximum(c - D, 0) / tot + lam * p
        p = p / p.sum()
        hit = (cand, p)
        if len(self._cache) < 200_000:
            self._cache[key] = hit
        return hit

    def prob(self, u: int, v: int, w: int) -> float:
        """Exact interpolated probability (used for perplexity; handles unknown words)."""
        if w < 0:
            return 1.0 / (self.total + len(self.vocab))
        p = float(self.p1[w])
        conts = self.bi.get(v)
        if conts:
            tot = self.bi_tot[v]
            p = max(conts.get(w, 0) - D, 0) / tot + D * len(conts) / tot * p
        conts = self.tri.get((u, v))
        if conts:
            tot = self.tri_tot[(u, v)]
            p = max(conts.get(w, 0) - D, 0) / tot + D * len(conts) / tot * p
        return p

    def perplexity(self, tokens: Sequence[str], context: Sequence[str] = ()) -> float:
        seq = [self.index.get(t, -1) for t in list(context) + list(tokens)]
        start = len(context)
        nll, n = 0.0, 0
        for t in range(max(start, 0), len(seq)):
            u = seq[t - 2] if t >= 2 else -1
            v = seq[t - 1] if t >= 1 else -1
            nll -= math.log(max(self.prob(u, v, seq[t]), 1e-12))
            n += 1
        return math.exp(nll / n) if n else float("nan")

    def entropy(self, u: int, v: int) -> float:
        _, p = self.distribution(u, v)
        return float(-(p * np.log(p)).sum())

    # ------------------------------------------------------------------ generation
    def generate(self, prompt: str, n_tokens: int = 200, *, watermark: WatermarkConfig | None = None,
                 temperature: float = 1.0, top_k: int = 40, seed: int | None = None) -> Generation:
        if n_tokens < 1:
            raise ValueError("n_tokens must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        rng = np.random.default_rng(seed)
        ptoks = tokenize(prompt) or ["the"]
        seq = [self.index.get(t, -1) for t in ptoks]
        ids_seq = [token_id(t) for t in ptoks]
        out: list[str] = []
        h = watermark.context_width if watermark else 1
        used_ctx: set[tuple] = set()
        for _ in range(n_tokens):
            u = seq[-2] if len(seq) >= 2 else -1
            v = seq[-1]
            cand, p = self.distribution(u, v)
            logits = np.log(p) / temperature
            ctx = ids_seq[-h:] if len(ids_seq) >= h else ([0] * (h - len(ids_seq)) + ids_seq)
            cand_ids = self.ids[cand]
            if watermark is not None and watermark.scheme == "kgw":
                logits = kgw_bias(watermark, ctx, cand_ids, logits)
            if top_k and top_k < len(cand):
                keep = np.argpartition(-logits, top_k)[:top_k]
                cand, logits, cand_ids = cand[keep], logits[keep], cand_ids[keep]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            # Gumbel sampling is deterministic per context, so a repeated context would repeat the
            # same token forever. Like SynthID-Text we sample normally when a context repeats.
            if watermark is not None and watermark.scheme == "gumbel" and tuple(ctx) not in used_ctx:
                used_ctx.add(tuple(ctx))
                j = gumbel_choice(watermark, ctx, cand_ids, probs)
            else:
                j = int(rng.choice(len(cand), p=probs))
            w = int(cand[j])
            seq.append(w)
            ids_seq.append(int(cand_ids[j]))
            out.append(self.vocab[w])
        return Generation(prompt, out, ptoks, watermark is not None, watermark.scheme if watermark else None,
                          self.casemap)


_DEFAULT: NGramLM | None = None


def default_lm() -> NGramLM:
    """The trigram LM trained on the bundled corpus (built once, ~2-4 s)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = NGramLM.from_text(corpus.load("train"))
    return _DEFAULT
