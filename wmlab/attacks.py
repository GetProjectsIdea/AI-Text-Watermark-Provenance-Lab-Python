"""Text transformations that real users (or attackers) apply after generation.

Every attack takes and returns a token list, is seeded, and never needs the watermark key or the
detector. They are crude stand-ins for real editing; use the optional HF backend or a human study
for stronger paraphrase experiments.
"""
from __future__ import annotations

import random
from typing import Callable

from .synonyms import synonyms
from .tokenize import detokenize, tokenize

PUNCT = set(".,;:!?\"'()-")
FUNCTION_WORDS = {"the", "a", "an", "very", "that", "just", "really", "quite", "so", "then"}


def _word_positions(tokens: list[str]) -> list[int]:
    return [i for i, t in enumerate(tokens) if t[0].isalnum()]


def substitute(tokens: list[str], rate: float, *, seed: int = 0, filler: list[str] | None = None) -> list[str]:
    """Replace ``rate`` of the words: a synonym when we know one, otherwise a random filler word."""
    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1]")
    rng = random.Random(seed)
    out = tokens[:]
    pos = _word_positions(out)
    k = round(rate * len(pos))
    pool = filler or ["thing", "matter", "place", "way", "time", "day", "word", "hand", "eye", "door"]
    for i in rng.sample(pos, k):
        alts = synonyms(out[i])
        out[i] = rng.choice(alts) if alts else rng.choice(pool)
        if " " in out[i]:
            out[i] = out[i].split()[0]
    return out


def delete(tokens: list[str], rate: float, *, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    pos = set(rng.sample(range(len(tokens)), round(rate * len(tokens))))
    return [t for i, t in enumerate(tokens) if i not in pos]


def insert(tokens: list[str], rate: float, *, seed: int = 0, filler: list[str] | None = None) -> list[str]:
    rng = random.Random(seed)
    pool = filler or ["indeed", "very", "quite", "still", "then", "really", "perhaps", "also"]
    out = []
    for t in tokens:
        out.append(t)
        if rng.random() < rate:
            out.append(rng.choice(pool))
    return out


def truncate(tokens: list[str], length: int) -> list[str]:
    return tokens[:max(0, length)]


def swap(tokens: list[str], rate: float, *, seed: int = 0) -> list[str]:
    """Swap neighbouring words (local re-ordering)."""
    rng = random.Random(seed)
    out = tokens[:]
    pos = _word_positions(out)
    for i in rng.sample(pos, round(rate * len(pos))):
        if i + 1 < len(out) and out[i + 1][0].isalnum():
            out[i], out[i + 1] = out[i + 1], out[i]
    return out


def paraphrase_toy(tokens: list[str], strength: float = 0.5, *, seed: int = 0) -> list[str]:
    """Rule-based 'paraphrase': drop filler words, swap synonyms, rotate clauses at commas.

    ``strength`` in [0, 1]. This is far weaker than an LLM paraphraser at preserving meaning and
    roughly comparable at destroying n-gram watermarks, so treat results as indicative only.
    """
    rng = random.Random(seed)
    out = [t for t in tokens if not (t in FUNCTION_WORDS and rng.random() < strength * 0.6)]
    out = [rng.choice(synonyms(t)).split()[0] if synonyms(t) and rng.random() < strength else t for t in out]
    # rotate clauses inside sentences
    sentences, cur = [], []
    for t in out:
        cur.append(t)
        if t in ".!?":
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    result = []
    for s in sentences:
        if "," in s[:-1] and rng.random() < strength:
            k = s.index(",")
            end = s[-1:] if s[-1] in ".!?" else []
            body = s[:-1] if end else s
            s = body[k + 1:] + [","] + body[:k] + end
        result.extend(s)
    return result


def mix(watermarked: list[str], human: list[str], fraction: float, *, seed: int = 0) -> list[str]:
    """Embed a watermarked span so it makes up ``fraction`` of a longer human text."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n_h = round(len(watermarked) * (1 - fraction) / fraction)
    h = human[:n_h]
    rng = random.Random(seed)
    cut = rng.randint(0, len(h)) if h else 0
    return h[:cut] + watermarked + h[cut:]


def zero_width(text: str, rate: float = 0.5, *, seed: int = 0) -> str:
    """Insert zero-width spaces inside words (an evasion trick; detection normalises it away)."""
    rng = random.Random(seed)
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < rate:
            out.append("\u200b")
    return "".join(out)


def homoglyph(text: str, rate: float = 0.3, *, seed: int = 0) -> str:
    """Swap Latin letters for Cyrillic look-alikes (another evasion trick that normalisation undoes)."""
    table = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441", "x": "\u0445"}
    rng = random.Random(seed)
    return "".join(table[c] if c in table and rng.random() < rate else c for c in text)


TOKEN_ATTACKS: dict[str, Callable[..., list[str]]] = {
    "substitute": substitute, "delete": delete, "insert": insert, "swap": swap, "paraphrase": paraphrase_toy,
}


def apply(name: str, text: str, rate: float, *, seed: int = 0) -> str:
    """Apply an attack by name to raw text (used by the CLI and web UI)."""
    if name in ("zero_width", "homoglyph"):
        return (zero_width if name == "zero_width" else homoglyph)(text, rate, seed=seed)
    if name == "truncate":
        toks = tokenize(text)
        return detokenize(truncate(toks, round(len(toks) * (1 - rate))))
    if name not in TOKEN_ATTACKS:
        raise ValueError(f"unknown attack {name!r}")
    return detokenize(TOKEN_ATTACKS[name](tokenize(text), rate, seed=seed))


ATTACK_NAMES = sorted(list(TOKEN_ATTACKS) + ["truncate", "zero_width", "homoglyph"])
