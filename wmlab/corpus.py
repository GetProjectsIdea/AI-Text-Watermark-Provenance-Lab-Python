"""Bundled public-domain corpora (Project Gutenberg texts, headers removed).

* ``train``  Pride and Prejudice + The Adventures of Sherlock Holmes: trains the toy n-gram LM.
* ``human``  Frankenstein + A Tale of Two Cities: held-out *human* text for false-positive tests
             (different authors from the training data).
* ``alice``  Alice's Adventures in Wonderland: a second, very different human style.
"""
from __future__ import annotations

import gzip
import random
import re
from functools import lru_cache
from importlib import resources

NAMES = ("train", "human", "alice")


@lru_cache(maxsize=None)
def load(name: str) -> str:
    if name not in NAMES:
        raise ValueError(f"unknown corpus {name!r}; choose from {NAMES}")
    data = resources.files("wmlab").joinpath("data", f"{name}.txt.gz").read_bytes()
    # Gutenberg marks italics with _underscores_; drop them so they do not become tokens
    return gzip.decompress(data).decode("utf-8").replace("_", "")


def paragraphs(name: str, min_words: int = 40) -> list[str]:
    """Prose paragraphs (joined lines) with at least ``min_words`` words."""
    out = []
    for block in re.split(r"\n\s*\n", load(name)):
        text = " ".join(block.split())
        if len(text.split()) < min_words:
            continue
        if text.isupper() or text.lower().startswith(("produced by", "chapter", "letter ")):
            continue
        out.append(text)
    return out


def sample_paragraphs(name: str, n: int, min_words: int = 40, seed: int = 0) -> list[str]:
    paras = paragraphs(name, min_words)
    rng = random.Random(seed)
    if n >= len(paras):
        return paras[:]
    return rng.sample(paras, n)


def prompts(n: int, words: int = 8, seed: int = 0, name: str = "human") -> list[str]:
    """Opening words of held-out paragraphs, used as generation prompts."""
    return [" ".join(p.split()[:words]) for p in sample_paragraphs(name, n, min_words=words + 5, seed=seed)]
