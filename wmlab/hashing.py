"""Keyed pseudo-random functions used by the watermark.

For a secret key and a context (the previous h token ids) we derive a 64-bit seed with BLAKE2b.
Each candidate token then gets a pseudo-random number u in [0, 1) from SplitMix64(seed xor id).

* KGW (green/red list): the token is "green" when u < gamma.
* Gumbel / Aaronson: u is the per-token random number r used by the exponential-minimum sampler.

Everything is vectorised with numpy uint64 arithmetic (wrap-around is intended).
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)
TWO64 = float(2**64)


def key_bytes(key: str | bytes) -> bytes:
    return key if isinstance(key, bytes) else key.encode("utf-8")


def context_seed(key: str | bytes, context: Sequence[int]) -> int:
    k = key_bytes(key)
    if not k:
        raise ValueError("watermark key must not be empty")
    if len(k) > 64:  # BLAKE2b keys are limited to 64 bytes
        k = hashlib.sha256(k).digest()
    h = hashlib.blake2b(digest_size=8, key=k)
    for c in context:
        h.update(int(c).to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest(), "little")


def splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + GOLDEN
        z = (z ^ (z >> np.uint64(30))) * M1
        z = (z ^ (z >> np.uint64(27))) * M2
        return z ^ (z >> np.uint64(31))


def uniforms(seed: int, ids: Sequence[int] | np.ndarray) -> np.ndarray:
    """Pseudo-random numbers in [0, 1) for each candidate id under one context seed."""
    arr = np.asarray(ids, dtype=np.uint64)
    with np.errstate(over="ignore"):
        mixed = splitmix64(arr * GOLDEN ^ np.uint64(seed))
    # top 53 bits -> float in [0,1)
    return (mixed >> np.uint64(11)).astype(np.float64) / float(2**53)


def green_mask(key: str | bytes, context: Sequence[int], ids, gamma: float) -> np.ndarray:
    return uniforms(context_seed(key, context), ids) < gamma
