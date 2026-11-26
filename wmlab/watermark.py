"""Text watermark schemes: embedding helpers and statistical detection.

Two schemes are implemented so students can compare them:

``kgw``    Green/red-list soft watermark (Kirchenbauer et al., ICML 2023). A keyed hash of the
           previous ``context_width`` tokens marks a fraction ``gamma`` of the vocabulary as
           green; ``delta`` is added to green logits before sampling. Detection counts green
           tokens and computes a one-proportion z-score.
``gumbel`` Exponential-minimum / Gumbel-max sampling (Aaronson 2022, Kuditipudi et al. 2024,
           the family SynthID-Text also builds on). Sampling picks argmax r_i^(1/p_i), which keeps
           the output distribution unchanged for a fixed key, and detection sums -ln(1 - r_t).

Detection only needs the key and the tokenizer, never the language model.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .hashing import context_seed, uniforms
from .tokenize import normalize, token_ids, tokenize

SCHEMES = ("kgw", "gumbel")
DEMO_KEY = "wmlab-demo-key-change-me"


@dataclass(frozen=True)
class WatermarkConfig:
    key: str = DEMO_KEY
    scheme: str = "kgw"
    gamma: float = 0.25
    delta: float = 2.0
    context_width: int = 0  # 0 = auto: 1 for kgw (LeftHash), 4 for gumbel (as in SynthID-Text)
    z_threshold: float = 4.0
    min_tokens: int = 25
    dedupe: bool = True  # score each (context, token) pair once (avoids FPR inflation on repeats)

    def __post_init__(self):
        if self.scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {self.scheme!r}; choose from {SCHEMES}")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if self.delta < 0:
            raise ValueError("delta must be >= 0")
        if self.context_width == 0:
            object.__setattr__(self, "context_width", 1 if self.scheme == "kgw" else 4)
        if not 1 <= self.context_width <= 8:
            raise ValueError("context_width must be 1..8")
        if not self.key:
            raise ValueError("key must not be empty")
        if self.min_tokens < 1:
            raise ValueError("min_tokens must be >= 1")

    @property
    def p_threshold(self) -> float:
        return normal_sf(self.z_threshold)

    def public_dict(self) -> dict:
        d = asdict(self)
        d.pop("key")
        return d


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def normal_isf(p: float) -> float:
    """Inverse survival function of N(0,1) (bisection; good to ~1e-10)."""
    if p <= 0:
        return float("inf")
    if p >= 1:
        return float("-inf")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if normal_sf(mid) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def gamma_sf_int(shape: int, x: float) -> float:
    """P(Gamma(shape, 1) >= x) for integer shape: e^-x * sum_{k<shape} x^k / k!  (log-space)."""
    if shape <= 0:
        return 1.0
    if x <= 0:
        return 1.0
    logx = math.log(x)
    terms = [k * logx - math.lgamma(k + 1) for k in range(shape)]
    m = max(terms)
    s = sum(math.exp(t - m) for t in terms)
    return min(1.0, math.exp(-x + m + math.log(s)))


# ----------------------------------------------------------------------------------- embedding

def kgw_bias(cfg: WatermarkConfig, context: Sequence[int], cand_ids, logits: np.ndarray) -> np.ndarray:
    """Return logits with ``delta`` added to green candidates."""
    u = uniforms(context_seed(cfg.key, context), cand_ids)
    return logits + cfg.delta * (u < cfg.gamma)


def gumbel_choice(cfg: WatermarkConfig, context: Sequence[int], cand_ids, probs: np.ndarray) -> int:
    """Index of the candidate chosen by exponential-minimum sampling: argmax log(r)/p."""
    r = uniforms(context_seed(cfg.key, context), cand_ids)
    r = np.clip(r, 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore"):
        score = np.where(probs > 0, np.log(r) / probs, -np.inf)
    return int(np.argmax(score))


# ----------------------------------------------------------------------------------- detection

@dataclass
class TokenFlag:
    token: str
    status: str  # "green" | "red" | "context" | "repeat"
    score: float | None = None


@dataclass
class Detection:
    scheme: str
    num_tokens: int
    num_scored: int
    num_green: int
    score: float
    z: float
    p_value: float
    detected: bool
    verdict: str
    flags: list[TokenFlag] = field(default_factory=list)

    @property
    def green_fraction(self) -> float:
        return self.num_green / self.num_scored if self.num_scored else 0.0

    def to_dict(self, with_flags: bool = False) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "flags"}
        d["green_fraction"] = round(self.green_fraction, 4)
        for k in ("score", "z"):
            d[k] = round(d[k], 4)
        if with_flags:
            d["flags"] = [asdict(f) for f in self.flags]
        return d


def _verdict(cfg: WatermarkConfig, scored: int, detected: bool, z: float) -> str:
    if scored < cfg.min_tokens:
        return f"too short to decide ({scored} scored tokens, need {cfg.min_tokens})"
    if detected:
        return f"watermark detected (z = {z:.2f} >= {cfg.z_threshold:g})"
    return f"no watermark evidence (z = {z:.2f} < {cfg.z_threshold:g})"


def detect_ids(ids: list[int], cfg: WatermarkConfig, *, display: list[str] | None = None,
               keep_flags: bool = True) -> Detection:
    """Score a sequence of token ids. Works for word ids and for HF tokenizer ids alike."""
    display = display if display is not None else [str(i) for i in ids]
    h = cfg.context_width
    seen: set[tuple] = set()
    flags: list[TokenFlag] = []
    green = 0
    scored = 0
    s_total = 0.0
    for t, tok in enumerate(display):
        if t < h:
            if keep_flags:
                flags.append(TokenFlag(tok, "context"))
            continue
        ctx = tuple(ids[t - h:t])
        pair = ctx + (ids[t],)
        if cfg.dedupe and pair in seen:
            if keep_flags:
                flags.append(TokenFlag(tok, "repeat"))
            continue
        seen.add(pair)
        u = float(uniforms(context_seed(cfg.key, ctx), [ids[t]])[0])
        scored += 1
        if cfg.scheme == "kgw":
            is_green = u < cfg.gamma
            green += is_green
            if keep_flags:
                flags.append(TokenFlag(tok, "green" if is_green else "red", round(u, 4)))
        else:
            s = -math.log(max(1e-12, 1.0 - u))
            s_total += s
            is_green = u > 0.5  # for display only: high r values are "watermark-like"
            green += is_green
            if keep_flags:
                flags.append(TokenFlag(tok, "green" if is_green else "red", round(s, 4)))
    if cfg.scheme == "kgw":
        if scored:
            z = (green - cfg.gamma * scored) / math.sqrt(scored * cfg.gamma * (1 - cfg.gamma))
        else:
            z = 0.0
        p = normal_sf(z)
        score = float(green)
    else:
        p = gamma_sf_int(scored, s_total) if scored else 1.0
        z = normal_isf(p) if scored else 0.0
        z = max(min(z, 40.0), -40.0)
        score = s_total
    detected = scored >= cfg.min_tokens and z >= cfg.z_threshold
    return Detection(cfg.scheme, len(display), scored, int(green), score, float(z), float(p), detected,
                     _verdict(cfg, scored, detected, z), flags if keep_flags else [])


def detect_tokens(tokens: list[str], cfg: WatermarkConfig, *, keep_flags: bool = True) -> Detection:
    return detect_ids(token_ids(tokens), cfg, display=tokens, keep_flags=keep_flags)


def detect(text: str, cfg: WatermarkConfig, *, clean: bool = True, keep_flags: bool = True) -> Detection:
    """Detect the watermark in raw text. ``clean`` undoes zero-width / homoglyph tricks first."""
    return detect_tokens(tokenize(text, clean=clean), cfg, keep_flags=keep_flags)


def z_scores(texts: Iterable[str], cfg: WatermarkConfig) -> list[float]:
    return [detect(t, cfg, keep_flags=False).z for t in texts]


def theoretical_fpr(z_threshold: float) -> float:
    return normal_sf(z_threshold)


__all__ = ["WatermarkConfig", "Detection", "TokenFlag", "detect", "detect_tokens", "detect_ids", "kgw_bias",
           "gumbel_choice", "normal_sf", "normal_isf", "gamma_sf_int", "SCHEMES", "DEMO_KEY", "normalize"]
