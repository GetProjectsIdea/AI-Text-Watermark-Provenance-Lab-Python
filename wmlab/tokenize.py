"""Word-level tokenizer shared by the toy language model and the detector.

The detector must see exactly the same tokens as the generator, so tokenisation is deliberately
simple and deterministic: lower-case words (with an optional apostrophe part) and single
punctuation marks. Every token maps to a stable 64-bit id with BLAKE2b, so the watermark works for
any word, including words the language model has never seen.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
# common Cyrillic / Greek look-alikes used to dodge detectors (homoglyph attack)
HOMOGLYPHS = str.maketrans({
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c", "\u0445": "x",
    "\u0443": "y", "\u0456": "i", "\u0455": "s", "\u0458": "j", "\u03bf": "o", "\u03b1": "a",
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2014": "-", "\u2013": "-",
})
NO_SPACE_BEFORE = set(".,;:!?)]}'%")
NO_SPACE_AFTER = set("([{")


def normalize(text: str) -> str:
    """Undo cheap evasion tricks before detection: NFKC, zero-width characters, homoglyphs."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ZERO_WIDTH).translate(HOMOGLYPHS)
    return text


def tokenize(text: str, *, clean: bool = True) -> list[str]:
    if clean:
        text = normalize(text)
    return TOKEN_RE.findall(text.lower())


def detokenize(tokens: list[str], casemap: dict[str, str] | None = None) -> str:
    """Join tokens into readable text. ``tokenize(detokenize(t)) == t`` for tokenizer output.

    ``casemap`` optionally restores capitalisation of names (``darcy`` -> ``Darcy``).
    """
    out: list[str] = []
    quote_open = False
    for tok in tokens:
        word = casemap.get(tok, tok) if casemap else tok
        if not out:
            out.append(word)
            quote_open = tok == '"'
            continue
        prev = out[-1]
        if tok == '"':
            out.append(" " + tok if not quote_open else tok)  # opening quote: space before; closing: none
            quote_open = not quote_open
        elif (prev.endswith('"') and quote_open) or tok in NO_SPACE_BEFORE or prev in NO_SPACE_AFTER:
            out.append(word)
        else:
            out.append(" " + word)
    text = "".join(out)
    # capitalise sentence starts for readability (the detector lower-cases anyway)
    text = re.sub(r"(^|[.!?]\"?\s+\"?)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r"\bi\b", "I", text)
    return text


def token_id(tok: str) -> int:
    return int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little")


def token_ids(tokens: list[str]) -> list[int]:
    return [token_id(t) for t in tokens]
