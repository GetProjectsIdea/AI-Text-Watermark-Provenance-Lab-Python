"""Reproducible watermark and provenance experiments -> CSV files + PNG plots + summary.md.

    python -m wmlab experiment all --out results            # full run (a few minutes on a laptop)
    python -m wmlab experiment all --out results --quick    # smoke run (seconds)

Every experiment uses fixed seeds, the bundled corpora and the CPU trigram LM, so numbers are
reproducible. Swap in the HF backend for transformer results.
"""
from __future__ import annotations

import csv
import io
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from . import attacks, corpus, provenance
from .lm import NGramLM, default_lm
from .tokenize import detokenize, tokenize
from .watermark import WatermarkConfig, detect_tokens, normal_sf

KEY = "experiment-key-2026"
EXPERIMENTS: dict[str, Callable] = {}


def experiment(fn):
    EXPERIMENTS[fn.__name__.removeprefix("exp_")] = fn
    return fn


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def _generate(lm: NGramLM, cfg: WatermarkConfig | None, n: int, length: int, seed: int = 0) -> list[list[str]]:
    return [lm.generate(p, length, watermark=cfg, seed=seed + i).tokens
            for i, p in enumerate(corpus.prompts(n, seed=seed))]


def _rate(dets) -> float:
    return sum(d.detected for d in dets) / len(dets) if dets else 0.0


def _mean(xs) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def _human_tokens(n: int, min_tokens: int, seed: int = 1, name: str = "human") -> list[list[str]]:
    out = []
    for p in corpus.sample_paragraphs(name, n * 3, min_words=min_tokens, seed=seed):
        t = tokenize(p)
        if len(t) >= min_tokens:
            out.append(t)
        if len(out) == n:
            break
    return out


# ------------------------------------------------------------------------------------ E1
@experiment
def exp_length(out: Path, n: int, lm: NGramLM) -> list[dict]:
    """Detection rate vs text length for several bias strengths, plus human false positives."""
    lengths = [16, 32, 64, 128, 200]
    rows = []
    human = _human_tokens(n, 200)
    for delta in (1.0, 2.0, 3.0, 4.0):
        cfg = WatermarkConfig(key=KEY, delta=delta, gamma=0.25, min_tokens=8)
        docs = _generate(lm, cfg, n, max(lengths), seed=100)
        for L in lengths:
            dets = [detect_tokens(d[:L], cfg, keep_flags=False) for d in docs]
            hum = [detect_tokens(h[:L], cfg, keep_flags=False) for h in human]
            rows.append({"scheme": "kgw", "delta": delta, "gamma": 0.25, "tokens": L, "n": len(docs),
                         "tpr_at_z4": round(_rate(dets), 4), "mean_z": round(_mean([d.z for d in dets]), 3),
                         "human_fpr_at_z4": round(_rate(hum), 4)})
    g = WatermarkConfig(key=KEY, scheme="gumbel", min_tokens=8)
    docs = _generate(lm, g, n, max(lengths), seed=100)
    for L in lengths:
        dets = [detect_tokens(d[:L], g, keep_flags=False) for d in docs]
        hum = [detect_tokens(h[:L], g, keep_flags=False) for h in human]
        rows.append({"scheme": "gumbel", "delta": "", "gamma": "", "tokens": L, "n": len(docs),
                     "tpr_at_z4": round(_rate(dets), 4), "mean_z": round(_mean([d.z for d in dets]), 3),
                     "human_fpr_at_z4": round(_rate(hum), 4)})
    write_csv(out / "e1_length.csv", rows)
    return rows


# ------------------------------------------------------------------------------------ E2
@experiment
def exp_robustness(out: Path, n: int, lm: NGramLM) -> list[dict]:
    """Detection after edits: substitution, deletion, insertion, swaps, toy paraphrase, mixing."""
    rows = []
    # long human documents (several paragraphs glued together) for the mixing attack
    stream = tokenize(corpus.load("human"))
    step = max(1, (len(stream) - 2000) // max(n, 1))
    human = [stream[i * step:i * step + 2000] for i in range(n)]
    filler = [w for w in lm.vocab if w.isalpha()][:3000]
    for cfg in (WatermarkConfig(key=KEY, delta=2.0), WatermarkConfig(key=KEY, scheme="gumbel")):
        docs = _generate(lm, cfg, n, 200, seed=200)
        label = "kgw d=2" if cfg.scheme == "kgw" else "gumbel"

        def run(attack, strength, fn):
            dets = [detect_tokens(fn(d, i), cfg, keep_flags=False) for i, d in enumerate(docs)]
            rows.append({"scheme": label, "attack": attack, "strength": strength, "n": len(dets),
                         "tpr_at_z4": round(_rate(dets), 4), "mean_z": round(_mean([d.z for d in dets]), 3)})

        for r in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7):
            run("substitute", r, lambda d, i, r=r: attacks.substitute(d, r, seed=i, filler=filler))
        for r in (0.1, 0.2, 0.3):
            run("delete", r, lambda d, i, r=r: attacks.delete(d, r, seed=i))
            run("insert", r, lambda d, i, r=r: attacks.insert(d, r, seed=i))
            run("swap", r, lambda d, i, r=r: attacks.swap(d, r, seed=i))
        for s in (0.3, 0.6, 1.0):
            run("paraphrase_toy", s, lambda d, i, s=s: attacks.paraphrase_toy(d, s, seed=i))
        for f in (0.5, 0.25, 0.1):
            run("mix_into_human", f, lambda d, i, f=f: attacks.mix(d, human[i % len(human)], f, seed=i))
        for clean in (False, True):
            def zw(d, i, clean=clean):
                return tokenize(attacks.zero_width(detokenize(d), 0.5, seed=i), clean=clean)
            run("zero_width" + ("+normalise" if clean else ""), 0.5, zw)
    write_csv(out / "e2_robustness.csv", rows)
    return rows


# ------------------------------------------------------------------------------------ E3
@experiment
def exp_fpr(out: Path, n: int, lm: NGramLM) -> list[dict]:
    """False positives on human text across keys, and the repeated-text trap (dedupe on/off)."""
    rows = []
    zs_all: list[float] = []
    for name in ("human", "alice"):
        human = _human_tokens(n * 2, 100, seed=3, name=name)
        for k in range(5):
            cfg = WatermarkConfig(key=f"fpr-key-{k}")
            zs = [detect_tokens(h, cfg, keep_flags=False).z for h in human]
            zs_all += zs
            for thr in (2.0, 3.0, 4.0):
                rows.append({"corpus": name, "key": k, "dedupe": True, "repeat": 1, "threshold": thr, "n": len(zs),
                             "empirical_fpr": round(sum(z >= thr for z in zs) / len(zs), 4),
                             "theoretical_fpr": f"{normal_sf(thr):.2e}"})
    human = _human_tokens(n * 2, 60, seed=4)
    for dedupe in (False, True):
        cfg = WatermarkConfig(key="fpr-key-0", dedupe=dedupe)
        for rep in (1, 3, 10):
            zs = [detect_tokens(h * rep, cfg, keep_flags=False).z for h in human]
            for thr in (2.0, 4.0):
                rows.append({"corpus": "human x repeat", "key": 0, "dedupe": dedupe, "repeat": rep, "threshold": thr,
                             "n": len(zs), "empirical_fpr": round(sum(z >= thr for z in zs) / len(zs), 4),
                             "theoretical_fpr": f"{normal_sf(thr):.2e}"})
    write_csv(out / "e3_fpr.csv", rows)
    write_csv(out / "e3_human_z.csv", [{"z": round(z, 4)} for z in zs_all])
    return rows


# ------------------------------------------------------------------------------------ E4
@experiment
def exp_quality(out: Path, n: int, lm: NGramLM) -> list[dict]:
    """Text quality cost: perplexity under the unwatermarked LM vs detection strength."""
    rows = []
    settings = [("none", None)] + [(f"kgw d={d:g}", WatermarkConfig(key=KEY, delta=d)) for d in (0.5, 1, 2, 4, 8)]
    settings.append(("gumbel", WatermarkConfig(key=KEY, scheme="gumbel")))
    det_cfg = WatermarkConfig(key=KEY, delta=2.0)
    for label, cfg in settings:
        gens = [lm.generate(p, 150, watermark=cfg, seed=300 + i) for i, p in enumerate(corpus.prompts(n, seed=300))]
        ppl = [lm.perplexity(g.tokens, g.prompt_tokens) for g in gens]
        dcfg = cfg or det_cfg
        z = [detect_tokens(g.tokens, dcfg, keep_flags=False).z for g in gens]
        rows.append({"setting": label, "n": len(gens), "median_ppl": round(statistics.median(ppl), 2),
                     "mean_z": round(_mean(z), 3), "tpr_at_z4": round(sum(x >= 4 for x in z) / len(z), 4)})
    write_csv(out / "e4_quality.csv", rows)
    return rows


# ------------------------------------------------------------------------------------ E5
@experiment
def exp_gamma(out: Path, n: int, lm: NGramLM) -> list[dict]:
    """Green-list size gamma and context width h: detection vs robustness to 20% substitution."""
    rows = []
    filler = [w for w in lm.vocab if w.isalpha()][:3000]
    for gamma in (0.1, 0.25, 0.5):
        for h in (1, 2, 4):
            cfg = WatermarkConfig(key=KEY, gamma=gamma, delta=2.0, context_width=h)
            docs = _generate(lm, cfg, n, 100, seed=400)
            clean = [detect_tokens(d, cfg, keep_flags=False) for d in docs]
            att = [detect_tokens(attacks.substitute(d, 0.2, seed=i, filler=filler), cfg, keep_flags=False)
                   for i, d in enumerate(docs)]
            rows.append({"gamma": gamma, "context_width": h, "n": len(docs), "tokens": 100,
                         "tpr_clean": round(_rate(clean), 4), "mean_z_clean": round(_mean([d.z for d in clean]), 3),
                         "tpr_sub20": round(_rate(att), 4), "mean_z_sub20": round(_mean([d.z for d in att]), 3)})
    write_csv(out / "e5_gamma_context.csv", rows)
    return rows


# ------------------------------------------------------------------------------------ E6
def _demo_image(seed: int = 0):
    from PIL import Image, ImageDraw
    import random
    rng = random.Random(seed)
    img = Image.new("RGB", (640, 400), (18, 24, 48))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rng.randint(0, 600), rng.randint(0, 360)
        r = rng.randint(10, 80)
        d.ellipse([x, y, x + r, y + r], fill=(rng.randint(40, 255), rng.randint(40, 255), rng.randint(40, 255)))
    d.rectangle([20, 330, 300, 380], fill=(255, 255, 255))
    d.text((30, 345), "AI-generated demo image", fill=(0, 0, 0))
    return img


@experiment
def exp_image_provenance(out: Path, n: int, lm: NGramLM | None = None) -> list[dict]:
    """Hard (pixel SHA-256) vs soft (dHash) binding under everyday image transformations."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from PIL import Image, ImageEnhance
    key = Ed25519PrivateKey.generate()
    rows = []
    trials = max(3, min(n, 20))
    transforms = {
        "unchanged (metadata stripped)": lambda im: im,
        "png re-save": lambda im: im,
        "jpeg q95": lambda im: _jpeg(im, 95),
        "jpeg q70": lambda im: _jpeg(im, 70),
        "resize 50%": lambda im: im.resize((im.width // 2, im.height // 2)),
        "crop 10%": lambda im: im.crop((32, 20, im.width - 32, im.height - 20)),
        "brightness +20%": lambda im: ImageEnhance.Brightness(im).enhance(1.2),
        "different image": None,
    }
    for name, fn in transforms.items():
        hard_ok = soft_ok = 0
        dists = []
        for t in range(trials):
            img = _demo_image(t)
            m = provenance.sign_manifest(provenance.build_manifest(
                "image", content_hash=provenance.pixel_hash(img), soft_hash=provenance.dhash(img)), key)
            new = _demo_image(t + 1000) if fn is None else fn(img.copy())
            v = provenance.verify_manifest(m, new)
            hard_ok += v.status == "valid"
            dist = 0 if v.soft_distance is None else v.soft_distance
            dists.append(dist)
            soft_ok += dist <= provenance.SOFT_MATCH_BITS
        rows.append({"transform": name, "n": trials, "hard_binding_valid": round(hard_ok / trials, 3),
                     "soft_binding_match": round(soft_ok / trials, 3), "mean_dhash_distance": round(_mean(dists), 2)})
    write_csv(out / "e6_image_provenance.csv", rows)
    return rows


def _jpeg(img, q):
    from PIL import Image
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue()))


# ------------------------------------------------------------------------------------ runner
def run(names: list[str], out: str | Path, *, quick: bool = False, plots: bool = True, log=print) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 12 if quick else 100
    lm = default_lm()
    results = {}
    for name in names:
        if name not in EXPERIMENTS:
            raise ValueError(f"unknown experiment {name!r}; choose from {sorted(EXPERIMENTS)} or 'all'")
        t = time.time()
        results[name] = EXPERIMENTS[name](out, n, lm)
        log(f"  {name:<18} {len(results[name]):>3} rows  {time.time() - t:5.1f}s")
    if plots:
        from .plots import make_plots
        for p in make_plots(out):
            log(f"  plot {p.name}")
    (out / "summary.md").write_text(summary(results, quick=quick), encoding="utf-8")
    return results


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def summary(results: dict, quick: bool = False) -> str:
    parts = [f"# wmlab experiment results{' (quick run)' if quick else ''}\n",
             "Toy trigram LM on public-domain text, CPU only. z threshold 4 (one-sided p ~ 3.2e-5).\n"]
    titles = {"length": "E1 Detection vs length", "robustness": "E2 Robustness to edits",
              "fpr": "E3 False positives on human text", "quality": "E4 Quality (perplexity) vs strength",
              "gamma": "E5 gamma and context width", "image_provenance": "E6 Image provenance bindings"}
    for name, rows in results.items():
        parts.append(f"\n## {titles.get(name, name)}\n\n{_md_table(rows)}\n")
    return "".join(parts)
