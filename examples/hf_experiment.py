"""Repeat the core experiment on a real transformer (optional; needs torch + transformers).

    pip install torch transformers
    python examples/hf_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct --n 12 --out docs/results

For each prompt it generates (a) a KGW-watermarked answer, (b) an unwatermarked answer, then
(c) asks the same model, *without* the watermark, to paraphrase (a), and (d) truncates (a) to 60
tokens. All four are scored with the tokenizer-level detector. CPU is fine for 0.5B models.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from wmlab.backends.hf import HFWatermarker
from wmlab.watermark import WatermarkConfig

PROMPTS = [
    "Explain how the monsoon affects farming in India.",
    "Write a short note on the importance of cybersecurity for college students.",
    "Describe the water cycle for a school science project.",
    "What are the benefits of learning Python for engineering students?",
    "Write a paragraph about the history of the Indian railways.",
    "Explain what a database index is and why it speeds up queries.",
    "Give advice to a first-year student on managing time during exams.",
    "Describe how solar panels generate electricity.",
    "Write a short essay on the role of libraries in a digital age.",
    "Explain the difference between machine learning and traditional programming.",
    "Describe a visit to a busy Indian market in the evening.",
    "Why is clean drinking water important for public health?",
    "Explain how UPI payments work in simple terms.",
    "Write about the benefits of reading newspapers daily.",
    "Explain what climate change is and one way students can help.",
    "Describe the main parts of a computer and what they do.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    cfg = WatermarkConfig(key="hf-experiment-key", delta=args.delta, gamma=0.25)
    wm = HFWatermarker(args.model, cfg)
    rows = []
    t0 = time.time()
    for i, prompt in enumerate(PROMPTS[: args.n]):
        a = wm.generate(prompt, args.tokens, seed=i)
        b = wm.generate(prompt, args.tokens, watermark=False, seed=i)
        c = wm.paraphrase(a["text"], seed=i)
        d = wm.tok.decode(a["ids"][:60], skip_special_tokens=True)
        for kind, text in (("watermarked", a["text"]), ("unwatermarked", b["text"]),
                           ("watermarked -> LLM paraphrase", c), ("watermarked -> first 60 tokens", d)):
            det = wm.detect(text)
            rows.append({"prompt": i, "kind": kind, "tokens": det.num_tokens, "scored": det.num_scored,
                         "z": round(det.z, 3), "detected": det.detected})
        print(f"[{i + 1}/{args.n}] {time.time() - t0:6.0f}s  z: wm={rows[-4]['z']:.1f} plain={rows[-3]['z']:.1f} "
              f"para={rows[-2]['z']:.1f} trunc={rows[-1]['z']:.1f}", flush=True)
        if i == 0:
            print("  sample watermarked:", a["text"][:300].replace("\n", " "))
            print("  sample paraphrase: ", c[:300].replace("\n", " "))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = args.model.split("/")[-1].lower()
    with (out / f"hf_{slug}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"# Real-model check: {args.model} (CPU)\n", f"KGW delta={args.delta}, gamma=0.25, {args.tokens} new tokens, "
             f"{args.n} prompts, z threshold 4. Runtime {time.time() - t0:.0f} s.\n",
             "| text | mean tokens | mean z | detected |", "|---|---|---|---|"]
    for kind in dict.fromkeys(r["kind"] for r in rows):
        rs = [r for r in rows if r["kind"] == kind]
        lines.append(f"| {kind} | {statistics.fmean(r['tokens'] for r in rs):.0f} | "
                     f"{statistics.fmean(r['z'] for r in rs):.2f} | {sum(r['detected'] for r in rs)}/{len(rs)} |")
    (out / f"hf_{slug}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
