"""Matplotlib figures for the experiment CSVs (Agg backend, no display needed)."""
from __future__ import annotations

import csv
import math
from pathlib import Path


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def make_plots(out: str | Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    made: list[Path] = []
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "font.size": 10})

    def save(fig, name):
        p = out / name
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        made.append(p)

    if (out / "e1_length.csv").exists():
        rows = _read(out / "e1_length.csv")
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for key in dict.fromkeys((r["scheme"], r["delta"]) for r in rows):
            rs = [r for r in rows if (r["scheme"], r["delta"]) == key]
            label = f"KGW delta={key[1]}" if key[0] == "kgw" else "Gumbel (distortion-free)"
            ax.plot([int(r["tokens"]) for r in rs], [float(r["tpr_at_z4"]) for r in rs], marker="o", label=label)
        hum = [r for r in rows if r["scheme"] == "kgw" and r["delta"] == "2.0"]
        ax.plot([int(r["tokens"]) for r in hum], [float(r["human_fpr_at_z4"]) for r in hum], "k--", label="human text (FPR)")
        ax.set(xlabel="tokens scored", ylabel="detected at z >= 4", title="E1: detection vs text length", ylim=(-0.02, 1.02))
        ax.legend(fontsize=8)
        save(fig, "e1_length.png")

    if (out / "e2_robustness.csv").exists():
        rows = _read(out / "e2_robustness.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, scheme in zip(axes, dict.fromkeys(r["scheme"] for r in rows)):
            for attack in ("substitute", "delete", "insert", "swap", "paraphrase_toy"):
                rs = [r for r in rows if r["scheme"] == scheme and r["attack"] == attack]
                if attack != "substitute":
                    base = [r for r in rows if r["scheme"] == scheme and r["attack"] == "substitute" and float(r["strength"]) == 0]
                    rs = base + rs
                ax.plot([float(r["strength"]) for r in rs], [float(r["mean_z"]) for r in rs], marker="o", label=attack)
            ax.axhline(4, color="k", ls="--", lw=1, label="threshold z=4")
            ax.set(xlabel="attack strength (fraction of words)", ylabel="mean z-score (200 tokens)", title=f"E2: {scheme}", ylim=(0, None))
            ax.legend(fontsize=8)
        save(fig, "e2_robustness.png")

    if (out / "e3_human_z.csv").exists():
        zs = [float(r["z"]) for r in _read(out / "e3_human_z.csv")]
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.hist(zs, bins=40, density=True, alpha=.7, label=f"human paragraphs x 5 keys (n={len(zs)})")
        xs = [i / 20 for i in range(-90, 91)]
        ax.plot(xs, [math.exp(-x * x / 2) / math.sqrt(2 * math.pi) for x in xs], "r-", label="N(0,1) theory")
        ax.axvline(4, color="k", ls="--", label="threshold z=4")
        ax.set(xlabel="z-score", ylabel="density", title="E3: z-scores of human text (no watermark)")
        ax.legend(fontsize=8)
        save(fig, "e3_fpr_human_z.png")

    if (out / "e4_quality.csv").exists():
        rows = _read(out / "e4_quality.csv")
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.scatter([float(r["mean_z"]) for r in rows], [float(r["median_ppl"]) for r in rows])
        for r in rows:
            ax.annotate(r["setting"], (float(r["mean_z"]), float(r["median_ppl"])), fontsize=8, xytext=(4, 3), textcoords="offset points")
        ax.set(xlabel="mean z-score (150 tokens)", ylabel="median perplexity (lower = more natural)", title="E4: detectability vs quality")
        save(fig, "e4_quality.png")

    if (out / "e6_image_provenance.csv").exists():
        rows = _read(out / "e6_image_provenance.csv")
        fig, ax = plt.subplots(figsize=(8, 4))
        names = [r["transform"] for r in rows]
        x = range(len(rows))
        ax.bar([i - .2 for i in x], [float(r["hard_binding_valid"]) for r in rows], width=.4, label="hard binding valid (SHA-256)")
        ax.bar([i + .2 for i in x], [float(r["soft_binding_match"]) for r in rows], width=.4, label="soft binding match (dHash<=10)")
        ax.set_xticks(list(x), names, rotation=25, ha="right", fontsize=8)
        ax.set(ylabel="fraction", title="E6: image provenance after transformations", ylim=(0, 1.05))
        ax.legend(fontsize=8)
        save(fig, "e6_image_provenance.png")
    return made
