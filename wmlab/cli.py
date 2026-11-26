"""Command-line interface: python -m wmlab <command> ...  (run with -h for help)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, attacks, provenance
from .watermark import DEMO_KEY, SCHEMES, WatermarkConfig, detect

HOME = Path(os.environ.get("WMLAB_HOME", Path.home() / ".wmlab"))
GREEN, RED, DIM, BOLD, YEL, CYAN, END = "\x1b[32m", "\x1b[31m", "\x1b[2m", "\x1b[1m", "\x1b[33m", "\x1b[36m", "\x1b[0m"


def _c(s: str, col: str, on: bool) -> str:
    col = col.replace("m\x1b[", ";")  # merge "bold" + "green" into one SGR sequence
    return f"{col}{s}{END}" if on else s


def _key(args) -> str:
    key = args.key or os.environ.get("WMLAB_KEY") or DEMO_KEY
    if key == DEMO_KEY and not getattr(args, "quiet_key", False):
        print(_c("note: using the public demo key; set WMLAB_KEY or --key for real experiments", YEL, args.color),
              file=sys.stderr)
    return key


def _cfg(args) -> WatermarkConfig:
    return WatermarkConfig(key=_key(args), scheme=args.scheme, gamma=args.gamma, delta=args.delta,
                           context_width=args.context_width, z_threshold=args.z, min_tokens=args.min_tokens,
                           dedupe=not args.no_dedupe)


def _read_text(src: str | None) -> str:
    if not src or src == "-":
        return sys.stdin.read()
    p = Path(src)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return src  # treat the argument itself as text


def _print_detection(d, color: bool, show_tokens: bool) -> None:
    if show_tokens:
        parts = []
        for f in d.flags:
            col = {"green": GREEN, "red": RED}.get(f.status, DIM)
            parts.append(_c(f.token, col, color))
        print(" ".join(parts))
        print()
    verdict_col = GREEN if d.detected else (YEL if "short" in d.verdict else CYAN)
    print(_c(d.verdict, BOLD + verdict_col, color))
    print(f"scheme={d.scheme}  tokens={d.num_tokens}  scored={d.num_scored}  green={d.num_green} "
          f"({d.green_fraction:.0%})  z={d.z:.2f}  p={d.p_value:.2e}")
    print(_c("A detected mark is evidence about one key, not proof of authorship; "
             "a missing mark does not prove a human wrote it.", DIM, color))


# ---------------------------------------------------------------------------------- commands

def cmd_generate(args) -> int:
    from .lm import default_lm
    cfg = None if args.no_watermark else _cfg(args)
    g = default_lm().generate(args.prompt, args.tokens, watermark=cfg, temperature=args.temperature,
                              top_k=args.top_k, seed=args.seed)
    text = g.text
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out} ({len(g.tokens)} tokens, {'watermarked ' + cfg.scheme if cfg else 'no watermark'})")
    else:
        print(text)
    return 0


def cmd_detect(args) -> int:
    cfg = _cfg(args)
    d = detect(_read_text(args.text), cfg, clean=not args.raw)
    if args.json:
        print(json.dumps(d.to_dict(with_flags=args.tokens), indent=2))
    else:
        _print_detection(d, args.color, args.tokens)
    return 0 if d.detected else 1


def cmd_attack(args) -> int:
    text = _read_text(args.text)
    out = attacks.apply(args.type, text, args.rate, seed=args.seed)
    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(out)
    return 0


def cmd_experiment(args) -> int:
    from .experiments import EXPERIMENTS, run
    names = list(EXPERIMENTS) if "all" in args.names else args.names
    print(f"running {', '.join(names)} -> {args.out} ({'quick' if args.quick else 'full'})")
    run(names, args.out, quick=args.quick, plots=not args.no_plots)
    print(f"done: {Path(args.out) / 'summary.md'}")
    return 0


def cmd_keygen(args) -> int:
    d = Path(args.dir)
    priv, pub = provenance.generate_keypair(d, args.name)
    k = provenance.load_public_key(pub)
    print(f"private key: {priv} (mode 600, keep it secret)")
    print(f"public key:  {pub}")
    print(f"key id:      {provenance.key_id(k)}")
    if args.trust:
        t = Path(args.trust)
        t.mkdir(parents=True, exist_ok=True)
        (t / pub.name).write_bytes(pub.read_bytes())
        print(f"added to trust list: {t / pub.name}")
    return 0


def cmd_sign(args) -> int:
    key = provenance.load_private_key(args.key_file)
    parent = json.loads(Path(args.parent).read_text(encoding="utf-8")) if args.parent else None
    wm = None
    if args.watermark_scheme:
        wm = {"scheme": args.watermark_scheme, "note": "text carries a statistical watermark; key held by the provider"}
    manifest, out = provenance.sign_file(args.file, key, title=args.title, source=args.source,
                                         generator=args.generator, action="c2pa.edited" if parent else "c2pa.created",
                                         watermark=wm, parent=parent, embed_png=not args.no_embed)
    print(f"signed {args.file}")
    print(f"  manifest: {out}" + ("  (also embedded in the PNG)" if Path(args.file).suffix.lower() == ".png" and not args.no_embed else ""))
    print(f"  signer:   {manifest['signature']['key_id']}")
    print(f"  binding:  {manifest['hard_binding']['alg']} {manifest['hard_binding']['hash'][:16]}...")
    return 0


def cmd_verify(args) -> int:
    v = provenance.verify_file(args.file, manifest_path=args.manifest, trust_dir=args.trust, parent_path=args.parent)
    if args.json:
        print(json.dumps(v.to_dict(), indent=2))
        return 0 if v.ok else 2
    col = GREEN if v.status == "valid" else (YEL if v.status == "valid_untrusted" else RED)
    print(_c(f"{v.status.upper()}", BOLD + col, args.color) + f"  {args.file}")
    if v.source_type:
        print(f"  source: {v.source_type}")
    for c in v.checks:
        print(f"  - {c}")
    return 0 if v.ok else 2


def cmd_web(args) -> int:
    from .web import create_app
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("refusing to listen on a non-loopback address; this UI is for local experiments", file=sys.stderr)
        return 2
    app = create_app(key=os.environ.get("WMLAB_KEY") or DEMO_KEY, home=HOME)
    print(f"wmlab web UI on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


def cmd_demo(args) -> int:
    from .lm import default_lm
    color = args.color
    cfg = WatermarkConfig(key="demo-" + DEMO_KEY, delta=2.0)
    lm = default_lm()
    prompt = "It was a dark and stormy night, and"
    print(_c("$ python -m wmlab generate --prompt '" + prompt + "' --tokens 120", GREEN, color))
    g = lm.generate(prompt, 120, watermark=cfg, seed=15)
    print(g.text, "\n")
    print(_c("$ python -m wmlab detect watermarked.txt", GREEN, color))
    _print_detection(detect(g.text, cfg), color, False)
    print()
    plain = lm.generate(prompt, 120, seed=15).text
    print(_c("$ python -m wmlab detect plain.txt   # same LM, no watermark", GREEN, color))
    _print_detection(detect(plain, cfg), color, False)
    print()
    for kind, rate in (("substitute", 0.2), ("substitute", 0.5), ("substitute", 0.7), ("paraphrase", 1.0),
                       ("zero_width", 0.5)):
        att = attacks.apply(kind, g.text, rate, seed=1)
        d = detect(att, cfg)
        print(f"after {kind:<10} {rate:>4.0%}: z = {d.z:6.2f}  -> {'detected' if d.detected else 'missed'}")
    return 0


def cmd_status(args) -> int:
    import platform
    print(f"wmlab {__version__}  python {platform.python_version()}")
    key = os.environ.get("WMLAB_KEY")
    print(f"WMLAB_KEY: {'set (' + str(len(key)) + ' chars, hidden)' if key else 'not set (demo key in use)'}")
    print(f"home: {HOME}")
    try:
        import transformers  # noqa: F401
        print("optional HF backend: available (torch + transformers installed)")
    except ImportError:
        print("optional HF backend: not installed (pip install torch transformers)")
    from . import corpus
    for n in corpus.NAMES:
        print(f"corpus {n:<6} {len(corpus.load(n)):>9,} chars")
    return 0


# ---------------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="wmlab", description="AI text watermarking + content provenance experiments")
    ap.add_argument("--version", action="version", version=f"wmlab {__version__}")
    ap.add_argument("--no-color", dest="color", action="store_false", help="disable ANSI colours")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def wm_opts(p):
        p.add_argument("--key", help="watermark key (default: $WMLAB_KEY or the public demo key)")
        p.add_argument("--scheme", choices=SCHEMES, default="kgw")
        p.add_argument("--gamma", type=float, default=0.25, help="green-list fraction (kgw)")
        p.add_argument("--delta", type=float, default=2.0, help="logit bias for green tokens (kgw)")
        p.add_argument("--context-width", type=int, default=0, help="previous tokens hashed (0 = auto)")
        p.add_argument("--z", type=float, default=4.0, help="z-score threshold")
        p.add_argument("--min-tokens", type=int, default=25)
        p.add_argument("--no-dedupe", action="store_true", help="score repeated (context, token) pairs again")

    p = sub.add_parser("generate", help="generate text with the toy trigram LM (watermarked by default)")
    p.add_argument("--prompt", default="It was a dark and stormy night")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-watermark", action="store_true")
    p.add_argument("--out", help="write to file instead of stdout")
    wm_opts(p)
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("detect", help="z-score / p-value watermark test for a file, text or stdin")
    p.add_argument("text", nargs="?", help="file path, literal text, or - for stdin")
    p.add_argument("--tokens", action="store_true", help="print tokens coloured green/red")
    p.add_argument("--json", action="store_true")
    p.add_argument("--raw", action="store_true", help="skip zero-width / homoglyph normalisation")
    wm_opts(p)
    p.set_defaults(fn=cmd_detect)

    p = sub.add_parser("attack", help="apply an edit/evasion transformation (for robustness tests)")
    p.add_argument("type", choices=attacks.ATTACK_NAMES)
    p.add_argument("text", nargs="?")
    p.add_argument("--rate", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_attack)

    from .experiments import EXPERIMENTS
    p = sub.add_parser("experiment", help="run experiments -> CSV + PNG plots + summary.md")
    p.add_argument("names", nargs="+", choices=sorted(EXPERIMENTS) + ["all"])
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true", help="small sample sizes (smoke test)")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(fn=cmd_experiment)

    p = sub.add_parser("keygen", help="create an Ed25519 signing key pair")
    p.add_argument("--dir", default=str(HOME / "keys"))
    p.add_argument("--name", default="signer")
    p.add_argument("--trust", help="also copy the public key into this trust-list folder")
    p.set_defaults(fn=cmd_keygen)

    p = sub.add_parser("sign", help="attach a signed C2PA-style manifest to a text or image file")
    p.add_argument("file")
    p.add_argument("--key-file", default=str(HOME / "keys" / "signer.key"))
    p.add_argument("--title", default="")
    p.add_argument("--source", choices=sorted(provenance.SOURCE_TYPES), default="ai")
    p.add_argument("--generator", default="", help="software agent, e.g. 'wmlab trigram LM'")
    p.add_argument("--parent", help="manifest of the file this one was edited from")
    p.add_argument("--watermark-scheme", choices=SCHEMES, help="record that the text is watermarked")
    p.add_argument("--no-embed", action="store_true", help="PNG: write sidecar only")
    p.set_defaults(fn=cmd_sign)

    p = sub.add_parser("verify", help="verify a file against its manifest (sidecar or embedded PNG)")
    p.add_argument("file")
    p.add_argument("--manifest")
    p.add_argument("--trust", help="folder of trusted *.pub keys")
    p.add_argument("--parent", help="expected parent manifest (edit history check)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("web", help="local Flask UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5088)
    p.set_defaults(fn=cmd_web)

    p = sub.add_parser("demo", help="60-second offline tour")
    p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("status", help="show configuration (never prints secrets)")
    p.set_defaults(fn=cmd_status)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    forced = os.environ.get("FORCE_COLOR", "") not in ("", "0")
    if args.color and not forced and (not sys.stdout.isatty() or os.environ.get("NO_COLOR")):
        args.color = False
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
