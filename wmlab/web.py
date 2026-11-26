"""Local Flask UI: Generate, Detect, Attack and Provenance tabs (127.0.0.1 only)."""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request

from . import __version__, attacks, provenance
from .watermark import DEMO_KEY, SCHEMES, WatermarkConfig, detect

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
MAX_TEXT = 20_000
MAX_TOKENS = 400


def hostname_of(host_header: str) -> str:
    h = (host_header or "").strip().lower()
    if h.startswith("["):
        return h[: h.find("]") + 1] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def _web_key(home: Path):
    """Ed25519 key used by the UI's Provenance tab (created on first use, mode 600)."""
    d = home / "web-keys"
    if not (d / "web.key").exists():
        provenance.generate_keypair(d, "web")
    return provenance.load_private_key(d / "web.key"), d


def create_app(key: str = DEMO_KEY, home: str | Path | None = None) -> Flask:
    home = Path(home or Path.home() / ".wmlab")
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256_000

    @app.before_request
    def guard():
        if hostname_of(request.host) not in LOCAL_HOSTS and not app.config.get("TESTING_ANY_HOST"):
            abort(403)  # blocks DNS rebinding
        if request.method == "POST" and request.headers.get("X-Wmlab") != "1":
            abort(403)  # CSRF guard: other sites cannot add custom headers without CORS

    @app.after_request
    def security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        return resp

    @app.errorhandler(ValueError)
    def value_error(exc):
        return jsonify(error=str(exc)), 400

    def body() -> dict:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError("send a JSON object")
        for k, v in data.items():
            if isinstance(v, str) and len(v) > MAX_TEXT:
                raise ValueError(f"field {k!r} is too long (max {MAX_TEXT} characters)")
        return data

    def cfg_from(data: dict) -> WatermarkConfig:
        scheme = data.get("scheme", "kgw")
        if scheme not in SCHEMES:
            raise ValueError("scheme must be kgw or gumbel")
        return WatermarkConfig(key=key, scheme=scheme, gamma=float(data.get("gamma", 0.25)),
                               delta=float(data.get("delta", 2.0)), context_width=int(data.get("context_width", 0)),
                               z_threshold=float(data.get("z", 4.0)))

    def text_of(data: dict, field: str = "text") -> str:
        t = data.get(field)
        if not isinstance(t, str) or not t.strip():
            raise ValueError(f"{field} is empty")
        return t

    @app.get("/")
    def index():
        return render_template("index.html", version=__version__, demo_key=key == DEMO_KEY,
                               attacks=attacks.ATTACK_NAMES)

    @app.get("/api/status")
    def status():
        return jsonify(version=__version__, demo_key=key == DEMO_KEY, schemes=list(SCHEMES))

    @app.post("/api/generate")
    def generate():
        from .lm import default_lm
        data = body()
        n = int(data.get("tokens", 150))
        if not 1 <= n <= MAX_TOKENS:
            raise ValueError(f"tokens must be 1..{MAX_TOKENS}")
        cfg = cfg_from(data) if data.get("watermark", True) else None
        seed = data.get("seed")
        g = default_lm().generate(str(data.get("prompt") or "It was a dark and stormy night")[:500], n,
                                  watermark=cfg, seed=int(seed) if seed not in (None, "") else None)
        det = detect(g.text, cfg or cfg_from(data))
        return jsonify(text=g.text, watermarked=cfg is not None, detection=det.to_dict(with_flags=True))

    @app.post("/api/detect")
    def detect_api():
        data = body()
        det = detect(text_of(data), cfg_from(data), clean=not data.get("raw"))
        return jsonify(detection=det.to_dict(with_flags=True))

    @app.post("/api/attack")
    def attack_api():
        data = body()
        text = text_of(data)
        kind = data.get("type", "substitute")
        if kind not in attacks.ATTACK_NAMES:
            raise ValueError("unknown attack")
        rate = float(data.get("rate", 0.2))
        if not 0 <= rate <= 1:
            raise ValueError("rate must be 0..1")
        cfg = cfg_from(data)
        attacked = attacks.apply(kind, text, rate, seed=int(data.get("seed", 0)))
        before = detect(text, cfg)
        after = detect(attacked, cfg)
        after_raw = detect(attacked, cfg, clean=False)
        return jsonify(text=attacked, before=before.to_dict(), after=after.to_dict(with_flags=True),
                       after_without_normalisation=after_raw.to_dict())

    @app.post("/api/sign")
    def sign_api():
        data = body()
        text = text_of(data)
        source = data.get("source", "ai")
        if source not in provenance.SOURCE_TYPES:
            raise ValueError("unknown source type")
        priv, _ = _web_key(home)
        wm = None
        if data.get("watermarked"):
            wm = {"scheme": data.get("scheme", "kgw"), "note": "statistical watermark; key held by the provider"}
        m = provenance.build_manifest("text", content_hash=provenance.text_hash(text),
                                      title=str(data.get("title", ""))[:200], source=source,
                                      generator=f"wmlab web {__version__}", watermark=wm)
        return jsonify(manifest=provenance.sign_manifest(m, priv))

    @app.post("/api/verify")
    def verify_api():
        data = body()
        text = text_of(data)
        raw = data.get("manifest")
        manifest = json.loads(raw) if isinstance(raw, str) else raw
        if manifest is not None and not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        _, keydir = _web_key(home)
        v = provenance.verify_manifest(manifest, text, trust=provenance.load_trust_list(keydir))
        return jsonify(verification=v.to_dict())

    return app
