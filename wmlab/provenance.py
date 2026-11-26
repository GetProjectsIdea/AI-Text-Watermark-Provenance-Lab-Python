"""C2PA-style signed provenance manifests for text and images (teaching implementation).

This is NOT a conformant C2PA implementation: real Content Credentials use JUMBF boxes, COSE
signatures and X.509 certificates from the C2PA trust list (use the official ``c2pa`` SDKs for
that). It keeps the same *ideas* in ~300 readable lines so students can experiment with them:

* a **claim** with assertions (``c2pa.actions`` with an IPTC ``digitalSourceType``, optional
  watermark info, ingredients for edit history),
* a **hard binding**: SHA-256 of normalised text, or of decoded image pixels (so stripping
  metadata does not break it, but any pixel change does),
* a **soft binding** for images: a 64-bit difference hash that survives re-encoding and resizing,
* an **Ed25519 signature** over canonical JSON, and a local **trust list** of public keys.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from . import __version__

SPEC = "wmlab-manifest/1"
PNG_KEY = "wmlab.manifest"
IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"
SOURCE_TYPES = {
    "ai": IPTC + "trainedAlgorithmicMedia",
    "ai-composite": IPTC + "compositeWithTrainedAlgorithmicMedia",
    "human": IPTC + "humanEdits",
    "capture": IPTC + "digitalCapture",
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
SOFT_MATCH_BITS = 10  # dHash Hamming distance at or below this = "probably the same picture"


# ------------------------------------------------------------------------------------- keys

def key_id(pub: Ed25519PublicKey) -> str:
    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def generate_keypair(directory: str | Path, name: str = "signer") -> tuple[Path, Path]:
    """Write ``name.key`` (PKCS#8 PEM, mode 0600) and ``name.pub`` into ``directory``."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    priv_path, pub_path = d / f"{name}.key", d / f"{name}.pub"
    if priv_path.exists():
        raise FileExistsError(f"{priv_path} already exists; refusing to overwrite a private key")
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())
    fd = os.open(priv_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)
    pub_path.write_bytes(priv.public_key().public_bytes(serialization.Encoding.PEM,
                                                        serialization.PublicFormat.SubjectPublicKeyInfo))
    return priv_path, pub_path


def load_private_key(path: str | Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("only Ed25519 private keys are supported")
    return key


def load_public_key(data: str | bytes | Path) -> Ed25519PublicKey:
    if isinstance(data, Path) or (isinstance(data, str) and not data.startswith(("-----", "ed25519:"))):
        data = Path(data).read_bytes()
    if isinstance(data, str) and data.startswith("ed25519:"):
        return Ed25519PublicKey.from_public_bytes(base64.b64decode(data[8:]))
    if isinstance(data, str):
        data = data.encode()
    key = serialization.load_pem_public_key(data)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("only Ed25519 public keys are supported")
    return key


def public_key_str(pub: Ed25519PublicKey) -> str:
    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return "ed25519:" + base64.b64encode(raw).decode()


def load_trust_list(directory: str | Path | None) -> dict[str, Ed25519PublicKey]:
    """All ``*.pub`` files in a folder, keyed by key id."""
    out: dict[str, Ed25519PublicKey] = {}
    if not directory or not Path(directory).is_dir():
        return out
    for p in sorted(Path(directory).glob("*.pub")):
        try:
            k = load_public_key(p)
        except Exception:  # noqa: BLE001 - skip unreadable files, report nothing secret
            continue
        out[key_id(k)] = k
    return out


# ------------------------------------------------------------------------------------- hashing

def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _open_image(src):
    from PIL import Image
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src))
    if isinstance(src, (str, Path)):
        return Image.open(src)
    return src


def pixel_hash(src) -> str:
    """SHA-256 over size + RGBA pixels: metadata-independent hard binding."""
    img = _open_image(src).convert("RGBA")
    h = hashlib.sha256(f"{img.width}x{img.height}:RGBA:".encode())
    h.update(img.tobytes())
    return h.hexdigest()


def dhash(src, size: int = 8) -> str:
    """64-bit difference hash (soft binding) as 16 hex chars."""
    from PIL import Image
    img = _open_image(src).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(img.tobytes())  # mode L: one byte per pixel
    bits = 0
    for row in range(size):
        for col in range(size):
            left, right = px[row * (size + 1) + col], px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return f"{bits:0{size * size // 4}x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------- manifests

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_manifest(kind: str, *, content_hash: str, title: str = "", source: str = "ai",
                   generator: str = "", action: str = "c2pa.created", soft_hash: str | None = None,
                   watermark: dict | None = None, parent: dict | None = None,
                   created: str | None = None) -> dict:
    if kind not in ("text", "image", "bytes"):
        raise ValueError("kind must be text, image or bytes")
    if source not in SOURCE_TYPES:
        raise ValueError(f"source must be one of {sorted(SOURCE_TYPES)}")
    act = {"action": action, "digitalSourceType": SOURCE_TYPES[source], "when": created or now_iso()}
    if generator:
        act["softwareAgent"] = generator
    assertions: list[dict] = [{"label": "c2pa.actions", "data": {"actions": [act]}}]
    if watermark:
        assertions.append({"label": "wmlab.watermark", "data": watermark})
    binding = {"text": "sha256-nfc-text", "image": "sha256-rgba-pixels", "bytes": "sha256-bytes"}[kind]
    m: dict[str, Any] = {
        "spec": SPEC,
        "claim_generator": f"wmlab/{__version__}",
        "title": title,
        "format": kind,
        "assertions": assertions,
        "hard_binding": {"alg": binding, "hash": content_hash},
        "ingredients": [],
    }
    if soft_hash:
        m["soft_binding"] = {"alg": "dhash64", "value": soft_hash}
    if parent:
        m["ingredients"].append({
            "relationship": "parentOf",
            "title": parent.get("title", ""),
            "manifest_sha256": hashlib.sha256(canonical(parent)).hexdigest(),
        })
    return m


def sign_manifest(manifest: dict, private_key: Ed25519PrivateKey) -> dict:
    body = {k: v for k, v in manifest.items() if k != "signature"}
    sig = private_key.sign(canonical(body))
    pub = private_key.public_key()
    return {**body, "signature": {"alg": "Ed25519", "key_id": key_id(pub), "public_key": public_key_str(pub),
                                  "value": base64.b64encode(sig).decode()}}


@dataclass
class Verification:
    status: str  # valid | valid_untrusted | content_mismatch | bad_signature | no_manifest | malformed
    ok: bool
    signer: str | None = None
    trusted: bool = False
    checks: list[str] = field(default_factory=list)
    soft_distance: int | None = None
    source_type: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _content_hash_for(manifest: dict, content) -> str:
    kind = manifest.get("format")
    if kind == "text":
        return text_hash(content if isinstance(content, str) else content.decode("utf-8"))
    if kind == "image":
        return pixel_hash(content)
    if isinstance(content, (str, Path)):
        content = Path(content).read_bytes()
    return bytes_hash(content)


def verify_manifest(manifest: dict | None, content, *, trust: dict[str, Ed25519PublicKey] | None = None,
                    parent: dict | None = None) -> Verification:
    """Check signature, hard binding, (optional) soft binding, trust list and parent link."""
    if not manifest:
        return Verification("no_manifest", False, checks=["no provenance manifest found"])
    try:
        sig = manifest["signature"]
        pub = load_public_key(sig["public_key"])
        raw_sig = base64.b64decode(sig["value"])
        body = {k: v for k, v in manifest.items() if k != "signature"}
        expected = manifest["hard_binding"]["hash"]
        if manifest.get("spec") != SPEC:
            raise ValueError("unknown spec")
    except Exception as e:  # noqa: BLE001
        return Verification("malformed", False, checks=[f"manifest is malformed: {type(e).__name__}"])
    v = Verification("valid", True, signer=key_id(pub))
    actions = next((a["data"]["actions"] for a in manifest.get("assertions", []) if a.get("label") == "c2pa.actions"), [])
    if actions:
        v.source_type = actions[-1].get("digitalSourceType", "").rsplit("/", 1)[-1] or None
    if sig.get("key_id") != v.signer:
        v.checks.append("key_id does not match embedded public key")
        v.status, v.ok = "bad_signature", False
        return v
    try:
        pub.verify(raw_sig, canonical(body))
        v.checks.append("signature: OK (Ed25519 over canonical JSON)")
    except InvalidSignature:
        v.checks.append("signature: INVALID - the manifest was changed after signing")
        v.status, v.ok = "bad_signature", False
        return v
    actual = _content_hash_for(manifest, content)
    if actual == expected:
        v.checks.append("hard binding: OK (content hash matches)")
    else:
        v.checks.append("hard binding: MISMATCH - the content was modified after signing")
        v.status, v.ok = "content_mismatch", False
        soft = manifest.get("soft_binding")
        if soft and manifest.get("format") == "image":
            try:
                v.soft_distance = hamming(soft["value"], dhash(content))
                same = v.soft_distance <= SOFT_MATCH_BITS
                v.checks.append(f"soft binding: dHash distance {v.soft_distance}/64 - "
                                + ("looks like a re-encoded/resized copy of the signed image" if same
                                   else "looks like a different image"))
            except Exception:  # noqa: BLE001
                v.checks.append("soft binding: could not compute")
    if parent is not None:
        want = hashlib.sha256(canonical(parent)).hexdigest()
        got = [i.get("manifest_sha256") for i in manifest.get("ingredients", [])]
        if want in got:
            v.checks.append("ingredient: OK (parent manifest is referenced)")
        else:
            v.checks.append("ingredient: parent manifest NOT referenced")
            v.ok = False
            if v.status == "valid":
                v.status = "content_mismatch"
    if trust is not None:
        v.trusted = v.signer in trust
        if v.trusted:
            v.checks.append(f"signer {v.signer}: trusted")
        else:
            v.checks.append(f"signer {v.signer}: NOT in trust list")
            if v.status == "valid":
                v.status = "valid_untrusted"
    return v


# ------------------------------------------------------------------------------------- files

def sidecar_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".manifest.json")


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXT


def sign_file(path: str | Path, private_key: Ed25519PrivateKey, *, title: str = "", source: str = "ai",
              generator: str = "", action: str = "c2pa.created", watermark: dict | None = None,
              parent: dict | None = None, embed_png: bool = True) -> tuple[dict, Path]:
    """Sign a text or image file. Writes a sidecar manifest; PNGs also get it embedded (iTXt)."""
    p = Path(path)
    if is_image(p):
        manifest = build_manifest("image", content_hash=pixel_hash(p), soft_hash=dhash(p), title=title or p.name,
                                  source=source, generator=generator, action=action, watermark=watermark,
                                  parent=parent)
    else:
        text = p.read_text(encoding="utf-8")
        manifest = build_manifest("text", content_hash=text_hash(text), title=title or p.name, source=source,
                                  generator=generator, action=action, watermark=watermark, parent=parent)
    signed = sign_manifest(manifest, private_key)
    out = sidecar_path(p)
    out.write_text(json.dumps(signed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if embed_png and p.suffix.lower() == ".png":
        embed_png_manifest(p, signed)
    return signed, out


def embed_png_manifest(path: str | Path, manifest: dict, out: str | Path | None = None) -> Path:
    from PIL import Image, PngImagePlugin
    img = Image.open(path)
    info = PngImagePlugin.PngInfo()
    for k, v in (img.text if hasattr(img, "text") else {}).items():
        if k != PNG_KEY:
            info.add_text(k, v)
    info.add_itxt(PNG_KEY, json.dumps(manifest, separators=(",", ":"), ensure_ascii=False))
    target = Path(out or path)
    img.save(target, pnginfo=info)
    return target


def read_png_manifest(path: str | Path) -> dict | None:
    from PIL import Image
    try:
        img = Image.open(path)
        raw = getattr(img, "text", {}).get(PNG_KEY)
    except Exception:  # noqa: BLE001
        return None
    return json.loads(raw) if raw else None


def find_manifest(path: str | Path, manifest_path: str | Path | None = None) -> dict | None:
    if manifest_path:
        return json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    side = sidecar_path(path)
    if side.exists():
        return json.loads(side.read_text(encoding="utf-8"))
    if Path(path).suffix.lower() == ".png":
        return read_png_manifest(path)
    return None


def verify_file(path: str | Path, *, manifest_path: str | Path | None = None, trust_dir: str | Path | None = None,
                parent_path: str | Path | None = None) -> Verification:
    p = Path(path)
    manifest = find_manifest(p, manifest_path)
    content: Any = p if (manifest and manifest.get("format") in ("image", "bytes")) else None
    if manifest and manifest.get("format") == "text":
        content = p.read_text(encoding="utf-8")
    parent = json.loads(Path(parent_path).read_text(encoding="utf-8")) if parent_path else None
    trust = load_trust_list(trust_dir) if trust_dir else None
    return verify_manifest(manifest, content, trust=trust, parent=parent)
