import json
import os
import stat

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image

from wmlab import provenance as pv


@pytest.fixture
def key():
    return Ed25519PrivateKey.generate()


@pytest.fixture
def img(tmp_path):
    from wmlab.experiments import _demo_image
    p = tmp_path / "pic.png"
    _demo_image(3).save(p)
    return p


def test_keygen_files(tmp_path):
    priv, pub = pv.generate_keypair(tmp_path, "me")
    assert priv.exists() and pub.exists()
    assert stat.S_IMODE(os.stat(priv).st_mode) == 0o600


def test_keygen_refuses_overwrite(tmp_path):
    pv.generate_keypair(tmp_path, "me")
    with pytest.raises(FileExistsError):
        pv.generate_keypair(tmp_path, "me")


def test_load_keys_roundtrip(tmp_path):
    priv, pub = pv.generate_keypair(tmp_path)
    k = pv.load_private_key(priv)
    assert pv.key_id(k.public_key()) == pv.key_id(pv.load_public_key(pub))


def test_public_key_str_roundtrip(key):
    s = pv.public_key_str(key.public_key())
    assert pv.key_id(pv.load_public_key(s)) == pv.key_id(key.public_key())


def test_key_id_format(key):
    kid = pv.key_id(key.public_key())
    assert len(kid) == 16 and int(kid, 16) >= 0


def test_canonical_sorted():
    assert pv.canonical({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_text_hash_normalises_newlines_and_trailing_spaces():
    assert pv.text_hash("a  \r\nb\n") == pv.text_hash("a\nb")


def test_text_hash_nfc():
    assert pv.text_hash("cafe\u0301") == pv.text_hash("caf\u00e9")


def test_text_hash_detects_change():
    assert pv.text_hash("hello world") != pv.text_hash("hello world!")


def test_build_manifest_fields():
    m = pv.build_manifest("text", content_hash="ab", title="t", source="ai", generator="g")
    assert m["spec"] == pv.SPEC and m["assertions"][0]["label"] == "c2pa.actions"
    assert m["assertions"][0]["data"]["actions"][0]["digitalSourceType"].endswith("trainedAlgorithmicMedia")


def test_build_manifest_validation():
    with pytest.raises(ValueError):
        pv.build_manifest("video", content_hash="x")
    with pytest.raises(ValueError):
        pv.build_manifest("text", content_hash="x", source="alien")


def test_watermark_assertion():
    m = pv.build_manifest("text", content_hash="x", watermark={"scheme": "kgw"})
    assert any(a["label"] == "wmlab.watermark" for a in m["assertions"])


def _signed_text(key, text="An AI wrote this paragraph."):
    return pv.sign_manifest(pv.build_manifest("text", content_hash=pv.text_hash(text)), key), text


def test_sign_and_verify_text(key):
    m, t = _signed_text(key)
    v = pv.verify_manifest(m, t)
    assert v.ok and v.status == "valid" and v.source_type == "trainedAlgorithmicMedia"


def test_verify_detects_text_edit(key):
    m, t = _signed_text(key)
    v = pv.verify_manifest(m, t.replace("AI", "human"))
    assert v.status == "content_mismatch" and not v.ok


def test_verify_detects_manifest_tamper(key):
    m, t = _signed_text(key)
    m["assertions"][0]["data"]["actions"][0]["digitalSourceType"] = pv.SOURCE_TYPES["human"]
    assert pv.verify_manifest(m, t).status == "bad_signature"


def test_verify_detects_swapped_public_key(key):
    m, t = _signed_text(key)
    other = Ed25519PrivateKey.generate().public_key()
    m["signature"]["public_key"] = pv.public_key_str(other)
    assert pv.verify_manifest(m, t).status == "bad_signature"


def test_resign_by_attacker_is_untrusted(key):
    m, t = _signed_text(key)
    attacker = Ed25519PrivateKey.generate()
    forged = pv.sign_manifest(pv.build_manifest("text", content_hash=pv.text_hash(t), source="human"), attacker)
    trust = {pv.key_id(key.public_key()): key.public_key()}
    v = pv.verify_manifest(forged, t, trust=trust)
    assert v.status == "valid_untrusted" and not v.trusted


def test_trusted_signer(key):
    m, t = _signed_text(key)
    v = pv.verify_manifest(m, t, trust={pv.key_id(key.public_key()): key.public_key()})
    assert v.trusted and v.status == "valid"


def test_no_manifest():
    assert pv.verify_manifest(None, "x").status == "no_manifest"


def test_malformed_manifest():
    assert pv.verify_manifest({"spec": "x"}, "x").status == "malformed"


def test_wrong_spec_malformed(key):
    m, t = _signed_text(key)
    m["spec"] = "other/9"
    assert pv.verify_manifest(m, t).status == "malformed"


def test_bad_signature_base64(key):
    m, t = _signed_text(key)
    m["signature"]["value"] = "AAAA"
    assert pv.verify_manifest(m, t).status == "bad_signature"


def test_parent_ingredient(key):
    parent, _ = _signed_text(key, "v1 text")
    child = pv.sign_manifest(pv.build_manifest("text", content_hash=pv.text_hash("v2 text"), action="c2pa.edited",
                                               parent=parent), key)
    assert pv.verify_manifest(child, "v2 text", parent=parent).ok
    other, _ = _signed_text(key, "unrelated")
    assert not pv.verify_manifest(child, "v2 text", parent=other).ok


def test_pixel_hash_ignores_metadata(img, tmp_path, key):
    h = pv.pixel_hash(img)
    m = pv.sign_manifest(pv.build_manifest("image", content_hash=h), key)
    pv.embed_png_manifest(img, m)
    assert pv.pixel_hash(img) == h


def test_pixel_hash_changes_with_pixels(img, tmp_path):
    im = Image.open(img).convert("RGB")
    im.putpixel((0, 0), (1, 2, 3))
    p2 = tmp_path / "b.png"
    im.save(p2)
    assert pv.pixel_hash(p2) != pv.pixel_hash(img)


def test_dhash_stable_under_jpeg(img, tmp_path):
    j = tmp_path / "pic.jpg"
    Image.open(img).convert("RGB").save(j, quality=70)
    assert pv.hamming(pv.dhash(img), pv.dhash(j)) <= pv.SOFT_MATCH_BITS


def test_dhash_differs_for_other_image(img, tmp_path):
    from wmlab.experiments import _demo_image
    o = tmp_path / "other.png"
    _demo_image(99).save(o)
    assert pv.hamming(pv.dhash(img), pv.dhash(o)) > pv.SOFT_MATCH_BITS


def test_hamming():
    assert pv.hamming("ff", "00") == 8 and pv.hamming("abc", "abc") == 0


def test_sign_file_text(tmp_path, key):
    f = tmp_path / "essay.txt"
    f.write_text("Generated by a model.\n")
    m, side = pv.sign_file(f, key)
    assert side.exists() and pv.verify_file(f).ok


def test_sign_file_text_tamper(tmp_path, key):
    f = tmp_path / "essay.txt"
    f.write_text("Generated by a model.\n")
    pv.sign_file(f, key)
    f.write_text("Written by me.\n")
    assert pv.verify_file(f).status == "content_mismatch"


def test_sign_png_embeds(img, key, tmp_path):
    pv.sign_file(img, key)
    assert pv.read_png_manifest(img) is not None
    pv.sidecar_path(img).unlink()
    assert pv.verify_file(img).ok  # falls back to the embedded manifest


def test_jpeg_copy_soft_binding(img, key, tmp_path):
    m, _ = pv.sign_file(img, key)
    j = tmp_path / "copy.jpg"
    Image.open(img).convert("RGB").save(j, quality=80)
    v = pv.verify_manifest(m, j)
    assert v.status == "content_mismatch" and v.soft_distance is not None and v.soft_distance <= pv.SOFT_MATCH_BITS
    assert any("re-encoded" in c for c in v.checks)


def test_verify_file_with_trust_dir(tmp_path):
    priv, pub = pv.generate_keypair(tmp_path / "keys")
    trust = tmp_path / "trust"
    trust.mkdir()
    (trust / pub.name).write_bytes(pub.read_bytes())
    f = tmp_path / "a.txt"
    f.write_text("hello")
    pv.sign_file(f, pv.load_private_key(priv))
    assert pv.verify_file(f, trust_dir=trust).trusted


def test_load_trust_list_skips_junk(tmp_path):
    (tmp_path / "junk.pub").write_text("not a key")
    assert pv.load_trust_list(tmp_path) == {}
    assert pv.load_trust_list(tmp_path / "missing") == {}


def test_bytes_manifest(tmp_path, key):
    f = tmp_path / "data.bin"
    f.write_bytes(b"\x00\x01")
    m = pv.sign_manifest(pv.build_manifest("bytes", content_hash=pv.bytes_hash(b"\x00\x01")), key)
    assert pv.verify_manifest(m, f).ok


def test_manifest_json_serialisable(key):
    m, _ = _signed_text(key)
    assert json.loads(json.dumps(m)) == m


def test_verification_to_dict(key):
    m, t = _signed_text(key)
    assert pv.verify_manifest(m, t).to_dict()["status"] == "valid"
