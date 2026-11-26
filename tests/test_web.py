import pytest

from wmlab.web import create_app, hostname_of

H = {"X-Wmlab": "1"}


@pytest.fixture
def client(tmp_path):
    app = create_app(key="web-test-key", home=tmp_path)
    return app.test_client()


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200 and b"wmlab" in r.data


def test_security_headers(client):
    r = client.get("/")
    assert "script-src 'self'" in r.headers["Content-Security-Policy"]
    assert r.headers["X-Frame-Options"] == "DENY"


def test_no_inline_styles_in_template(client):
    assert b"style=" not in client.get("/").data


def test_host_check(client):
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 403


def test_localhost_with_port_ok(client):
    assert client.get("/", headers={"Host": "localhost:5088"}).status_code == 200


def test_csrf_header_required(client):
    assert client.post("/api/detect", json={"text": "x"}).status_code == 403


def test_status(client):
    j = client.get("/api/status").get_json()
    assert j["demo_key"] is False and "kgw" in j["schemes"]


def test_generate_and_detect(client):
    j = client.post("/api/generate", json={"tokens": 120, "seed": 3}, headers=H).get_json()
    assert j["watermarked"] and j["detection"]["detected"]
    d = client.post("/api/detect", json={"text": j["text"]}, headers=H).get_json()["detection"]
    assert d["detected"] and d["flags"]


def test_generate_without_watermark(client):
    j = client.post("/api/generate", json={"tokens": 120, "seed": 3, "watermark": False}, headers=H).get_json()
    assert not j["watermarked"] and not j["detection"]["detected"]


def test_generate_gumbel(client):
    j = client.post("/api/generate", json={"tokens": 120, "seed": 3, "scheme": "gumbel"}, headers=H).get_json()
    assert j["detection"]["scheme"] == "gumbel" and j["detection"]["detected"]


def test_generate_token_limit(client):
    assert client.post("/api/generate", json={"tokens": 5000}, headers=H).status_code == 400


def test_bad_scheme(client):
    assert client.post("/api/detect", json={"text": "x y z", "scheme": "nope"}, headers=H).status_code == 400


def test_empty_text(client):
    assert client.post("/api/detect", json={"text": "  "}, headers=H).status_code == 400


def test_not_json(client):
    assert client.post("/api/detect", data="x", headers=H).status_code == 400


def test_text_too_long(client):
    assert client.post("/api/detect", json={"text": "a " * 15000}, headers=H).status_code == 400


def test_attack_endpoint(client):
    t = client.post("/api/generate", json={"tokens": 150, "seed": 3}, headers=H).get_json()["text"]
    j = client.post("/api/attack", json={"text": t, "type": "zero_width", "rate": 0.5}, headers=H).get_json()
    assert j["after"]["detected"] and not j["after_without_normalisation"]["detected"]


def test_attack_bad_rate(client):
    assert client.post("/api/attack", json={"text": "a b c", "rate": 3}, headers=H).status_code == 400


def test_attack_unknown(client):
    assert client.post("/api/attack", json={"text": "a b c", "type": "x"}, headers=H).status_code == 400


def test_sign_verify_tamper(client):
    text = "This paragraph was produced by a language model."
    m = client.post("/api/sign", json={"text": text, "watermarked": True}, headers=H).get_json()["manifest"]
    v = client.post("/api/verify", json={"text": text, "manifest": m}, headers=H).get_json()["verification"]
    assert v["status"] == "valid" and v["trusted"]
    v = client.post("/api/verify", json={"text": text + " Edited.", "manifest": m}, headers=H).get_json()["verification"]
    assert v["status"] == "content_mismatch"


def test_verify_manifest_as_string(client):
    import json
    text = "abc def"
    m = client.post("/api/sign", json={"text": text}, headers=H).get_json()["manifest"]
    v = client.post("/api/verify", json={"text": text, "manifest": json.dumps(m)}, headers=H).get_json()
    assert v["verification"]["ok"]


def test_verify_without_manifest(client):
    v = client.post("/api/verify", json={"text": "abc", "manifest": None}, headers=H).get_json()["verification"]
    assert v["status"] == "no_manifest"


def test_sign_bad_source(client):
    assert client.post("/api/sign", json={"text": "a", "source": "x"}, headers=H).status_code == 400


def test_key_never_returned(client):
    j = client.post("/api/generate", json={"tokens": 30, "seed": 1}, headers=H).get_data(as_text=True)
    assert "web-test-key" not in j


@pytest.mark.parametrize("h,want", [("localhost:5088", "localhost"), ("[::1]:5088", "[::1]"), ("127.0.0.1", "127.0.0.1")])
def test_hostname_of(h, want):
    assert hostname_of(h) == want
