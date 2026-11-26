import json

import pytest

from wmlab.cli import main


def run(capsys, *args):
    code = main(["--no-color", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "wmlab" in capsys.readouterr().out


def test_generate_and_detect(tmp_path, capsys):
    f = tmp_path / "wm.txt"
    code, out, err = run(capsys, "generate", "--tokens", "120", "--seed", "1", "--out", str(f), "--key", "cli-key")
    assert code == 0 and f.exists()
    code, out, _ = run(capsys, "detect", str(f), "--key", "cli-key")
    assert code == 0 and "watermark detected" in out


def test_detect_wrong_key_exit_1(tmp_path, capsys):
    f = tmp_path / "wm.txt"
    run(capsys, "generate", "--tokens", "120", "--seed", "1", "--out", str(f), "--key", "cli-key")
    code, out, _ = run(capsys, "detect", str(f), "--key", "other")
    assert code == 1 and "no watermark" in out


def test_detect_json(tmp_path, capsys):
    f = tmp_path / "wm.txt"
    run(capsys, "generate", "--tokens", "80", "--seed", "1", "--out", str(f), "--key", "k")
    code, out, _ = run(capsys, "detect", str(f), "--key", "k", "--json", "--tokens")
    d = json.loads(out)
    assert d["detected"] and "flags" in d


def test_detect_literal_text(capsys):
    code, out, _ = run(capsys, "detect", "just a few words", "--key", "k")
    assert code == 1 and "too short" in out


def test_demo_key_warning(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("WMLAB_KEY", raising=False)
    code, out, err = run(capsys, "detect", "hello there")
    assert "demo key" in err


def test_env_key(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("WMLAB_KEY", "env-key")
    f = tmp_path / "wm.txt"
    run(capsys, "generate", "--tokens", "100", "--seed", "2", "--out", str(f))
    code, out, err = run(capsys, "detect", str(f))
    assert code == 0 and "demo key" not in err


def test_generate_stdout_no_watermark(capsys):
    code, out, _ = run(capsys, "generate", "--tokens", "30", "--seed", "1", "--no-watermark")
    assert code == 0 and len(out.split()) > 10


def test_attack(tmp_path, capsys):
    f = tmp_path / "wm.txt"
    run(capsys, "generate", "--tokens", "80", "--seed", "1", "--out", str(f), "--key", "k")
    code, out, _ = run(capsys, "attack", "substitute", str(f), "--rate", "0.3")
    assert code == 0 and out.strip() != f.read_text().strip()


def test_keygen_sign_verify(tmp_path, capsys):
    keys, trust = tmp_path / "keys", tmp_path / "trust"
    code, out, _ = run(capsys, "keygen", "--dir", str(keys), "--trust", str(trust))
    assert code == 0 and "key id" in out
    doc = tmp_path / "a.txt"
    doc.write_text("AI text")
    code, out, _ = run(capsys, "sign", str(doc), "--key-file", str(keys / "signer.key"), "--watermark-scheme", "kgw")
    assert code == 0
    code, out, _ = run(capsys, "verify", str(doc), "--trust", str(trust))
    assert code == 0 and "VALID" in out
    doc.write_text("AI text edited")
    code, out, _ = run(capsys, "verify", str(doc), "--json")
    assert code == 2 and json.loads(out)["status"] == "content_mismatch"


def test_sign_with_parent(tmp_path, capsys):
    keys = tmp_path / "keys"
    run(capsys, "keygen", "--dir", str(keys))
    v1 = tmp_path / "v1.txt"
    v1.write_text("draft")
    run(capsys, "sign", str(v1), "--key-file", str(keys / "signer.key"))
    v2 = tmp_path / "v2.txt"
    v2.write_text("final")
    code, _, _ = run(capsys, "sign", str(v2), "--key-file", str(keys / "signer.key"), "--parent", str(v1) + ".manifest.json")
    assert code == 0
    code, out, _ = run(capsys, "verify", str(v2), "--parent", str(v1) + ".manifest.json")
    assert code == 0 and "ingredient: OK" in out


def test_verify_no_manifest(tmp_path, capsys):
    f = tmp_path / "x.txt"
    f.write_text("x")
    code, out, _ = run(capsys, "verify", str(f))
    assert code == 2 and "NO_MANIFEST" in out


def test_keygen_twice_errors(tmp_path, capsys):
    run(capsys, "keygen", "--dir", str(tmp_path))
    code, _, err = run(capsys, "keygen", "--dir", str(tmp_path))
    assert code == 2 and "refusing" in err


def test_experiment_quick(tmp_path, capsys):
    code, out, _ = run(capsys, "experiment", "image_provenance", "--out", str(tmp_path), "--quick", "--no-plots")
    assert code == 0 and (tmp_path / "e6_image_provenance.csv").exists()


def test_demo(capsys):
    code, out, _ = run(capsys, "demo")
    assert code == 0 and "watermark detected" in out and "after substitute" in out


def test_status_hides_key(capsys, monkeypatch):
    monkeypatch.setenv("WMLAB_KEY", "super-secret-value")
    code, out, _ = run(capsys, "status")
    assert code == 0 and "super-secret-value" not in out and "hidden" in out


def test_web_refuses_public_host(capsys):
    code, _, err = run(capsys, "web", "--host", "0.0.0.0")
    assert code == 2 and "refusing" in err


def test_bad_gamma_error(capsys):
    code, _, err = run(capsys, "detect", "text", "--gamma", "2")
    assert code == 2 and "gamma" in err
