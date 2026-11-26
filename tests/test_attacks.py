import pytest

from wmlab import attacks
from wmlab.tokenize import tokenize
from wmlab.watermark import detect

TOKS = tokenize("The quick brown fox said that the old house was very dark , and the night was cold . "
                "She walked home slowly , because she was afraid of the strange light .")


def test_substitute_zero_is_identity():
    assert attacks.substitute(TOKS, 0.0) == TOKS


def test_substitute_changes_expected_count():
    out = attacks.substitute(TOKS, 0.5, seed=1)
    words = [i for i, t in enumerate(TOKS) if t[0].isalnum()]
    changed = sum(TOKS[i] != out[i] for i in words)
    assert changed >= len(words) * 0.4


def test_substitute_keeps_length():
    assert len(attacks.substitute(TOKS, 0.3)) == len(TOKS)


def test_substitute_uses_synonyms():
    out = attacks.substitute(["said"], 1.0, seed=0)
    assert out[0] in {"remarked", "replied", "answered", "observed"}


def test_substitute_rate_validation():
    with pytest.raises(ValueError):
        attacks.substitute(TOKS, 1.5)


def test_substitute_deterministic():
    assert attacks.substitute(TOKS, 0.3, seed=4) == attacks.substitute(TOKS, 0.3, seed=4)


def test_delete_shortens():
    assert len(attacks.delete(TOKS, 0.2)) == len(TOKS) - round(0.2 * len(TOKS))


def test_insert_lengthens():
    assert len(attacks.insert(TOKS, 0.5, seed=2)) > len(TOKS)


def test_truncate():
    assert attacks.truncate(TOKS, 5) == TOKS[:5]
    assert attacks.truncate(TOKS, -1) == []


def test_swap_keeps_multiset():
    out = attacks.swap(TOKS, 0.3, seed=1)
    assert sorted(out) == sorted(TOKS) and out != TOKS


def test_paraphrase_changes_text():
    out = attacks.paraphrase_toy(TOKS, 1.0, seed=1)
    assert out != TOKS


def test_paraphrase_zero_strength_identity():
    assert attacks.paraphrase_toy(TOKS, 0.0, seed=1) == TOKS


def test_mix_fraction():
    wm = ["w"] * 100
    human = ["h"] * 1000
    out = attacks.mix(wm, human, 0.25)
    assert out.count("w") == 100 and out.count("h") == 300


def test_mix_validation():
    with pytest.raises(ValueError):
        attacks.mix(["a"], ["b"], 0)


def test_zero_width_inserts():
    assert "\u200b" in attacks.zero_width("hello world", 1.0)


def test_homoglyph_changes():
    assert attacks.homoglyph("aaaa eeee", 1.0) != "aaaa eeee"


@pytest.mark.parametrize("name", attacks.ATTACK_NAMES)
def test_apply_all(name, wm_text):
    out = attacks.apply(name, wm_text, 0.3, seed=1)
    assert isinstance(out, str) and out


def test_apply_unknown():
    with pytest.raises(ValueError):
        attacks.apply("nope", "text", 0.1)


def test_light_substitution_survives(wm_text, cfg):
    assert detect(attacks.apply("substitute", wm_text, 0.1, seed=1), cfg).detected


def test_heavy_substitution_lowers_z(wm_text, cfg):
    z0 = detect(wm_text, cfg).z
    z1 = detect(attacks.apply("substitute", wm_text, 0.6, seed=1), cfg).z
    assert z1 < z0 * 0.6


def test_truncation_lowers_z(wm_text, cfg):
    assert detect(attacks.apply("truncate", wm_text, 0.8), cfg).z < detect(wm_text, cfg).z
