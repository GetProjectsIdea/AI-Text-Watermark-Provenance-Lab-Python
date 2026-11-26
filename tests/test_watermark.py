import math

import pytest

from wmlab import corpus
from wmlab.watermark import (WatermarkConfig, detect, detect_ids, detect_tokens, gamma_sf_int, normal_isf,
                             normal_sf, theoretical_fpr)


def test_config_defaults():
    c = WatermarkConfig()
    assert c.scheme == "kgw" and c.context_width == 1


def test_gumbel_auto_context_width():
    assert WatermarkConfig(scheme="gumbel").context_width == 4


def test_explicit_context_width():
    assert WatermarkConfig(context_width=3).context_width == 3


@pytest.mark.parametrize("kw", [{"gamma": 0}, {"gamma": 1}, {"delta": -1}, {"scheme": "x"}, {"key": ""},
                                {"context_width": 9}, {"min_tokens": 0}])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        WatermarkConfig(**kw)


def test_public_dict_hides_key():
    assert "key" not in WatermarkConfig(key="secret").public_dict()


def test_normal_sf_values():
    assert normal_sf(0) == pytest.approx(0.5)
    assert normal_sf(4) == pytest.approx(3.167e-5, rel=1e-3)


def test_normal_isf_inverse():
    for p in (0.5, 0.05, 1e-3, 1e-8):
        assert normal_sf(normal_isf(p)) == pytest.approx(p, rel=1e-6)


def test_normal_isf_edges():
    assert normal_isf(0) == math.inf and normal_isf(1) == -math.inf


def test_gamma_sf_shape1_is_exponential():
    assert gamma_sf_int(1, 2.0) == pytest.approx(math.exp(-2.0))


def test_gamma_sf_mean():
    # median of Gamma(100,1) is ~99.67
    assert gamma_sf_int(100, 99.67) == pytest.approx(0.5, abs=0.01)


def test_gamma_sf_edges():
    assert gamma_sf_int(0, 5) == 1.0 and gamma_sf_int(5, 0) == 1.0


def test_theoretical_fpr():
    assert theoretical_fpr(4.0) == normal_sf(4.0)


def test_detects_watermarked(wm_text, cfg):
    d = detect(wm_text, cfg)
    assert d.detected and d.z > 6


def test_plain_not_detected(plain_text, cfg):
    d = detect(plain_text, cfg)
    assert not d.detected and d.z < 4


def test_wrong_key_not_detected(wm_text):
    assert not detect(wm_text, WatermarkConfig(key="another-key")).detected


def test_green_fraction_above_gamma(wm_text, cfg):
    assert detect(wm_text, cfg).green_fraction > 0.45


def test_short_text_verdict(cfg):
    d = detect("a short line of text", cfg)
    assert not d.detected and "too short" in d.verdict


def test_empty_text(cfg):
    d = detect("", cfg)
    assert d.num_scored == 0 and d.z == 0 and not d.detected


def test_flags_cover_tokens(wm_text, cfg):
    d = detect(wm_text, cfg)
    assert len(d.flags) == d.num_tokens
    assert d.flags[0].status == "context"


def test_flag_counts_match(wm_text, cfg):
    d = detect(wm_text, cfg)
    assert sum(f.status == "green" for f in d.flags) == d.num_green


def test_keep_flags_false(wm_text, cfg):
    assert detect(wm_text, cfg, keep_flags=False).flags == []


def test_to_dict(wm_text, cfg):
    d = detect(wm_text, cfg).to_dict(with_flags=True)
    assert {"z", "p_value", "verdict", "flags", "green_fraction"} <= set(d)


def test_dedupe_marks_repeats(cfg):
    toks = ("the cat sat on the mat . " * 10).split()
    d = detect_tokens(toks, cfg)
    assert any(f.status == "repeat" for f in d.flags)
    assert d.num_scored < len(toks) // 2


def test_no_dedupe_scores_everything():
    c = WatermarkConfig(key="k", dedupe=False)
    toks = ("the cat sat on the mat . " * 10).split()
    assert detect_tokens(toks, c).num_scored == len(toks) - 1


def test_repetition_inflates_z_without_dedupe():
    human = corpus.sample_paragraphs("human", 1, min_words=80, seed=2)[0]
    no = WatermarkConfig(key="fpr-key-0", dedupe=False)
    yes = WatermarkConfig(key="fpr-key-0", dedupe=True)
    z1 = detect(human, no).z
    z10 = detect(" ".join([human] * 10), no).z
    assert abs(z10) == pytest.approx(abs(z1) * math.sqrt(10), rel=0.2)
    assert detect(" ".join([human] * 10), yes).z == pytest.approx(detect(human, yes).z, abs=0.6)


def test_detect_ids_numeric_display():
    d = detect_ids([1, 2, 3, 4, 5], WatermarkConfig(key="k", min_tokens=1))
    assert d.num_scored == 4 and d.flags[1].token == "2"


def test_human_false_positive_rate_low():
    c = WatermarkConfig(key="fpr-test")
    paras = corpus.sample_paragraphs("human", 200, min_words=80, seed=9)
    zs = [detect(p, c, keep_flags=False).z for p in paras]
    assert sum(z >= 4 for z in zs) == 0
    assert abs(sum(zs) / len(zs)) < 0.35


def test_gumbel_roundtrip(lm):
    c = WatermarkConfig(key="g", scheme="gumbel")
    t = lm.generate("The morning was", 150, watermark=c, seed=3).text
    assert detect(t, c).detected
    assert not detect(t, WatermarkConfig(key="other", scheme="gumbel")).detected


def test_gumbel_plain_not_detected(plain_text):
    assert not detect(plain_text, WatermarkConfig(key="g", scheme="gumbel")).detected


def test_cross_scheme_does_not_detect(wm_text):
    # KGW text checked with a Gumbel detector (same key) must not be flagged
    assert not detect(wm_text, WatermarkConfig(key="test-key", scheme="gumbel")).detected


def test_zero_width_attack_defeated_by_normalisation(wm_text, cfg):
    from wmlab.attacks import zero_width
    att = zero_width(wm_text, 0.5)
    assert detect(att, cfg).detected
    assert not detect(att, cfg, clean=False).detected


def test_homoglyph_attack_defeated(wm_text, cfg):
    from wmlab.attacks import homoglyph
    att = homoglyph(wm_text, 0.5)
    assert detect(att, cfg).detected


def test_higher_threshold_reduces_detection(wm_text):
    lo = WatermarkConfig(key="test-key", z_threshold=2)
    hi = WatermarkConfig(key="test-key", z_threshold=100)
    assert detect(wm_text, lo).detected and not detect(wm_text, hi).detected


def test_p_threshold():
    assert WatermarkConfig(z_threshold=4).p_threshold == pytest.approx(3.17e-5, rel=1e-2)
