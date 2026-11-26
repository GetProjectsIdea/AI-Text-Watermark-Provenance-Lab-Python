import math

import numpy as np
import pytest

from wmlab.lm import NGramLM, truecase_map
from wmlab.tokenize import tokenize
from wmlab.watermark import WatermarkConfig, detect

TOY = "the cat sat on the mat . the dog sat on the log . the cat ate the fish ."


@pytest.fixture
def toy():
    return NGramLM.from_text(TOY, top_unigrams=5)


def test_vocab(toy):
    assert "cat" in toy.vocab and len(toy.vocab) == len(set(tokenize(TOY)))


def test_distribution_sums_to_one(toy):
    cand, p = toy.distribution(toy.index["the"], toy.index["cat"])
    assert p.sum() == pytest.approx(1.0) and (p > 0).all()


def test_distribution_prefers_seen(toy):
    cand, p = toy.distribution(toy.index["sat"], toy.index["on"])
    best = toy.vocab[cand[int(np.argmax(p))]]
    assert best == "the"


def test_distribution_cached(toy):
    a = toy.distribution(1, 2)
    assert toy.distribution(1, 2) is a


def test_prob_unknown(toy):
    assert 0 < toy.prob(-1, -1, -1) < 0.1


def test_prob_seen_higher_than_unseen(toy):
    i = toy.index
    assert toy.prob(i["sat"], i["on"], i["the"]) > toy.prob(i["sat"], i["on"], i["fish"])


def test_perplexity_finite(toy):
    ppl = toy.perplexity(tokenize("the cat sat on the mat ."))
    assert 1 < ppl < 50


def test_perplexity_empty(toy):
    assert math.isnan(toy.perplexity([]))


def test_entropy_positive(toy):
    assert toy.entropy(toy.index["the"], toy.index["cat"]) > 0


def test_generate_length(lm):
    g = lm.generate("Once upon a time", 50, seed=1)
    assert len(g.tokens) == 50


def test_generate_reproducible(lm):
    a = lm.generate("Once upon a time", 40, seed=5).tokens
    b = lm.generate("Once upon a time", 40, seed=5).tokens
    assert a == b


def test_generate_seed_changes_output(lm):
    assert lm.generate("Once", 40, seed=1).tokens != lm.generate("Once", 40, seed=2).tokens


def test_generate_unknown_prompt_ok(lm):
    assert len(lm.generate("zzqx qqzz", 10, seed=1).tokens) == 10


def test_generate_empty_prompt_ok(lm):
    assert len(lm.generate("", 10, seed=1).tokens) == 10


@pytest.mark.parametrize("bad", [{"n_tokens": 0}, {"temperature": 0}])
def test_generate_validation(lm, bad):
    kw = {"n_tokens": 10, **bad}
    with pytest.raises(ValueError):
        lm.generate("x", kw.pop("n_tokens"), **kw)


def test_generation_metadata(lm, cfg):
    g = lm.generate("hello there", 20, watermark=cfg, seed=1)
    assert g.watermarked and g.scheme == "kgw" and g.full_tokens[:2] == ["hello", "there"]


def test_text_roundtrips(lm, cfg):
    g = lm.generate("It was", 120, watermark=cfg, seed=4)
    assert tokenize(g.text) == g.tokens


def test_watermark_raises_perplexity_a_little(lm):
    plain = [lm.perplexity(lm.generate("The", 120, seed=i).tokens) for i in range(8)]
    strong = WatermarkConfig(key="k", delta=8)
    marked = [lm.perplexity(lm.generate("The", 120, watermark=strong, seed=i).tokens) for i in range(8)]
    assert np.median(marked) > np.median(plain)


def test_gumbel_does_not_loop(lm):
    g = lm.generate("It was a dark night", 200, watermark=WatermarkConfig(key="k", scheme="gumbel", context_width=1), seed=1)
    assert len(set(g.tokens)) > 40


def test_temperature_low_is_less_diverse(lm):
    hot = lm.generate("The", 150, temperature=1.5, seed=1).tokens
    cold = lm.generate("The", 150, temperature=0.3, seed=1).tokens
    assert len(set(cold)) < len(set(hot))


def test_top_k_one_is_deterministic(lm):
    assert lm.generate("The", 30, top_k=1, seed=1).tokens == lm.generate("The", 30, top_k=1, seed=2).tokens


def test_context_width_2_roundtrip(lm):
    c = WatermarkConfig(key="k", context_width=2)
    assert detect(lm.generate("The house", 150, watermark=c, seed=2).text, c).detected


def test_truecase_map():
    m = truecase_map("we met Darcy . later Darcy came . and Darcy left . the cat")
    assert m.get("darcy") == "Darcy" and "cat" not in m


def test_default_lm_casemap(lm):
    assert lm.casemap.get("elizabeth") == "Elizabeth"
