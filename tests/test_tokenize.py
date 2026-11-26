import pytest

from wmlab import corpus
from wmlab.tokenize import detokenize, normalize, token_id, token_ids, tokenize


def test_basic_tokens():
    assert tokenize("Hello, World!") == ["hello", ",", "world", "!"]


def test_apostrophes_kept_in_word():
    assert tokenize("Don't stop") == ["don't", "stop"]


def test_lowercases():
    assert tokenize("MiXeD") == ["mixed"]


def test_numbers():
    assert tokenize("in 2026 we") == ["in", "2026", "we"]


def test_zero_width_removed():
    assert tokenize("wa\u200bter\u200dmark") == ["watermark"]


def test_zero_width_kept_when_raw():
    assert tokenize("wa\u200bter", clean=False) != ["water"]


def test_homoglyphs_mapped():
    assert tokenize("\u0441\u0430t") == ["cat"]


def test_curly_quotes_normalised():
    assert normalize("\u2019") == "'"


def test_nfkc():
    assert normalize("\ufb01ne") == "fine"


def test_detokenize_punctuation():
    assert detokenize(["hello", ",", "world", "."]) == "Hello, world."


def test_detokenize_capitalises_i():
    assert detokenize(["then", "i", "went"]) == "Then I went"


def test_detokenize_quotes_pair():
    assert detokenize(["he", "said", '"', "yes", '"', "."]) == 'He said "yes".'


def test_detokenize_casemap():
    assert detokenize(["mr", ".", "darcy"], {"darcy": "Darcy", "mr": "Mr"}) == "Mr. Darcy"


@pytest.mark.parametrize("name", ["human", "alice", "train"])
def test_roundtrip_on_corpus(name):
    for p in corpus.sample_paragraphs(name, 60, seed=5):
        toks = tokenize(p)
        assert tokenize(detokenize(toks)) == toks


def test_token_id_stable():
    assert token_id("the") == token_id("the")
    assert 0 <= token_id("the") < 2**64


def test_token_ids_distinct():
    ids = token_ids(["a", "b", "c", "the", "of"])
    assert len(set(ids)) == 5
