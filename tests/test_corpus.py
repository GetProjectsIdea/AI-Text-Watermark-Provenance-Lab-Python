import pytest

from wmlab import corpus


@pytest.mark.parametrize("name", corpus.NAMES)
def test_load(name):
    t = corpus.load(name)
    assert len(t) > 100_000 and "Project Gutenberg License" not in t and "_" not in t


def test_unknown():
    with pytest.raises(ValueError):
        corpus.load("nope")


def test_paragraph_min_words():
    assert all(len(p.split()) >= 50 for p in corpus.paragraphs("human", 50))


def test_sample_deterministic():
    assert corpus.sample_paragraphs("human", 5, seed=1) == corpus.sample_paragraphs("human", 5, seed=1)


def test_prompts():
    ps = corpus.prompts(5, words=6)
    assert len(ps) == 5 and all(len(p.split()) == 6 for p in ps)


def test_train_and_human_differ():
    assert "Elizabeth" in corpus.load("train") and "Elizabeth" in corpus.load("human")  # both novels use the name
    assert "Frankenstein" in corpus.load("human") and "Frankenstein" not in corpus.load("train")
