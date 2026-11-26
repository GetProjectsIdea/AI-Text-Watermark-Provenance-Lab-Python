import numpy as np
import pytest

from wmlab.hashing import context_seed, green_mask, splitmix64, uniforms


def test_seed_deterministic():
    assert context_seed("k", [1, 2]) == context_seed("k", [1, 2])


def test_seed_depends_on_key():
    assert context_seed("k1", [1]) != context_seed("k2", [1])


def test_seed_depends_on_context():
    assert context_seed("k", [1]) != context_seed("k", [2])


def test_seed_context_order_matters():
    assert context_seed("k", [1, 2]) != context_seed("k", [2, 1])


def test_empty_key_rejected():
    with pytest.raises(ValueError):
        context_seed("", [1])


def test_long_key_ok_and_distinct():
    a, b = "x" * 100, "x" * 99 + "y"
    assert context_seed(a, [1]) != context_seed(b, [1])


def test_bytes_key_same_as_str():
    assert context_seed(b"abc", [3]) == context_seed("abc", [3])


def test_uniforms_range():
    u = uniforms(123, np.arange(10000, dtype=np.uint64))
    assert u.min() >= 0 and u.max() < 1


def test_uniforms_mean_near_half():
    u = uniforms(99, np.arange(50000, dtype=np.uint64))
    assert abs(u.mean() - 0.5) < 0.01


def test_uniforms_deterministic():
    ids = np.arange(100, dtype=np.uint64)
    assert np.array_equal(uniforms(5, ids), uniforms(5, ids))


def test_uniforms_change_with_seed():
    ids = np.arange(100, dtype=np.uint64)
    assert not np.array_equal(uniforms(5, ids), uniforms(6, ids))


@pytest.mark.parametrize("gamma", [0.1, 0.25, 0.5, 0.75])
def test_green_fraction_matches_gamma(gamma):
    m = green_mask("key", [42], np.arange(40000, dtype=np.uint64), gamma)
    assert abs(m.mean() - gamma) < 0.01


def test_splitmix_no_warning():
    with np.errstate(all="raise"):
        splitmix64(np.array([2**64 - 1], dtype=np.uint64))


def test_large_ids_ok():
    u = uniforms(1, [2**64 - 1, 2**63])
    assert len(u) == 2
