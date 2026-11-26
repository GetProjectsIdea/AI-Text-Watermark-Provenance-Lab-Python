import pytest

from wmlab.lm import default_lm
from wmlab.watermark import WatermarkConfig


@pytest.fixture(scope="session")
def lm():
    return default_lm()


@pytest.fixture(scope="session")
def cfg():
    return WatermarkConfig(key="test-key", delta=2.0, gamma=0.25)


@pytest.fixture(scope="session")
def wm_text(lm, cfg):
    return lm.generate("It was a dark and stormy night", 200, watermark=cfg, seed=11).text


@pytest.fixture(scope="session")
def plain_text(lm):
    return lm.generate("It was a dark and stormy night", 200, seed=11).text
