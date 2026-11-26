import csv

import pytest

from wmlab import experiments


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    res = experiments.run(list(experiments.EXPERIMENTS), out, quick=True, log=lambda *_: None)
    return out, res


def test_all_csvs_written(results):
    out, _ = results
    for f in ("e1_length.csv", "e2_robustness.csv", "e3_fpr.csv", "e3_human_z.csv", "e4_quality.csv",
              "e5_gamma_context.csv", "e6_image_provenance.csv"):
        assert (out / f).exists(), f


def test_plots_written(results):
    out, _ = results
    assert len(list(out.glob("*.png"))) >= 5


def test_summary_written(results):
    out, _ = results
    s = (out / "summary.md").read_text()
    assert "E1 Detection vs length" in s and "E6" in s


def test_length_monotone_z(results):
    _, res = results
    rows = [r for r in res["length"] if r["scheme"] == "kgw" and r["delta"] == 2.0]
    zs = [r["mean_z"] for r in rows]
    assert zs == sorted(zs)


def test_length_no_human_false_positives(results):
    _, res = results
    assert all(r["human_fpr_at_z4"] == 0 for r in res["length"])


def test_robustness_substitution_degrades(results):
    _, res = results
    rows = [r for r in res["robustness"] if r["scheme"] == "kgw d=2" and r["attack"] == "substitute"]
    assert rows[0]["mean_z"] > rows[-1]["mean_z"]


def test_robustness_zero_width_normalisation(results):
    _, res = results
    rows = {r["attack"]: r for r in res["robustness"] if r["scheme"] == "kgw d=2"}
    assert rows["zero_width"]["tpr_at_z4"] < rows["zero_width+normalise"]["tpr_at_z4"]


def test_fpr_repeat_trap(results):
    _, res = results
    rows = [r for r in res["fpr"] if r["corpus"] == "human x repeat" and r["threshold"] == 2.0 and r["repeat"] == 10]
    no = next(r for r in rows if r["dedupe"] is False)
    yes = next(r for r in rows if r["dedupe"] is True)
    assert no["empirical_fpr"] >= yes["empirical_fpr"]


def test_quality_tradeoff(results):
    _, res = results
    rows = {r["setting"]: r for r in res["quality"]}
    assert rows["kgw d=8"]["median_ppl"] > rows["none"]["median_ppl"]
    assert rows["gumbel"]["tpr_at_z4"] >= 0.9


def test_image_provenance_rows(results):
    _, res = results
    rows = {r["transform"]: r for r in res["image_provenance"]}
    assert rows["unchanged (metadata stripped)"]["hard_binding_valid"] == 1.0
    assert rows["jpeg q70"]["hard_binding_valid"] == 0.0 and rows["jpeg q70"]["soft_binding_match"] == 1.0
    assert rows["different image"]["soft_binding_match"] == 0.0


def test_csv_readable(results):
    out, _ = results
    with (out / "e2_robustness.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert {"scheme", "attack", "strength", "tpr_at_z4", "mean_z"} <= set(rows[0])


def test_unknown_experiment(tmp_path):
    with pytest.raises(ValueError):
        experiments.run(["nope"], tmp_path, quick=True, plots=False, log=lambda *_: None)
