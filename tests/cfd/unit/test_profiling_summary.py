import pytest

from cfd.profiling.summary import VariableSummary


class TestVariableSummary:
    def test_empty_values_gives_zeroed_summary(self):
        summary = VariableSummary.from_values("empty", [])
        assert summary.count == 0
        assert summary.total == 0.0
        assert summary.mean == 0.0
        assert summary.min == 0.0
        assert summary.max == 0.0
        assert summary.std == 0.0
        assert summary.p95 == 0.0
        assert summary.p99 == 0.0

    def test_basic_statistics(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        summary = VariableSummary.from_values("basic", values)
        assert summary.count == 5
        assert summary.total == pytest.approx(15.0)
        assert summary.mean == pytest.approx(3.0)
        assert summary.min == pytest.approx(1.0)
        assert summary.max == pytest.approx(5.0)

    def test_constant_values_have_zero_std(self):
        summary = VariableSummary.from_values("const", [2.0, 2.0, 2.0])
        assert summary.std == pytest.approx(0.0)

    def test_to_dict_structure(self):
        summary = VariableSummary.from_values("basic", [1.0, 2.0, 3.0])
        d = summary.to_dict()
        assert set(d.keys()) == {"count", "total", "mean", "min", "max", "std", "p95", "p99"}
        assert d["count"] == 3

    def test_percentiles_of_uniform_distribution(self):
        values = list(range(1, 101))  # 1..100
        summary = VariableSummary.from_values("uniform", values)
        assert summary.p95 == pytest.approx(95.05, abs=0.5)
        assert summary.p99 == pytest.approx(99.01, abs=0.5)
