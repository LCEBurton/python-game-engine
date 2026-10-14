import time
import pytest

from cfd.profiling.timing import TimingProfiler


class TestTimingProfilerDisabled:
    def test_disabled_when_no_timings_given(self):
        profiler = TimingProfiler()
        assert profiler.enabled is False

    def test_scope_is_noop_when_disabled(self):
        profiler = TimingProfiler()
        with profiler.scope("simulation.substep"):
            pass
        assert profiler.get_total_ns("simulation.substep") == 0.0
        assert profiler.durations_ns == {}

    def test_scope_noop_for_unlisted_name(self):
        profiler = TimingProfiler(enabled_timings=["simulation.substep"])
        with profiler.scope("some.other.scope"):
            pass
        assert "some.other.scope" not in profiler.durations_ns


class TestTimingProfilerEnabled:
    def test_enabled_when_timings_given(self):
        profiler = TimingProfiler(enabled_timings=["simulation.substep"])
        assert profiler.enabled is True

    def test_scope_records_duration(self):
        profiler = TimingProfiler(enabled_timings=["simulation.substep"])
        with profiler.scope("simulation.substep"):
            time.sleep(0.001)
        assert len(profiler.durations_ns["simulation.substep"]) == 1
        assert profiler.durations_ns["simulation.substep"][0] > 0

    def test_multiple_calls_accumulate_separate_samples(self):
        profiler = TimingProfiler(enabled_timings=["simulation.substep"])
        for _ in range(3):
            with profiler.scope("simulation.substep"):
                pass
        assert len(profiler.durations_ns["simulation.substep"]) == 3

    def test_get_total_ms_sums_durations(self):
        profiler = TimingProfiler(enabled_timings=["simulation.substep"])
        profiler.durations_ns["simulation.substep"] = [1_000_000, 2_000_000]
        assert profiler.get_total_ns("simulation.substep") == 3_000_000

    def test_get_avg_ms_averages_durations(self):
        profiler = TimingProfiler(enabled_timings=["simulation.substep"])
        profiler.durations_ns["simulation.substep"] = [1_000_000, 3_000_000]
        assert profiler.get_avg_ns("simulation.substep") == 2_000_000

    def test_get_avg_ms_empty_is_zero(self):
        profiler = TimingProfiler(enabled_timings=["simulation.substep"])
        assert profiler.get_avg_ns("simulation.substep") == 0.0

    def test_report_lines_contains_scope_name(self):
        profiler = TimingProfiler(enabled_timings=["simulation.substep"])
        with profiler.scope("simulation.substep"):
            pass
        lines = profiler.report_lines()
        assert any("simulation.substep" in line for line in lines)

    def test_to_dict_structure(self):
        profiler = TimingProfiler(enabled_timings=["simulation.substep"])
        profiler.durations_ns["simulation.substep"] = [1_000_000, 1_000_000]
        d = profiler.to_dict()
        assert d["simulation.substep"]["calls"] == 2
        assert d["simulation.substep"]["total_ms"] == pytest.approx(2.0)
        assert d["simulation.substep"]["avg_ms"] == pytest.approx(1.0)

    def test_reset_clears_durations(self):
        profiler = TimingProfiler(enabled_timings=["simulation.substep"])
        with profiler.scope("simulation.substep"):
            pass
        profiler.reset()
        assert profiler.durations_ns == {}
