import numpy as np
import pytest

from cfd.profiling.result import ProfilerResult
from cfd.profiling.timing import TimingProfiler
from cfd.profiling.metrics import MetricsRecorder
from cfd.profiling.snapshots import SnapshotRecorder
from cfd.profiling.config import MetricsConfig, FieldMetricConfig, ScalarMetricConfig


class TestProfilerResultFromRecorders:
    def test_summarizes_timings(self):
        timing = TimingProfiler(enabled_timings=["simulation.substep"])
        timing.durations_ns["simulation.substep"] = [1_000_000, 2_000_000]

        metrics = MetricsRecorder(MetricsConfig(), grid_shape=(4, 4))
        snapshots = SnapshotRecorder(enabled=False)

        result = ProfilerResult.from_recorders(timing, metrics, snapshots)

        assert len(result.timings) == 1
        assert result.timings[0].name == "simulation.substep"
        assert result.timings[0].mean == pytest.approx(1.5)

    def test_summarizes_scalars_and_fields(self):
        config = MetricsConfig(
            fields=[FieldMetricConfig(source="pressure", stages=["final"], reducers=["rms"], scopes=["all"])],
            scalars=[ScalarMetricConfig(name="solver.iterations")],
        )
        metrics = MetricsRecorder(config, grid_shape=(4, 4))
        metrics.tick(0)
        metrics.store("solver.iterations", 5)
        metrics.capture("pressure", "final", np.full((4, 4), 2.0))
        metrics.process()

        timing = TimingProfiler()
        snapshots = SnapshotRecorder(enabled=False)

        result = ProfilerResult.from_recorders(timing, metrics, snapshots)

        assert any(s.name == "solver.iterations" for s in result.scalars)
        assert any(f.name == "pressure.final.all.rms" for f in result.fields)

    def test_to_dict_structure(self):
        timing = TimingProfiler()
        metrics = MetricsRecorder(MetricsConfig(), grid_shape=(4, 4))
        snapshots = SnapshotRecorder(enabled=False)

        result = ProfilerResult.from_recorders(timing, metrics, snapshots)
        d = result.to_dict()

        assert set(d.keys()) == {"timings", "scalars", "fields", "snapshots"}
