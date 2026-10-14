import numpy as np
import pytest

from cfd.profiling.metrics import MetricsRecorder, _build_scope_mask, _build_boundary_mask
from cfd.profiling.config import MetricsConfig, FieldMetricConfig, ScalarMetricConfig


class TestScopeMasks:
    def test_all_scope_selects_everything(self):
        arr = np.arange(16).reshape(4, 4)
        mask = _build_scope_mask("all", arr.shape)
        assert np.array_equal(arr[mask], arr)

    def test_interior_scope_excludes_edges(self):
        arr = np.arange(16).reshape(4, 4)
        mask = _build_scope_mask("interior", arr.shape)
        assert arr[mask].shape == (2, 2)

    def test_boundary_mask_excludes_interior(self):
        shape = (4, 4)
        mask = _build_boundary_mask(shape)
        assert mask[1:-1, 1:-1].sum() == 0
        assert mask[0, :].all()
        assert mask[-1, :].all()

    def test_unknown_scope_raises(self):
        with pytest.raises(ValueError, match="Unknown scope"):
            _build_scope_mask("bogus", (4, 4))


class TestMetricsRecorderDisabled:
    def test_disabled_when_no_config(self):
        recorder = MetricsRecorder(MetricsConfig(), grid_shape=(4, 4))
        assert recorder.enabled is False

    def test_store_is_noop_when_disabled(self):
        recorder = MetricsRecorder(MetricsConfig(), grid_shape=(4, 4))
        recorder.store("solver.iterations", 5)
        assert recorder.scalar_results == {}

    def test_capture_is_noop_when_disabled(self):
        recorder = MetricsRecorder(MetricsConfig(), grid_shape=(4, 4))
        recorder.capture("pressure", "final", np.ones((4, 4)))
        assert recorder._captured_this_frame == set()


class TestMetricsRecorderScalars:
    def test_store_records_value_matching_every(self):
        config = MetricsConfig(scalars=[ScalarMetricConfig(name="solver.iterations", every=1)])
        recorder = MetricsRecorder(config, grid_shape=(4, 4))
        recorder.tick(0)
        recorder.store("solver.iterations", 10)
        assert recorder.scalar_results["solver.iterations"] == [10.0]

    def test_store_respects_every_interval(self):
        config = MetricsConfig(scalars=[ScalarMetricConfig(name="solver.iterations", every=2)])
        recorder = MetricsRecorder(config, grid_shape=(4, 4))
        for frame in range(4):
            recorder.tick(frame)
            recorder.store("solver.iterations", frame)
        # Only frames 0 and 2 should be recorded (frame % 2 == 0)
        assert recorder.scalar_results["solver.iterations"] == [0.0, 2.0]

    def test_store_unknown_name_ignored(self):
        config = MetricsConfig(scalars=[ScalarMetricConfig(name="solver.iterations")])
        recorder = MetricsRecorder(config, grid_shape=(4, 4))
        recorder.store("not_configured", 1.0)
        assert "not_configured" not in recorder.scalar_results


class TestMetricsRecorderFields:
    def make_recorder(self, every=1, reducers=None, scopes=None):
        config = MetricsConfig(fields=[
            FieldMetricConfig(
                source="pressure",
                stages=["final"],
                reducers=reducers or ["rms"],
                scopes=scopes or ["all"],
                every=every,
            )
        ])
        return MetricsRecorder(config, grid_shape=(4, 4))

    def test_capture_unconfigured_source_stage_ignored(self):
        recorder = self.make_recorder()
        recorder.tick(0)
        recorder.capture("divergence", "before_projection", np.ones((4, 4)))
        assert recorder._captured_this_frame == set()

    def test_capture_records_buffer(self):
        recorder = self.make_recorder()
        recorder.tick(0)
        arr = np.full((4, 4), 2.0)
        recorder.capture("pressure", "final", arr)
        assert ("pressure", "final") in recorder._captured_this_frame
        assert np.array_equal(recorder._buffers[("pressure", "final")], arr)

    def test_capture_respects_every_interval(self):
        recorder = self.make_recorder(every=2)
        recorder.tick(1)
        recorder.capture("pressure", "final", np.ones((4, 4)))
        assert recorder._captured_this_frame == set()

    def test_process_reduces_captured_fields(self):
        recorder = self.make_recorder(reducers=["rms", "mean"], scopes=["all"])
        recorder.tick(0)
        recorder.capture("pressure", "final", np.full((4, 4), 3.0))
        recorder.process()

        assert recorder.field_results[("pressure", "final", "all", "rms")] == [3.0]
        assert recorder.field_results[("pressure", "final", "all", "mean")] == [3.0]
        # captured set cleared after processing
        assert recorder._captured_this_frame == set()

    def test_process_without_capture_adds_nothing(self):
        recorder = self.make_recorder()
        recorder.process()
        assert recorder.field_results == {}


class TestMetricsRecorderReportingAndReset:
    def test_to_dict_structure(self):
        config = MetricsConfig(scalars=[ScalarMetricConfig(name="solver.iterations")])
        recorder = MetricsRecorder(config, grid_shape=(4, 4))
        recorder.tick(0)
        recorder.store("solver.iterations", 4)
        recorder.store("solver.iterations", 6)
        d = recorder.to_dict()
        assert d["scalars"]["solver.iterations"]["calls"] == 2
        assert d["scalars"]["solver.iterations"]["avg"] == pytest.approx(5.0)

    def test_report_lines_includes_scalar_name(self):
        config = MetricsConfig(scalars=[ScalarMetricConfig(name="solver.iterations")])
        recorder = MetricsRecorder(config, grid_shape=(4, 4))
        recorder.tick(0)
        recorder.store("solver.iterations", 4)
        lines = recorder.report_lines()
        assert any("solver.iterations" in line for line in lines)

    def test_reset_clears_results_but_keeps_buffers(self):
        config = MetricsConfig(
            fields=[FieldMetricConfig(source="pressure", stages=["final"], reducers=["rms"], scopes=["all"])],
            scalars=[ScalarMetricConfig(name="solver.iterations")],
        )
        recorder = MetricsRecorder(config, grid_shape=(4, 4))
        recorder.tick(0)
        recorder.store("solver.iterations", 1)
        recorder.capture("pressure", "final", np.ones((4, 4)))
        recorder.process()

        recorder.reset()
        assert recorder.scalar_results == {}
        assert recorder.field_results == {}
        assert ("pressure", "final") in recorder._buffers  # buffer preserved
