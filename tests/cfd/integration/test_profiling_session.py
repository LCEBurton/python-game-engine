import json
import numpy as np
import pytest

from cfd.profiling.session import ProfileSession
from cfd.profiling.config import ProfileConfig, MetricsConfig, FieldMetricConfig, ScalarMetricConfig


def make_session(grid_shape=(16, 16)):
    config = ProfileConfig(
        timings=["simulation.substep", "simulation.pressure.solve"],
        metrics=MetricsConfig(
            fields=[
                FieldMetricConfig(
                    source="pressure",
                    stages=["final"],
                    reducers=["rms", "max_abs"],
                    scopes=["all", "interior"],
                    every=1,
                )
            ],
            scalars=[ScalarMetricConfig(name="solver.iterations", every=1)],
        ),
        snapshots=[],
    )
    return ProfileSession(config=config, grid_shape=grid_shape)


class TestProfileSessionComposition:
    def test_enabled_reflects_sub_recorders(self):
        session = make_session()
        assert session.enabled is True

    def test_disabled_session_reports_disabled(self, capsys):
        session = ProfileSession(config=ProfileConfig(), grid_shape=(4, 4))
        assert session.enabled is False
        session.report()
        captured = capsys.readouterr()
        assert "Profiling is disabled" in captured.out

    def test_tick_advances_frame_and_propagates(self):
        session = make_session()
        session.tick()
        session.tick()
        assert session._frame == 2
        assert session.metrics._frame == 2
        assert session.snapshots.enabled is False  # no snapshots configured

    def test_timing_scope_records_duration(self):
        session = make_session()
        with session.timing.scope("simulation.substep"):
            pass
        assert len(session.timing.durations_ns["simulation.substep"]) == 1

    def test_metrics_store_and_capture_flow(self):
        session = make_session()
        session.tick()
        session.metrics.store("solver.iterations", 7)
        session.metrics.capture("pressure", "final", np.full((16, 16), 2.0))
        session.metrics.process()

        assert session.metrics.scalar_results["solver.iterations"] == [7.0]
        assert session.metrics.field_results[("pressure", "final", "all", "rms")] == [2.0]

    def test_to_dict_contains_all_sections(self):
        session = make_session()
        session.tick()
        with session.timing.scope("simulation.substep"):
            pass
        session.metrics.store("solver.iterations", 3)
        session.metrics.capture("pressure", "final", np.ones((16, 16)))
        session.metrics.process()

        d = session.to_dict()
        assert set(d.keys()) == {"metadata", "checkpoints", "variables", "snapshots"}
        assert "simulation.substep" in d["checkpoints"]
        assert "solver.iterations" in d["variables"]["scalars"]

    def test_to_result_summarizes_all_recorders(self):
        session = make_session()
        session.tick()
        with session.timing.scope("simulation.substep"):
            pass
        session.metrics.store("solver.iterations", 4)
        session.metrics.capture("pressure", "final", np.full((16, 16), 1.0))
        session.metrics.process()

        result = session.to_result()
        assert any(t.name == "simulation.substep" for t in result.timings)
        assert any(s.name == "solver.iterations" for s in result.scalars)
        assert any(f.name.startswith("pressure.final") for f in result.fields)

    def test_export_json_writes_file_with_metadata(self, tmp_path):
        session = make_session()
        session.tick()
        with session.timing.scope("simulation.substep"):
            pass

        path = tmp_path / "profile.json"
        session.export_json(str(path))

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "schema_version" in data["metadata"]
        assert "simulation.substep" in data["checkpoints"]

    def test_export_json_skips_when_disabled(self, tmp_path, capsys):
        session = ProfileSession(config=ProfileConfig(), grid_shape=(4, 4))
        path = tmp_path / "profile.json"
        session.export_json(str(path))

        assert not path.exists()
        captured = capsys.readouterr()
        assert "Profiling is disabled" in captured.out

    def test_report_prints_timing_and_variable_sections(self, capsys):
        session = make_session()
        with session.timing.scope("simulation.substep"):
            pass
        session.report()
        captured = capsys.readouterr()
        assert "Timing report:" in captured.out
        assert "Variable report:" in captured.out


class TestProfileSessionDisabled:
    def test_disabled_uses_default_grid_shape(self):
        session = ProfileSession.disabled()
        assert session.metrics.grid_shape == (1, 1)

    def test_disabled_is_fully_disabled(self):
        session = ProfileSession.disabled()
        assert session.enabled is False
        assert session.timing.enabled is False
        assert session.metrics.enabled is False
        assert session.snapshots.enabled is False

    def test_disabled_accepts_custom_grid_shape(self):
        session = ProfileSession.disabled(grid_shape=(16, 16))
        assert session.metrics.grid_shape == (16, 16)

    def test_disabled_scope_and_store_are_noops(self):
        session = ProfileSession.disabled()
        with session.timing.scope("simulation.substep"):
            pass
        session.metrics.store("solver.iterations", 5)
        assert session.timing.durations_ns == {}
        assert session.metrics.scalar_results == {}


class TestProfileSessionFull:
    def test_full_uses_default_grid_shape(self):
        session = ProfileSession.full()
        assert session.metrics.grid_shape == (1, 1)

    def test_full_is_fully_enabled(self):
        session = ProfileSession.full()
        assert session.enabled is True
        assert session.timing.enabled is True
        assert session.metrics.enabled is True

    def test_full_accepts_custom_grid_shape(self):
        session = ProfileSession.full(grid_shape=(16, 16))
        assert session.metrics.grid_shape == (16, 16)

    def test_full_records_timing_scope(self):
        session = ProfileSession.full()
        with session.timing.scope("simulation.substep"):
            pass
        assert len(session.timing.durations_ns["simulation.substep"]) == 1

    def test_full_records_configured_field_metric(self):
        session = ProfileSession.full(grid_shape=(4, 4))
        session.tick()
        session.metrics.capture("pressure", "final", np.full((4, 4), 2.0))
        session.metrics.process()
        assert session.metrics.field_results[("pressure", "final", "all", "rms")] == [2.0]

    def test_full_records_configured_scalar_metric(self):
        session = ProfileSession.full()
        session.tick()
        session.metrics.store("solver.iterations", 3)
        assert session.metrics.scalar_results["solver.iterations"] == [3.0]
