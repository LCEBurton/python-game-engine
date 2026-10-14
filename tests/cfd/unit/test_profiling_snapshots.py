import numpy as np

from cfd.profiling.snapshots import SnapshotRecorder


class TestSnapshotRecorderDisabled:
    def test_tick_is_noop_when_disabled(self):
        recorder = SnapshotRecorder(enabled=False)
        recorder.register_snapshot("dt", frame=5, getter=lambda: 0.016)
        recorder.tick(5)
        assert recorder.get_value("dt") is None


class TestSnapshotRecorderScalars:
    def test_register_and_capture_scalar_snapshot(self):
        recorder = SnapshotRecorder()
        recorder.register_snapshot("dt", frame=10, getter=lambda: 0.016)
        recorder.tick(9)
        assert recorder.get_value("dt") is None
        recorder.tick(10)
        assert recorder.get_value("dt") == 0.016

    def test_snapshot_only_captured_on_exact_frame(self):
        recorder = SnapshotRecorder()
        recorder.register_snapshot("dt", frame=10, getter=lambda: 0.016)
        recorder.tick(11)
        assert recorder.get_value("dt") is None

    def test_get_value_unregistered_returns_none(self):
        recorder = SnapshotRecorder()
        assert recorder.get_value("missing") is None


class TestSnapshotRecorderFields:
    def test_register_and_capture_field_snapshot(self):
        recorder = SnapshotRecorder()
        field = np.ones((4, 4))
        recorder.register_field_snapshot("pressure", frame=3, getter=lambda: field)
        recorder.tick(3)
        assert recorder.get_field("pressure") is not None
        assert np.array_equal(recorder.get_field("pressure"), field) # type: ignore

    def test_get_field_unregistered_returns_none(self):
        recorder = SnapshotRecorder()
        assert recorder.get_field("missing") is None


class TestSnapshotRecorderReportingAndReset:
    def test_to_dict_includes_only_scalar_values(self):
        recorder = SnapshotRecorder()
        recorder.register_snapshot("dt", frame=1, getter=lambda: 0.5)
        recorder.register_field_snapshot("pressure", frame=1, getter=lambda: np.ones((2, 2)))
        recorder.tick(1)

        d = recorder.to_dict()
        assert d == {"dt": 0.5}

    def test_reset_clears_all_state(self):
        recorder = SnapshotRecorder()
        recorder.register_snapshot("dt", frame=1, getter=lambda: 0.5)
        recorder.tick(1)
        recorder.reset()

        assert recorder.get_value("dt") is None
        assert recorder._snapshots == {}
        assert recorder._field_snapshots == {}
