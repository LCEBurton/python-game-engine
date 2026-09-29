import numpy as np


class SnapshotRecorder:
    """
    Captures scalar values and 2D/3D array fields at specific simulation
    frames, for later inspection/visualization. Independent of timing data.

    Usage:
        recorder.register_snapshot("dt", frame=100, getter=lambda: sim.dt)
        recorder.register_field_snapshot("pressure", frame=100, getter=lambda: sim.pressure)
        recorder.tick(current_frame)
        ...
        field = recorder.get_field("pressure")
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._snapshots = {}        # name -> (frame, getter)
        self._field_snapshots = {}  # name -> (frame, getter)
        self._values = {}           # name -> captured scalar
        self._fields = {}           # name -> captured field

    def register_snapshot(self, name: str, frame: int, getter):
        self._snapshots[name] = (frame, getter)

    def register_field_snapshot(self, name: str, frame: int, getter):
        self._field_snapshots[name] = (frame, getter)

    def get_value(self, name: str):
        return self._values.get(name, None)

    def get_field(self, name: str):
        return self._fields.get(name, None)

    def tick(self, frame: int):
        if not self.enabled:
            return

        for name, (target_frame, getter) in self._snapshots.items():
            if frame == target_frame:
                value = getter()
                self._values[name] = float(value) if np.isscalar(value) else value # type: ignore

        for name, (target_frame, getter) in self._field_snapshots.items():
            if frame == target_frame:
                value = getter()
                self._fields[name] = float(value) if np.isscalar(value) else value # type: ignore

    def to_dict(self) -> dict:
        # Fields are typically large arrays; exclude from JSON metrics export
        # by default. Only scalar snapshot values are included here.
        return {name: value for name, value in self._values.items()}

    def reset(self):
        self._snapshots.clear()
        self._field_snapshots.clear()
        self._values.clear()
        self._fields.clear()
