from collections import defaultdict
import numpy as np
from numpy import floating
from .config import MetricsConfig, FieldMetricConfig, REDUCERS, SCOPES


def _build_scope_mask(scope: str, shape: tuple[int, ...]):
    """
    Build an index expression for a given scope over an array of the given
    shape. Returned as something usable directly as a numpy index, e.g.
    arr[_build_scope_mask("interior", arr.shape)].
    """
    if scope == "all":
        return tuple(slice(None) for _ in shape)
    if scope == "interior":
        return tuple(slice(1, -1) for _ in shape)
    if scope == "boundary":
        return _build_boundary_mask(shape)
    raise ValueError(f"Unknown scope: {scope!r}")


def _build_boundary_mask(shape: tuple[int, ...]) -> np.ndarray:
    """Boolean mask selecting only the boundary cells of an array."""
    mask = np.zeros(shape, dtype=bool)
    mask[...] = True
    interior_idx = tuple(slice(1, -1) for _ in shape)
    mask[interior_idx] = False
    return mask


class MetricsRecorder:
    """
    Records scalar metrics (store) and field captures (capture), then
    reduces field captures to scalars during an explicit process() step.
    Frame-interval gating ("every") is driven by tick(frame), called once
    per frame (e.g. from the simulation's update loop), mirroring
    SnapshotRecorder's tick() convention.
    Usage:
        recorder.tick(frame)
        recorder.store("solver.iterations", iterations)
        recorder.capture("pressure", "initial", pressure_array)
        ...
        recorder.process()
    """

    def __init__(self, config: MetricsConfig, grid_shape: tuple[int, ...]):
        self.config = config
        self.grid_shape = grid_shape
        self.enabled = bool(config.fields or config.scalars)

        self._frame = 0

        # Scalar metric storage: name -> list of recorded values
        self.scalar_results: dict[str, list[float]] = defaultdict(list)

        # Lookup: scalar name -> ScalarMetricConfig, for interval/enabled checks
        self._scalar_configs = {s.name: s for s in config.scalars}

        # Lookup: (source, stage) -> FieldMetricConfig, for capture gating
        self._field_configs: dict[tuple[str, str], FieldMetricConfig] = {}
        for field_cfg in config.fields:
            for stage in field_cfg.stages:
                self._field_configs[(field_cfg.source, stage)] = field_cfg

        # Preallocated capture buffers: (source, stage) -> ndarray
        self._buffers: dict[tuple[str, str], np.ndarray] = {
            key: np.zeros(grid_shape, dtype=np.float64)
            for key in self._field_configs
        }
        # Tracks whether a buffer was actually written this frame
        self._captured_this_frame: set[tuple[str, str]] = set()

        # Precomputed scope masks, keyed by scope name
        self._scope_masks = {scope: _build_scope_mask(scope, grid_shape) for scope in SCOPES}

        # Processed field metric results: (source, stage, scope, reducer) -> list of values
        self.field_results: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)

    def tick(self, frame: int):
        """Advance the current frame number, used for 'every' interval gating."""
        self._frame = frame

    # --- Scalar path ---

    def store(self, name: str, value: floating | float):
        if not self.enabled:
            return
        cfg = self._scalar_configs.get(name)
        if cfg is None:
            return
        if self._frame % cfg.every != 0:
            return

        self.scalar_results[name].append(float(value))

    # --- Field capture path ---

    def capture(self, source: str, stage: str, array: np.ndarray):
        if not self.enabled:
            return
        key = (source, stage)
        cfg = self._field_configs.get(key)
        if cfg is None:
            return
        if self._frame % cfg.every != 0:
            return
        np.copyto(self._buffers[key], array)
        self._captured_this_frame.add(key)

    # --- Deferred processing ---

    def process(self):
        """
        Apply scope masking and reducers to all captured field buffers from
        this frame. Intended to be called outside the hot path, wrapped in
        a separate "profiling.metrics" timing scope by the caller.
        """
        if not self.enabled:
            return

        for key in self._captured_this_frame:
            source, stage = key
            cfg = self._field_configs[key]
            buffer = self._buffers[key]

            for scope in cfg.scopes:
                mask = self._scope_masks[scope]
                for reducer_name in cfg.reducers:
                    reducer = REDUCERS[reducer_name]
                    value = reducer(buffer[mask])
                    self.field_results[(source, stage, scope, reducer_name)].append(value)

        self._captured_this_frame.clear()

    # --- Reporting / export ---

    def report_lines(self) -> list[str]:
        lines = []
        for name, values in sorted(self.scalar_results.items()):
            calls = len(values)
            avg = sum(values) / calls if calls else 0.0
            lines.append(
                f"{name:20s} | {calls:5d} calls | "
                f"{sum(values):.3f} total | {avg:.3f} avg"
            )
        for (source, stage, scope, reducer), values in sorted(self.field_results.items()):
            avg = sum(values) / len(values) if values else 0.0
            lines.append(
                f"{source}.{stage}.{scope}.{reducer:10s} | {len(values):5d} samples | {avg:.6f} avg"
            )
        return lines

    def to_dict(self) -> dict:
        result = {
            "scalars": {},
            "fields": {},
        }
        for name, values in self.scalar_results.items():
            calls = len(values)
            result["scalars"][name] = {
                "total": sum(values),
                "calls": calls,
                "avg": (sum(values) / calls) if calls else 0.0,
            }
        for (source, stage, scope, reducer), values in self.field_results.items():
            key = f"{source}.{stage}.{scope}.{reducer}"
            calls = len(values)
            result["fields"][key] = {
                "calls": calls,
                "avg": (sum(values) / calls) if calls else 0.0,
                "last": values[-1] if values else None,
            }
        return result

    def reset(self):
        """Clear recorded samples/captures but preserve config and buffers."""
        self.scalar_results.clear()
        self.field_results.clear()
        self._captured_this_frame.clear()
        # Buffers themselves are NOT reallocated, just left to be overwritten
        # on next capture().

