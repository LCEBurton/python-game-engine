import json

from .metrics import MetricsRecorder
from .schema import build_metadata
from .snapshots import SnapshotRecorder
from .timing import TimingProfiler
from .config import ProfileConfig
from .result import ProfilerResult



class ProfileSession:
    """
    Composes timing, metrics, and snapshot recording into a single object,
    with unified reporting and JSON export/import.

    Usage:
        profiler = ProfileSession()
        with profiler.timing.scope("pressure_solve"):
            ...
        profiler.metrics.store("residual_norm", value)
        profiler.snapshots.tick(frame)
        profiler.export_json("benchmarks/run.json")
    """

    def __init__(self, config: ProfileConfig, grid_shape: tuple[int, ...]):
        self.config = config
        self.timing = TimingProfiler(enabled_timings=config.timings)
        self.metrics = MetricsRecorder(config=config.metrics, grid_shape=grid_shape)
        self.snapshots = SnapshotRecorder(enabled=bool(config.snapshots))
        self._frame = 0

    @property
    def enabled(self) -> bool:
        return (
            self.timing.enabled
            or self.metrics.enabled
            or self.snapshots.enabled
        )

    @classmethod
    def disabled(cls, grid_shape: tuple[int, ...]=(1, 1)) -> "ProfileSession":
        """Build a fully disabled ProfileSession."""
        return cls(ProfileConfig.disabled(), grid_shape=grid_shape)

    @classmethod
    def full(cls, grid_shape: tuple[int, ...]=(1, 1)) -> "ProfileSession":
        """Build a fully enabled ProfileSession."""
        return cls(ProfileConfig.full(), grid_shape=grid_shape)

    def tick(self):
        """Advance the current frame number, used for 'every' interval gating."""
        self._frame += 1
        self.metrics.tick(self._frame)
        self.snapshots.tick(self._frame)

    def report(self):
        if not self.enabled:
            print("Profiling is disabled; no data recorded.")
            return
        print(self.to_string())

    def to_string(self) -> str:
        lines = ["Timing report:"]
        lines.extend(self.timing.report_lines())
        lines.append("\nVariable report:")
        lines.extend(self.metrics.report_lines())
        return "\n".join(lines)

    def to_dict(self, extra_metadata: dict | None = None) -> dict:
        return {
            "metadata": build_metadata(extra_metadata),
            "checkpoints": self.timing.to_dict(),
            "variables": self.metrics.to_dict(),
            "snapshots": self.snapshots.to_dict(),
        }

    def to_result(self) -> ProfilerResult:

        return ProfilerResult.from_recorders(
            timing=self.timing,
            metrics=self.metrics,
            snapshots=self.snapshots,
        )

    def export_json(self, filepath: str, indent: int = 2,
                     extra_metadata: dict | None = None):
        """
        Export the current session state to a JSON file for record-keeping.

        Args:
            filepath: Path to write the JSON file to.
            indent: JSON indentation level (default 2, use None for compact).
            extra_metadata: Optional additional metadata (e.g. backend info
                for external C++/GPU profiler merges).
        """
        if not self.enabled:
            print("Profiling is disabled; skipping export.")
            return

        with open(filepath, "w") as f:
            json.dump(self.to_dict(extra_metadata), f, indent=indent)

#    def merge_external_json(self, filepath: str, prefix: str):
#        """
#        Merge an externally-produced profile JSON (e.g. from a C++ or GPU
#        kernel profiler) into this session's timing data, namespaced under
#        the given prefix (e.g. "cpp" -> "cpp/pressure_solve/jacobi_kernel").
#
#        The external JSON is expected to follow the same "checkpoints"
#        schema as `TimingProfiler.to_dict()`.
#        """
#        with open(filepath) as f:
#            data = json.load(f)
#
#        for name, entry in data.get("checkpoints", {}).items():
#            full_name = f"{prefix}/{name}"
#            self.timing.total_ns[full_name] += int(entry["total_ms"] * 1_000_000)
#            self.timing.calls[full_name] += entry["calls"]
#            if entry.get("location"):
#                self.timing._locations.setdefault(full_name, entry["location"])
#
    @staticmethod
    def load_json(filepath: str) -> dict:
        """Load a previously exported profile JSON file for comparison."""
        with open(filepath) as f:
            return json.load(f)

    def reset(self):
        self.timing.reset()
        self.metrics.reset()
        self.snapshots.reset()


