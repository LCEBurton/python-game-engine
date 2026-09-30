import json

from .metrics import MetricsRecorder
from .schema import build_metadata
from .snapshots import SnapshotRecorder
from .timing import TimingProfiler


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

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self.timing = TimingProfiler(enabled=enabled)
        self.metrics = MetricsRecorder(enabled=enabled)
        self.snapshots = SnapshotRecorder(enabled=enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool):
        """
        Enable or disable profiling at runtime, e.g. when toggling a debug
        flag. Propagates to all sub-recorders.
        """
        self._enabled = enabled
        self.timing.enabled = enabled
        self.metrics.enabled = enabled
        self.snapshots.enabled = enabled

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

    def merge_external_json(self, filepath: str, prefix: str):
        """
        Merge an externally-produced profile JSON (e.g. from a C++ or GPU
        kernel profiler) into this session's timing data, namespaced under
        the given prefix (e.g. "cpp" -> "cpp/pressure_solve/jacobi_kernel").

        The external JSON is expected to follow the same "checkpoints"
        schema as `TimingProfiler.to_dict()`.
        """
        with open(filepath) as f:
            data = json.load(f)

        for name, entry in data.get("checkpoints", {}).items():
            full_name = f"{prefix}/{name}"
            self.timing.total_ns[full_name] += int(entry["total_ms"] * 1_000_000)
            self.timing.calls[full_name] += entry["calls"]
            if entry.get("location"):
                self.timing._locations.setdefault(full_name, entry["location"])

    @staticmethod
    def load_json(filepath: str) -> dict:
        """Load a previously exported profile JSON file for comparison."""
        with open(filepath) as f:
            return json.load(f)

    def reset(self):
        self.timing.reset()
        self.metrics.reset()
        self.snapshots.reset()


