from dataclasses import dataclass, field
from .summary import VariableSummary
from .metrics import MetricsRecorder
from .timing import TimingProfiler
from .snapshots import SnapshotRecorder

@dataclass
class ProfilerResult:
    """
    Summarized profiling results for a single run, suitable for attaching
    to a BenchmarkRun/RunResult and for JSON export. Field/scalar metrics
    and timings are reduced to VariableSummary statistics; snapshots are
    kept as raw captured values (nothing to summarize for a single sample).
    """
    timings: list[VariableSummary] = field(default_factory=list)
    scalars: list[VariableSummary] = field(default_factory=list)
    fields: list[VariableSummary] = field(default_factory=list)
    snapshots: dict = field(default_factory=dict)

    @classmethod
    def from_recorders(
        cls,
        timing: TimingProfiler,
        metrics: MetricsRecorder,
        snapshots: SnapshotRecorder,
    ) -> "ProfilerResult":
        """Build a ProfilerResult by summarizing the given recorders' data."""
        timing_summaries = [
            VariableSummary.from_values(
                name, [ns / 1_000_000 for ns in durations_ns]
            )
            for name, durations_ns in timing.durations_ns.items()
        ]

        scalar_summaries = [
            VariableSummary.from_values(name, values)
            for name, values in metrics.scalar_results.items()
        ]

        field_summaries = [
            VariableSummary.from_values(
                f"{source}.{stage}.{scope}.{reducer}", values
            )
            for (source, stage, scope, reducer), values in metrics.field_results.items()
        ]

        return cls(
            timings=timing_summaries,
            scalars=scalar_summaries,
            fields=field_summaries,
            snapshots=snapshots.to_dict(),
        )

    def to_dict(self) -> dict:
        return {
            "timings": {s.name: s.to_dict() for s in self.timings},
            "scalars": {s.name: s.to_dict() for s in self.scalars},
            "fields": {s.name: s.to_dict() for s in self.fields},
            "snapshots": self.snapshots,
        }
