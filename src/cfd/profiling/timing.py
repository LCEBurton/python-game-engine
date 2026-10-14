from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter_ns
class TimingProfiler:
    """
    Tracks elapsed time for named, static scopes. Each call's duration is
    recorded individually (in milliseconds) to support percentile/variance
    statistics, not just totals/averages.
    Usage:
        with profiler.scope("simulation.pressure.solve"):
            ...
    """

    def __init__(self, enabled_timings: list[str] | None = None):
        self.enabled_timings = set(enabled_timings or [])
        self.enabled = bool(self.enabled_timings)
        self.durations_ns: dict[str, list[float]] = defaultdict(list)
        self._scope_stack = []  # unused by current scope(), kept for future debug_scope

    # --- Original stack-based dynamic scope implementation (kept for
    # potential future debug_scope use) ---
    # @contextmanager
    # def scope(self, name: str):
    #     if not self.enabled:
    #         yield
    #         return
    #
    #     full_name = "/".join(self._scope_stack + [name])
    #     self._record_location(full_name)
    #
    #     self._scope_stack.append(name)
    #     start = perf_counter_ns()
    #     try:
    #         yield
    #     finally:
    #         elapsed_ns = perf_counter_ns() - start
    #         self.total_ns[full_name] += elapsed_ns
    #         self.calls[full_name] += 1
    #         self._scope_stack.pop()

    @contextmanager
    def scope(self, name: str):
        """
        Time a named, static scope. `name` must be a fixed, dot-separated
        identifier from the TIMINGS vocabulary (e.g. "simulation.pressure.solve").
        No-ops entirely if the timing is not enabled for this profiler.
        """
        if not self.enabled or name not in self.enabled_timings:
            yield
            return

        start = perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = perf_counter_ns() - start
            self.durations_ns[name].append(elapsed_ns)

    def get_total_ns(self, name: str) -> float:
        if name in self.durations_ns:
            return sum(self.durations_ns[name])
        else:
            return 0.0


    def get_avg_ns(self, name: str) -> float:
        values = self.durations_ns[name]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def report_lines(self):
        lines = []
        for name, values in sorted(
                self.durations_ns.items(), key=lambda item: sum(item[1]), reverse=True):
            calls = len(values)
            total_ms = sum(values) / 1_000_000
            avg_ms = total_ms / calls if calls else 0.0
            lines.append(
                f"{name:30s} | {calls:5d} calls | "
                f"{total_ms:.3f} ms total | "
                f"{avg_ms:.3f} ms avg"
            )
        return lines

    def to_dict(self) -> dict:
        result = {}
        for name, values in self.durations_ns.items():
            calls = len(values)
            total_ms = sum(values) / 1_000_000
            result[name] = {
                "total_ms": total_ms,
                "calls": calls,
                "avg_ms": (total_ms / calls) if calls else 0.0,
            }
        return result

    def reset(self):
        self.durations_ns.clear()
        self._scope_stack.clear()

