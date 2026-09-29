import inspect
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter_ns


def _get_caller_info(skip=2):
    """
    Inspect the call stack to find where a timing call was invoked from.

    Args:
        skip: Number of frames to skip (profiler internals) to reach the caller.

    Returns:
        A dict with file, function, and line info, or {} if unavailable.
    """
    stack = inspect.stack()
    if len(stack) <= skip:
        return {}
    frame_info = stack[skip]
    return {
        "file": frame_info.filename,
        "function": frame_info.function,
        "line": frame_info.lineno,
    }


class TimingProfiler:
    """
    Tracks elapsed time for named, optionally nested, scopes.

    Usage:
        with profiler.scope("pressure_solve"):
            ...
            with profiler.scope("jacobi_iterate"):
                ...
    Nested scopes are automatically joined into hierarchical names, e.g.
    "pressure_solve/jacobi_iterate".
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.total_ns = defaultdict(int)
        self.calls = defaultdict(int)
        self._locations = {}  # name -> caller info (first-seen)
        self._scope_stack = []  # active nested scope names

    @contextmanager
    def scope(self, name: str):
        if not self.enabled:
            yield
            return

        full_name = "/".join(self._scope_stack + [name])
        self._record_location(full_name)

        self._scope_stack.append(name)
        start = perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = perf_counter_ns() - start
            self.total_ns[full_name] += elapsed_ns
            self.calls[full_name] += 1
            self._scope_stack.pop()

    def _record_location(self, name: str, skip: int = 3):
        if name not in self._locations:
            info = _get_caller_info(skip=skip)
            if info:
                self._locations[name] = info

    def get_total_ms(self, name: str) -> float:
        return self.total_ns[name] / 1_000_000

    def get_avg_ms(self, name: str) -> float:
        calls = self.calls[name]
        if calls == 0:
            return 0.0
        return self.total_ns[name] / calls / 1_000_000

    def report_lines(self):
        lines = []
        for name, total_ns in sorted(
                self.total_ns.items(), key=lambda item: item[1], reverse=True):
            calls = self.calls[name]
            avg_ms = total_ns / calls / 1_000_000
            loc = self._locations.get(name)
            loc_str = f" ({loc['function']}:{loc['line']})" if loc else ""
            lines.append(
                f"{name:30s} | {calls:5d} calls | "
                f"{total_ns / 1_000_000:.3f} ms total | "
                f"{avg_ms:.3f} ms avg{loc_str}"
            )
        return lines

    def to_dict(self) -> dict:
        result = {}
        for name, total_ns in self.total_ns.items():
            calls = self.calls[name]
            result[name] = {
                "total_ms": total_ns / 1_000_000,
                "calls": calls,
                "avg_ms": (total_ns / calls / 1_000_000) if calls else 0.0,
                "location": self._locations.get(name),
            }
        return result

    def reset(self):
        self.total_ns.clear()
        self.calls.clear()
        self._locations.clear()
        self._scope_stack.clear()
