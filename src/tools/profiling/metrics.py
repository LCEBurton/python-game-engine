
from collections import defaultdict

from numpy import floating

from .timing import _get_caller_info


class MetricsRecorder:
    """
    Tracks arbitrary named scalar values over time (e.g. residual norms,
    CFL numbers, iteration counts) independent of timing data.

    Usage:
        recorder.store("residual_norm", value)
        avg = recorder.get_avg("residual_norm")
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.total_vars = defaultdict(float)
        self.var_calls = defaultdict(int)
        self._locations = {}

    def store(self, name: str, value: floating | float):
        if not self.enabled:
            return

        self.total_vars[name] += float(value)
        self.var_calls[name] += 1
        self._record_location(name)

    def _record_location(self, name: str, skip: int = 3):
        if name not in self._locations:
            info = _get_caller_info(skip=skip)
            if info:
                self._locations[name] = info

    def get(self, name: str):
        return self.total_vars[name], self.var_calls[name]

    def get_avg(self, name: str) -> float:
        total, calls = self.get(name)
        if calls == 0:
            return 0.0
        return total / calls

    def report_lines(self) -> list[str]:
        lines = []
        for name, total_var in sorted(
                self.total_vars.items(), key=lambda item: item[0], reverse=True):
            calls = self.var_calls[name]
            avg_var = total_var / calls if calls else 0.0
            lines.append(
                f"{name:20s} | {calls:5d} calls | "
                f"{total_var:.3f} total | {avg_var:.3f} avg"
            )
        return lines

    def to_dict(self) -> dict:
        result = {}
        for name, total_var in self.total_vars.items():
            calls = self.var_calls[name]
            result[name] = {
                "total": total_var,
                "calls": calls,
                "avg": (total_var / calls) if calls else 0.0,
                "location": self._locations.get(name),
            }
        return result

    def reset(self):
        self.total_vars.clear()
        self.var_calls.clear()
        self._locations.clear()
