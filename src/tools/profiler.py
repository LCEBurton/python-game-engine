from time import perf_counter_ns
from collections import defaultdict


class Profiler:
    def __init__(self):
        self.total_ns = defaultdict(int)
        self.total_vars = defaultdict(float)
        self.calls = defaultdict(int)
        self._last_time = None

    def start(self):
        self._last_time = perf_counter_ns()

    def checkpoint(self, name: str):
        now = perf_counter_ns()

        elapsed_ns = now - self._last_time # type: ignore
        self.total_ns[name] += elapsed_ns
        self.calls[name] += 1

        self._last_time = now

    def store(self, name, value):
        self.total_vars[name] += value # type: ignore
        self.calls[name] += 1


    def report(self):
        for name, total_ns in sorted(
                self.total_ns.items(),
                key=lambda item: item[1],
                reverse=True):
            calls = self.calls[name]
            avg_ms = total_ns / calls / 1_000_000
            print(f"{name:20s} | {calls:5d} calls | {total_ns / 1_000_000:.3f} ms total | {avg_ms:.3f} ms avg")

        print("\nVariable profiling report:")

        for name, total_var in sorted(
                self.total_vars.items(),
                key=lambda item: item[0],
                reverse=True):
            calls = self.calls[name]
            avg_var = total_var / calls
            print(f"{name:20s} | {calls:5d} calls | {total_var:.3f} total | {avg_var:.3f} avg")

    def to_string(self):
        report_lines = []
        for name, total_ns in sorted(
                self.total_ns.items(),
                key=lambda item: item[1],
                reverse=True):
            calls = self.calls[name]
            avg_ms = total_ns / calls / 1_000_000
            report_lines.append(f"{name:20s} | {calls:5d} calls | {total_ns / 1_000_000:.3f} ms total | {avg_ms:.3f} ms avg")

        report_lines.append("\nVariable profiling report:")

        for name, total_var in sorted(
                self.total_vars.items(),
                key=lambda item: item[0],
                reverse=True):
            calls = self.calls[name]
            avg_var = total_var / calls
            report_lines.append(f"{name:20s} | {calls:5d} calls | {total_var:.3f} total | {avg_var:.3f} avg")

        return "\n".join(report_lines)



    def reset(self):
        self.total_ns.clear()
        self.calls.clear()
            
