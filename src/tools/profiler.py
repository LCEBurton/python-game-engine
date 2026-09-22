from time import perf_counter_ns
from collections import defaultdict


class Profiler:
    def __init__(self):
        self.total_ns = defaultdict(int)
        self.total_vars = defaultdict(float)
        self.calls = defaultdict(int)
        self.var_calls = defaultdict(int)
        self._last_time = None
        self._snapshots = {} # name -> (frame, getter)
        self._field_snapshots = {} # name -> (frame, getter)
        self._fields = {} # name -> captured field

    def start(self):
        self._last_time = perf_counter_ns()

    def checkpoint(self, name: str):
        now = perf_counter_ns()

        elapsed_ns = now - self._last_time  # type: ignore
        self.total_ns[name] += elapsed_ns
        self.calls[name] += 1

        self._last_time = now

    def get_checkpoint_avg_ms(self, name: str):
        """Average time in milliseconds for a given checkpoint."""
        calls = self.calls[name]
        if calls == 0:
            return 0.0
        return self.total_ns[name] / calls / 1_000_000

    def get_checkpoint_total_ms(self, name: str):
        """Total time in milliseconds for a given checkpoint."""
        return self.total_ns[name] / 1_000_000

    def get_variable(self, name: str):
        return self.total_vars[name], self.var_calls[name]

    def get_variable_avg(self, name: str):
        total_vars, calls = self.get_variable(name)
        if calls == 0:
            return 0.0
        return total_vars / calls

    def store(self, name, value):
        self.total_vars[name] += value  # type: ignore
        self.var_calls[name] += 1

    def register_snapshot(self, name: str, frame: int, getter):
        """
        Register a value to be captured at a specific frame number.

        Args:
            name: Name to store the captured value under (retrieve via get_variable_avg).
            frame: The frame number at which to capture the value.
            getter: A zero-argument callable returning the value to store at that frame.
        """
        self._snapshots[name] = (frame, getter)

    def register_field_snapshot(self, name: str, frame: int, field_getter):
        """
        Register a 2D array field to be captured (copied) at a specific frame number.
        Retrieve via get_field(name).

        Args:
            name: Name to store the captured field under.
            frame: The frame number at which to capture the field.
            getter: A zero-argument callable returning a numpy array to snapshot.
        """
        self._field_snapshots[name] = (frame, field_getter)

    def get_field(self, name: str):
        """
        Retrieve a previously captured field snapshot.

        Args:
            name: Name of the field snapshot to retrieve.

        Returns:
            A numpy array containing the captured field data.
        """
        return self._fields.get(name, None)

    def tick(self, frame: int):
        """
        Advance the profiler's notion of the current frame, capturing any
        registered snapshots whose target frame matches.
        """
        for name, (target_frame, getter) in self._snapshots.items():
            if frame == target_frame:
                self.store(name, getter())

        for name, (target_frame, getter) in self._field_snapshots.items():
            if frame == target_frame:
                self._fields[name] = getter().copy()

    def report(self):
        print(self.to_string())

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
            calls = self.var_calls[name]
            avg_var = total_var / calls if calls else 0.0
            report_lines.append(f"{name:20s} | {calls:5d} calls | {total_var:.3f} total | {avg_var:.3f} avg")

        return "\n".join(report_lines)

    def reset(self):
        self.total_ns.clear()
        self.total_vars.clear()
        self.calls.clear()
        self.var_calls.clear()
        self._snapshots.clear()

