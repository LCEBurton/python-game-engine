from dataclasses import dataclass
from typing import Union

import numpy as np
@dataclass
class VariableSummary:
    """
    Statistical summary of a list of recorded float values (e.g. scalar
    metric samples, field-reduced metric samples, or timing durations).
    """
    name: str
    count: int
    total: float
    mean: float
    min: float
    max: float
    std: float
    p95: float
    p99: float

    @classmethod
    def from_values(cls, name: str, values: Union[list[float], list[int]]) -> "VariableSummary":
        """Build a VariableSummary from a list of recorded values."""
        if not values:
            return cls(
                name=name, count=0, total=0.0, mean=0.0,
                min=0.0, max=0.0, std=0.0, p95=0.0, p99=0.0,
            )

        arr = np.asarray(values, dtype=np.float64)
        return cls(
            name=name,
            count=arr.size,
            total=float(arr.sum()),
            mean=float(arr.mean()),
            min=float(arr.min()),
            max=float(arr.max()),
            std=float(arr.std()),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
        )

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "total": self.total,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "std": self.std,
            "p95": self.p95,
            "p99": self.p99,
        }
