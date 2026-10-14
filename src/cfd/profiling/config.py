from dataclasses import dataclass, field
from typing import Callable
import numpy as np
SCHEMA_VERSION = 1


@dataclass
class FieldMetricConfig:
    source: str
    stages: list[str] = field(default_factory=list)
    reducers: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    every: int = 1

    def validate_config(self) -> None:
        """Validate this field metric configuration."""
        if self.source not in SOURCES:
            raise ValueError(f"Unknown field metric source: {self.source!r}. Must be one of {SOURCES}.")

        allowed_stages = SOURCE_STAGES.get(self.source, set())
        invalid_stages = set(self.stages) - allowed_stages
        if invalid_stages:
            raise ValueError(
                f"Invalid stages {sorted(invalid_stages)} for source {self.source!r}. "
                f"Must be a subset of {sorted(allowed_stages)}."
            )

        invalid_reducers = set(self.reducers) - REDUCERS.keys()
        if invalid_reducers:
            raise ValueError(
                f"Invalid reducers {sorted(invalid_reducers)}. Must be a subset of {sorted(REDUCERS.keys())}."
            )

        invalid_scopes = set(self.scopes) - set(SCOPES)
        if invalid_scopes:
            raise ValueError(
                f"Invalid scopes {sorted(invalid_scopes)}. Must be a subset of {SCOPES}."
            )

        if self.every <= 0:
            raise ValueError(f"'every' must be a positive integer, got {self.every}.")

    def to_dict(self) -> dict:
        """Convert this field metric configuration to a dictionary."""
        return {
            "source": self.source,
            "stages": self.stages,
            "reducers": self.reducers,
            "scopes": self.scopes,
            "every": self.every,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FieldMetricConfig":
        """Load a FieldMetricConfig from a dict."""
        return cls(
            source=data["source"],
            stages=data.get("stages", []),
            reducers=data.get("reducers", []),
            scopes=data.get("scopes", []),
            every=data.get("every", 1),
        )

@dataclass
class ScalarMetricConfig:
    name: str
    every: int = 1

    def validate_config(self) -> None:
        """Validate this scalar metric configuration."""
        if self.name not in SCALAR_METRICS:
            raise ValueError(f"Unknown scalar metric: {self.name!r}. Must be one of {SCALAR_METRICS}.")
        if self.every <= 0:
            raise ValueError(f"'every' must be a positive integer, got {self.every}.")

    def to_dict(self) -> dict:
        """Convert this scalar metric configuration to a dictionary."""
        return {
            "name": self.name,
            "every": self.every,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScalarMetricConfig":
        """Load a ScalarMetricConfig from a dict."""
        return cls(
            name=data["name"],
            every=data.get("every", 1),
        )

@dataclass
class MetricsConfig:
    fields: list[FieldMetricConfig] = field(default_factory=list)
    scalars: list[ScalarMetricConfig] = field(default_factory=list)

    def validate_config(self) -> None:
        """Validate the metrics configuration."""
        for field_metric in self.fields:
            field_metric.validate_config()
        for scalar_metric in self.scalars:
            scalar_metric.validate_config()

    def to_dict(self) -> dict:
        """Convert this metrics configuration to a dictionary."""
        return {
            "fields": [f.to_dict() for f in self.fields],
            "scalars": [s.to_dict() for s in self.scalars],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetricsConfig":
        """Load a MetricsConfig from a dict."""
        return cls(
            fields=[FieldMetricConfig(**f) for f in data.get("fields", [])],
            scalars=[ScalarMetricConfig(**s) for s in data.get("scalars", [])],
        )


@dataclass
class ProfileConfig:
    timings: list[str] = field(default_factory=list)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    snapshots: list[str] = field(default_factory=list)

    def validate_config(self) -> None:
        """Validate the profiling configuration."""
        invalid_timings = set(self.timings) - set(TIMINGS)
        if invalid_timings:
            raise ValueError(
                f"Invalid timings {sorted(invalid_timings)}. Must be a subset of {TIMINGS}."
            )

        if not isinstance(self.metrics, MetricsConfig):
            raise ValueError("metrics must be an instance of MetricsConfig.")
        self.metrics.validate_config()

        # TODO: validate snapshots once snapshot config schema is finalized.

    @classmethod
    def disabled(cls) -> "ProfileConfig":
        """
        Build a fully disabled ProfileConfig: no timings, no metrics, no
        snapshots. Useful for tests/mocks where profiling overhead or
        output is undesired but a ProfileSession is still required.
        """
        return cls(
            timings=[],
            metrics=MetricsConfig(fields=[], scalars=[]),
            snapshots=[],
        )

    @classmethod
    def full(cls) -> "ProfileConfig":
        """
        Build a fully enabled ProfileConfig: every known timing scope,
        every field metric source/stage/reducer/scope combination, and
        every scalar metric, all sampled every frame. Useful for
        exhaustive debugging/benchmarking runs where overhead is not a
        concern.
        """
        field_metrics = [
            FieldMetricConfig(
                source=source,
                stages=sorted(stages),
                reducers=sorted(REDUCERS.keys()),
                scopes=list(SCOPES),
                every=1,
            )
            for source, stages in SOURCE_STAGES.items()
            if source in SOURCES
        ]

        scalar_metrics = [
            ScalarMetricConfig(name=name, every=1)
            for name in SCALAR_METRICS
        ]

        return cls(
            timings=list(TIMINGS),
            metrics=MetricsConfig(fields=field_metrics, scalars=scalar_metrics),
            snapshots=[],  # TODO: populate once snapshot config schema is finalized.
        )

    def to_dict(self) -> dict:
        """Convert this profiling configuration to a dictionary."""
        return {
            "timings": self.timings,
            "metrics": self.metrics.to_dict(),
            "snapshots": self.snapshots,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProfileConfig":
        """
        Load a ProfileConfig from a dict.

        Args:
            data (dict): Dictionary of config options.

        Returns:
            ProfileConfig: The loaded profiling configuration.
        """
        return cls(
            timings=data.get("timings", []),
            metrics=MetricsConfig.from_dict(data.get("metrics", {})),
            snapshots=data.get("snapshots", []),
        )

    @classmethod
    def from_json(cls, path: str) -> "ProfileConfig":
        """
        Load a ProfileConfig from a JSON file.

        Args:
            path (str): Path to the JSON file.

        Returns:
            ProfileConfig: The loaded profiling configuration.
        """
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

SCALAR_METRICS = [ "solver.iterations" ]


TIMINGS = [
        "simulation.substep",
        "simulation.emitters",
        "simulation.forces",
        "simulation.advection.velocity",
        "simulation.advection.density",
        "simulation.advection.temperature",
        "simulation.cooling",
        "simulation.dissipation",
        "simulation.divergence",
        "simulation.pressure.solve",
        "simulation.pressure.apply",
        "profiling.capture",
        "profiling.metrics",
]

SOURCES = [
    "divergence",
    "pressure",
    "pressure_residual",
    "velocity_u",
    "velocity_v",
]

SOURCE_STAGES = {
    "divergence":
        {
            "before_projection",
            "after_projection"
        },
    "pressure":
        {
            "final",
        },
    "pressure_residual":
        {
            "final",
        },
}

def rms(arr: np.ndarray) -> float:
    """Compute the root mean square of an array."""
    return float(np.sqrt(np.mean(np.square(arr))))

def mean(arr: np.ndarray) -> float:
    """Compute the mean of an array."""
    return float(np.mean(arr))

def mean_abs(arr: np.ndarray) -> float:
    """Compute the mean of the absolute values of an array."""
    return float(np.mean(np.abs(arr)))

def _max(arr: np.ndarray) -> float:
    """Compute the maximum of an array."""
    return float(np.max(arr))

def max_abs(arr: np.ndarray) -> float:
    """Compute the maximum of the absolute values of an array."""
    return float(np.max(np.abs(arr)))

def std(arr: np.ndarray) -> float:
    """Compute the standard deviation of an array."""
    return float(np.std(arr))

REDUCERS: dict[str, Callable[[np.ndarray], float]] = {
    "rms": rms,
    "mean": mean,
    "mean_abs": mean_abs,
    "max": _max,
    "max_abs": max_abs,
    "std": std,
}

SCOPES = [ "all", "interior", "boundary" ]



