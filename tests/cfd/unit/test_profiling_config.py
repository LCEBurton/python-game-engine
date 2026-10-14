import json
import pytest

from cfd.profiling.config import (
    ProfileConfig,
    MetricsConfig,
    FieldMetricConfig,
    ScalarMetricConfig,
    TIMINGS,
    SOURCES,
    SOURCE_STAGES,
    SCALAR_METRICS,
    SCOPES,
    REDUCERS,
)


class TestFieldMetricConfig:
    def test_valid_config_does_not_raise(self):
        cfg = FieldMetricConfig(
            source="divergence",
            stages=["before_projection", "after_projection"],
            reducers=["rms", "mean"],
            scopes=["all", "interior"],
            every=2,
        )
        cfg.validate_config()  # should not raise

    def test_defaults(self):
        cfg = FieldMetricConfig(source="pressure")
        assert cfg.stages == []
        assert cfg.reducers == []
        assert cfg.scopes == []
        assert cfg.every == 1

    def test_unknown_source_raises(self):
        cfg = FieldMetricConfig(source="not_a_source")
        with pytest.raises(ValueError, match="Unknown field metric source"):
            cfg.validate_config()

    def test_invalid_stage_for_source_raises(self):
        cfg = FieldMetricConfig(source="divergence", stages=["final"])
        with pytest.raises(ValueError, match="Invalid stages"):
            cfg.validate_config()

    def test_invalid_reducer_raises(self):
        cfg = FieldMetricConfig(source="pressure", stages=["final"], reducers=["bogus"])
        with pytest.raises(ValueError, match="Invalid reducers"):
            cfg.validate_config()

    def test_invalid_scope_raises(self):
        cfg = FieldMetricConfig(
            source="pressure", stages=["final"], reducers=["rms"], scopes=["bogus"]
        )
        with pytest.raises(ValueError, match="Invalid scopes"):
            cfg.validate_config()

    @pytest.mark.parametrize("every", [0, -1])
    def test_non_positive_every_raises(self, every):
        cfg = FieldMetricConfig(source="pressure", stages=["final"], every=every)
        with pytest.raises(ValueError, match="'every' must be a positive integer"):
            cfg.validate_config()

    def test_to_dict_from_dict_round_trip(self):
        cfg = FieldMetricConfig(
            source="divergence",
            stages=["before_projection"],
            reducers=["rms"],
            scopes=["all"],
            every=3,
        )
        restored = FieldMetricConfig.from_dict(cfg.to_dict())
        assert restored == cfg


class TestScalarMetricConfig:
    def test_valid_config_does_not_raise(self):
        cfg = ScalarMetricConfig(name="solver.iterations")
        cfg.validate_config()

    def test_unknown_name_raises(self):
        cfg = ScalarMetricConfig(name="not_a_metric")
        with pytest.raises(ValueError, match="Unknown scalar metric"):
            cfg.validate_config()

    @pytest.mark.parametrize("every", [0, -5])
    def test_non_positive_every_raises(self, every):
        cfg = ScalarMetricConfig(name="solver.iterations", every=every)
        with pytest.raises(ValueError, match="'every' must be a positive integer"):
            cfg.validate_config()

    def test_to_dict_from_dict_round_trip(self):
        cfg = ScalarMetricConfig(name="solver.iterations", every=5)
        restored = ScalarMetricConfig.from_dict(cfg.to_dict())
        assert restored == cfg


class TestMetricsConfig:
    def test_valid_config_does_not_raise(self):
        cfg = MetricsConfig(
            fields=[FieldMetricConfig(source="pressure", stages=["final"], reducers=["rms"], scopes=["all"])],
            scalars=[ScalarMetricConfig(name="solver.iterations")],
        )
        cfg.validate_config()

    def test_invalid_nested_field_propagates(self):
        cfg = MetricsConfig(fields=[FieldMetricConfig(source="bogus")])
        with pytest.raises(ValueError, match="Unknown field metric source"):
            cfg.validate_config()

    def test_invalid_nested_scalar_propagates(self):
        cfg = MetricsConfig(scalars=[ScalarMetricConfig(name="bogus")])
        with pytest.raises(ValueError, match="Unknown scalar metric"):
            cfg.validate_config()

    def test_to_dict_from_dict_round_trip(self):
        cfg = MetricsConfig(
            fields=[FieldMetricConfig(source="pressure", stages=["final"], reducers=["rms"], scopes=["all"])],
            scalars=[ScalarMetricConfig(name="solver.iterations", every=2)],
        )
        restored = MetricsConfig.from_dict(cfg.to_dict())
        assert restored == cfg

    def test_from_dict_empty_defaults(self):
        cfg = MetricsConfig.from_dict({})
        assert cfg.fields == []
        assert cfg.scalars == []


class TestProfileConfig:
    def test_default_config_is_valid(self):
        cfg = ProfileConfig()
        cfg.validate_config()  # should not raise

    def test_valid_timings_do_not_raise(self):
        cfg = ProfileConfig(timings=list(TIMINGS))
        cfg.validate_config()

    def test_invalid_timing_raises(self):
        cfg = ProfileConfig(timings=["not.a.timing"])
        with pytest.raises(ValueError, match="Invalid timings"):
            cfg.validate_config()

    def test_metrics_wrong_type_raises(self):
        cfg = ProfileConfig()
        cfg.metrics = "not a MetricsConfig" # type: ignore
        with pytest.raises(ValueError, match="metrics must be an instance of MetricsConfig"):
            cfg.validate_config()

    def test_invalid_nested_metrics_propagates(self):
        cfg = ProfileConfig(metrics=MetricsConfig(scalars=[ScalarMetricConfig(name="bogus")]))
        with pytest.raises(ValueError, match="Unknown scalar metric"):
            cfg.validate_config()

    def test_to_dict_from_dict_round_trip(self):
        cfg = ProfileConfig(
            timings=["simulation.substep"],
            metrics=MetricsConfig(scalars=[ScalarMetricConfig(name="solver.iterations")]),
            snapshots=["pressure"],
        )
        restored = ProfileConfig.from_dict(cfg.to_dict())
        assert restored == cfg
        restored.validate_config()

    def test_from_dict_defaults_when_missing_sections(self):
        cfg = ProfileConfig.from_dict({})
        assert cfg.timings == []
        assert cfg.snapshots == []
        cfg.validate_config()

    def test_json_round_trip(self, tmp_path):
        cfg = ProfileConfig(timings=["simulation.substep"])
        path = tmp_path / "profile_config.json"
        with open(path, "w") as f:
            json.dump(cfg.to_dict(), f)

        loaded = ProfileConfig.from_json(str(path))
        assert loaded == cfg


class TestReducers:
    def test_rms_of_constant_array(self):
        import numpy as np
        arr = np.full((4, 4), 3.0)
        assert REDUCERS["rms"](arr) == pytest.approx(3.0)

    def test_mean_abs(self):
        import numpy as np
        arr = np.array([-1.0, 1.0, -2.0, 2.0])
        assert REDUCERS["mean_abs"](arr) == pytest.approx(1.5)

    def test_max_abs_picks_largest_magnitude_negative(self):
        import numpy as np
        arr = np.array([-5.0, 1.0, 2.0])
        assert REDUCERS["max_abs"](arr) == pytest.approx(5.0)

    def test_std_of_constant_array_is_zero(self):
        import numpy as np
        arr = np.full((3, 3), 7.0)
        assert REDUCERS["std"](arr) == pytest.approx(0.0)


class TestProfileConfigDisabled:
    def test_disabled_has_no_timings_metrics_or_snapshots(self):
        cfg = ProfileConfig.disabled()
        assert cfg.timings == []
        assert cfg.metrics.fields == []
        assert cfg.metrics.scalars == []
        assert cfg.snapshots == []

    def test_disabled_is_valid(self):
        ProfileConfig.disabled().validate_config()

    def test_disabled_equals_default_config(self):
        assert ProfileConfig.disabled() == ProfileConfig()


class TestProfileConfigFull:
    def test_full_includes_all_timings(self):
        cfg = ProfileConfig.full()
        assert set(cfg.timings) == set(TIMINGS)

    def test_full_includes_all_scalar_metrics(self):
        cfg = ProfileConfig.full()
        names = {s.name for s in cfg.metrics.scalars}
        assert names == set(SCALAR_METRICS)

    def test_full_field_metrics_cover_known_sources_with_stages(self):
        cfg = ProfileConfig.full()
        sources = {f.source for f in cfg.metrics.fields}
        assert sources == {s for s in SOURCES if s in SOURCE_STAGES}

    def test_full_field_metrics_use_all_reducers_and_scopes(self):
        cfg = ProfileConfig.full()
        for field_cfg in cfg.metrics.fields:
            assert set(field_cfg.reducers) == set(REDUCERS.keys())
            assert set(field_cfg.scopes) == set(SCOPES)
            assert field_cfg.every == 1

    def test_full_field_metrics_stages_match_source_stages(self):
        cfg = ProfileConfig.full()
        for field_cfg in cfg.metrics.fields:
            assert set(field_cfg.stages) == SOURCE_STAGES[field_cfg.source]

    def test_full_is_valid(self):
        ProfileConfig.full().validate_config()
