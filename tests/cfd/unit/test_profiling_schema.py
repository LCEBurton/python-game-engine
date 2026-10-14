from unittest.mock import patch

from cfd.profiling.schema import build_metadata, get_git_commit, SCHEMA_VERSION


class TestGetGitCommit:
    def test_returns_string_on_success(self):
        with patch("subprocess.check_output", return_value=b"abcdef1234\n"):
            assert get_git_commit() == "abcdef1234"

    def test_returns_none_on_failure(self):
        with patch("subprocess.check_output", side_effect=Exception("not a git repo")):
            assert get_git_commit() is None


class TestBuildMetadata:
    def test_contains_required_keys(self):
        with patch("cfd.profiling.schema.get_git_commit", return_value="deadbeef"):
            metadata = build_metadata()
        assert metadata["schema_version"] == SCHEMA_VERSION
        assert "timestamp" in metadata
        assert "python_version" in metadata
        assert "platform" in metadata
        assert metadata["git_commit"] == "deadbeef"

    def test_extra_metadata_is_merged(self):
        with patch("cfd.profiling.schema.get_git_commit", return_value=None):
            metadata = build_metadata(extra={"backend": "cuda"})
        assert metadata["backend"] == "cuda"

    def test_extra_metadata_can_override_defaults(self):
        with patch("cfd.profiling.schema.get_git_commit", return_value=None):
            metadata = build_metadata(extra={"schema_version": 999})
        assert metadata["schema_version"] == 999
