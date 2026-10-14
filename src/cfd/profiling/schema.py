import platform
import subprocess
from datetime import datetime, timezone

SCHEMA_VERSION = 1


def get_git_commit():
    """Return the current git commit hash, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def build_metadata(extra: dict | None = None) -> dict:
    """
    Build the standard metadata block included in every exported profile JSON.

    Args:
        extra: Optional additional metadata fields to merge in (e.g. backend
            info such as "cpp" or "cuda" for external profilers).
    """
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(),
    }
    if extra:
        metadata.update(extra)
    return metadata
