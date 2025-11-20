import os
from pathlib import Path
from typing import Optional


def load_env_file(path: Optional[str] = None) -> None:
    """
    Populate os.environ with key/value pairs from a simple .env file.
    Existing environment variables take precedence over file values.
    """
    env_path = Path(path or ".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        os.environ.setdefault(key, value)
