"""Path resolution and import bootstrap.

Every path used by the training scripts is derived from the location of the
``dataset/`` directory on disk, never from the current working directory. This
makes the commands in ``TRAINING.md`` behave identically on Linux, Windows, and
macOS, and whether they are launched from ``dataset/`` or from the repository
root.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``training/paths.py`` -> ``training/`` -> ``dataset/``
DATASET_DIR: Path = Path(__file__).resolve().parents[1]
REPO_DIR: Path = DATASET_DIR.parent

# Make the flat pipeline modules importable regardless of the launch directory.
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))


def resolve(value: str | Path) -> Path:
    """Resolve a possibly relative path against ``dataset/``."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else DATASET_DIR / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
