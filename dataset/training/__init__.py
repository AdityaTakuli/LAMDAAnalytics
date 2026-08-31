"""Reusable, device-aware training components for the country-month models.

The package sits next to the existing flat pipeline modules (``common.py``,
``fuse_dataset.py``, ``model_gcn.py``, ``model_tgn.py``) and reuses them
directly. It adds nothing to the data contract: every table, feature name,
label definition, and split rule is the one already documented in
``dataset/README.md``.

Import order matters. ``training.paths`` must be imported before any module
that imports a flat pipeline module, because it puts ``dataset/`` on
``sys.path`` so ``import common`` resolves whether the entry point was started
from ``dataset/``, from the repository root, or from an absolute path.
"""

from __future__ import annotations

from training import paths as paths  # noqa: F401  (import for the sys.path side effect)

__all__ = ["paths"]
__version__ = "1.0.0"
