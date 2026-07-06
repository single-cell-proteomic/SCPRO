from __future__ import annotations

import importlib
from typing import Any


def optional_import(module: str, *, extra: str | None = None) -> Any:
    """Import an optional dependency with a helpful error message."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        install_hint = f" Install scpro[{extra}]" if extra else " Install the missing dependency"
        raise ModuleNotFoundError(
            f"Optional dependency '{module}' is required for this operation."
            f"{install_hint} and try again."
        ) from exc
