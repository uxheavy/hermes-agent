"""Hermes G1 image shim: never load ambient dotenv credentials."""

from __future__ import annotations

from typing import Any


def load_dotenv(*args: Any, **kwargs: Any) -> bool:
    """Keep the immutable isolated child free of ambient dotenv state."""

    del args, kwargs
    return False


__all__ = ["load_dotenv"]
