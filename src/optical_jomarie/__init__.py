# src/optical_jomarie/__init__.py
"""optical_jomarie package init"""
from importlib.metadata import version as _version

__all__ = ["absorption", "elliott", "general", "PL"]
try:
    __version__ = _version("optical_jomarie")
except Exception:
    __version__ = "0.0.0"
