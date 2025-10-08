"""My3DIS core package."""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - package metadata optional
    __version__ = version("my3dis")
except PackageNotFoundError:  # pragma: no cover - local source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
