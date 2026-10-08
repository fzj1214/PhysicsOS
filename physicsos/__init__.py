"""PhysicsOS core package."""

__all__ = ["__version__"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("physicsos")
except PackageNotFoundError:
    __version__ = "0.1.31"
