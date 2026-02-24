from .value_classifier import (
    ValueClassifier
)

__all__ = [
    "ValueClassifier"
]

from importlib import metadata

__version__ = metadata.version(__package__)  # type: ignore
