from .value_classifier import (
    ValueEval24Classifier
)

__all__ = [
    "ValueEval24Classifier"
]

from importlib import metadata

__version__ = metadata.version(__package__)  # type: ignore
