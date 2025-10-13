from .value_classifier import (
    ValueClassifier, combine_attained_and_constrained, combine_detailed_values, write_predictions, coarse_values, values
)

__all__ = [
    "ValueClassifier",
    "combine_attained_and_constrained",
    "combine_detailed_values",
    "write_predictions",
    "coarse_values",
    "values"
]

from importlib import metadata

__version__ = metadata.version(__package__)  # type: ignore
