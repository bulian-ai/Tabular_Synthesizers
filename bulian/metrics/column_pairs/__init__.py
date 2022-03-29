"""Metrics to compare column pairs."""

from . import statistical

#from column_pairs import statistical
from .base import ColumnPairsMetric
from .statistical.kl_divergence import (
    ContinuousKLDivergence, DiscreteKLDivergence)

__all__ = [
    'statistical',
    'ColumnPairsMetric',
    'ContinuousKLDivergence',
    'DiscreteKLDivergence',
]
