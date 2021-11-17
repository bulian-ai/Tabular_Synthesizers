"""Metrics Goal Enumeration."""

from enum import Enum


class MetricType(Enum):
    """Goal Enumeration.

    This enumerates the ``goal`` for a metric; the value of a metric can be ignored,
    minimized, or maximized.
    """
    
    IGNORE = 'ignore'
    LIKELIHOOD = 'Likelihood Metric'
    DETECTION = 'Real vs Synthetic Dectection Metric'
    EFFICACY = 'ML Efficacy Metric: R-Sq or F1' 
    STATISTICAL = 'Statistical Test Metric'
    ENTROPY = 'Distribution Similarity Metric'