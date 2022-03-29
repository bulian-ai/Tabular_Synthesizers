"""Metrics for multi table datasets."""

from ...metrics.multi_table import detection, multi_single_table
from ...metrics.multi_table.base import MultiTableMetric
from ...metrics.multi_table.detection.base import DetectionMetric
from ...metrics.multi_table.detection.parent_child import (
    LogisticParentChildDetection, ParentChildDetectionMetric, SVCParentChildDetection)
from ...metrics.multi_table.multi_single_table import (
    BNLikelihood, BNLogLikelihood, CSTest, KSTest, KSTestExtended, LogisticDetection,
    MultiSingleTableMetric, SVCDetection)

__all__ = [
    'detection',
    'multi_single_table',
    'MultiTableMetric',
    'DetectionMetric',
    'ParentChildDetectionMetric',
    'LogisticParentChildDetection',
    'SVCParentChildDetection',
    'BNLikelihood',
    'BNLogLikelihood',
    'CSTest',
    'KSTest',
    'KSTestExtended',
    'LogisticDetection',
    'SVCDetection',
    'MultiSingleTableMetric',
]
