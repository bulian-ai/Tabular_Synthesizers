"""Metrics for single table datasets."""

from ...metrics.single_table import (
    base, bayesian_network, detection, efficacy, gaussian_mixture, multi_single_column)
from ...metrics.single_table.base import SingleTableMetric
from ...metrics.single_table.bayesian_network import BNLikelihood, BNLogLikelihood
from ...metrics.single_table.detection.base import DetectionMetric
from ...metrics.single_table.detection.sklearn import (
    LogisticDetection, ScikitLearnClassifierDetectionMetric, SVCDetection)
from ...metrics.single_table.efficacy.base import MLEfficacyMetric
from ...metrics.single_table.efficacy.binary import (
    BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier, BinaryEfficacyMetric,
    BinaryLogisticRegression, BinaryMLPClassifier)
from ...metrics.single_table.efficacy.multiclass import (
    MulticlassDecisionTreeClassifier, MulticlassEfficacyMetric, MulticlassMLPClassifier)
from ...metrics.single_table.efficacy.regression import (
    LinearRegression, MLPRegressor, RegressionEfficacyMetric)
from ...metrics.single_table.gaussian_mixture import GMLogLikelihood
from ...metrics.single_table.multi_column_pairs import (
    ContinuousKLDivergence, DiscreteKLDivergence, MultiColumnPairsMetric)
from ...metrics.single_table.multi_single_column import (
    CSTest, KSTest, KSTestExtended, MultiSingleColumnMetric)

__all__ = [
    'bayesian_network',
    'base',
    'detection',
    'efficacy',
    'gaussian_mixture',
    'multi_single_column',
    'SingleTableMetric',
    'BNLikelihood',
    'BNLogLikelihood',
    'DetectionMetric',
    'LogisticDetection',
    'SVCDetection',
    'ScikitLearnClassifierDetectionMetric',
    'MLEfficacyMetric',
    'BinaryEfficacyMetric',
    'BinaryDecisionTreeClassifier',
    'BinaryAdaBoostClassifier',
    'BinaryLogisticRegression',
    'BinaryMLPClassifier',
    'MulticlassEfficacyMetric',
    'MulticlassDecisionTreeClassifier',
    'MulticlassMLPClassifier',
    'RegressionEfficacyMetric',
    'LinearRegression',
    'MLPRegressor',
    'GMLogLikelihood',
    'MultiColumnPairsMetric',
    'ContinuousKLDivergence',
    'DiscreteKLDivergence',
    'MultiSingleColumnMetric',
    'CSTest',
    'KSTest',
    'KSTestExtended',
]
