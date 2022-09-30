import unittest
import numpy as np

from bulian.metrics.demo import sample_relational_demo
from bulian.metrics.multi_table.detection.parent_child import LogisticParentChildDetection
from bulian.metrics.multi_table.multi_single_table import CSTest, BNLogLikelihood
from bulian.metrics.multi_table.statistical.cardinality_shape_similarity import CardinalityShapeSimilarity
from bulian.metrics.multi_table.statistical.cardinality_statistic_similarity import CardinalityStatisticSimilarity

class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self.metadata, tables = sample_relational_demo()
        sample_transcations = tables['transactions'].iloc[:5]
        sample_sessions = tables['sessions'].iloc[:5]
        sample_users = tables['users'].iloc[:5]

        real_trasactions = tables['transactions'].iloc[5:]
        real_sessions = tables['sessions'].iloc[5:]
        real_users = tables['users'].iloc[5:]

        self.sample_data = {
            'users': sample_users,
            'sessions': sample_sessions,
            'transactions': sample_transcations,
        }

        self.real_data = {
            'users': real_users,
            'sessions': real_sessions,
            'transactions': real_trasactions
        }


class multi_single_table_metrics_tests(BaseTestCase):
    def cs_test(self):
        val = CSTest.compute(real_data=self.real_data, synthetic_data=self.sample_data, metadata=self.metadata)
        self.assertEqual(type(val), np.float64)

        nomralized_score = CSTest.normalize(val)
        self.assertEqual(nomralized_score, val)

    def cs_test_compute_breakdown(self):
        val = CSTest.compute(real_data=self.real_data, synthetic_data=self.sample_data, metadata=self.metadata)
        breakdown = CSTest.compute_breakdown(real_data=self.real_data, synthetic_data=self.sample_data, metadata=self.metadata)
        scores = list(breakdown.values())
        if len(scores) > 0 and isinstance(scores[0], dict):
            scores = [
                result['score'] for table_scores in scores for result in table_scores.values()
                if 'score' in result
            ]
        self.assertEqual(np.nanmean(scores), val)
    
    def bnloglikelihood_normarlize(self):
        val = BNLogLikelihood.compute(real_data=self.real_data, synthetic_data=self.sample_data, metadata=self.metadata)
        self.assertEqual(type(val), np.float64)

        nomralized_score = CSTest.normalize(val)
        self.assertEqual(type(nomralized_score), np.float64)

class multi_table_detection_metrics_tests(BaseTestCase):
    def logistic_parent_child_test(self):
        val = LogisticParentChildDetection.compute(real_data=self.real_data, synthetic_data=self.sample_data, metadata=self.metadata)
        self.assertEqual(type(val), np.float64)
    
    def logistic_parent_child_metadata_error_test(self):
        self.assertRaises(ValueError, LogisticParentChildDetection.compute, self.real_data, self.sample_data)

class cardinality_shape_similarity_tests(BaseTestCase):
    def cardinality_shape_similarity_compute_test(self):
        val = CardinalityShapeSimilarity.compute(real_data=self.real_data, synthetic_data=self.sample_data, metadata=self.metadata)
        self.assertEqual(type(val), float)

        nomralized_score = CardinalityShapeSimilarity.normalize(val)
        self.assertEqual(nomralized_score, val)

class cardinality_statistic_similarity_tests(BaseTestCase):
    def cardinality_statistic_similarity_compute_test(self):
        val = CardinalityStatisticSimilarity.compute(real_data=self.real_data, synthetic_data=self.sample_data, metadata=self.metadata)
        self.assertEqual(type(val), np.float64)

        nomralized_score = CardinalityShapeSimilarity.normalize(val)
        self.assertEqual(nomralized_score, val)
