from curses import meta
import pandas as pd
import numpy as np
import unittest
from bulian.metadata.dataset import Metadata
from bulian.metrics.single_table.base import SingleTableMetric
from bulian.metrics.single_table.efficacy.mlefficacy import MLEfficacy
from bulian.metrics.single_table.efficacy.multiclass import MulticlassMLPClassifier
from bulian.metrics.single_table.bayesian_network import BNLikelihood
from bulian.metrics.single_table.bayesian_network import BNLogLikelihood
from bulian.metrics.single_table.detection.sklearn import LogisticDetection, SVCDetection, ScikitLearnClassifierDetectionMetric
from bulian.metrics.single_table.efficacy.binary import BinaryDecisionTreeClassifier
from bulian.metrics.single_table.efficacy.regression import LinearRegression
from bulian.metrics.single_table.gaussian_mixture import GMLogLikelihood
from bulian.metrics.single_table.multi_column_pairs import ContinuousKLDivergence, DiscreteKLDivergence
from bulian.metrics.single_table.multi_single_column import CSTest, KSTest, KSTestExtended
from bulian.metrics.single_table.privacy.cap import CAPAttacker, CategoricalCAP, CategoricalZeroCAP
from bulian.metrics.single_table.privacy.categorical_sklearn import CategoricalKNN, CategoricalNB, CategoricalRF, CategoricalSVM
from bulian.metrics.single_table.privacy.numerical_sklearn import NumericalLR, NumericalMLP
from bulian.metrics.single_table.privacy.radius_nearest_neighbor import InverseCDFCutoff, NumericalRadiusNearestNeighbor, NumericalRadiusNearestNeighborAttacker

class BaseTestClass(unittest.TestCase):
    def setUp(self):
        self.real_data = pd.DataFrame(
            {
                '1':[0.6417366765311319, 0.4995232533994396, 0.7048236572174155, 0.01483747235772337, 0.8785052529521522, 0.7899419323434513, 0.6491719454951946, 0.8280835960643933, 0.5778327106751368, 0.309529058151795, 0.8033292868913439, 0.45642783362372885, 0.07560689463557335, 0.7325691884959649, 0.4736925521335209, 0.3611024877244906, 0.9803347507946942, 0.08558737496794311, 0.888512878083493, 0.1306618189016644],
                '2':[True, False, False, False, True, False, False, False, False, True, False, True, False, False, True, False, False, True, True, False],
                '3':['1', '1', '1', '1', '2', '2', '1', '2', '1', '2', '1', '1', '2', '2', '2', '2', '2', '2', '1', '2'],
                '4':[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0]

            }
        )
        self.synthetic_data = pd.DataFrame(
            {
                '1':[0.997361358781321, 0.15696424392733377, 0.8117729422949825, 0.15026548286480323, 0.0401690143637321, 0.6091175881875124, 0.46626052754000824, 0.35556690374849353, 0.3113300851082139, 0.0935274041188554],
                '2':[True, False, False, True, True, False, False, True, False, True],
                '3':['1', '1', '2', '2', '1', '1', '2', '1', '2', '1'],
                '4':[0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
            }
        )

        
class base_metric_test(BaseTestClass):
    def single_table_metric_subclasses_test(self):
        val = SingleTableMetric.get_subclasses()
        self.assertEqual(type(val), dict)
        self.assertIsNotNone(val)


class bayesian_network_test(BaseTestClass):

    def bnlikelihood_test(self):
        val = BNLikelihood().compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        self.assertEqual(type(val), np.float64)

    def bnloglikelihood_test(self):
        val = BNLogLikelihood().compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        self.assertEqual(type(val), np.float64)
    
    def bnloglikelihood_normalization_test(self):
        val = BNLogLikelihood().compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        normalized_score = BNLogLikelihood().normalize(raw_score=val)
        self.assertTrue(normalized_score>0)
        self.assertTrue(normalized_score<1)

    def bnloglikelihood_failed_normalization(self):
        self.assertRaises(ValueError, BNLogLikelihood().normalize, 5)


class gaussian_mixture_test(BaseTestClass):
    def gmlog_likelihood_test(self):
        val = GMLogLikelihood().compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        self.assertEqual(type(val), np.float64)
    
    def gmlog_likelihood_normalization_test(self):
        val = GMLogLikelihood().compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        try:
            normalized_score = BNLogLikelihood().normalize(raw_score=val)
            self.assertTrue(normalized_score>0)
            self.assertTrue(normalized_score<1)
        except Exception as e:
            self.assertEqual(ValueError, type(e))


class cs_test(BaseTestClass):
    def cs_test_compute_test(self):
        val = CSTest.compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        self.assertEqual(type(val), np.float64)
    
    def cs_test_normalization_test(self):
        val = CSTest.compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        normalized_score = CSTest.normalize(raw_score=val)
        self.assertTrue(normalized_score>0.0)
        self.assertTrue(normalized_score<1.0)


class ks_test(BaseTestClass):
    def ks_test_compute_test(self):
        val = KSTest.compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        self.assertEqual(type(val), np.float64)
    
    def ks_test_normalization_test(self):
        val = KSTest.compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        normalized_score = KSTest.normalize(raw_score=val)
        self.assertTrue(normalized_score>0.0)
        self.assertTrue(normalized_score<1.0)


class ks_test_extented(BaseTestClass):
    def ks_test_extended_compute_test(self):
        val = KSTestExtended.compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        self.assertEqual(type(val), np.float64)


# TODO Fix NAN value
class continuous_kldivergence_test(BaseTestClass):
    def continuous_kldivergence_compute_test(self):
        val = ContinuousKLDivergence.compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        self.assertEqual(type(val), np.float64)

# TODO Fix NAN value
class discretekldivergence_test(BaseTestClass):
    def discretekldivergence_compute_test(self):
        val = DiscreteKLDivergence.compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        self.assertEqual(type(val), np.float64)

# Detection Metrics
class sklearn_classifier_detection_test(BaseTestClass):
    def sklearn_classifier_detection_compute_test(self):
        self.assertRaises(NotImplementedError, ScikitLearnClassifierDetectionMetric().compute, self.real_data, self.synthetic_data)


class logistic_detection_test(BaseTestClass):
    def logistic_detection_compute_test(self):
        val = LogisticDetection.compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        self.assertEqual(type(val), np.float64)

class svc_detection_test(BaseTestClass):
    def svc_detection_compute_test(self):
        val = SVCDetection.compute(real_data=self.real_data, synthetic_data=self.synthetic_data)
        self.assertEqual(type(val), np.float64)

#Efficacy Metrics
class binary_decision_tree_classifier_test(BaseTestClass):
    def binary_decision_tree_classifier_compute_test(self):
        val = BinaryDecisionTreeClassifier.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, target='3')
        normalized_score = BinaryDecisionTreeClassifier.normalize(raw_score=val)
        self.assertEqual(type(val), np.float64)
        self.assertEqual(val, normalized_score)
    
    def no_target_test(self):
        self.assertRaises(TypeError, BinaryDecisionTreeClassifier.compute, self.real_data, self.synthetic_data)

# TODO Fix error
class mlefficacy_test(BaseTestClass):
    def mlefficacy_compute_test(self):
        metadata = Metadata()
        metadata.add_table(name="1", data=self.real_data)
        val = MLEfficacy.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, target='2', metadata=metadata)
        self.assertEqual(type(val), np.float64)

class multiclass_test(BaseTestClass):
    def multiclass_compute_test(self):
        val = MulticlassMLPClassifier.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, target='2')
        normalized_score = MulticlassMLPClassifier.normalize(raw_score=val)
        self.assertEqual(type(val), np.float64)
        self.assertEqual(val, normalized_score)


class regression_efficacy_test(BaseTestClass):
    def regression_efficacy_compute_test(self):
        val = LinearRegression.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, target='3')
        normalized_score = LinearRegression.normalize(raw_score=val)
        self.assertEqual(type(val), np.float64)
        self.assertTrue(normalized_score>-np.inf)
        self.assertTrue(normalized_score<1)

# Privacy Metrics

class categorical_cap_test(BaseTestClass):
    def categorical_cap_compute_test(self):
        val = CategoricalCAP.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, key_fields=["2"], sensitive_fields=["2"])
        self.assertEqual(type(val), float)

        model = CAPAttacker()
        model.fit(self.real_data, key_fields=["2"], sensitive_fields=["2"])
        self.assertIsNone(model.predict(key_data=None))
    
    def categorical_cap_key_field_error_test(self):
        self.assertRaises(TypeError, CategoricalCAP.compute, self.real_data, self.synthetic_data)

class categorical_zero_cap_test(BaseTestClass):
    def categorical_zero_cap_compute_test(self):
        val = CategoricalZeroCAP.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, key_fields=["2"], sensitive_fields=["2"])
        self.assertEqual(type(val), float)

class categorical_generalized_cap_test(BaseTestClass):
    def categorical_generalized_cap_compute_test(self):
        val = CategoricalZeroCAP.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, key_fields=["2"], sensitive_fields=["2"])
        self.assertEqual(type(val), float)


class categorical_nb_test(BaseTestClass):
    def categorical_nb_compute_test(self):
        val = CategoricalNB.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, key_fields=["2"], sensitive_fields=["2"])
        self.assertEqual(type(val), float)

class categorical_knn_test(BaseTestClass):
    def categorical_knn_compute_test(self):
        val = CategoricalKNN.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, key_fields=["2"], sensitive_fields=["2"])
        self.assertEqual(type(val), float)

class categorical_rf_test(BaseTestClass):
    def categorical_rf_compute_test(self):
        val = CategoricalRF.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, key_fields=["2"], sensitive_fields=["2"])
        self.assertEqual(type(val), float)

class categorical_svm_test(BaseTestClass):
    def categorical_svm_compute_test(self):
        val = CategoricalSVM.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, key_fields=["2"], sensitive_fields=["2"])
        self.assertEqual(type(val), float)

class numerical_lr_test(BaseTestClass):
    def numerical_lr_compute_test(self):
        val = NumericalLR.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, key_fields=["1", "4"], sensitive_fields=["1", "4"])
        self.assertEqual(type(val), float)

class numerical_mlp_test(BaseTestClass):
    def numerical_mlp_compute_test(self):
        val = NumericalMLP.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, key_fields=["1", "4"], sensitive_fields=["1", "4"])
        self.assertEqual(type(val), float)

class numerical_radius_nn_test(BaseTestClass):
    def numerical_radius_nn_compute_test(self):
        val = NumericalRadiusNearestNeighbor.compute(real_data=self.real_data, synthetic_data=self.synthetic_data, key_fields=["1", "4"], sensitive_fields=["1", "4"])
        self.assertEqual(type(val), float)
