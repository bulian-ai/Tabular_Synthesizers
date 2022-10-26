import unittest
import warnings

from metrics.multi_table_metric_tests import cardinality_shape_similarity_tests, cardinality_statistic_similarity_tests, multi_single_table_metrics_tests, multi_table_detection_metrics_tests
from metrics.single_table_metric_tests import (
    base_metric_test,
    bayesian_network_test,
    binary_decision_tree_classifier_test,
    categorical_generalized_cap_test,
    categorical_knn_test,
    categorical_nb_test,
    categorical_rf_test,
    categorical_svm_test,
    categorical_zero_cap_test,
    categorical_cap_test,
    discretekldivergence_test,
    gaussian_mixture_test,
    cs_test, ks_test,
    ks_test_extented,
    continuous_kldivergence_test,
    logistic_detection_test,
    mlefficacy_test,
    multiclass_test,
    numerical_lr_test,
    numerical_mlp_test,
    numerical_radius_nn_test,
    regression_efficacy_test,
    sklearn_classifier_detection_test,
    svc_detection_test
)
from tabular.twin_synthesizer import PrivateTwinSynthesizerTest, TwinSynthesizerTest

warnings.filterwarnings('ignore')
testsuit = unittest.TestSuite()

def single_table_tests(testsuit):
    testsuit.addTest(base_metric_test("single_table_metric_subclasses_test"))

    testsuit.addTest(bayesian_network_test("bnlikelihood_test"))
    testsuit.addTest(bayesian_network_test("bnloglikelihood_test"))
    testsuit.addTest(bayesian_network_test("bnloglikelihood_normalization_test"))
    testsuit.addTest(bayesian_network_test("bnloglikelihood_failed_normalization"))

    testsuit.addTest(gaussian_mixture_test("gmlog_likelihood_test"))
    testsuit.addTest(gaussian_mixture_test("gmlog_likelihood_normalization_test"))

    testsuit.addTest(cs_test("cs_test_compute_test"))
    testsuit.addTest(cs_test("cs_test_normalization_test"))

    testsuit.addTest(ks_test("ks_test_compute_test"))
    testsuit.addTest(ks_test("ks_test_normalization_test"))

    testsuit.addTest(ks_test_extented("ks_test_extended_compute_test"))

    testsuit.addTest(continuous_kldivergence_test("continuous_kldivergence_compute_test"))
    testsuit.addTest(discretekldivergence_test("discretekldivergence_compute_test"))

    testsuit.addTest(sklearn_classifier_detection_test("sklearn_classifier_detection_compute_test"))

    testsuit.addTest(logistic_detection_test("logistic_detection_compute_test"))
    testsuit.addTest(svc_detection_test("svc_detection_compute_test"))

    testsuit.addTest(binary_decision_tree_classifier_test("binary_decision_tree_classifier_compute_test"))
    testsuit.addTest(binary_decision_tree_classifier_test("no_target_test"))
    #testsuit.addTest(mlefficacy_test("mlefficacy_compute_test"))
    testsuit.addTest(multiclass_test("multiclass_compute_test"))
    testsuit.addTest(regression_efficacy_test("regression_efficacy_compute_test"))

    testsuit.addTest(categorical_cap_test("categorical_cap_compute_test"))
    testsuit.addTest(categorical_cap_test("categorical_cap_key_field_error_test"))
    testsuit.addTest(categorical_zero_cap_test("categorical_zero_cap_compute_test"))
    testsuit.addTest(categorical_generalized_cap_test("categorical_generalized_cap_compute_test"))
    testsuit.addTest(categorical_nb_test("categorical_nb_compute_test"))
    testsuit.addTest(categorical_knn_test("categorical_knn_compute_test"))
    testsuit.addTest(categorical_rf_test("categorical_rf_compute_test"))
    testsuit.addTest(categorical_svm_test("categorical_svm_compute_test"))

    #testsuit.addTest(numerical_lr_test("numerical_lr_compute_test"))
    #testsuit.addTest(numerical_mlp_test("numerical_mlp_compute_test"))
    #testsuit.addTest(numerical_radius_nn_test("numerical_radius_nn_compute_test"))

def multi_table_test(testsuit):
    testsuit.addTest(multi_single_table_metrics_tests('cs_test'))
    testsuit.addTest(multi_single_table_metrics_tests('cs_test_compute_breakdown'))
    #testsuit.addTest(multi_single_table_metrics_tests('bnloglikelihood_normarlize'))

    testsuit.addTest(multi_table_detection_metrics_tests('logistic_parent_child_test'))
    testsuit.addTest(multi_table_detection_metrics_tests('logistic_parent_child_metadata_error_test'))

    testsuit.addTest(cardinality_shape_similarity_tests('cardinality_shape_similarity_compute_test'))
    testsuit.addTest(cardinality_statistic_similarity_tests('cardinality_statistic_similarity_compute_test'))


def twin_synthesizer_tests(testsuit):
    testsuit.addTest(TwinSynthesizerTest('test_fit'))
    #testsuit.addTest(PrivateTwinSynthesizerTest('test_fit'))

single_table_tests(testsuit)
multi_table_test(testsuit)
twin_synthesizer_tests(testsuit)

runner = unittest.TextTestRunner()
runner.run(testsuit)

