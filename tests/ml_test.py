# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing ml module
"""

import os
import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from pyds import constants
from tests import data_generators, tests_constants


class MLTestCase(unittest.TestCase):
    classification_X, classification_y = None, None
    regression_X, regression_y = None, None
    logger = None
    is_warnings_traced = False

    def setUp(self):
        import traceback
        import warnings
        import sys
        import logging.config

        # setting log configuration
        log_conf_path = os.path.abspath(constants.LOGGER_CONFIGURATION_RELATIVE_PATH)
        logging.config.fileConfig(log_conf_path)
        self.logger = logging.getLogger(__name__)

        def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
            traceback.print_stack()

            log = file if hasattr(file, 'write') else sys.stderr
            log.write(warnings.formatwarning(message, category, filename, lineno, line))

        if self.is_warnings_traced:
            warnings.showwarning = warn_with_traceback

        temp_X, temp_y = make_classification(n_features=20, n_redundant=3, n_samples=1000, n_informative=8,
                                             random_state=1, n_clusters_per_class=1, n_classes=4)
        self.classification_X, self.classification_y = pd.DataFrame(temp_X), pd.Series(temp_y)
        # , bias=0.7, tail_strength=0.5, noise=0.05
        temp_X, temp_y = make_regression(n_samples=1000, n_features=20, n_informative=10, n_targets=1)
        self.regression_X, self.regression_y = pd.DataFrame(temp_X), pd.Series(temp_y)

    def test_classify(self):
        from pyds import ml
        self.logger.info("start classify")
        X_train, X_test, y_train, y_test = train_test_split(self.classification_X, self.classification_y,
                                                            test_size=constants.TEST_SPLIT_SIZE)
        best_model, predictions, best_score = ml.classify(X_train, X_test, y_train)
        self.logger.info("best model : \n %s" % best_model)
        self.logger.info("predictions : \n %s" % predictions)
        self.logger.info("best score : \n %s" % best_score)
        self.assertIsNotNone(best_model)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertGreater(best_score, tests_constants.CLASSIFICATION_SCORE_THRESHOLD)

    def test_regress(self):
        from pyds import ml
        X_train, X_test, y_train, y_test = train_test_split(self.regression_X, self.regression_y,
                                                            test_size=constants.TEST_SPLIT_SIZE)
        best_model, predictions, best_score = ml.regress(X_train, X_test, y_train)
        self.assertIsNotNone(best_model)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertGreater(best_score, tests_constants.REGRESSION_SCORE_THRESHOLD)

    def test_create_clusters(self):
        from pyds import ml
        centers = [[1, 1], [-1, -1], [1, -1]]
        densities = [0.2, 0.35, 0.5]
        X, labels_true = data_generators.make_var_density_blobs(n_samples=750, centers=centers, cluster_std=densities)
        clustering_algorithms = ml.create_clusters(X, X.columns.tolist(), n_clusters=3)
        self.assertTrue(clustering_algorithms != set())
        clustering_algorithms = ml.create_clusters(X, X.columns.tolist())
        self.assertTrue(clustering_algorithms != set())

    def test_reduce_dimensions(self):
        from pyds import ml
        df = data_generators.generate_random_data(100, 5)
        reducer_to_results = ml.reduce_dimensions(df)
        self.assertTrue(reducer_to_results != {})
        for reducer, results in reducer_to_results.items():
            self.assertIsInstance(results, pd.DataFrame)
            self.assertGreaterEqual(len(df.columns), len(results.columns))

    def test_detect_anomalies(self):
        from pyds import ml
        centers = [[1, 1], [-1, -1], [1, -1]]
        densities = [0.2, 0.35, 0.5]
        X, labels_true = data_generators.make_var_density_blobs(n_samples=750, centers=centers, cluster_std=densities)
        if_outliers = ml.detect_anomalies_with_isolation_forest(X, contamination=tests_constants.CONTAMINATION)
        hdb_outliers = ml.detect_anomalies_with_hdbscan(X, contamination=tests_constants.CONTAMINATION)
        self.assertAlmostEquals(len(if_outliers), tests_constants.CONTAMINATION * X.shape[0])
        self.assertAlmostEquals(len(hdb_outliers), tests_constants.CONTAMINATION * X.shape[0])


if __name__ == '__main__':
    unittest.main()
